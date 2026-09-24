"""eval_auc.py — AUC-ROC / PR-AUC post-hoc de um checkpoint, por partição e pooled.

Por que existe: o confusion_matrix_<tag>.json guarda TP/TN/FP/FN já
limiarizados em 0,5 — é UM ponto da curva ROC, não dá para reconstruir a
área a partir dele. AUC precisa do score contínuo por ponto. Este script
recarrega um checkpoint (federado OU centralizado) e roda o MESMO
task.evaluate() usado pelos clientes, partição por partição, guardando os
scores; depois calcula:

  * per_pi  — AUC em cada partição (comparável ao que cada Pi vê);
  * pooled  — AUC sobre a UNIÃO dos pontos das partições avaliadas.
              É ESTE o número a comparar entre arquiteturas: AUC não é
              aditivo, então a média de AUCs por cliente (o que o FedAvg
              faz com a métrica) ≠ AUC do conjunto;
  * macro   — média simples dos AUCs por partição (visão "cada cliente
              pesa igual", útil para discutir heterogeneidade Non-IID).

Comparação justa federado × centralizado: rode este script nos DOIS
checkpoints com os MESMOS --pis, --max-shards e --max-windows. Como
task.evaluate() amostra shards de forma determinística (linspace) e rotula
em runtime com funções determinísticas, os dois modelos são avaliados
exatamente nos mesmos pontos.

Uso (no servidor, a partir do diretório do app — o mesmo cwd do preflight):

    python eval_auc.py --ckpt best_model_global_full_real.pth \
        --name fed_full_real --data-root /mnt/.../EnergyBench-Anomaly \
        --max-shards 15

    # centralizado de um Pi específico, mesma avaliação:
    python eval_auc.py --ckpt /caminho/central_pi3/best_model.pth \
        --name central_pi3 --data-root ... --max-shards 15 --pis 3

--reuse: reaproveita scores_pi<N>.npz já calculados (ex.: o script caiu
no pi4), desde que o checkpoint (sha256) e a amostragem sejam os mesmos —
verificado pelo .meta.json ao lado de cada .npz.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

import task


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _aucs(y: np.ndarray, p: np.ndarray) -> tuple[float | None, float | None]:
    """(roc_auc, pr_auc); None se só houver uma classe (AUC indefinido)."""
    if np.unique(y).size < 2:
        return None, None
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, type=Path,
                    help="state_dict (best_model_global.pth, best_model.pth…)")
    ap.add_argument("--name", required=True,
                    help="rótulo do experimento, ex.: fed_full_real, central_pi3")
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--pis", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--max-shards", type=int, required=True,
                    help="MESMO valor do run avaliado (15 nos runs completos)")
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="só para F1/precision/recall de referência")
    ap.add_argument("--out-dir", type=Path, default=Path("auc_posthoc"))
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"[ERRO] checkpoint não existe: {args.ckpt}")
    out = args.out_dir / args.name
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = task.get_model(task.load_config(), device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device),
                          strict=True)
    model.eval()
    sha = _sha256(args.ckpt)
    print(f"[auc] ckpt={args.ckpt} sha256={sha[:12]}… device={device}")
    print(f"[auc] pis={args.pis} max_shards={args.max_shards} "
          f"max_windows={args.max_windows} -> {out}")

    sampling = {"sha256": sha, "max_shards": args.max_shards,
                "max_windows": args.max_windows,
                "data_root": str(args.data_root)}
    per_pi: dict[str, dict] = {}
    ys, ps = [], []
    for pi in args.pis:
        npz = out / f"scores_pi{pi}.npz"
        meta = out / f"scores_pi{pi}.meta.json"
        reuse_ok = False
        if args.reuse and npz.exists() and meta.exists():
            old = json.loads(meta.read_text())
            reuse_ok = all(old.get(k) == v for k, v in sampling.items())
            if not reuse_ok:
                print(f"[auc] pi={pi}: .npz existe mas de outro ckpt/"
                      f"amostragem — recalculando.")
        if reuse_ok:
            print(f"[auc] pi={pi}: reaproveitando {npz.name}")
        else:
            t0 = time.time()
            print(f"[auc] pi={pi}: avaliando…", flush=True)
            task.evaluate(model, args.data_root, device,
                          threshold=args.threshold,
                          tag=f"{args.name}_pi{pi}", pi=pi,
                          max_shards=args.max_shards,
                          max_windows=args.max_windows,
                          metrics_dir=out, round_no=None,
                          scores_out=npz)
            _write_json(meta, {**sampling, "pi": pi,
                               "eval_wall_s": round(time.time() - t0, 1)})
            print(f"[auc] pi={pi}: {time.time() - t0:.0f} s", flush=True)

        d = np.load(npz)
        y, p = d["y"].astype(np.uint8), d["p"].astype(np.float32)
        roc, pr = _aucs(y, p)
        per_pi[str(pi)] = {"roc_auc": roc, "pr_auc": pr,
                           "n_pontos": int(y.size),
                           "taxa_anomalia": float(y.mean())}
        print(f"[auc] pi={pi}: roc_auc={roc} pr_auc={pr} n={y.size}")
        ys.append(y)
        ps.append(p)

    y = np.concatenate(ys)
    p = np.concatenate(ps)
    roc, pr = _aucs(y, p)
    pred = (p >= args.threshold).astype(np.uint8)
    rocs = [v["roc_auc"] for v in per_pi.values() if v["roc_auc"] is not None]
    prs = [v["pr_auc"] for v in per_pi.values() if v["pr_auc"] is not None]

    result = {
        "name": args.name,
        "ckpt": str(args.ckpt),
        "ckpt_sha256": sha,
        "pis": args.pis,
        "max_shards": args.max_shards,
        "max_windows": args.max_windows,
        "pooled": {
            "roc_auc": roc,
            "pr_auc": pr,
            # linha de base do PR-AUC = prevalência (classificador aleatório)
            "pr_auc_baseline": float(y.mean()),
            "n_pontos": int(y.size),
            "threshold": args.threshold,
            "f1": float(f1_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
        },
        "macro": {
            "roc_auc": float(np.mean(rocs)) if rocs else None,
            "pr_auc": float(np.mean(prs)) if prs else None,
            "n_particoes_validas": len(rocs),
        },
        "per_pi": per_pi,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    res_path = out / f"auc_{args.name}.json"
    _write_json(res_path, result)
    print(f"[auc] POOLED roc_auc={roc} pr_auc={pr} "
          f"(baseline PR={y.mean():.4f}) -> {res_path}")


if __name__ == "__main__":
    main()