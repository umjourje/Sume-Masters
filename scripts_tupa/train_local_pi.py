# -*- coding: utf-8 -*-
"""train_local_pi.py — Treino local CENTRALIZADO no Raspberry Pi (ou no
servidor), sobre os shards de UM país (ex.: Espanha/GoiEner) acessados
via NAS, partindo de um checkpoint pré-treinado (v0_both ou v0_real), e
terminando com MATRIZ DE CONFUSÃO da classificação binária de anomalias
sobre o teste do país (rotulado em tempo de execução — desenho anti-leak).

Reusa as peças JÁ VALIDADAS do pipeline (nada de código de treino novo):
  - ShardTemporalDataset + streaming por shard (RAM ~1 shard; o eager NÃO
    cabe: Espanha ≈ 118 GB de shards vs 16 GB de RAM no Pi);
  - run_epoch (mesma perda: alfa/beta dinâmicos + BCE com pos_weight);
  - label_windows_batch (rotulagem intra-janela no teste, em execução).

Uso (rodar DUAS vezes, trocando --init-checkpoint e --tag):
  python train_local_pi.py \
      --windows-root /mnt/nas/EnergyBench-Anomaly/02_windows/Hourly \
      --pi 1 \
      --init-checkpoint /mnt/nas/.../04_models/v0_final/best_model.pth \
      --epochs 30 --max-shards 15 --tag esp_v0both --outdir /mnt/nas/local_runs

Os shards NÃO ficam em subpasta por subconjunto: são arquivos soltos em
<windows_root>/<split>/<Setor>/ nomeados <Subconjunto>.wNN.partKKK.pt.
Por isso a seleção é por GRUPO (via --pi, lendo country_map.py, ou via
--groups manual), nunca por varredura de diretório — que capturaria
todos os países daquele setor.

Saídas em <outdir>/<tag>/: best_local.pth, progress.json,
confusion_matrix.json, confusion_matrix.png, metrics.json.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

# config.py exige RAW_ROOT/OUT_ROOT no ambiente; num Pi de treino local
# esses caminhos não são usados de fato (os dados vêm de --windows-root),
# então definimos valores neutros ANTES de importar módulos do pipeline.
#
# ATENÇÃO — a ÚNICA variável que realmente precisa estar correta no Pi é
# WLSTMIX_DIR, apontando para a pasta do repositório W-LSTMix que contém
# models/W_LSTMix.py (o backbone é importado de lá por model_hybrid.py).
# Sem ela, o script falha no import com mensagem explícita.
# Ex.:  export WLSTMIX_DIR=/mnt/nas/W-LSTMix
os.environ.setdefault("RAW_ROOT", "/tmp/raw_unused")
os.environ.setdefault("OUT_ROOT", "/tmp/out_unused")


def _add_repo_to_path(repo_dir: str | None) -> None:
    aqui = Path(__file__).resolve().parent
    candidatos = ([Path(repo_dir)] if repo_dir else []) + [
        aqui, aqui.parent, aqui.parent / "pipeline",
        aqui.parent.parent / "scripts_tupa", Path.cwd()]
    for c in candidatos:
        if (c / "step6_train.py").exists():
            sys.path.insert(0, str(c))
            return
    raise FileNotFoundError(
        "step6_train.py não encontrado; use --repo-dir apontando para a "
        "pasta scripts_tupa do repositório.")


def _shards_de_grupos(windows_root: Path, split: str,
                      grupos: list[str]) -> list[Path]:
    """Seleciona os shards de subconjuntos ESPECÍFICOS.

    Os shards NÃO ficam em subpasta por subconjunto: são arquivos soltos
    em <windows_root>/<split>/<Setor>/ com o nome do subconjunto como
    prefixo (<Nome>.wNN.partKKK.pt). Portanto, apontar para a pasta
    <Setor> e varrer *.pt pegaria TODOS os países daquele setor — erro
    silencioso e grave num experimento por país.

    O ponto literal depois do nome evita colisão de prefixo (ex.: 'REED'
    não captura 'REEDD'; 'NEST-Commercial' e 'NEST-Residential' já ficam
    em setores distintos).
    """
    achados: list[Path] = []
    for grupo in grupos:                       # "Setor/Nome"
        setor, _, nome = grupo.partition("/")
        d = windows_root / split / setor
        if not d.exists():
            print(f"  [AVISO] pasta inexistente: {d}")
            continue
        f = sorted(d.glob(f"{nome}.w*.part*.pt"))
        if not f:
            print(f"  [AVISO] nenhum shard de '{nome}' em {d}")
        achados += f
    return achados


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-root", type=Path, required=True,
                    help="pasta 02_windows/<resolução>, ex.: "
                         "/mnt/nas/EnergyBench-Anomaly/02_windows/Hourly")
    ap.add_argument("--pi", type=int, choices=[1, 2, 3, 4, 5], default=None,
                    help="usa o balde de países deste nó, conforme "
                         "country_map.py (recomendado)")
    ap.add_argument("--groups", nargs="+", default=None,
                    help="alternativa manual ao --pi: lista de "
                         "'Setor/Subconjunto', ex.: Residential/GoiEner")
    ap.add_argument("--country-map-dir", type=str, default=None,
                    help="pasta do country_map.py, se --pi for usado e o "
                         "arquivo não estiver junto ao script")
    ap.add_argument("--init-checkpoint", type=Path, required=True,
                    help="pesos iniciais (best do v0_both OU do v0_real)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10,
                    help="early stopping sobre a validação intra-shard")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-shards", type=int, default=0,
                    help="0 = todos; use p/ calibrar tempo no Pi "
                         "(ex.: 15 shards ≈ 3M janelas)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("./local_runs"))
    ap.add_argument("--repo-dir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    _add_repo_to_path(a.repo_dir)
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from step6_train import ShardTemporalDataset, run_epoch  # já validados
    from step4_5_labels_v2 import label_windows_batch
    from model_hybrid import HybridWLSTMix
    from config import CFG

    out = a.outdir / a.tag
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)

    # ---------------- modelo + checkpoint inicial (strict) ----------------
    model = HybridWLSTMix(device).to(device)
    sd = torch.load(a.init_checkpoint, map_location=device)
    model.load_state_dict(sd, strict=True)   # divergência falha AQUI, não na época 30
    print(f"[local] checkpoint inicial: {a.init_checkpoint}")

    # ------------- resolução dos grupos (quais subconjuntos usar) -------------
    if a.pi and a.groups:
        raise SystemExit("use --pi OU --groups, não os dois")
    if a.pi:
        aqui = Path(__file__).resolve().parent
        for c in ([Path(a.country_map_dir)] if a.country_map_dir else []) + [
                aqui, aqui.parent, aqui.parent.parent, Path.cwd()]:
            if (c / "country_map.py").exists():
                sys.path.insert(0, str(c))
                break
        else:
            raise FileNotFoundError(
                "country_map.py não encontrado — use --country-map-dir "
                "ou passe os grupos manualmente com --groups")
        from country_map import groups_for_pi, pais_do_pi
        grupos = groups_for_pi(a.pi)
        print(f"[local] Pi{a.pi} -> países {pais_do_pi(a.pi)}")
    elif a.groups:
        grupos = a.groups
    else:
        raise SystemExit("informe --pi N ou --groups Setor/Nome [...]")
    print(f"[local] {len(grupos)} grupo(s): {', '.join(grupos)}")

    train_shards = _shards_de_grupos(a.windows_root, "train", grupos)
    test_shards = _shards_de_grupos(a.windows_root, "test", grupos)
    if not train_shards or not test_shards:
        raise FileNotFoundError(
            f"shards não encontrados (train={len(train_shards)}, "
            f"test={len(test_shards)}) — confira --windows-root e se o "
            f"passo 2-3 rodou para estes grupos em AMBOS os splits")
    if a.max_shards > 0 and len(train_shards) > a.max_shards:
        rng = np.random.default_rng(a.seed)
        idx = sorted(rng.choice(len(train_shards), a.max_shards,
                                replace=False).tolist())
        train_shards = [train_shards[i] for i in idx]
    print(f"[local] {len(train_shards)} shards de treino | "
          f"{len(test_shards)} de teste | device={device}")

    # ---------------- treino em STREAMING por shard ----------------
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    best, wait = float("inf"), 0
    rng = np.random.default_rng(a.seed)

    def _loader(ds, shuffle):
        # workers=0 SEMPRE no streaming (lição do vazamento de /dev/shm):
        # o shard já está em RAM; IPC não traz ganho, só risco.
        return DataLoader(ds, batch_size=CFG.batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=(device.type == "cuda"))

    for ep in range(a.epochs):
        t0 = time.time()
        ordem = rng.permutation(len(train_shards))
        tr_sum = tr_n = va_sum = va_n = 0.0
        for si, k in enumerate(ordem, 1):
            tr_ds = ShardTemporalDataset(train_shards[k], "train", a.val_frac)
            va_ds = ShardTemporalDataset(train_shards[k], "val", a.val_frac)
            if len(tr_ds):
                tr = run_epoch(model, _loader(tr_ds, True), mse, bce,
                               device, scaler, opt)
                tr_sum += tr * len(tr_ds); tr_n += len(tr_ds)
            if len(va_ds):
                va = run_epoch(model, _loader(va_ds, False), mse, bce,
                               device, scaler)
                va_sum += va * len(va_ds); va_n += len(va_ds)
            if si % 5 == 0 or si == len(ordem):
                _atomic_json(out / "progress.json", {
                    "epoca": ep + 1, "epocas_alvo": a.epochs,
                    "shard": si, "shards_totais": len(ordem),
                    "train_parcial": round(tr_sum / max(tr_n, 1), 6),
                    "val_parcial": round(va_sum / max(va_n, 1), 6),
                    "melhor_val": None if best == float("inf") else round(best, 6),
                    "early_stop_wait": f"{wait}/{a.patience}",
                    "seg_epoca": round(time.time() - t0, 1)})
        va_ep = va_sum / max(va_n, 1)
        print(f"[local] época {ep+1}/{a.epochs}: "
              f"train={tr_sum/max(tr_n,1):.4f} val={va_ep:.4f} "
              f"({time.time()-t0:.0f}s)")
        if va_ep < best:
            best, wait = va_ep, 0
            tmp = out / "best_local.pth.tmp"
            torch.save(model.state_dict(), tmp)
            tmp.replace(out / "best_local.pth")
        else:
            wait += 1
            if wait >= a.patience:
                print(f"[local] early stopping na época {ep+1} "
                      f"(paciência {a.patience})")
                break

    model.load_state_dict(torch.load(out / "best_local.pth",
                                     map_location=device))

    # ------- avaliação no TESTE: rótulo em execução + matriz de confusão -------
    model.eval()
    B = CFG.backcast_length
    tp = tn = fp = fn = 0
    with torch.no_grad():
        for sp in test_shards:
            pack = torch.load(sp, map_location="cpu")
            x = pack["x"].numpy()
            trend = pack["trend"].numpy()
            y = label_windows_batch(x, trend)          # rótulo intra-janela
            y_f = torch.tensor(y[:, B:], dtype=torch.float32)
            tn_in = pack["trend_norm"][:, :B].to(device)
            sn_in = pack["season_norm"][:, :B].to(device)
            _, _, logits = model(tn_in, sn_in)
            pred = (torch.sigmoid(logits).cpu() > 0.5).float()
            tp += int(((pred == 1) & (y_f == 1)).sum())
            tn += int(((pred == 0) & (y_f == 0)).sum())
            fp += int(((pred == 1) & (y_f == 0)).sum())
            fn += int(((pred == 0) & (y_f == 1)).sum())

    total = tp + tn + fp + fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    cm = {"tag": a.tag, "init_checkpoint": str(a.init_checkpoint),
          "TP": tp, "TN": tn, "FP": fp, "FN": fn, "total_pontos": total,
          "precision": round(precision, 6), "recall": round(recall, 6),
          "f1": round(f1, 6),
          "accuracy": round((tp + tn) / max(total, 1), 6),
          "taxa_anomalia_teste": round((tp + fn) / max(total, 1), 6)}
    _atomic_json(out / "confusion_matrix.json", cm)
    _atomic_json(out / "metrics.json", {**cm, "melhor_val_treino": best})
    print(json.dumps(cm, indent=2))

    # heatmap simples da matriz (opcional; só precisa de matplotlib)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        m = np.array([[tn, fp], [fn, tp]], dtype=float)
        figc, axc = plt.subplots(figsize=(4.2, 3.6))
        im = axc.imshow(m, cmap="Blues")
        for (i, j), v in np.ndenumerate(m):
            axc.text(j, i, f"{int(v):,}", ha="center", va="center",
                     color="black" if v < m.max() * 0.6 else "white")
        axc.set_xticks([0, 1], ["Normal", "Anomalia"])
        axc.set_yticks([0, 1], ["Normal", "Anomalia"])
        axc.set_xlabel("Predito"); axc.set_ylabel("Real")
        axc.set_title(f"Matriz de confusão — {a.tag}")
        figc.colorbar(im, shrink=0.8)
        figc.tight_layout()
        figc.savefig(out / "confusion_matrix.png", dpi=160)
        plt.close(figc)
        print(f"[local] matriz salva em {out/'confusion_matrix.png'}")
    except ImportError:
        print("[local] matplotlib ausente — só o JSON foi gerado")


if __name__ == "__main__":
    main()