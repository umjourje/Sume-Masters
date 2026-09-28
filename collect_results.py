"""collect_results.py — junta AUC-ROC e taxa de anomalia a partir dos
artefatos JÁ SALVOS (sem recarregar modelo nem dados).

Fontes aceitas (todas opcionais; use as que você tiver):

  --log      log de texto do `flwr run --stream` (ou `flwr log`): extrai,
             por rodada, a linha "Aggregated evaluate MetricRecord".
  --summary  run_summary_<tag>.json do servidor (eval_history por rodada).
  --cm       confusion_matrix_<tag>.json de cada Pi (aceita glob/pasta).
  --npz      scores_pi<N>.npz (y, p) do eval_auc.py -> AUC POOLED exato.
  --auc      auc_<nome>.json já produzido pelo eval_auc.py.

O que cada número significa (importante p/ a dissertação):

  * anomaly_rate do Flower = média de y.mean() dos clientes PONDERADA por
    num-examples (= nº de JANELAS), não por nº de PONTOS. É próxima, mas
    não idêntica, à prevalência real do conjunto de teste.
  * anomaly_rate "pooled" (exata) = soma(TP+FN) / soma(total_pontos) das
    confusion_matrix dos Pis, ou y.mean() das .npz concatenadas.
  * roc_auc do Flower = média ponderada dos AUCs por cliente. AUC não é
    aditivo -> isso NÃO é o AUC do conjunto. O AUC comparável entre
    federado e centralizado é o POOLED (--npz / --auc, via eval_auc.py).
  * acurácia: o Flower NÃO agrega acurácia (task.evaluate não a devolve
    no MetricRecord); ela vem das confusion_matrix de cada Pi, das .npz
    ou do auc_*.json. Pooled = soma(TP+TN) / soma(total_pontos).
    Sempre ao lado da linha de base da classe majoritária
    max(prev, 1-prev): com poucas anomalias, "tudo normal" já acerta muito.

Uso:
    python collect_results.py --log flower_run.log \
        --summary run_summary_full_real.json \
        --cm "metrics/confusion_matrix_*.json" \
        --npz auc_posthoc/fed_full_real --out resultados_fed.json
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

# Formato do repr do MetricRecord varia entre versões do flwr (dict com
# aspas simples/duplas). A regex aceita ambos: 'chave': valor.
_RE_ROUND = re.compile(r"\[ROUND (\d+)/\d+\]")
_RE_AGG = re.compile(r"Aggregated evaluate MetricRecord:\s*(.*)$")
_RE_KV = re.compile(r"['\"]?([\w\-]+)['\"]?\s*:\s*"
                    r"(nan|NaN|inf|-inf|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def _expand(patterns: list[str], suffix: str) -> list[Path]:
    out: list[Path] = []
    for pat in patterns or []:
        p = Path(pat)
        if p.is_dir():
            out += sorted(p.glob(f"*{suffix}"))
        else:
            out += sorted(Path(x) for x in glob.glob(pat))
    return out


def from_log(path: Path) -> list[dict]:
    rows, rnd = [], None
    for line in path.read_text(errors="replace").splitlines():
        m = _RE_ROUND.search(line)
        if m:
            rnd = int(m.group(1))
        m = _RE_AGG.search(line)
        if m:
            kv = {k: float(v) for k, v in _RE_KV.findall(m.group(1))}
            rows.append({"round": rnd, **kv})
    return rows


def from_summary(path: Path) -> list[dict]:
    return json.loads(path.read_text()).get("eval_history", [])


def _base(prev: float) -> float:
    """Acurácia de quem sempre prevê a classe majoritária."""
    return max(prev, 1.0 - prev)


def _mean(vals) -> float | None:
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def from_cms(paths: list[Path]) -> dict:
    per, pos, hit, tot = {}, 0, 0, 0
    for p in paths:
        cm = json.loads(p.read_text())
        tp, tn = int(cm["TP"]), int(cm["TN"])
        fp, fn = int(cm["FP"]), int(cm["FN"])
        n = int(cm.get("total_pontos", tp + tn + fp + fn))
        pos, hit, tot = pos + tp + fn, hit + tp + tn, tot + n
        prev = (tp + fn) / max(n, 1)
        per[p.name] = {"pi": cm.get("pi"), "n_pontos": n,
                       "taxa_anomalia": prev,
                       "accuracy": (tp + tn) / max(n, 1),
                       "accuracy_baseline_majoritaria": _base(prev),
                       "roc_auc": cm.get("roc_auc"),
                       "pr_auc": cm.get("pr_auc")}
    pooled_prev = pos / tot if tot else None
    return {"per_arquivo": per,
            "pooled_taxa_anomalia": pooled_prev,
            "pooled_accuracy": hit / tot if tot else None,
            "pooled_accuracy_baseline": (_base(pooled_prev)
                                         if pooled_prev is not None else None),
            "macro_taxa_anomalia": _mean(v["taxa_anomalia"] for v in per.values()),
            "macro_accuracy": _mean(v["accuracy"] for v in per.values()),
            "macro_roc_auc": _mean(v["roc_auc"] for v in per.values()),
            "n_pontos_total": tot}


def from_npz(paths: list[Path], thr: float) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    ys, ps, per = [], [], {}
    for p in paths:
        d = np.load(p)
        y, s = d["y"].astype(np.uint8), d["p"].astype(np.float32)
        two = np.unique(y).size == 2
        prev = float(y.mean())
        per[p.name] = {"n_pontos": int(y.size), "taxa_anomalia": prev,
                       "accuracy": float(((s >= thr) == y).mean()),
                       "accuracy_baseline_majoritaria": _base(prev),
                       "roc_auc": float(roc_auc_score(y, s)) if two else None}
        ys.append(y); ps.append(s)
    if not ys:
        return {}
    y, s = np.concatenate(ys), np.concatenate(ps)
    two = np.unique(y).size == 2
    prev = float(y.mean())
    return {"per_arquivo": per,
            "threshold": thr,
            "pooled_roc_auc": float(roc_auc_score(y, s)) if two else None,
            "pooled_pr_auc": float(average_precision_score(y, s)) if two else None,
            "pooled_taxa_anomalia": prev,
            "pooled_accuracy": float(((s >= thr) == y).mean()),
            "pooled_accuracy_baseline": _base(prev),
            "macro_accuracy": _mean(v["accuracy"] for v in per.values()),
            "n_pontos_total": int(y.size)}


def _tabela(titulo: str, per: dict) -> None:
    """Uma linha por Rasp (arquivo), p/ ler direto no terminal."""
    print(f"\n{titulo}")
    print(f"  {'arquivo':<34}{'n_pontos':>10}{'taxa_anom':>11}"
          f"{'acc':>9}{'base':>9}{'roc_auc':>9}")
    for nome, v in per.items():
        roc = v.get("roc_auc")
        print(f"  {nome:<34}{v['n_pontos']:>10}{v['taxa_anomalia']:>11.4f}"
              f"{v['accuracy']:>9.4f}{v['accuracy_baseline_majoritaria']:>9.4f}"
              f"{(f'{roc:.4f}' if roc is not None else '—'):>9}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log", type=Path)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--cm", nargs="*", default=[])
    ap.add_argument("--npz", nargs="*", default=[])
    ap.add_argument("--auc", type=Path)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="limiar p/ acurácia a partir das .npz (padrão 0,5, "
                         "o mesmo de task.evaluate)")
    ap.add_argument("--out", type=Path, default=Path("resultados.json"))
    a = ap.parse_args()

    res: dict = {}
    if a.log:
        res["flower_log"] = from_log(a.log)
    if a.summary:
        res["run_summary"] = from_summary(a.summary)
    hist = res.get("flower_log") or res.get("run_summary") or []
    rates = [r["anomaly_rate"] for r in hist if "anomaly_rate" in r]
    if rates:
        # Rótulos são determinísticos e não dependem do modelo -> a taxa
        # deve ser igual em toda rodada; o desvio serve de sanidade.
        res["flower_anomaly_rate"] = {"media_rodadas": float(np.mean(rates)),
                                      "desvio_rodadas": float(np.std(rates)),
                                      "n_rodadas": len(rates)}
        aucs = [r["roc_auc"] for r in hist
                if "roc_auc" in r and np.isfinite(r["roc_auc"])]
        if aucs:
            res["flower_roc_auc_ultima_rodada"] = aucs[-1]
    if a.cm:
        res["confusion_matrices"] = from_cms(_expand(a.cm, ".json"))
    if a.npz:
        res["scores_npz"] = from_npz(_expand(a.npz, ".npz"), a.threshold)
    if a.auc:
        res["eval_auc"] = json.loads(a.auc.read_text())

    a.out.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print("=" * 60)
    if "flower_anomaly_rate" in res:
        f = res["flower_anomaly_rate"]
        print(f"anomaly_rate (Flower, pond. por janelas): "
              f"{f['media_rodadas']:.6f}  (desvio entre rodadas "
              f"{f['desvio_rodadas']:.2e}, {f['n_rodadas']} rodadas)")
    if res.get("confusion_matrices", {}).get("per_arquivo"):
        c = res["confusion_matrices"]
        _tabela("confusion_matrix por Rasp (modelo da ÚLTIMA rodada):",
                c["per_arquivo"])
        print(f"  POOLED  taxa={c['pooled_taxa_anomalia']:.6f}  "
              f"acc={c['pooled_accuracy']:.4f} "
              f"(base {c['pooled_accuracy_baseline']:.4f})  "
              f"n={c['n_pontos_total']}")
        print(f"  MACRO   acc={c['macro_accuracy']:.4f}  "
              f"roc_auc={c['macro_roc_auc']}")
    if res.get("scores_npz"):
        s = res["scores_npz"]
        _tabela(f"scores .npz por Rasp (limiar {s['threshold']}):",
                s["per_arquivo"])
        print(f"  POOLED  roc_auc={s['pooled_roc_auc']}  "
              f"pr_auc={s['pooled_pr_auc']}  "
              f"acc={s['pooled_accuracy']:.4f} "
              f"(base {s['pooled_accuracy_baseline']:.4f})  "
              f"taxa={s['pooled_taxa_anomalia']:.6f}")
    if "eval_auc" in res:
        e = res["eval_auc"]
        acc = e["pooled"].get("accuracy")
        print(f"\neval_auc.py  POOLED roc_auc={e['pooled']['roc_auc']}  "
              f"acc={acc}  | MACRO roc_auc={e['macro']['roc_auc']}  "
              f"acc={e['macro'].get('accuracy')}")
        if acc is None:
            print("  (auc_*.json antigo, sem acurácia: rode eval_auc.py "
                  "--reuse — reaproveita as .npz, não reavalia o modelo)")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()