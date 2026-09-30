"""montar_resultados.py — monta (e CONFERE) o dicionário `data_input` do
plot_matriz_confusao.py a partir das avaliações post-hoc do eval_auc.py.

NÃO avalia modelo nenhum: só lê o que o eval_auc.py já gravou em cada
pasta de avaliação (<out-dir>/<name>/):

    scores_pi<N>.npz             y (rótulo fundido por ponto) e p (score)
    scores_pi<N>.meta.json       sha256 do ckpt + amostragem
    confusion_matrix_<name>_pi<N>.json   gravado por task.evaluate (float64)
    auc_<name>.json              resumo pooled/macro/per_pi do eval_auc.py

Para cada cenário recalcula do zero, a partir das .npz, a matriz POOLED
(soma dos pontos de todas as partições), recall, acurácia e AUC-ROC, e
confere contra as outras duas fontes independentes:

  [A] soma de TP/TN/FP/FN dos confusion_matrix_<name>_pi<N>.json
      (task.evaluate limiariza p em float64; a .npz guarda p em float32 —
      um ponto com p a ~1e-7 de 0,5 pode cair do outro lado; diferença de
      0 ou poucos pontos é esperada, mais que isso NÃO é);
  [B] auc_<name>.json (pooled.roc_auc, recall, accuracy);
  [C] ENTRE cenários: mesmas partições, mesmo max_shards/max_windows e
      vetor de rótulos y BYTE-IDÊNTICO — prova que os 4 modelos foram
      avaliados exatamente nos mesmos pontos (a comparação é válida);
  [D] opcional (--fed-cm): confronta com os confusion_matrix_<TAG>.json
      que os Pis gravaram durante o run federado (trazidos pelo
      `collect`). Eles refletem o modelo da ÚLTIMA rodada avaliada; só
      batem com o best_model_global se a melhor rodada foi a última.

Cada --cen aceita DOIS tipos de pasta:
  * a pasta de avaliação do eval_auc.py (<out-dir>/<name>/, a que contém
    scores_pi<N>.npz) — caso dos checkpoints federados;
  * a pasta de um run do train_local_pi.py (<outdir>/<tag>/). O script
    acha sozinho <outdir>/<tag>/auc/<tag>/ e, do confusion_matrix.json do
    run, copia os metadados de treino (wall_time, épocas, CPU/RAM,
    init_checkpoint) — NUNCA as contagens legadas, que não são
    comparáveis. Confere também que cen1/cen3 partiram do v0 certo.

--extra cenN=arquivo.json acrescenta os mesmos metadados (se existirem)
a partir de qualquer json, ex.: o summary_<TAG>.json de um Pi federado.

Uso (no servidor, com o mesmo PYTHON_SERVER do smoke_test_fed.sh):

    python montar_resultados.py \
        --cen cen1=runs_central/real/pi1_esp_v0real \
        --cen cen2=metrics_full_real/auc/full_real \
        --cen cen3=runs_central/real_synth/pi1_esp_v0both \
        --cen cen4=metrics_full_both/auc/full_both \
        --fed-cm "cen2=metrics_full_real/pi*/confusion_matrix_full_real.json" \
        --fed-cm "cen4=metrics_full_both/pi*/confusion_matrix_full_both.json" \
        --out resultados_matrizes.json

    # só a partição da Espanha (Pi1), p/ a visão "terreno do centralizado":
    python montar_resultados.py ... --pis 1 --out resultados_matrizes_pi1.json

Saída: o dicionário `data_input` como literal Python pronto para colar no
plot_matriz_confusao.py (stdout) e o mesmo conteúdo + o relatório de
conferência em --out. Sai com código 1 se houver qualquer [FALHA].
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import pprint
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

TITULOS = {
    "cen1": "Matriz de Confusão - Cenário I\nCentralizado / Real",
    "cen2": "Matriz de Confusão - Cenário II\nFederado / Real",
    "cen3": "Matriz de Confusão - Cenário III\nCentralizado / Real + Sintético",
    "cen4": "Matriz de Confusão - Cenário IV\nFederado / Real + Sintético",
}
# v0 esperado de cada cenário centralizado (substring do init_checkpoint)
V0_ESPERADO = {"cen1": "v0_real", "cen3": "v0_final"}
# metadados de execução copiados para o dict (o plot não os usa; ficam
# para tabelas de custo). Contagens/métricas NUNCA vêm daqui.
META_KEYS = ("init_checkpoint", "wall_time_s", "wall_time_treino_s",
             "epocas_executadas", "val_mode", "cpu_pct_avg", "load1_avg",
             "ram_used_gb_avg", "ram_used_gb_max", "ram_total_gb")
TOL = 1e-9          # tolerância p/ comparar floats recalculados
TOL_PONTOS = 10     # diferença aceitável em contagens (efeito float32 x float64)
_RE_PI = re.compile(r"scores_pi(\d+)\.npz$")


def _counts(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = p >= thr
    pos = y == 1
    return {"TP": int(np.sum(pos & pred)), "TN": int(np.sum(~pos & ~pred)),
            "FP": int(np.sum(~pos & pred)), "FN": int(np.sum(pos & ~pred))}


def _metricas(c: dict) -> dict:
    tp, tn, fp, fn = c["TP"], c["TN"], c["FP"], c["FN"]
    n = tp + tn + fp + fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    prev = (tp + fn) / n if n else 0.0
    return {"total_pontos": n,
            "accuracy": (tp + tn) / n if n else 0.0,
            "recall": rec,
            "precision": prec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "taxa_anomalia_teste": prev,
            "accuracy_baseline_majoritaria": max(prev, 1 - prev)}


class Relatorio:
    def __init__(self) -> None:
        self.linhas: list[str] = []
        self.falhas = 0

    def ok(self, msg: str) -> None:
        self.linhas.append(f"  [OK]    {msg}")

    def aviso(self, msg: str) -> None:
        self.linhas.append(f"  [AVISO] {msg}")

    def falha(self, msg: str) -> None:
        self.falhas += 1
        self.linhas.append(f"  [FALHA] {msg}")

    def titulo(self, msg: str) -> None:
        self.linhas.append(f"\n== {msg}")


def avaliar_cenario(cen: str, pasta: Path, pis_filtro: list[int] | None,
                    thr: float, rel: Relatorio) -> dict:
    rel.titulo(f"{cen}: {pasta}")
    if not pasta.is_dir():
        rel.falha(f"pasta inexistente: {pasta}")
        return {}
    npzs = {int(_RE_PI.search(p.name).group(1)): p
            for p in pasta.glob("scores_pi*.npz") if _RE_PI.search(p.name)}
    if pis_filtro:
        faltam = [pi for pi in pis_filtro if pi not in npzs]
        if faltam:
            rel.falha(f"sem scores_pi<N>.npz para pi(s) {faltam}")
        npzs = {pi: npzs[pi] for pi in pis_filtro if pi in npzs}
    if not npzs:
        rel.falha("nenhuma .npz encontrada")
        return {}
    pis = sorted(npzs)

    # --- metadados: mesmo checkpoint e mesma amostragem em todas as partições
    metas = {}
    for pi in pis:
        mp = pasta / f"scores_pi{pi}.meta.json"
        metas[pi] = json.loads(mp.read_text()) if mp.exists() else {}
        if not metas[pi]:
            rel.aviso(f"pi{pi}: {mp.name} ausente — não dá p/ provar o ckpt")
    shas = {m.get("sha256") for m in metas.values() if m}
    amostr = {(m.get("max_shards"), m.get("max_windows")) for m in metas.values() if m}
    if len(shas) > 1:
        rel.falha(f"partições avaliadas com checkpoints DIFERENTES: {shas}")
    elif shas:
        rel.ok(f"mesmo ckpt em {len(pis)} partição(ões): sha256={next(iter(shas))[:12]}…")
    if len(amostr) > 1:
        rel.falha(f"amostragens diferentes entre partições: {amostr}")

    # --- recálculo a partir das .npz ---
    ys, ps, per_pi = [], [], {}
    soma_cm = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    tem_cm_json = True
    for pi in pis:
        d = np.load(npzs[pi])
        y, p = d["y"].astype(np.uint8), d["p"].astype(np.float32)
        c = _counts(y, p, thr)
        roc = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None
        per_pi[pi] = {**c, "roc_auc": roc, "n": int(y.size)}
        ys.append(y); ps.append(p)

        # [A] confusion_matrix gravado pelo task.evaluate dentro do eval_auc
        cands = sorted(pasta.glob(f"confusion_matrix_*_pi{pi}.json"))
        if not cands:
            tem_cm_json = False
            rel.aviso(f"pi{pi}: confusion_matrix_*_pi{pi}.json ausente — "
                      f"fonte [A] indisponível (ex.: .npz reaproveitada)")
            continue
        cmj = json.loads(cands[-1].read_text())
        dif = {k: int(cmj[k]) - c[k] for k in c}
        maxd = max(abs(v) for v in dif.values())
        if maxd == 0:
            rel.ok(f"pi{pi}: TP/TN/FP/FN da .npz == {cands[-1].name}")
        elif maxd <= TOL_PONTOS:
            rel.aviso(f"pi{pi}: diferença de {maxd} ponto(s) vs {cands[-1].name} "
                      f"(float32 x float64 no limiar) — desprezível: {dif}")
        else:
            rel.falha(f"pi{pi}: contagens NÃO batem com {cands[-1].name}: {dif}")
        if cmj.get("roc_auc") is not None and roc is not None:
            if abs(cmj["roc_auc"] - roc) > 1e-5:   # json arredonda em 6 casas
                rel.falha(f"pi{pi}: roc_auc {cmj['roc_auc']} (json) x {roc:.6f} (npz)")
        for k in soma_cm:
            soma_cm[k] += int(cmj[k])

    y = np.concatenate(ys); p = np.concatenate(ps)
    c = _counts(y, p, thr)
    m = _metricas(c)
    roc = float(roc_auc_score(y, p))
    pr = float(average_precision_score(y, p))
    rel.ok(f"POOLED recalculado: n={m['total_pontos']:,} TP={c['TP']:,} "
           f"TN={c['TN']:,} FP={c['FP']:,} FN={c['FN']:,}")
    rel.ok(f"recall = TP/(TP+FN) = {c['TP']:,}/{c['TP'] + c['FN']:,} = {m['recall']:.6f}")
    rel.ok(f"acc = (TP+TN)/n = {m['accuracy']:.6f} "
           f"(base classe majoritária {m['accuracy_baseline_majoritaria']:.6f})")
    rel.ok(f"AUC-ROC pooled = {roc:.6f} | PR-AUC = {pr:.6f} "
           f"(base PR = prevalência {m['taxa_anomalia_teste']:.6f})")
    if tem_cm_json:
        dif = {k: soma_cm[k] - c[k] for k in c}
        maxd = max(abs(v) for v in dif.values())
        (rel.ok if maxd == 0 else rel.aviso if maxd <= TOL_PONTOS * len(pis)
         else rel.falha)(f"[A] soma dos confusion_matrix por Pi x pooled: dif={dif}")

    # [B] auc_<name>.json
    aucs = sorted(pasta.glob("auc_*.json"))
    if not aucs:
        rel.aviso("auc_<name>.json ausente — fonte [B] indisponível")
    else:
        a = json.loads(aucs[-1].read_text())
        if sorted(a.get("pis", [])) != pis:
            rel.aviso(f"[B] {aucs[-1].name} foi gerado com pis={a.get('pis')}, "
                      f"aqui usei {pis} — pooled dele NÃO é comparável a este; "
                      f"conferindo só per_pi")
        else:
            pa = a["pooled"]
            for k, v in (("roc_auc", roc), ("recall", m["recall"]),
                         ("accuracy", m["accuracy"])):
                if pa.get(k) is None:
                    rel.aviso(f"[B] {k} ausente em {aucs[-1].name} (json antigo; "
                              f"rode eval_auc.py --reuse)")
                elif abs(pa[k] - v) > TOL:
                    rel.falha(f"[B] {k}: {pa[k]} (auc json) x {v} (recalculado)")
                else:
                    rel.ok(f"[B] {k} == {aucs[-1].name}")
        for pi in pis:
            r = a.get("per_pi", {}).get(str(pi), {}).get("roc_auc")
            if r is not None and per_pi[pi]["roc_auc"] is not None and \
                    abs(r - per_pi[pi]["roc_auc"]) > TOL:
                rel.falha(f"[B] pi{pi} roc_auc {r} x {per_pi[pi]['roc_auc']}")

    meta0 = next((mm for mm in metas.values() if mm), {})
    return {
        "_y_hash": hashlib.sha256(y.tobytes()).hexdigest(),
        "_per_pi": per_pi,
        "dict": {
            "titulo": TITULOS.get(cen, cen),
            "tag": pasta.name,
            **c,
            "total_pontos": m["total_pontos"],
            "accuracy": round(m["accuracy"], 6),
            "recall": round(m["recall"], 6),
            "aucroc": round(roc, 6),
            "precision": round(m["precision"], 6),
            "f1": round(m["f1"], 6),
            "pr_auc": round(pr, 6),
            "taxa_anomalia_teste": round(m["taxa_anomalia_teste"], 6),
            "accuracy_baseline_majoritaria": round(m["accuracy_baseline_majoritaria"], 6),
            "threshold": thr,
            "pis": pis,
            "max_shards": meta0.get("max_shards"),
            "max_windows": meta0.get("max_windows"),
            "ckpt_sha256": meta0.get("sha256"),
        },
    }


def conferir_entre_cenarios(res: dict, rel: Relatorio) -> None:
    rel.titulo("[C] comparabilidade ENTRE cenários")
    ok = {k: v for k, v in res.items() if v}
    if len(ok) < 2:
        rel.aviso("menos de 2 cenários válidos — nada a comparar")
        return
    for campo in ("pis", "max_shards", "max_windows", "total_pontos"):
        vals = {k: str(v["dict"][campo]) for k, v in ok.items()}
        if len(set(vals.values())) == 1:
            rel.ok(f"{campo} idêntico: {next(iter(vals.values()))}")
        else:
            rel.falha(f"{campo} DIFERE entre cenários: {vals}")
    hashes = {k: v["_y_hash"][:12] for k, v in ok.items()}
    if len(set(hashes.values())) == 1:
        rel.ok(f"vetor de rótulos y byte-idêntico nos {len(ok)} cenários "
               f"(sha256 {next(iter(hashes.values()))}…) -> mesmos pontos avaliados")
    else:
        rel.falha(f"rótulos y DIFEREM entre cenários: {hashes} — avaliações "
                  f"em pontos diferentes, comparação inválida")
    shas = {k: (v["dict"]["ckpt_sha256"] or "?")[:12] for k, v in ok.items()}
    if len(set(shas.values())) < len(shas):
        rel.falha(f"dois cenários com o MESMO checkpoint: {shas}")
    else:
        rel.ok(f"checkpoints distintos: {shas}")


def conferir_fed_cm(res: dict, specs: list[str], rel: Relatorio) -> None:
    for spec in specs:
        cen, _, pat = spec.partition("=")
        rel.titulo(f"[D] {cen}: confusion_matrix do run federado ({pat})")
        if not res.get(cen):
            rel.aviso(f"{cen} sem avaliação válida — pulando")
            continue
        per_pi = res[cen]["_per_pi"]
        arqs = sorted(glob.glob(pat))
        if not arqs:
            rel.aviso("nenhum arquivo casou com o padrão")
            continue
        for a in arqs:
            cm = json.loads(Path(a).read_text())
            pi = int(cm.get("pi", 0))
            if pi not in per_pi:
                continue
            if (cm.get("max_shards"), cm.get("max_windows")) != (
                    res[cen]["dict"]["max_shards"], res[cen]["dict"]["max_windows"]):
                rel.aviso(f"pi{pi}: amostragem do run ({cm.get('max_shards')},"
                          f"{cm.get('max_windows')}) difere da avaliação — sem comparação")
                continue
            dif = {k: int(cm[k]) - per_pi[pi][k] for k in ("TP", "TN", "FP", "FN")}
            if max(abs(v) for v in dif.values()) <= TOL_PONTOS:
                rel.ok(f"pi{pi}: run federado == best_model_global (melhor rodada "
                       f"= última avaliada)")
            else:
                rel.aviso(f"pi{pi}: run federado != best_model_global, dif={dif}. "
                          f"Esperado se a melhor rodada NÃO foi a última (o json do "
                          f"Pi é sobrescrito a cada rodada). Confira em "
                          f"run_summary_<tag>.json qual rodada teve o menor test_loss.")
            if cm.get("n_shards") is not None and cm.get("total_pontos") != per_pi[pi]["n"]:
                rel.aviso(f"pi{pi}: total_pontos {cm.get('total_pontos')} (run) x "
                          f"{per_pi[pi]['n']} (avaliação) — pontos diferentes!")


def _meta_de(path: Path) -> dict:
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    return {k: d[k] for k in META_KEYS if d.get(k) is not None}


def _sha256_arquivo(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolver_pasta(pasta: Path) -> tuple[Path, Path | None]:
    """(pasta_de_avaliação, pasta_do_run_train_local_pi | None)."""
    if any(pasta.glob("scores_pi*.npz")):
        return pasta, None
    auc = pasta / "auc"
    if auc.is_dir():
        cand = auc / pasta.name
        if not any(cand.glob("scores_pi*.npz")):
            subs = [d for d in auc.iterdir()
                    if d.is_dir() and any(d.glob("scores_pi*.npz"))]
            cand = subs[0] if len(subs) == 1 else None
        if cand is not None:
            return cand, pasta
    return pasta, None


def anexar_run_central(cen: str, info: dict, run_dir: Path,
                       rel: Relatorio) -> None:
    """Metadados do train_local_pi.py + provas de que a avaliação é do
    checkpoint desse run e que ele partiu do v0 certo."""
    rel.titulo(f"{cen}: run do train_local_pi.py em {run_dir}")
    d = info["dict"]
    meta = _meta_de(run_dir / "confusion_matrix.json")
    if not meta:
        rel.aviso("confusion_matrix.json do run ausente — sem metadados de treino")
    for k, v in meta.items():
        d.setdefault(k, v)
    best = run_dir / "best_local.pth"
    if best.exists():
        sha = _sha256_arquivo(best)
        if sha == d.get("ckpt_sha256"):
            rel.ok(f"avaliação é do best_local.pth deste run (sha256 {sha[:12]}…)")
        else:
            rel.falha(f"best_local.pth do run (sha256 {sha[:12]}…) != checkpoint "
                      f"avaliado ({str(d.get('ckpt_sha256'))[:12]}…) — o run foi "
                      f"refeito depois da avaliação? Rode de novo com --eval-only")
    else:
        rel.aviso("best_local.pth ausente na pasta do run — não dá p/ provar "
                  "qual checkpoint foi avaliado além do sha256 do meta")
    esp = V0_ESPERADO.get(cen)
    ini = str(d.get("init_checkpoint", ""))
    if esp and ini:
        (rel.ok if esp in ini else rel.falha)(
            f"init_checkpoint = {ini} (esperado conter '{esp}')")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cen", action="append", required=True,
                    help="cenN=PASTA (pasta <out-dir>/<name> do eval_auc.py)")
    ap.add_argument("--pis", type=int, nargs="+", default=None,
                    help="restringe o pooled a estas partições (padrão: todas "
                         "as .npz presentes)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--fed-cm", action="append", default=[],
                    help='cenN=GLOB dos confusion_matrix_<TAG>.json do collect')
    ap.add_argument("--extra", action="append", default=[],
                    help="cenN=arquivo.json: copia metadados de execução "
                         f"({', '.join(META_KEYS[1:4])}…) se existirem")
    ap.add_argument("--out", type=Path, default=Path("resultados_matrizes.json"))
    a = ap.parse_args()

    rel = Relatorio()
    res: dict = {}
    for spec in a.cen:
        cen, _, pasta = spec.partition("=")
        pasta_eval, run_dir = resolver_pasta(Path(pasta).expanduser())
        res[cen] = avaliar_cenario(cen, pasta_eval, a.pis, a.threshold, rel)
        if res[cen] and run_dir is not None:
            res[cen]["dict"]["tag"] = run_dir.name
            anexar_run_central(cen, res[cen], run_dir, rel)
        elif cen in V0_ESPERADO and res[cen]:
            rel.aviso(f"{cen} é centralizado mas foi passado como pasta de "
                      f"avaliação solta — passe a pasta do run "
                      f"(<outdir>/<tag>) para conferir o v0 e o checkpoint")
    for spec in a.extra:
        cen, _, arq = spec.partition("=")
        if res.get(cen):
            for k, v in _meta_de(Path(arq)).items():
                res[cen]["dict"].setdefault(k, v)
    conferir_entre_cenarios(res, rel)
    conferir_fed_cm(res, a.fed_cm, rel)

    data_input = {k: (v["dict"] if v else {}) for k, v in res.items()}
    a.out.write_text(json.dumps({"data_input": data_input,
                                 "conferencia": rel.linhas,
                                 "falhas": rel.falhas},
                                indent=2, ensure_ascii=False))
    print("\n".join(rel.linhas))
    print(f"\n{'=' * 70}\n{rel.falhas} falha(s). Relatório completo em {a.out}")
    if rel.falhas:
        print("NÃO use o dicionário abaixo antes de resolver as falhas.")
    # literal PYTHON (não JSON): None/True em vez de null/true, cola direto
    print("\n# ---- cole em plot_matriz_confusao.py ----")
    print("data_input = " + pprint.pformat(data_input, sort_dicts=False,
                                           width=100))
    sys.exit(1 if rel.falhas else 0)


if __name__ == "__main__":
    main()