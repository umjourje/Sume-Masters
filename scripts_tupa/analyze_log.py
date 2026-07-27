"""analyze_log.py — Parser dos logs do pipeline (perf_log) para diagnóstico
de gargalos e do que aconteceu numa execução.

Lê um arquivo de log (ou o mais recente de <OUT_ROOT>/logs/) e resume:
  * linha do tempo dos eventos principais (grupos, shards, épocas);
  * trajetória de recursos (RSS, RAM disponível, load, disco) — em
    particular, se a RAM estava subindo monotonicamente (indício de
    estouro iminente) e onde a execução parou;
  * tempos por fase quando presentes.

Uso:
    python analyze_log.py                    # analisa o log mais recente
    python analyze_log.py <arquivo.log>
    python analyze_log.py --step step6_train # mais recente daquele passo
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from config import CFG
    _LOGS = CFG.out_root / "logs"
except Exception:
    _LOGS = None

_RE_RES = re.compile(
    r"\[recursos @ (?P<ctx>[^\]]+)\].*?RSS=(?P<rss>[\d.]+)(?P<ru>[KMGT]?B)"
    r".*?RAM_disp=(?P<ram>[\d.]+)(?P<rmu>[KMGT]?B)"
    r".*?load=(?P<load>[\d./?]+).*?disco_livre=(?P<disk>[\d.]+)(?P<du>[KMGT]?B)"
    r"(?:.*?t\+(?P<t>[\dhms.]+))?")
_UNIT = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30, "TB": 2**40}


def _to_gb(v: str, u: str) -> float:
    return float(v) * _UNIT.get(u, 1) / 2**30


def analyze(path: Path) -> None:
    lines = path.read_text(errors="replace").splitlines()
    print(f"# Análise de {path}")
    print(f"# {len(lines)} linhas\n")

    # 1) Cabeçalho e término
    header = [l for l in lines if l.startswith("#")]
    for h in header:
        print(h)
    finished = any("# fim:" in l for l in lines)
    print(f"\n[status] execução {'CONCLUÍDA' if finished else 'INTERROMPIDA '
          '(sem linha de término — possível kill/OOM ou ainda em curso)'}\n")

    # 2) Trajetória de recursos
    res = []
    for l in lines:
        m = _RE_RES.search(l)
        if m:
            res.append((m["ctx"], _to_gb(m["rss"], m["ru"]),
                        _to_gb(m["ram"], m["rmu"]), m["load"],
                        _to_gb(m["disk"], m["du"]), m["t"] or "?"))
    if res:
        print("[recursos] evolução (RSS = memória do processo):")
        print(f"  {'contexto':<28}{'RSS(G)':>9}{'RAM_disp(G)':>13}"
              f"{'load':>10}{'disco(G)':>10}{'t+':>10}")
        for ctx, rss, ram, load, disk, t in res:
            print(f"  {ctx[:28]:<28} {rss:>8.1f} {ram:>11.1f} "
                  f"{load:>10} {disk:>9.0f} {t:>9}")
        rss0, rssN = res[0][1], res[-1][1]
        ramN = res[-1][2]
        print(f"\n  RSS: {rss0:.1f}G -> {rssN:.1f}G "
              f"(Δ {rssN - rss0:+.1f}G)")
        if len(res) >= 3 and all(res[i][1] <= res[i+1][1]
                                 for i in range(len(res)-1)):
            print("  [!] RSS monotonicamente CRESCENTE — padrão de "
                  "acúmulo/estouro de memória.")
        if ramN < 5:
            print(f"  [!] RAM disponível no fim: {ramN:.1f}G — perto do "
                  f"esgotamento (risco de OOM kill).")
    else:
        print("[recursos] nenhum snapshot [recursos] encontrado — o processo "
              "pode ter morrido ANTES do primeiro snapshot (ex.: durante o "
              "carregamento de dados, antes do laço instrumentado).")

    # 3) Eventos de treino / grupos / shards
    def _tail(pred, label, n=8):
        hits = [l for l in lines if pred(l)]
        if hits:
            print(f"\n[{label}] {len(hits)} evento(s); últimos:")
            for l in hits[-n:]:
                print("  " + l.strip()[:110])

    _tail(lambda l: "fold" in l and "ep" in l, "épocas")
    _tail(lambda l: "estágio" in l or "fonte '" in l or "combinado" in l,
          "carregamento/estágios")
    _tail(lambda l: "shard" in l.lower() and "grav" in l.lower(), "shards")
    _tail(lambda l: "FIM" in l or "interrompid" in l.lower(), "conclusão")

    # 4) Última linha não-vazia (onde parou)
    last = next((l for l in reversed(lines) if l.strip()), "")
    print(f"\n[última linha do log] {last.strip()[:120]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", nargs="?", type=Path)
    ap.add_argument("--step", type=str, default=None)
    a = ap.parse_args()
    if a.logfile:
        path = a.logfile
    elif _LOGS and _LOGS.exists():
        pats = f"{a.step}_*.log" if a.step else "*.log"
        logs = sorted(_LOGS.glob(pats), key=lambda p: p.stat().st_mtime)
        if not logs:
            sys.exit(f"Nenhum log em {_LOGS} (padrão {pats})")
        path = logs[-1]
    else:
        sys.exit("Informe o arquivo de log ou configure OUT_ROOT.")
    analyze(path)


if __name__ == "__main__":
    main()