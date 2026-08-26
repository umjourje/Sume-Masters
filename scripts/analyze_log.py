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


def _parse_t(t: str) -> float | None:
    """Converte 't+1h03m12s' / '5m02s' / '27.0s' em segundos (float)."""
    if not t or t == "?":
        return None
    m = re.match(r"(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", t)
    if not m or not any(m.groups()):
        return None
    h, mi, s = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mi * 60 + s


_RE_PROGRESS = re.compile(
    r"\[step6:(?P<stage>\w+)\] ep (?P<ep>\d+)/(?P<epn>\d+) "
    r"shard (?P<sh>\d+)/(?P<shn>\d+) \| train=[\d.]+ val=[\d.]+ "
    r"\((?P<el>[^,]+),")

_RE_FOLD = re.compile(
    r"\[step6:(?P<stage>\w+)\] fold (?P<fold>\d+)/(?P<foldn>\d+) "
    r"ep (?P<ep>\d+): train=[\d.]+ val=[\d.]+ \((?P<el>[^)]+)\)")


def _estimate_eta_folds(lines) -> bool:
    """Projeta o tempo restante para o formato 'fold F/N ep E: ... (dur)'
    (rolling-origin, usado no fine-tuning). O 'dur' impresso é a duração
    DAQUELA ÉPOCA (não cumulativo — t_e é resetado a cada época no
    código), então a duração de um fold completo é a SOMA das durações
    de suas épocas — abordagem robusta ao early stopping (não presume
    nº fixo de épocas por fold). Retorna True se encontrou dados."""
    hits = []
    for l in lines:
        m = _RE_FOLD.search(l)
        if m:
            el = _parse_t(m["el"])
            if el is not None:
                hits.append({"stage": m["stage"], "fold": int(m["fold"]),
                            "foldn": int(m["foldn"]), "ep": int(m["ep"]),
                            "el": el})
    if not hits:
        return False
    stage, foldn = hits[-1]["stage"], hits[-1]["foldn"]
    by_fold = {}
    for h in hits:
        by_fold.setdefault(h["fold"], []).append(h["el"])
    folds_sorted = sorted(by_fold)
    cur_fold = folds_sorted[-1]
    completed = [sum(by_fold[f]) for f in folds_sorted[:-1]]  # folds fechados
    elapsed_cur = sum(by_fold[cur_fold])
    avg_fold = sum(completed) / len(completed) if completed else elapsed_cur
    remaining_cur = max(0.0, avg_fold - elapsed_cur)
    remaining_folds_after = max(0, foldn - cur_fold)
    eta = remaining_cur + remaining_folds_after * avg_fold

    print(f"\n[ETA — estágio '{stage}'] fold atual: {cur_fold}/{foldn} "
          f"(época {hits[-1]['ep']} em andamento)")
    if completed:
        print(f"  duração média de fold(s) concluído(s) "
              f"({len(completed)}): {_fmt_hms(avg_fold)}")
    print(f"  tempo decorrido no fold atual: {_fmt_hms(elapsed_cur)}")
    print(f"  ETA para o FIM do estágio '{stage}': "
          f"~{_fmt_hms(eta)} a partir de agora")
    print(f"  [!] Estimativa aproximada: o nº de épocas por fold varia "
          f"com o early stopping, então usa-se a duração TOTAL média dos "
          f"folds já concluídos como referência, não um nº fixo de "
          f"épocas.")
    return True



def _fmt_hms(seconds: float) -> str:
    if seconds < 0:
        return "?"
    d, rem = divmod(int(seconds), 86400)
    h, rem = divmod(rem, 3600)
    m, _ = divmod(rem, 60)
    if d:
        return f"{d}d{h:02d}h{m:02d}m"
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m"


def _estimate_eta(lines) -> None:
    """Projeta o tempo restante a partir das linhas de progresso do
    streaming ('ep E/N shard S/T | ... (elapsed, ...'). Usa a duração de
    épocas JÁ CONCLUÍDAS (quando houver) para uma média real; a época em
    curso usa a taxa observada (shards processados / tempo decorrido)
    extrapolada para o que falta. Cobre apenas o ESTÁGIO ATUAL do log —
    um estágio seguinte (ex.: fine-tuning após pré-treino) não tem como
    ser projetado a partir daqui, e isso é dito explicitamente."""
    hits = []
    for l in lines:
        m = _RE_PROGRESS.search(l)
        if m:
            el = _parse_t(m["el"])
            if el is not None:
                hits.append({"stage": m["stage"], "ep": int(m["ep"]),
                            "epn": int(m["epn"]), "sh": int(m["sh"]),
                            "shn": int(m["shn"]), "elapsed": el})
    if not hits:
        return
    stage, epn, shn = hits[-1]["stage"], hits[-1]["epn"], hits[-1]["shn"]
    by_ep = {}
    for h in hits:
        by_ep.setdefault(h["ep"], []).append(h)
    eps_sorted = sorted(by_ep)
    cur_ep = eps_sorted[-1]
    cur_pts = by_ep[cur_ep]

    # Duração de épocas CONCLUÍDAS (todo ep < cur_ep tem, por definição,
    # terminado — sua última amostra registrada é o fim daquela época).
    completed_durs = [by_ep[e][-1]["elapsed"] for e in eps_sorted[:-1]]

    # Taxa da época EM CURSO: regressão linear shard~elapsed nos pontos
    # observados (robusta a espaçamento irregular entre snapshots).
    xs = [p["elapsed"] for p in cur_pts]
    ys = [p["sh"] for p in cur_pts]
    if len(xs) >= 2:
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs) or 1e-9
        rate = num / den                                  # shards/segundo
    else:
        rate = ys[-1] / max(xs[-1], 1e-9)
    if rate <= 0:
        print("\n[ETA] taxa de progresso não-positiva — sem dados "
              "suficientes para projetar.")
        return

    sh_last, el_last = cur_pts[-1]["sh"], cur_pts[-1]["elapsed"]
    remaining_in_epoch = (shn - sh_last) / rate
    # Duração total estimada da época atual (para projetar as SEGUINTES):
    full_epoch_est = (sum(completed_durs) / len(completed_durs)
                      if completed_durs else shn / rate)
    remaining_epochs_after = max(0, epn - cur_ep)
    eta_stage = remaining_in_epoch + remaining_epochs_after * full_epoch_est

    print(f"\n[ETA — estágio '{stage}'] "
          f"época atual: {cur_ep}/{epn} | shard {sh_last}/{shn} "
          f"({100*sh_last/shn:.1f}%)")
    print(f"  taxa observada nesta época: {rate*60:.2f} shards/min")
    if completed_durs:
        print(f"  duração média de época(s) concluída(s) "
              f"({len(completed_durs)}): {_fmt_hms(sum(completed_durs)/len(completed_durs))}")
    print(f"  tempo p/ terminar a época atual: {_fmt_hms(remaining_in_epoch)}")
    if remaining_epochs_after:
        print(f"  + {remaining_epochs_after} época(s) subsequente(s) "
              f"(~{_fmt_hms(full_epoch_est)} cada, estimado)")
    print(f"  ETA para o FIM do estágio '{stage}': "
          f"~{_fmt_hms(eta_stage)} a partir de agora")
    print(f"  [!] Cobre só o estágio '{stage}'. Se houver um estágio "
          f"seguinte no seu comando (ex.: fine-tuning após pré-treino), "
          f"o tempo dele NÃO está incluído aqui — este log ainda não "
          f"tem dados para projetá-lo. O early stopping também pode "
          f"encerrar antes das {epn} épocas, encurtando este total.")


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

        # LINHA CRUCIAL: tendência de RAM_disp e projeção de esgotamento.
        # RSS do processo pode ficar PLANO mesmo com vazamento em memória
        # que não é RSS (ex.: /dev/shm sob a estratégia 'file_system') —
        # é RAM_disp caindo que denuncia isso. Ajusta uma reta aos últimos
        # pontos e projeta quando chegaria a zero, se a tendência persistir.
        pts = [(( _parse_t(t)), ram) for _, _, ram, _, _, t in res]
        pts = [(x, y) for x, y in pts if x is not None]
        if len(pts) >= 4:
            xs = [p[0] for p in pts[-12:]]     # últimos ~12 snapshots
            ys = [p[1] for p in pts[-12:]]
            n = len(xs)
            mx, my = sum(xs) / n, sum(ys) / n
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            den = sum((x - mx) ** 2 for x in xs) or 1e-9
            slope = num / den                  # GB por segundo
            slope_min = slope * 60
            print(f"\n  Tendência de RAM_disp (últimos {n} pontos): "
                  f"{slope_min:+.3f} GB/min")
            # LIMIAR: abaixo de 0,05 GB/min (~72GB/dia) é ruído de medição
            # (flutuação normal de cache/alocador), não vazamento — evita
            # alarme falso quando a janela cruza uma queda ÚNICA (ex.: o
            # carregamento eager do fine-tuning) seguida de platô estável.
            EPS = 0.05
            if abs(slope_min) < EPS:
                print("  RAM_disp ESTÁVEL — variação dentro do ruído "
                      "normal, sem indício de vazamento.")
            elif slope_min < 0:
                t_zero = ys[-1] / (-slope)
                if t_zero < 10 * 86400:         # só alarma se < ~10 dias
                    h = int(t_zero // 3600); mrest = int((t_zero % 3600) // 60)
                    print(f"  [!] EM QUEDA — no ritmo atual, RAM_disp "
                          f"chegaria a 0 em ~{h}h{mrest:02d}m. Se isso NÃO "
                          f"for esperado (ex.: RSS do processo está "
                          f"plano), é sinal de vazamento fora do processo "
                          f"principal (ex.: /dev/shm) — considere "
                          f"reiniciar com a versão corrigida antes de "
                          f"perder o progresso num crash não controlado.")
                else:
                    print(f"  Leve tendência de queda, mas em ritmo tão "
                          f"lento (projeção de esgotamento: "
                          f"{_fmt_hms(t_zero)}) que não é motivo de "
                          f"preocupação — provavelmente ruído residual de "
                          f"uma queda pontual (ex.: um carregamento) "
                          f"dentro da janela analisada, não um vazamento "
                          f"contínuo. Continue monitorando.")
            else:
                print("  RAM_disp em RECUPERAÇÃO — sem risco aparente.")
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

    # Cobre os DOIS vocabulários de progresso de treino:
    #  - rolling-origin (_fit):    "fold J/N ep E: train=... val=..."
    #  - streaming (_fit_scale):   "ep E/N shard S/T | train=... val=..."
    _tail(lambda l: ("[step6:" in l and "train=" in l and "val=" in l),
          "progresso de treino (época/shard)")
    if not _estimate_eta_folds(lines):
        _estimate_eta(lines)
    _tail(lambda l: "época" in l.lower() and "completa" in l.lower(),
          "épocas concluídas")
    _tail(lambda l: "estágio" in l or "fonte '" in l or "combinado" in l
          or ": streaming" in l or "shards (amostra" in l,
          "carregamento/estágios")
    _tail(lambda l: "shard" in l.lower() and "grav" in l.lower(),
          "shards gravados (passo 2-3)")
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
