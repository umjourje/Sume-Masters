"""smoke_report.py — Consolida o smoke test federado e extrapola o ETA do
treino completo (roda no SERVIDOR).

Entradas: os arquivos summary_<tag>.json que cada Pi grava via
fed_monitor.RunMonitor (traga-os por rsync/scp, ou aponte para o diretório
de métricas do servidor se o summary for embutido no MetricRecord).

Saídas:
  * Tabela por dispositivo: wall time, CPU%, RAM, vazão (unid/min);
  * ETA extrapolado do treino federado COMPLETO, no modelo síncrono:
        T_rodada ≈ max_i(T_cliente_i) + overhead_agregação
        T_total  ≈ R × E/E_smoke × T_rodada_ajustada
    (o max é o straggler — coerente com a decisão de balanceamento LPT
     registrada no projeto).

Uso:
    python smoke_report.py --summaries "metrics/*/summary_smoke.json" \
        --full-units pi1=10091558 pi2=7282780 pi3=3507067 pi4=2489317 pi5=2489409 \
        --rounds 10 --local-epochs 1 --smoke-epochs 1

    # --full-units: total de UNIDADES do treino completo por dispositivo,
    #   na MESMA unidade usada em RunMonitor.tick() (janelas, batches ou
    #   shards). Os números acima são ilustrativos — use os reais do
    #   prepare_and_ship_pi_data.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


def fmt_h(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    h = seconds / 3600.0
    if h < 1:
        return f"{seconds/60:.1f} min"
    if h < 48:
        return f"{h:.1f} h"
    return f"{h/24:.1f} dias"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--summaries", nargs="+", required=True,
                    help="Arquivos summary_*.json (aceita glob entre aspas).")
    ap.add_argument("--full-units", nargs="*", default=[],
                    metavar="HOST=N",
                    help="Total de unidades do treino completo por host, "
                         "na mesma unidade de RunMonitor.tick().")
    ap.add_argument("--rounds", type=int, default=None,
                    help="Rodadas federadas planejadas (R).")
    ap.add_argument("--local-epochs", type=int, default=1,
                    help="Épocas locais por rodada no treino completo (E).")
    ap.add_argument("--smoke-epochs", type=int, default=1,
                    help="Épocas locais usadas no smoke test.")
    ap.add_argument("--agg-overhead-s", type=float, default=30.0,
                    help="Overhead estimado de comunicação+agregação por "
                         "rodada (s). Meça no próprio smoke: "
                         "T_rodada_servidor − max(T_cliente).")
    ap.add_argument("--json-out", default=None,
                    help="Se definido, grava o relatório também em JSON.")
    args = ap.parse_args()

    paths: list[Path] = []
    for pat in args.summaries:
        expanded = glob.glob(pat)
        paths.extend(Path(p) for p in (expanded or [pat]))
    paths = [p for p in paths if p.exists()]
    if not paths:
        print("[erro] nenhum summary encontrado", file=sys.stderr)
        return 2

    full_units: dict[str, int] = {}
    for spec in args.full_units:
        host, _, n = spec.partition("=")
        try:
            full_units[host] = int(n)
        except ValueError:
            print(f"[erro] --full-units inválido: {spec}", file=sys.stderr)
            return 2

    rows = []
    for p in sorted(paths):
        try:
            s = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"[aviso] {p}: ilegível ({e})", file=sys.stderr)
            continue
        host = s.get("hostname") or p.parent.name
        rate = s.get("units_per_min")  # unid/min medidas no smoke
        total = full_units.get(host) or full_units.get(p.parent.name)
        # tempo de UMA época local completa neste host, extrapolado
        t_epoch_s = (total / rate * 60.0) if (rate and total) else None
        rows.append({
            "host": host, "file": str(p),
            "wall_time_s": s.get("wall_time_s"),
            "cpu_pct_avg": s.get("cpu_pct_avg"),
            "load1_avg": s.get("load1_avg"),
            "ram_used_gb_avg": s.get("ram_used_gb_avg"),
            "ram_used_gb_max": s.get("ram_used_gb_max"),
            "rss_gb_max": s.get("rss_gb_max"),
            "units_per_min": rate,
            "smoke_units": s.get("done_units"),
            "full_units": total,
            "t_full_epoch_s": t_epoch_s,
        })

    if not rows:
        print("[erro] nenhum summary legível", file=sys.stderr)
        return 2

    # ---------- tabela ----------
    hdr = (f"{'host':<10} {'wall':>9} {'CPU%':>6} {'load1':>6} "
           f"{'RAMavg':>7} {'RAMmax':>7} {'RSSmax':>7} "
           f"{'unid/min':>10} {'época cheia':>12}")
    print("\n=== SMOKE TEST — recursos e vazão por dispositivo ===")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def n(v, f="{:.1f}"):
            return "—" if v is None else f.format(v)
        print(f"{r['host']:<10} {fmt_h(r['wall_time_s']):>9} "
              f"{n(r['cpu_pct_avg']):>6} {n(r['load1_avg'], '{:.2f}'):>6} "
              f"{n(r['ram_used_gb_avg']):>7} {n(r['ram_used_gb_max']):>7} "
              f"{n(r['rss_gb_max'], '{:.2f}'):>7} "
              f"{n(r['units_per_min']):>10} {fmt_h(r['t_full_epoch_s']):>12}")

    # ---------- extrapolação ----------
    report: dict = {"devices": rows}
    epochs_known = [r["t_full_epoch_s"] for r in rows if r["t_full_epoch_s"]]
    if epochs_known and args.rounds:
        # síncrono: a rodada dura o tempo do cliente mais lento (straggler)
        t_round = max(epochs_known) * args.local_epochs + args.agg_overhead_s
        t_total = t_round * args.rounds
        straggler = max(rows, key=lambda r: r["t_full_epoch_s"] or 0)["host"]
        print("\n=== EXTRAPOLAÇÃO — treino federado completo ===")
        print(f"Straggler previsto : {straggler}")
        print(f"Tempo por rodada   : {fmt_h(t_round)} "
              f"(E={args.local_epochs} época(s) local(is) "
              f"+ {args.agg_overhead_s:.0f}s de agregação)")
        print(f"ETA total (R={args.rounds}) : {fmt_h(t_total)}")
        print("\n[nota] extrapolação linear a partir da vazão do smoke test; "
              "válida porque o custo por unidade é ~constante no streaming "
              "de shards. Sensível a: térmica do Pi (throttling), I/O do "
              "cartão SD e tamanho do modelo trafegado.")
        report["extrapolation"] = {
            "straggler": straggler,
            "t_round_s": t_round,
            "t_total_s": t_total,
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "agg_overhead_s": args.agg_overhead_s,
        }
    else:
        missing = [r["host"] for r in rows if not r["t_full_epoch_s"]]
        print("\n[aviso] extrapolação de ETA não calculada — informe "
              "--rounds e --full-units para: " + ", ".join(missing))

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\n[ok] relatório JSON em {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    