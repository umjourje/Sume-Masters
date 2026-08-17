"""plot_loss.py — Gráficos EDITÁVEIS da loss ao longo do tempo/passos.

Consome os JSONL gerados pelo fed_monitor.RunMonitor (loss_<tag>.jsonl),
de um ou mais dispositivos, e produz:

  * loss_vs_step.<fmt>   — loss × passo de treinamento (uma série por
                           cliente/tag; média móvel opcional);
  * loss_vs_time.<fmt>   — loss × tempo decorrido (minutos);
  * loss_data.csv        — os dados subjacentes, para regenerar/editar
                           o gráfico em qualquer ferramenta.

"Editável" aqui significa três coisas, todas atendidas:
  1. Saída vetorial SVG (texto como texto, editável em Inkscape/Illustrator)
     — matplotlib com `svg.fonttype = "none"`;
  2. Saída PGF opcional (--fmt pgf) para edição direta no LaTeX da
     dissertação;
  3. CSV com os dados crus, de modo que o gráfico nunca é um beco sem saída.

Uso:
    python plot_loss.py --inputs metrics/pi1/loss_smoke.jsonl \
                        metrics/pi2/loss_smoke.jsonl \
                        --out plots/ --fmt svg pdf --smooth 25

    # também aceita glob:
    python plot_loss.py --inputs "metrics/*/loss_*.jsonl" --out plots/

Dependência: matplotlib (roda no SERVIDOR; não precisa estar nos Pis).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[aviso] {path}:{lineno} — linha JSON inválida, ignorada",
                      file=sys.stderr)
    return rows


def moving_average(values: list[float], k: int) -> list[float]:
    if k <= 1:
        return values
    out, acc = [], 0.0
    from collections import deque
    win: deque[float] = deque(maxlen=k)
    for v in values:
        win.append(v)
        out.append(sum(win) / len(win))
    return out


def series_label(path: Path, rows: list[dict]) -> str:
    """Rótulo da série: <pasta-pai>/<tag do arquivo>."""
    tag = path.stem.replace("loss_", "")
    parent = path.parent.name
    stages = sorted({r.get("stage", "?") for r in rows})
    return f"{parent}:{tag}" + (f" ({','.join(stages)})" if stages != ["fit"] else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Arquivos loss_*.jsonl (aceita glob entre aspas).")
    ap.add_argument("--out", default="plots", help="Diretório de saída.")
    ap.add_argument("--fmt", nargs="+", default=["svg", "pdf"],
                    choices=["svg", "pdf", "png", "pgf"],
                    help="Formatos de saída (svg é o editável por padrão).")
    ap.add_argument("--smooth", type=int, default=1,
                    help="Janela da média móvel (1 = sem suavização).")
    ap.add_argument("--stage", default=None,
                    help="Filtra por estágio (ex.: fit, evaluate).")
    ap.add_argument("--logy", action="store_true", help="Eixo y em escala log.")
    args = ap.parse_args()

    # matplotlib importado após o parse para --help funcionar sem a lib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "svg.fonttype": "none",      # texto permanece <text> no SVG (editável)
        "pdf.fonttype": 42,          # TrueType no PDF (editável, não outline)
        "figure.dpi": 120,
        "font.size": 10,
    })

    paths: list[Path] = []
    for pat in args.inputs:
        expanded = glob.glob(pat)
        paths.extend(Path(p) for p in (expanded or [pat]))
    paths = [p for p in paths if p.exists()]
    if not paths:
        print("[erro] nenhum arquivo de entrada encontrado", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows_csv: list[dict] = []
    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))

    for path in sorted(paths):
        rows = load_jsonl(path)
        if args.stage:
            rows = [r for r in rows if r.get("stage") == args.stage]
        rows = [r for r in rows if "loss" in r and "ts" in r]
        if not rows:
            print(f"[aviso] {path}: sem registros utilizáveis", file=sys.stderr)
            continue
        rows.sort(key=lambda r: r["ts"])
        label = series_label(path, rows)

        t0 = rows[0]["ts"]
        losses = [float(r["loss"]) for r in rows]
        steps = [int(r.get("step", i)) for i, r in enumerate(rows)]
        # passo global monotônico (caso step reinicie a cada época/rodada)
        gsteps, offset, prev = [], 0, None
        for s in steps:
            if prev is not None and s <= prev:
                offset += prev + 1
            gsteps.append(s + offset)
            prev = s
        minutes = [(r["ts"] - t0) / 60.0 for r in rows]
        smoothed = moving_average(losses, args.smooth)

        ax1.plot(gsteps, smoothed, linewidth=1.2, label=label)
        ax2.plot(minutes, smoothed, linewidth=1.2, label=label)
        if args.smooth > 1:
            ax1.plot(gsteps, losses, linewidth=0.4, alpha=0.25,
                     color=ax1.lines[-1].get_color())
            ax2.plot(minutes, losses, linewidth=0.4, alpha=0.25,
                     color=ax2.lines[-1].get_color())

        for r, g, m in zip(rows, gsteps, minutes):
            all_rows_csv.append({
                "series": label, "global_step": g, "minutes": round(m, 4),
                "loss": r["loss"], "stage": r.get("stage"),
                "round": r.get("round"), "epoch": r.get("epoch"),
                "step": r.get("step"), "ts": r["ts"],
            })

    for ax, xlabel in ((ax1, "Passo de treinamento (global)"),
                       (ax2, "Tempo decorrido (min)")):
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Loss")
        if args.logy:
            ax.set_yscale("log")
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.legend(fontsize=8)

    fig1.tight_layout()
    fig2.tight_layout()
    for fmt in args.fmt:
        fig1.savefig(out_dir / f"loss_vs_step.{fmt}")
        fig2.savefig(out_dir / f"loss_vs_time.{fmt}")

    csv_path = out_dir / "loss_data.csv"
    if all_rows_csv:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows_csv[0].keys()))
            w.writeheader()
            w.writerows(all_rows_csv)

    print(f"[ok] {len(paths)} série(s) → {out_dir}/loss_vs_step.*, "
          f"loss_vs_time.*, {csv_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())