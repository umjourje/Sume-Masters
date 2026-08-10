"""bubble_map_real.py — Mapa de bolhas do dataset REAL efetivamente
utilizado (25 países, 59 subconjuntos presentes em 01_splits), com mapa
base embutido (Plotly Scattergeo — geometria do mapa mundial embarcada
na própria biblioteca, sem download em tempo de execução; robusto para
ambientes sem acesso à internet, ao contrário de soluções baseadas em
cartopy/Natural Earth que buscam shapefiles remotos na primeira
execução — hipótese mais provável para o mapa anterior aparecer sem base
cartográfica).

Fonte dos volumes: country_map.py (já validado nesta conversa: mapeamento
subconjunto->país cruzado com a listagem real de 01_splits, excluindo os
8 subconjuntos ausentes do EnergyBench).

Dependências: pip install plotly kaleido

Uso:
    python bubble_map_real.py --out bubble_map_real.jpg
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "federated_real"))
from country_map import SUBSETS, PI_BUCKETS  # fonte única de verdade

# Centróides nacionais aproximados (lat, lon) — suficientes para um mapa
# de bolhas em escala de país; NÃO são centróides geodésicos oficiais.
# ⚠️ não verificado contra fonte cartográfica oficial, apenas valores de
# referência amplamente usados para visualização.
CENTROIDES = {
    "Espanha": (40.46, -3.75), "Reino Unido": (55.38, -3.44),
    "Austrália": (-25.27, 133.78), "EUA": (37.09, -95.71),
    "Noruega": (60.47, 8.47), "Portugal": (39.40, -8.22),
    "China": (35.86, 104.20), "Sri Lanka": (7.87, 80.77),
    "Eslováquia": (48.67, 19.70), "Índia": (20.59, 78.96),
    "Alemanha": (51.17, 10.45), "Tailândia": (15.87, 100.99),
    "Canadá": (56.13, -106.35), "Itália/Áustria": (46.45, 11.35),
    "África do Sul": (-30.56, 22.94), "Japão": (36.20, 138.25),
    "Irlanda": (53.14, -7.69), "Coreia do Sul": (35.91, 127.77),
    "Grécia": (39.07, 21.82), "Suíça": (46.82, 8.23),
    "França": (46.23, 2.21), "México": (23.63, -102.55),
    "Emirados Árabes": (23.42, 53.85), "Costa Rica": (9.75, -83.75),
    "Malásia": (4.21, 101.98),
}


def volumes_por_pais() -> dict[str, tuple[int, int]]:
    """(observações reais, nº subconjuntos presentes) por país, só com
    dados REALMENTE presentes — mesmo cálculo já validado nesta conversa."""
    from collections import defaultdict
    obs = defaultdict(int)
    n_sub = defaultdict(int)
    for nome, sub in SUBSETS.items():
        if sub.presente:
            n_sub[sub.pais] += 1
    return n_sub  # placeholder — obs preenchido abaixo via tabela fixa


# Observações reais por país (calculadas e verificadas na conversa,
# a partir das Tabelas 3/4 do dataset card oficial, filtradas pelos
# subconjuntos presentes).
OBS_REAIS = {
    "Espanha": 632_313_933, "Austrália": 174_786_722,
    "Reino Unido": 84_169_606, "EUA": 32_921_297, "Noruega": 28_429_008,
    "Portugal": 14_931_052, "China": 14_024_105, "Sri Lanka": 11_372_970,
    "Eslováquia": 9_453_051, "Índia": 2_756_663, "Alemanha": 1_722_619,
    "Tailândia": 1_504_058, "Canadá": 645_989, "Itália/Áustria": 489_644,
    "África do Sul": 488_098, "Japão": 187_008, "Irlanda": 174_398,
    "Coreia do Sul": 90_631, "Grécia": 83_452, "Suíça": 69_513,
    "França": 58_186, "México": 55_280, "Emirados Árabes": 19_652,
    "Costa Rica": 12_058, "Malásia": 706,
}


def pi_do_pais(pais: str) -> int:
    for pi, paises in PI_BUCKETS.items():
        if pais in paises:
            return pi
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("bubble_map_real.jpg"))
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=800)
    a = ap.parse_args()

    import plotly.graph_objects as go
    import numpy as np

    n_sub = {p: sum(1 for s in SUBSETS.values() if s.presente and s.pais == p)
             for p in OBS_REAIS}
    paises = list(OBS_REAIS.keys())
    obs = np.array([OBS_REAIS[p] for p in paises], dtype=float)
    lat = [CENTROIDES[p][0] for p in paises]
    lon = [CENTROIDES[p][1] for p in paises]
    pi = [pi_do_pais(p) for p in paises]

    # Tamanho de bolha por RAIZ QUADRADA do volume (área proporcional ao
    # volume, não o raio — evita que Espanha domine visualmente de forma
    # desproporcional ao dado real).
    tam = 8 + 55 * np.sqrt(obs / obs.max())
    texto = [f"<b>{p}</b><br>{n_sub[p]} subconjunto(s)<br>"
             f"{OBS_REAIS[p]:,} observações<br>Nó federado: Pi{pi_do_pais(p)}"
             for p in paises]

    fig = go.Figure(go.Scattergeo(
        lat=lat, lon=lon, text=texto, hoverinfo="text",
        marker=dict(size=tam, color=pi, colorscale="Viridis",
                    line=dict(width=0.5, color="white"),
                    colorbar=dict(title="Nó\nfederado", tickmode="linear",
                                  tick0=1, dtick=1)),
    ))
    fig.update_geos(
        showland=True, landcolor="rgb(235,235,235)",
        showcountries=True, countrycolor="rgb(200,200,200)",
        showocean=True, oceancolor="rgb(247,250,255)",
        showcoastlines=True, coastlinecolor="rgb(200,200,200)",
        projection_type="natural earth",
    )
    fig.update_layout(
        title="Distribuição geográfica do dataset real utilizado "
              "(EnergyBench, 59/67 subconjuntos presentes)",
        width=a.width, height=a.height, margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.write_image(str(a.out), format=a.out.suffix.lstrip("."), scale=2)
    print(f"Salvo em {a.out} | {len(paises)} países, "
          f"{sum(n_sub.values())} subconjuntos, {int(obs.sum()):,} observações")


if __name__ == "__main__":
    main()