"""bubble_map_real.py — Mapa de bolhas do dataset REAL efetivamente
utilizado (25 países, 59 subconjuntos presentes em 01_splits).

VERSÃO SEM PLOTLY/KALEIDO: kaleido empacota um Chromium para exportação
estática do Plotly — fonte comum de problemas em servidor (download de
binário, sandboxing, memória). Esta versão usa geopandas (geometria
vetorial real do Natural Earth) + matplotlib, que salva JPG/PNG
nativamente, sem nenhum navegador embutido.

Fonte dos volumes: country_map.py (já validado: mapeamento
subconjunto->país cruzado com a listagem real de 01_splits, excluindo os
8 subconjuntos ausentes do EnergyBench).

Dependências: pip install geopandas matplotlib
  * Se "geopandas<1.0": usa o shapefile Natural Earth BUNDLED no pacote
    (nenhum download em tempo de execução).
  * Se "geopandas>=1.0" (removeu o dataset embutido): o script cai para
    baixar o shapefile lowres do Natural Earth uma única vez (requer
    rede nesse caso específico — não relacionado ao problema do
    kaleido/Chromium). Use --shapefile para apontar um arquivo local e
    evitar qualquer download.

Uso:
    python bubble_map_real.py --out bubble_map_real.jpg
    python bubble_map_real.py --out bubble_map_real.jpg \
        --shapefile /caminho/ne_110m_admin_0_countries.shp
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path


def _localizar_country_map(explicito: str | None) -> Path:
    """Busca country_map.py em candidatos plausíveis, em vez de presumir
    uma estrutura de pastas fixa. Ordem: (1) --country-map-dir; (2) mesma
    pasta do script; (3) pai/avô/bisavô da pasta do script; (4) variável
    de ambiente COUNTRY_MAP_DIR."""
    aqui = Path(__file__).resolve().parent
    candidatos = []
    if explicito:
        candidatos.append(Path(explicito))
    if os.environ.get("COUNTRY_MAP_DIR"):
        candidatos.append(Path(os.environ["COUNTRY_MAP_DIR"]))
    candidatos += [aqui, aqui.parent, aqui.parent.parent,
                  aqui.parent.parent.parent, aqui / "federated_real",
                  aqui.parent / "federated_real"]
    for c in candidatos:
        if (c / "country_map.py").exists():
            return c
    raise FileNotFoundError(
        "Não encontrei country_map.py. Tentado em: "
        + ", ".join(str(c) for c in candidatos)
        + ". Use --country-map-dir /caminho/exato ou defina a variável "
          "de ambiente COUNTRY_MAP_DIR, ou copie country_map.py para a "
          "mesma pasta deste script.")


# Centróides nacionais aproximados (lat, lon) — suficientes para um mapa
# de bolhas em escala de país; NÃO são centróides geodésicos oficiais.
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

# Observações reais por país (Tabelas 3/4 do dataset card oficial,
# filtradas pelos subconjuntos presentes em 01_splits).
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

NE_LOWRES_URL = ("https://naciscdn.org/naturalearth/110m/cultural/"
                 "ne_110m_admin_0_countries.zip")


def _carregar_mundo(shapefile_explicito: str | None):
    import geopandas as gpd
    if shapefile_explicito:
        print(f"[bubble_map] usando shapefile local: {shapefile_explicito}")
        return gpd.read_file(shapefile_explicito)
    try:
        # geopandas < 1.0: dataset Natural Earth embutido no pacote,
        # sem download.
        caminho = gpd.datasets.get_path("naturalearth_lowres")
        print(f"[bubble_map] usando dataset embutido do geopandas: {caminho}")
        return gpd.read_file(caminho)
    except Exception as e:
        print(f"[bubble_map][AVISO] dataset embutido indisponível "
              f"({type(e).__name__}: {e}). geopandas>=1.0 removeu o "
              f"dataset embutido — tentando baixar o shapefile lowres "
              f"do Natural Earth (requer rede; SEM relação com o "
              f"problema do kaleido/Chromium).")
        return gpd.read_file(NE_LOWRES_URL)


def pi_do_pais(pais: str, pi_buckets: dict) -> int:
    for pi, paises in pi_buckets.items():
        if pais in paises:
            return pi
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("bubble_map_real.jpg"))
    ap.add_argument("--width", type=float, default=14.0, help="polegadas")
    ap.add_argument("--height", type=float, default=8.0, help="polegadas")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--country-map-dir", type=str, default=None,
                    help="pasta que contém country_map.py, se não estiver "
                         "em nenhum dos caminhos buscados automaticamente")
    ap.add_argument("--shapefile", type=str, default=None,
                    help="caminho local para um shapefile de países "
                         "(evita qualquer download); ex.: "
                         "ne_110m_admin_0_countries.shp")
    a = ap.parse_args()

    pasta = _localizar_country_map(a.country_map_dir)
    sys.path.insert(0, str(pasta))
    from country_map import PI_BUCKETS  # fonte única de verdade
    print(f"[bubble_map_real] country_map.py localizado em: {pasta}")

    import matplotlib
    matplotlib.use("Agg")  # renderização sem display, sem navegador
    import matplotlib.pyplot as plt
    import numpy as np

    mundo = _carregar_mundo(a.shapefile)

    paises = list(OBS_REAIS.keys())
    obs = np.array([OBS_REAIS[p] for p in paises], dtype=float)
    lon = [CENTROIDES[p][1] for p in paises]
    lat = [CENTROIDES[p][0] for p in paises]
    pi = np.array([pi_do_pais(p, PI_BUCKETS) for p in paises])

    # Tamanho de bolha por RAIZ QUADRADA do volume (área proporcional ao
    # volume, não o raio).
    tam = 30 + 3500 * np.sqrt(obs / obs.max())

    fig, ax = plt.subplots(figsize=(a.width, a.height))
    mundo.plot(ax=ax, color="#ededed", edgecolor="#bbbbbb", linewidth=0.5)
    sc = ax.scatter(lon, lat, s=tam, c=pi, cmap="viridis",
                    alpha=0.75, edgecolors="white", linewidths=0.8, zorder=3)
    for p, x, y in zip(paises, lon, lat):
        ax.annotate(p, (x, y), fontsize=6, ha="center", va="center",
                   xytext=(0, 0), textcoords="offset points", zorder=4)

    cbar = fig.colorbar(sc, ax=ax, orientation="vertical", shrink=0.6,
                        ticks=sorted(set(pi)))
    cbar.set_label("Nó federado (Pi)")
    ax.set_title("Distribuição geográfica do dataset real utilizado "
                "(EnergyBench, 59/67 subconjuntos presentes)", fontsize=12)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(a.out, dpi=a.dpi, format=a.out.suffix.lstrip("."))
    plt.close(fig)
    print(f"Salvo em {a.out} | {len(paises)} países, {int(obs.sum()):,} "
          f"observações")


if __name__ == "__main__":
    main()