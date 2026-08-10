# -*- coding: utf-8 -*-
"""
Mapa de bolhas — densidade de observações por localização (EnergyBench)
=======================================================================

Deriva do "Metadata-dataset.csv" (pasta Dataset_V0.0/metadata) um CSV com
lat/lon + nº de observações por sub-dataset ou por localização, e gera um
bubble plot mundial cuja área E COR da bolha codificam a intensidade de
observações (nunca ambas para dimensões diferentes — ver nota abaixo).

ALTERAÇÕES nesta versão (mantendo geocodificação offline, estrutura e a
saída HTML/Plotly do script original intocadas):

  1. FILTRO DE DADOS REALMENTE UTILIZADOS: 8 subconjuntos que constam do
     Metadata-dataset.csv oficial mas NÃO estão presentes em 01_splits
     (não foram baixados/processados) são excluídos ANTES de qualquer
     agregação — o mapa reflete o dataset efetivamente usado no
     treinamento, não o catálogo completo do EnergyBench.
  2. COR = INTENSIDADE, não mais Tipo (Comercial/Residencial) nem nó
     federado: a cor da bolha agora é log10(n_obs), na mesma escala de
     variável que o tamanho — um "bubble map de intensidade" simples,
     sem dimensão categórica extra. Colormap perceptualmente uniforme
     ('plasma') para dar variação visível mesmo com a desigualdade
     extrema de volume entre países (Espanha ≈ 62% do total).
  3. MAPA BASE REAL na saída estática (matplotlib): a função original
     nunca desenhou geometria de países — apenas uma grade de
     lat/lon. Trocado por geopandas (contorno vetorial real dos
     países), SEM depender de kaleido/Chromium (esse nunca foi o
     problema aqui: a função estática é pura matplotlib, não passa
     pelo pipeline de exportação do Plotly). A saída HTML
     (plot_plotly) permanece EXATAMENTE como estava — já está boa.
  4. MENOS SOBREPOSIÇÃO: jitter aumentado para localizações com
     coordenadas coincidentes/próximas, transparência ajustada, e
     ordem de desenho preservada (maior primeiro, menor por cima) para
     que bolhas pequenas não fiquem escondidas atrás de bolhas grandes.

Saídas:
  - <outdir>/map_data.csv .......... CSV derivado (Alias, Location, lat, lon,
                                      n_obs, n_buildings, Type)
  - <outdir>/bubble_map.html ....... mapa interativo (plotly, inalterado)
  - <outdir>/bubble_map.png ........ mapa estático (matplotlib + geopandas)

Uso:
    python bubble_map.py --metadata Metadata-dataset.csv \
        --by location --size-scale log --outdir ./eda_outputs_map

Dependências: pandas, numpy, matplotlib, geopandas (mapa estático);
              plotly opcional (mapa HTML).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Subconjuntos AUSENTES do 01_splits real (constam do Metadata-dataset.csv
# oficial, mas não foram baixados/processados) — excluídos do mapa para
# refletir o dataset REALMENTE utilizado. Mesma lista já validada e
# documentada em country_map.py / decisoes_operacionalizacao.tex.
ALIASES_AUSENTES = {
    "SAVE", "UKST", "METER", "HES", "NESEMP",   # Reino Unido (5)
    "NEEA",                                      # EUA (1)
    "ECRG-Commercial", "ECRG-Residential",       # Polônia (2)
}

# --------------------------------------------------------------------------
# Geocodificação offline
# --------------------------------------------------------------------------
# Centróides aproximados de país (iso3) — suficientes p/ bubble map mundial.
ISO_COORDS = {
    "USA": (39.8, -98.6), "IND": (22.4, 79.0), "CHN": (35.0, 103.8),
    "KOR": (36.4, 127.9), "CHE": (46.8, 8.2), "THA": (15.1, 101.0),
    "MYS": (3.8, 109.7), "PRT": (39.6, -8.0), "GBR": (54.0, -2.5),
    "ARE": (24.0, 54.0), "ZAF": (-29.0, 25.1), "POL": (52.1, 19.4),
    "AUS": (-25.7, 134.5), "DEU": (51.1, 10.4), "ESP": (40.2, -3.6),
    "IRL": (53.2, -8.1), "GRC": (39.1, 22.9), "JPN": (36.6, 138.0),
    "CAN": (56.1, -106.3), "ITA": (42.8, 12.1), "NOR": (64.6, 12.7),
    "FRA": (46.6, 2.5), "SVK": (48.7, 19.5), "MEX": (23.9, -102.5),
    "CRI": (9.9, -84.2), "LKA": (7.7, 80.7), "AUT": (47.6, 14.1),
}

# Localizações subnacionais/específicas presentes no CSV do EnergyBench
# (a chave é comparada em minúsculas, por 'contém').
PLACE_COORDS = {
    "portland": (45.52, -122.68),        # NEEA
    "cambridge": (52.20, 0.12),          # ULE (Cambridge, GBR)
    "sharjah": (25.35, 55.42),           # IOT
    "phoenix": (33.45, -112.07),         # HB
    "california": (36.78, -119.42),      # Honda SMART Home
    "british columbia": (53.73, -127.65),# HUE
    "scotland": (56.49, -4.20),          # NESEMP
    "great britain": (54.00, -2.50),     # UKST
    "southern china": (23.13, 113.26),   # EWELD/IPC (Guangdong aprox.)
    "sceaux": (48.78, 2.29),             # IHEPC
    "sri lanka": (7.70, 80.70),          # RSL
    "italy, austria": (46.60, 13.85),    # GREEND (fronteira ITA/AUT)
    "usa, europe": (39.8, -98.6),        # PES (multi-região -> EUA)
}


def geocode(location: str, iso: str) -> tuple[float, float] | tuple[None, None]:
    loc = str(location).strip().lower()
    for key, (la, lo) in PLACE_COORDS.items():
        if key in loc:
            return la, lo
    iso = str(iso).strip().upper()
    if iso in ISO_COORDS:
        return ISO_COORDS[iso]
    return None, None


# --------------------------------------------------------------------------
# Derivação do CSV
# --------------------------------------------------------------------------
def build_map_data(metadata_csv: Path, by: str) -> pd.DataFrame:
    md = pd.read_csv(metadata_csv)
    md.columns = [c.strip() for c in md.columns]

    # FILTRO: só os subconjuntos REALMENTE presentes em 01_splits.
    antes = len(md)
    ausentes_no_csv = md[md["Alias"].isin(ALIASES_AUSENTES)]
    md = md[~md["Alias"].isin(ALIASES_AUSENTES)].copy()
    if len(ausentes_no_csv):
        def _num(x):
            try:
                return float(str(x).replace(",", ""))
            except (ValueError, TypeError):
                return 0.0
        obs_excluidas = ausentes_no_csv["No. of Obs"].map(_num).sum()
        print(f"[filtro] {antes - len(md)}/{antes} subconjuntos excluídos "
              f"(ausentes em 01_splits): "
              f"{', '.join(sorted(ausentes_no_csv['Alias']))} "
              f"(~{obs_excluidas:,.0f} observações não utilizadas)")

    def _num(x):
        try:
            return float(str(x).replace(",", ""))
        except (ValueError, TypeError):
            return np.nan

    df = pd.DataFrame({
        "Alias": md["Alias"],
        "Type": md["Type"],
        "Location": md["Location"],
        "iso": md["iso"],
        "n_obs": md["No. of Obs"].map(_num),
        "n_buildings": md["No. of Buildings"].map(_num),
    })

    coords = df.apply(lambda r: geocode(r["Location"], r["iso"]), axis=1)
    df["lat"] = [c[0] for c in coords]
    df["lon"] = [c[1] for c in coords]

    missing = df[df["lat"].isna()]
    if len(missing):
        print("[AVISO] Localizações sem coordenadas (adicione em PLACE/ISO_COORDS):")
        print(missing[["Alias", "Location", "iso"]].to_string(index=False))
    df = df.dropna(subset=["lat", "lon", "n_obs"])

    rng = np.random.default_rng(0)
    if by == "location":
        df = (df.groupby(["Location", "iso", "lat", "lon"], as_index=False)
              .agg(n_obs=("n_obs", "sum"),
                   n_buildings=("n_buildings", "sum"),
                   n_datasets=("Alias", "count"),
                   datasets=("Alias", lambda s: "; ".join(s))))
        df["label"] = df["Location"]
    else:  # by == "dataset"
        df["label"] = df["Alias"]

    # Jitter para localizações com coordenadas coincidentes/próximas —
    # aumentado (antes ±1.5°) para reduzir sobreposição visual entre
    # bolhas vizinhas, já que o filtro de subconjuntos ausentes reduz a
    # densidade de pontos em alguns clusters (ex.: Reino Unido, que
    # perdeu 5 dos 8 subconjuntos residenciais).
    dup = df.duplicated(subset=["lat", "lon"], keep=False)
    df.loc[dup, "lat"] += rng.uniform(-2.5, 2.5, dup.sum())
    df.loc[dup, "lon"] += rng.uniform(-2.5, 2.5, dup.sum())

    return df.sort_values("n_obs", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def _bubble_sizes(n_obs: pd.Series, scale: str, max_pt: float = 2500.0) -> np.ndarray:
    v = n_obs.to_numpy(dtype=float)
    if scale == "log":
        w = np.log10(np.clip(v, 1, None))
    else:
        w = v
    return max_pt * (w / w.max()) ** 2 + 15  # área ∝ peso²; piso p/ visibilidade


def _carregar_mundo(shapefile_explicito: str | None):
    """Geometria real dos países via geopandas — NÃO relacionado ao
    kaleido/Chromium (matplotlib puro, sem navegador embutido)."""
    import geopandas as gpd
    if shapefile_explicito:
        print(f"[bubble_map] usando shapefile local: {shapefile_explicito}")
        return gpd.read_file(shapefile_explicito)
    try:
        caminho = gpd.datasets.get_path("naturalearth_lowres")  # geopandas <1.0
        print(f"[bubble_map] usando dataset embutido do geopandas: {caminho}")
        return gpd.read_file(caminho)
    except Exception as e:
        url = ("https://naciscdn.org/naturalearth/110m/cultural/"
               "ne_110m_admin_0_countries.zip")
        print(f"[bubble_map][AVISO] dataset embutido indisponível "
              f"({type(e).__name__}) — geopandas>=1.0 removeu o dataset "
              f"embutido. Baixando shapefile lowres do Natural Earth "
              f"(requer rede; sem relação com o problema do kaleido). "
              f"Para evitar isso, use --shapefile /caminho/local.shp.")
        return gpd.read_file(url)


def plot_matplotlib(df: pd.DataFrame, outdir: Path, scale: str,
                    shapefile: str | None) -> None:
    sizes = _bubble_sizes(df["n_obs"], scale)
    # COR = INTENSIDADE (log10 de n_obs), colormap perceptualmente
    # uniforme — dá variação visível mesmo com Espanha em ~62% do total,
    # o que uma escala linear de cor tornaria quase monocromática para
    # todos os demais países.
    intensidade = np.log10(np.clip(df["n_obs"].to_numpy(dtype=float), 1, None))

    mundo = _carregar_mundo(shapefile)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    mundo.plot(ax=ax, color="#ededed", edgecolor="#bbbbbb", linewidth=0.5,
              zorder=1)

    sc = ax.scatter(df["lon"], df["lat"], s=sizes, c=intensidade,
                    cmap="plasma", alpha=0.72, edgecolors="k",
                    linewidths=0.5, zorder=3)

    # rótulos apenas nas maiores bolhas, p/ não poluir
    for _, r in df.head(12).iterrows():
        ax.annotate(f"{r['label']}\n{r['n_obs']/1e6:.1f}M",
                   (r["lon"], r["lat"]), fontsize=7,
                   xytext=(4, 4), textcoords="offset points", zorder=4)

    cbar = fig.colorbar(sc, ax=ax, orientation="vertical", shrink=0.7)
    cbar.set_label("Observações (log$_{10}$)")

    ax.set_xlim(-180, 180); ax.set_ylim(-60, 85)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("EnergyBench — densidade de observações por localização "
                f", escala {scale})")

    fig.tight_layout()
    fig.savefig(outdir / "bubble_map.png", dpi=150)
    plt.close(fig)
    print(f"Mapa estático: {outdir/'bubble_map.png'}")


def plot_plotly(df: pd.DataFrame, outdir: Path, scale: str) -> bool:
    """INALTERADO — a saída HTML já está boa, segundo o usuário."""
    try:
        import plotly.express as px
    except ImportError:
        print("[INFO] plotly não instalado — apenas o PNG foi gerado "
              "(pip install plotly para o mapa interativo).")
        return False

    size = np.log10(np.clip(df["n_obs"], 1, None)) if scale == "log" else df["n_obs"]
    fig = px.scatter_geo(
        df.assign(_size=size),
        lat="lat", lon="lon", size="_size", size_max=45,
        color="Type" if "Type" in df.columns else None,
        hover_name="label",
        hover_data={"n_obs": ":,", "n_buildings": ":,",
                   "lat": False, "lon": False, "_size": False},
        projection="natural earth",
        title="EnergyBench — densidade de observações por localização "
              f"(bolha ∝ nº de observações, escala {scale})",
    )
    fig.write_html(outdir / "bubble_map.html", include_plotlyjs="cdn")
    print(f"Mapa interativo: {outdir/'bubble_map.html'}")
    return True


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Bubble map de observações (EnergyBench)")
    ap.add_argument("--metadata", type=Path, required=True,
                    help="Caminho do Metadata-dataset.csv")
    ap.add_argument("--by", choices=["location", "dataset"], default="location")
    ap.add_argument("--size-scale", choices=["log", "linear"], default="log",
                    help="'log' recomendado: as obs variam de ~7e2 a ~6e8")
    ap.add_argument("--outdir", type=Path, default=Path("eda_outputs_map"))
    ap.add_argument("--shapefile", type=str, default=None,
                    help="shapefile local de países (evita qualquer "
                         "download); ex.: ne_110m_admin_0_countries.shp")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = build_map_data(args.metadata, args.by)
    df.to_csv(args.outdir / "map_data.csv", index=False)
    print(f"CSV derivado ({len(df)} linhas): {args.outdir/'map_data.csv'}")
    print(df.head(10)[["label", "lat", "lon", "n_obs"]].to_string(index=False))

    plot_matplotlib(df, args.outdir, args.size_scale, args.shapefile)
    plot_plotly(df, args.outdir, args.size_scale)


if __name__ == "__main__":
    main()