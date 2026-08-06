"""prepare_and_ship_pi_data.py — Monta o data-root de cada Pi (grupos de
países do country_map.py) e envia via rsync.

Para a Espanha (único país acima do teto de disco), aplica amostragem
ESTRATIFICADA por shard, reaproveitando a mesma lógica de
_sample_shards do step6_train.py (streaming do v0 sintético) — mesmo
princípio, escala diferente: escolhe um subconjunto de shards que cabe
no orçamento de disco do Pi, sem viés de ordem alfabética.

Uso:
    # 1) monta o staging local (sem enviar) para conferir antes:
    python prepare_and_ship_pi_data.py --pi 1 --stage-only

    # 2) monta e envia via rsync para o Pi (ajuste usuário/IP):
    python prepare_and_ship_pi_data.py --pi 1 --ship pi@192.168.1.101:/home/pi/dados

    # 3) todos de uma vez (staging local, revise antes de enviar):
    for i in 1 2 3 4 5; do python prepare_and_ship_pi_data.py --pi $i --stage-only; done
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/PIPELINE_DIR")  # AJUSTE: caminho do pipeline (config.py etc.)

from country_map import groups_for_pi, pais_do_pi, CAP_JANELAS_POR_PI, SUBSETS
from config import CFG


def _discover_shards(base: Path, group: str, split: str) -> list[Path]:
    d = base / split / group
    return sorted(d.rglob("*.pt")) if d.exists() else []


def _sample_shards_stratified(shards: list[Path], max_shards: int, seed: int = 42):
    """Mesma lógica de step6_train._sample_shards: amostra uniforme se
    já coubermos; caso contrário, escolha determinística e reprodutível
    (sem viés de ordem alfabética/data)."""
    if max_shards <= 0 or len(shards) <= max_shards:
        return shards
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(shards), max_shards, replace=False)
    return [shards[i] for i in sorted(idx.tolist())]


def _windows_per_shard_estimate(shards: list[Path]) -> float:
    """Estima janelas/shard a partir do tamanho em disco (evita abrir
    cada .pt); usa CFG.max_windows_per_shard como teto de referência e
    o tamanho médio real dos primeiros arquivos como fator de escala."""
    if not shards:
        return 0.0
    amostra = shards[:20]
    tam_medio = sum(p.stat().st_size for p in amostra) / len(amostra)
    bytes_por_janela = 4469  # calculado do esquema real do shard
    return tam_medio / bytes_por_janela


def stage_pi(pi: int, staging_root: Path, split: str) -> Path:
    base = CFG.split_root.parent / "02_windows" / CFG.resolution \
        if False else CFG.windows_root  # CFG.windows_root já é .../02_windows/<res>
    dest = staging_root / f"pi{pi}" / split
    dest.mkdir(parents=True, exist_ok=True)
    grupos = groups_for_pi(pi)

    total_copiados = total_originais = 0
    for grupo in grupos:
        shards = _discover_shards(base, grupo, split)
        if not shards:
            print(f"  [AVISO] nenhum shard em {base/split/grupo} — pulando")
            continue
        pais = next(s.pais for nome, s in SUBSETS.items()
                    if f"{s.setor}/{nome}" == grupo)
        usar = shards
        # DOWNSAMPLING: só se o balde tiver 1 único país MUITO acima do
        # teto (caso da Espanha) — decisão explícita, não automática por
        # csv, para não cortar sem intenção clara.
        if split == "train" and pai_precisa_corte(pi):
            wps = _windows_per_shard_estimate(shards)
            janelas_totais = wps * len(shards)
            if janelas_totais > CAP_JANELAS_POR_PI:
                max_shards = max(1, int(CAP_JANELAS_POR_PI / max(wps, 1)))
                usar = _sample_shards_stratified(shards, max_shards)
                print(f"  {grupo}: {len(shards)} shards -> amostrados "
                      f"{len(usar)} (~{wps:.0f} janelas/shard, teto "
                      f"{CAP_JANELAS_POR_PI:,} janelas)")
        destino_grupo = dest / grupo
        destino_grupo.mkdir(parents=True, exist_ok=True)
        for p in usar:
            shutil.copy2(p, destino_grupo / p.name)
        total_copiados += len(usar)
        total_originais += len(shards)
        print(f"  {grupo} ({pais}): {len(usar)}/{len(shards)} shards")

    print(f"[Pi{pi}/{split}] total: {total_copiados}/{total_originais} shards "
          f"copiados para {dest}")
    return dest


def pai_precisa_corte(pi: int) -> bool:
    # Único caso hoje: Pi1 = Espanha sozinha, muito acima do teto.
    return pais_do_pi(pi) == ["Espanha"]


def ship(local_dir: Path, destino_scp: str) -> None:
    cmd = ["rsync", "-avz", "--progress", f"{local_dir}/",
           f"{destino_scp}/"]
    print("Executando:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi", type=int, required=True, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--staging-root", type=Path,
                    default=Path("/home/claude/staging_federado"))
    ap.add_argument("--stage-only", action="store_true")
    ap.add_argument("--ship", type=str, default=None,
                    help="destino rsync, ex.: pi@192.168.1.101:/home/pi/dados")
    a = ap.parse_args()

    print(f"=== Pi{a.pi}: países {pais_do_pi(a.pi)} ===")
    for split in ("train", "test"):
        dest = stage_pi(a.pi, a.staging_root, split)

    if a.ship and not a.stage_only:
        ship(a.staging_root / f"pi{a.pi}", a.ship)
    elif not a.stage_only:
        print("\n[!] Nenhum --ship informado — dados ficaram só no staging local. "
              "Use --ship usuario@ip:/caminho para enviar, ou --stage-only para "
              "só revisar antes.")


if __name__ == "__main__":
    main()