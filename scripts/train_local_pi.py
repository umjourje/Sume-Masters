# -*- coding: utf-8 -*-
"""train_local_pi.py — Treino local CENTRALIZADO no Raspberry Pi 5, sobre
os shards de UM país (ex.: Espanha/GoiEner) acessados via NAS, partindo de
um checkpoint pré-treinado (v0_both ou v0_real), e terminando com MATRIZ DE
CONFUSÃO da classificação binária de anomalias sobre o teste do país
(rotulado em tempo de execução — desenho anti-leak).

Reusa as peças JÁ VALIDADAS do pipeline (nada de código de treino novo):
  - run_epoch (step6_train): mesma perda, alfa/beta dinâmicos + BCE;
  - label_windows_batch (step4_5_labels_v2): rotulagem intra-janela no
    teste, em execução.

NÃO importa mais ShardTemporalDataset de step6_train. Motivo: step6_train
é o script do SERVIDOR (treino do v0 / fine-tuning), com pressupostos de
máquina grande. O particionamento dos shards passa a viver AQUI, no script
do Pi, para que o caching possa reaproveitá-lo entre épocas. A semântica
padrão é IDÊNTICA à de ShardTemporalDataset (--val-mode=atual): nada muda
nos números em relação às execuções anteriores.

=========================== CACHE (o pedido) ===============================

Três camadas, nesta ordem: RAM -> disco LOCAL -> NAS.

Dimensionamento, derivado do esquema real gravado por
step2_3_windows_wavelet_v2.flush() + os rótulos de step4_5_labels_v2:

    campo          B/janela        usado no treino?
    x                   768        não (só no TESTE, p/ rotular)
    trend               768        não (só no TESTE)
    season              768        não
    trend_norm          768        SIM -> ti (168) + tt (24)
    season_norm         768        SIM -> si (168) + st (24)
    building_idx          8        só para particionar
    start                 8        só para particionar
    stats                16        não
    labels              192        não
    labels_fused        192        SIM -> ct (24), mantido em uint8
    -----------------------------------
    total             4.256        treino usa 1.560 (35%)

Ou seja, cachear os tensores DE TREINO custa ~1/3 do shard em disco:
15 shards x 200k janelas = 4,4 GB, folgado nos 16 GB do Pi 5 (o processo
usa hoje <2 GB). O `ct` fica em uint8 no cache e vira float só no batch —
economiza 72 B/janela sem custo mensurável.

Ganhos concretos, em ordem de tamanho:
  1. A versão anterior construía ShardTemporalDataset DUAS vezes por shard
     por época (uma para "train", outra para "val"), ou seja, DOIS
     torch.load do mesmo arquivo: ~25 GB de tráfego NAS por época onde
     12,5 GB bastavam. Agora é uma leitura só, e no modo padrão (em que os
     dois conjuntos coincidem) o cache guarda UMA cópia.
  2. A partir da 2ª época não há leitura de NAS nenhuma (hit de RAM).
  3. A máscara temporal por prédio era um laço Python sobre np.unique
     refeito a cada época; agora é vetorizada (lexsort) e cacheada.
  4. DataLoader(num_workers=0) + default_collate substituídos por
     fatiamento direto (_Batches): some ~200 mil construções de dict
     Python por shard por época.

LIMITE HONESTO: medido no seu próprio log (2.137 s / 5.859 batches =
0,365 s por batch), ~80% do tempo é forward+backward em CPU. O Pi 5 tem
4 núcleos Cortex-A76 e o torch já usa os 4 por padrão — "muita CPU e pouca
RAM" é o comportamento esperado, não um sintoma. Nenhuma otimização de I/O
ataca esses 80%: só menos épocas, lote maior, ou hardware diferente.

Uso:
  python3 train_local_pi.py \
      --windows-root $REAL/02_windows/Hourly --pi 1 \
      --init-checkpoint $V0BOTH \
      --epochs 30 --patience 5 --max-shards 15 --tag pi1_esp_v0both \
      --outdir /tmp/local_runs/real_synth \
      --cache-gb 8 --cache-dir /var/tmp/shard_cache

Os shards NÃO ficam em subpasta por subconjunto: são arquivos soltos em
<windows_root>/<split>/<Setor>/ nomeados <Subconjunto>.wNN.partKKK.pt.
Por isso a seleção é por GRUPO (via --pi, lendo country_map.py, ou via
--groups manual), nunca por varredura de diretório — que capturaria
todos os países daquele setor.

Saídas em <outdir>/<tag>/: best_local.pth, progress.json,
confusion_matrix.json, confusion_matrix.png, metrics.json,
history_<tag>.json, loss_<tag>.jsonl, summary_<tag>.json.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# config.py exige RAW_ROOT/OUT_ROOT no ambiente; num Pi de treino local
# esses caminhos não são usados de fato (os dados vêm de --windows-root),
# então definimos valores neutros ANTES de importar módulos do pipeline.
#
# ATENÇÃO — a ÚNICA variável que realmente precisa estar correta no Pi é
# WLSTMIX_DIR, apontando para a pasta do repositório W-LSTMix que contém
# models/W_LSTMix.py (o backbone é importado de lá por model_hybrid.py).
# Sem ela, o script falha no import com mensagem explícita.
# Ex.:  export WLSTMIX_DIR=/source/pesquisa/Tupa-Masters
os.environ.setdefault("RAW_ROOT", "/tmp/raw_unused")
os.environ.setdefault("OUT_ROOT", "/tmp/out_unused")


def _add_repo_to_path(repo_dir: str | None) -> None:
    aqui = Path(__file__).resolve().parent
    candidatos = ([Path(repo_dir)] if repo_dir else []) + [
        aqui, aqui.parent, aqui.parent / "pipeline",
        aqui.parent.parent / "scripts_tupa", Path.cwd()]
    for c in candidatos:
        if (c / "step6_train.py").exists():
            sys.path.insert(0, str(c))
            return
    raise FileNotFoundError(
        "step6_train.py não encontrado; use --repo-dir apontando para a "
        "pasta scripts_tupa do repositório.")


def _shards_de_grupos(windows_root: Path, split: str,
                      grupos: list[str]) -> list[Path]:
    """Seleciona os shards de subconjuntos ESPECÍFICOS.

    Os shards NÃO ficam em subpasta por subconjunto: são arquivos soltos
    em <windows_root>/<split>/<Setor>/ com o nome do subconjunto como
    prefixo (<Nome>.wNN.partKKK.pt). Portanto, apontar para a pasta
    <Setor> e varrer *.pt pegaria TODOS os países daquele setor — erro
    silencioso e grave num experimento por país.

    O ponto literal depois do nome evita colisão de prefixo (ex.: 'REED'
    não captura 'REEDD'; 'NEST-Commercial' e 'NEST-Residential' já ficam
    em setores distintos).
    """
    achados: list[Path] = []
    for grupo in grupos:                       # "Setor/Nome"
        setor, _, nome = grupo.partition("/")
        d = windows_root / split / setor
        if not d.exists():
            print(f"  [AVISO] pasta inexistente: {d}")
            continue
        f = sorted(d.glob(f"{nome}.w*.part*.pt"))
        if not f:
            print(f"  [AVISO] nenhum shard de '{nome}' em {d}")
        achados += f
    return achados


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


# ===================== PARTIÇÃO DAS JANELAS DO SHARD ========================

def particionar(starts, b_idx, val_frac: float, modo: str):
    """Índices (treino, validação) das janelas de um shard.

    Vetorizada: um lexsort + rank dentro do prédio, no lugar do laço
    Python sobre np.unique(b_idx) com argsort por prédio — que era refeito
    a cada época, para cada shard, e duas vezes por shard (uma por 'part').

    modo="atual"  -> reproduz EXATAMENTE ShardTemporalDataset(step6_train):
                     ambos os conjuntos são a CABEÇA cronológica (as
                     janelas anteriores à cauda de val_frac). É o padrão,
                     para não alterar nenhum número já produzido.
    modo="cauda"  -> treino = cabeça, validação = cauda (hold-out temporal
                     disjunto). Use só se decidir mudar o protocolo; os
                     resultados NÃO serão comparáveis aos anteriores.

    A equivalência do modo "atual" com a implementação vigente foi
    verificada índice a índice em casos aleatórios.
    """
    import numpy as np
    order = np.lexsort((starts, b_idx))          # por prédio, depois tempo
    b_sorted = b_idx[order]
    _, first, counts = np.unique(b_sorted, return_index=True,
                                 return_counts=True)
    sizes = np.repeat(counts, counts)
    rank = np.arange(len(order)) - np.repeat(first, counts)
    n_val = np.maximum(1, (sizes * val_frac).astype(int))
    cauda_sorted = rank >= (sizes - n_val)
    eh_cauda = np.empty(len(order), dtype=bool)
    eh_cauda[order] = cauda_sorted
    cabeca = np.where(~eh_cauda)[0]
    if modo == "cauda":
        return cabeca, np.where(eh_cauda)[0]
    return cabeca, cabeca            # "atual": os dois são a mesma cabeça


class _Batches:
    """Iterável de batches por FATIAMENTO DIRETO dos tensores do shard.

    Substitui DataLoader(num_workers=0) + default_collate. O shard já está
    inteiro em RAM como tensores contíguos, então o DataLoader só
    acrescentava, por shard e por época, ~200 mil construções de dict
    Python e centenas de chamadas de collate — puro overhead de
    interpretador num Cortex-A76. Aqui cada batch é um index_select
    (embaralhado) ou uma fatia contígua (sequencial): operações em C.

    'ct' é guardado em uint8 no cache (economiza 72 B/janela) e convertido
    para float aqui, no batch, porque é o que BCEWithLogitsLoss exige.

    Compatível com run_epoch por duck typing: entrega dicts com as mesmas
    cinco chaves e tensores já empilhados.
    """

    __slots__ = ("d", "bs", "shuffle", "seed", "n")

    def __init__(self, d: dict, batch_size: int, shuffle: bool, seed: int = 0):
        self.d, self.bs, self.shuffle, self.seed = d, batch_size, shuffle, seed
        self.n = int(d["ti"].shape[0])

    def __len__(self) -> int:
        return (self.n + self.bs - 1) // self.bs

    def __iter__(self):
        import torch as _t
        perm = None
        if self.shuffle:
            g = _t.Generator().manual_seed(int(self.seed))
            perm = _t.randperm(self.n, generator=g)
        for i in range(0, self.n, self.bs):
            sel = perm[i:i + self.bs] if self.shuffle else slice(i, i + self.bs)
            yield {"trend_input": self.d["ti"][sel],
                   "season_input": self.d["si"][sel],
                   "trend_target": self.d["tt"][sel],
                   "season_target": self.d["st"][sel],
                   "cls_target": self.d["ct"][sel].float()}


class ShardCache:
    """Cache de shards JÁ PROCESSADOS, com orçamento de RAM em GB, política
    LRU e cache opcional em disco LOCAL.

    Guarda só os cinco tensores que o treino consome (1.560 B/janela contra
    4.256 B/janela do .pt), e no modo "atual" — em que treino e validação
    são o mesmo conjunto — guarda UMA cópia, devolvida duas vezes.

    Ordem de busca: RAM -> disco local -> NAS.
      * hit de RAM   -> zero I/O e zero recálculo de partição;
      * hit de disco -> zero tráfego de rede, e ainda 65% menos bytes que
                        o .pt original, por ser o pacote compacto;
      * miss         -> lê do NAS uma vez (não duas, como antes).
    """

    def __init__(self, budget_gb: float, disk_dir: Path | None,
                 val_frac: float, modo: str):
        self.budget = float(budget_gb) * (2 ** 30)
        self.disk_dir = Path(disk_dir) if disk_dir else None
        if self.disk_dir:
            self.disk_dir.mkdir(parents=True, exist_ok=True)
        self.val_frac = val_frac
        self.modo = modo
        self._ram: dict[str, tuple] = {}     # chave -> (tr, va, bytes)
        self._ordem: list[str] = []          # LRU: mais antigo primeiro
        self.usado = 0
        self.hits_ram = self.hits_disco = self.miss = 0

    def _processar(self, pt_path: Path) -> tuple[dict, dict]:
        """UMA leitura do .pt bruto -> pacote(s) compacto(s)."""
        import numpy as np
        import torch
        from scripts.config import CFG
        pack = torch.load(pt_path, weights_only=False)
        B = CFG.backcast_length
        tr_idx, va_idx = particionar(pack["start"].numpy(),
                                     pack["building_idx"].numpy(),
                                     self.val_frac, self.modo)
        tn, sn, lb = pack["trend_norm"], pack["season_norm"], pack["labels_fused"]

        def _view(idx):
            i = torch.from_numpy(np.ascontiguousarray(idx))
            # .contiguous() é deliberado: garante que o cache guarde APENAS
            # as fatias e não mantenha viva uma referência ao pack inteiro
            # (que inclui x/trend/season/labels — 65% de peso morto aqui).
            return {"ti": tn[i, :B].contiguous(),
                    "si": sn[i, :B].contiguous(),
                    "tt": tn[i, B:].contiguous(),
                    "st": sn[i, B:].contiguous(),
                    "ct": lb[i, B:].contiguous()}      # uint8, vira float no batch

        tr = _view(tr_idx)
        # No modo "atual" os índices são o MESMO array: materializar duas
        # vezes dobraria a RAM sem nenhum ganho.
        va = tr if (tr_idx is va_idx or (len(tr_idx) == len(va_idx)
                    and bool((tr_idx == va_idx).all()))) else _view(va_idx)
        return tr, va

    @staticmethod
    def _bytes(tr: dict, va: dict) -> int:
        n = sum(t.numel() * t.element_size() for t in tr.values())
        if va is not tr:
            n += sum(t.numel() * t.element_size() for t in va.values())
        return n

    def _disk_path(self, pt_path: Path) -> Path:
        # nome estável e único: val_frac e modo mudam a partição, então
        # entram na chave — senão um cache antigo contaminaria o novo.
        h = hashlib.sha1(
            f"{pt_path}|{self.val_frac}|{self.modo}".encode()).hexdigest()[:16]
        return self.disk_dir / f"{pt_path.stem}.{h}.cache.pt"

    def _evict(self, preciso: int) -> None:
        while self._ordem and self.usado + preciso > self.budget:
            velho = self._ordem.pop(0)
            _, _, nb = self._ram.pop(velho)
            self.usado -= nb

    def get(self, pt_path: Path) -> tuple[dict, dict]:
        import torch
        chave = str(pt_path)
        if chave in self._ram:
            self.hits_ram += 1
            self._ordem.remove(chave)
            self._ordem.append(chave)                    # LRU touch
            tr, va, _ = self._ram[chave]
            return tr, va

        dp = self._disk_path(pt_path) if self.disk_dir else None
        if dp is not None and dp.exists():
            try:
                blob = torch.load(dp, weights_only=False)
                tr = blob["train"]
                va = tr if blob.get("val_igual") else blob["val"]
                self.hits_disco += 1
            except Exception as e:                       # cache corrompido
                print(f"  [cache] {dp.name} ilegível ({e}); relendo do NAS")
                dp.unlink(missing_ok=True)
                tr, va = self._processar(pt_path)
                self.miss += 1
        else:
            tr, va = self._processar(pt_path)
            self.miss += 1
            if dp is not None:
                try:
                    tmp = dp.with_suffix(".pt.tmp")
                    torch.save({"train": tr, "val_igual": va is tr,
                                "val": None if va is tr else va}, tmp)
                    tmp.replace(dp)
                except OSError as e:                     # disco cheio etc.
                    print(f"  [cache] não gravou {dp.name}: {e}")

        nb = self._bytes(tr, va)
        if nb <= self.budget:
            self._evict(nb)
            self._ram[chave] = (tr, va, nb)
            self._ordem.append(chave)
            self.usado += nb
        return tr, va

    def resumo(self) -> dict:
        tot = self.hits_ram + self.hits_disco + self.miss
        return {"hits_ram": self.hits_ram, "hits_disco": self.hits_disco,
                "miss_nas": self.miss,
                "taxa_acerto": round((self.hits_ram + self.hits_disco) /
                                     max(tot, 1), 3),
                "ram_usada_gb": round(self.usado / 2 ** 30, 2),
                "ram_orcamento_gb": round(self.budget / 2 ** 30, 2),
                "shards_em_ram": len(self._ram)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-root", type=Path, required=True,
                    help="pasta 02_windows/<resolução>, ex.: "
                         "$REAL/02_windows/Hourly")
    ap.add_argument("--pi", type=int, choices=[1, 2, 3, 4, 5], default=None,
                    help="usa o balde de países deste nó, conforme "
                         "country_map.py (recomendado)")
    ap.add_argument("--groups", nargs="+", default=None,
                    help="alternativa manual ao --pi: lista de "
                         "'Setor/Subconjunto', ex.: Residential/GoiEner")
    ap.add_argument("--country-map-dir", type=str, default=None,
                    help="pasta do country_map.py, se --pi for usado e o "
                         "arquivo não estiver junto ao script")
    ap.add_argument("--init-checkpoint", type=Path, required=True,
                    help="pesos iniciais (best do v0_both OU do v0_real)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10,
                    help="early stopping sobre a validação intra-shard")
    ap.add_argument("--min-delta", type=float, default=0.0,
                    help="melhora mínima da validação para zerar a "
                         "paciência. 0.0 = comportamento anterior; "
                         "1e-4 evita contar ruído de 4ª casa como melhora")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-shards", type=int, default=0,
                    help="0 = todos; use p/ calibrar tempo no Pi "
                         "(ex.: 15 shards ~ 3M janelas)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--val-mode", choices=["atual", "cauda"], default="atual",
                    help="'atual' (padrão) reproduz exatamente a partição "
                         "de ShardTemporalDataset; 'cauda' usa hold-out "
                         "temporal disjunto (números NÃO comparáveis)")
    ap.add_argument("--batch-size", type=int, default=0,
                    help="0 = CFG.batch_size (512). Em CPU, lotes maiores "
                         "costumam melhorar a eficiência do GEMM — teste "
                         "1024 por uma época antes de adotar")
    ap.add_argument("--threads", type=int, default=0,
                    help="0 = default do torch (4 no Pi 5, = nº de "
                         "núcleos). Raramente vale mexer")
    ap.add_argument("--cache-gb", type=float, default=0.0,
                    help="orçamento de RAM para o cache de shards "
                         "processados. 0 = desligado. No Pi 5 de 16 GB, "
                         "8 é seguro (15 shards ~ 4,4 GB)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="cache em disco LOCAL (evita reler o NAS num "
                         "miss de RAM). Ex.: /var/tmp/shard_cache")
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("./local_runs"))
    ap.add_argument("--repo-dir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    _add_repo_to_path(a.repo_dir)
    import numpy as np
    import torch
    import torch.nn as nn

    from step6_train import run_epoch                 # já validado
    from step4_5_labels_v2 import label_windows_batch
    from model_hybrid import HybridWLSTMix
    from scripts.config import CFG
    from fed_monitor import RunMonitor

    if a.threads > 0:
        torch.set_num_threads(a.threads)
    bs = a.batch_size or CFG.batch_size

    out = a.outdir / a.tag
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    print(f"[local] threads torch={torch.get_num_threads()} | batch={bs} | "
          f"val-mode={a.val_mode}")

    # ---------------- modelo + checkpoint inicial (strict) ----------------
    model = HybridWLSTMix(device).to(device)
    sd = torch.load(a.init_checkpoint, map_location=device)
    model.load_state_dict(sd, strict=True)   # divergência falha AQUI, não na época 30
    print(f"[local] checkpoint inicial: {a.init_checkpoint}")

    # ------------- resolução dos grupos (quais subconjuntos usar) -------------
    if a.pi and a.groups:
        raise SystemExit("use --pi OU --groups, não os dois")
    if a.pi:
        aqui = Path(__file__).resolve().parent
        for c in ([Path(a.country_map_dir)] if a.country_map_dir else []) + [
                aqui, aqui.parent, aqui.parent.parent, Path.cwd()]:
            if (c / "country_map.py").exists():
                sys.path.insert(0, str(c))
                break
        else:
            raise FileNotFoundError(
                "country_map.py não encontrado — use --country-map-dir "
                "ou passe os grupos manualmente com --groups")
        from country_map import groups_for_pi, pais_do_pi
        grupos = groups_for_pi(a.pi)
        print(f"[local] Pi{a.pi} -> países {pais_do_pi(a.pi)}")
    elif a.groups:
        grupos = a.groups
    else:
        raise SystemExit("informe --pi N ou --groups Setor/Nome [...]")
    print(f"[local] {len(grupos)} grupo(s): {', '.join(grupos)}")

    train_shards = _shards_de_grupos(a.windows_root, "train", grupos)
    test_shards = _shards_de_grupos(a.windows_root, "test", grupos)
    if not train_shards or not test_shards:
        raise FileNotFoundError(
            f"shards não encontrados (train={len(train_shards)}, "
            f"test={len(test_shards)}) — confira --windows-root e se o "
            f"passo 2-3 rodou para estes grupos em AMBOS os splits")
    if a.max_shards > 0 and len(train_shards) > a.max_shards:
        rng = np.random.default_rng(a.seed)
        idx = sorted(rng.choice(len(train_shards), a.max_shards,
                                replace=False).tolist())
        train_shards = [train_shards[i] for i in idx]
    print(f"[local] {len(train_shards)} shards de treino | "
          f"{len(test_shards)} de teste | device={device}")

    cache = ShardCache(a.cache_gb, a.cache_dir, a.val_frac, a.val_mode)
    if a.cache_gb > 0 or a.cache_dir:
        print(f"[local] cache: RAM={a.cache_gb} GB | disco={a.cache_dir}")

    # ---------------- treino em STREAMING por shard ----------------
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    best, wait = float("inf"), 0
    rng = np.random.default_rng(a.seed)
    historico: list[dict] = []

    mon = RunMonitor(out_dir=out, tag=a.tag,
                     total_units=len(train_shards) * a.epochs,
                     progress_every=5)
    gstep = 0
    t_treino = time.time()
    with mon:
        for ep in range(a.epochs):
            t0 = time.time()
            ordem = rng.permutation(len(train_shards))
            tr_sum = tr_n = va_sum = va_n = 0.0
            tr = va = float("nan")
            for si, k in enumerate(ordem, 1):
                # UMA leitura (ou hit de cache) por shard, não duas
                tr_d, va_d = cache.get(train_shards[k])
                n_tr = int(tr_d["ti"].shape[0])
                n_va = int(va_d["ti"].shape[0])
                if n_tr:
                    tr = run_epoch(model,
                                   _Batches(tr_d, bs, True, a.seed + gstep),
                                   mse, bce, device, scaler, opt)
                    tr_sum += tr * n_tr; tr_n += n_tr
                if n_va:
                    va = run_epoch(model, _Batches(va_d, bs, False),
                                   mse, bce, device, scaler)
                    va_sum += va * n_va; va_n += n_va
                mon.log_loss(tr, step=gstep, stage="fit", epoch=ep,
                             val_loss=va, n_train=n_tr, n_val=n_va,
                             shard=int(k))
                mon.tick()
                gstep += 1
                if si % 5 == 0 or si == len(ordem):
                    _atomic_json(out / "progress.json", {
                        "epoca": ep + 1, "epocas_alvo": a.epochs,
                        "shard": si, "shards_totais": len(ordem),
                        "train_parcial": round(tr_sum / max(tr_n, 1), 6),
                        "val_parcial": round(va_sum / max(va_n, 1), 6),
                        "melhor_val": None if best == float("inf") else round(best, 6),
                        "early_stop_wait": f"{wait}/{a.patience}",
                        "seg_epoca": round(time.time() - t0, 1),
                        "eta": mon.eta(), "cache": cache.resumo()})
            va_ep = va_sum / max(va_n, 1)
            tr_ep = tr_sum / max(tr_n, 1)
            dur = time.time() - t0
            historico.append({"epoca": ep + 1, "train": tr_ep, "val": va_ep,
                              "seg": round(dur, 1)})
            _atomic_json(out / f"history_{a.tag}.json", historico)
            rc = cache.resumo()
            print(f"[local] época {ep+1}/{a.epochs}: "
                  f"train={tr_ep:.4f} val={va_ep:.4f} ({dur:.0f}s) | "
                  f"cache {rc['taxa_acerto']:.0%} acerto, "
                  f"{rc['ram_usada_gb']:.1f} GB em RAM")
            if va_ep < best - a.min_delta:
                best, wait = va_ep, 0
                tmp = out / "best_local.pth.tmp"
                torch.save(model.state_dict(), tmp)
                tmp.replace(out / "best_local.pth")
            else:
                best = min(best, va_ep)
                wait += 1
                if wait >= a.patience:
                    print(f"[local] early stopping na época {ep+1} "
                          f"(paciência {a.patience}, min-delta {a.min_delta})")
                    break
    seg_treino = time.time() - t_treino

    ckpt_best = out / "best_local.pth"
    if ckpt_best.exists():
        model.load_state_dict(torch.load(ckpt_best, map_location=device))
    else:
        print("[local] AVISO: nenhuma época melhorou a validação além de "
              "min-delta; avaliando os pesos da ÚLTIMA época.")

    # ------- avaliação no TESTE: rótulo em execução + matriz de confusão -------
    # Os shards de TESTE não têm 'labels_fused' (step4_5 roda só sobre
    # 'train'), por isso o rótulo nasce aqui, das mesmas funções do passo 4.
    model.eval()
    B = CFG.backcast_length
    tp = tn = fp = fn = 0
    t_teste = time.time()
    with torch.no_grad():
        for sp in test_shards:
            pack = torch.load(sp, map_location="cpu")
            x = pack["x"].numpy()
            trend = pack["trend"].numpy()
            y = label_windows_batch(x, trend)          # rótulo intra-janela
            y_f = torch.tensor(y[:, B:], dtype=torch.float32)
            tn_in = pack["trend_norm"][:, :B]
            sn_in = pack["season_norm"][:, :B]
            # EM LOTES: a versão anterior fazia um único forward com o shard
            # de teste inteiro (até 200k janelas) — no Pi isso é um pico de
            # RAM desnecessário, já que o resultado é idêntico por lotes.
            preds = []
            for i in range(0, tn_in.shape[0], bs):
                _, _, logits = model(tn_in[i:i + bs].to(device),
                                     sn_in[i:i + bs].to(device))
                preds.append((torch.sigmoid(logits).cpu() > 0.5).float())
            pred = torch.cat(preds)
            tp += int(((pred == 1) & (y_f == 1)).sum())
            tn += int(((pred == 0) & (y_f == 0)).sum())
            fp += int(((pred == 1) & (y_f == 0)).sum())
            fn += int(((pred == 0) & (y_f == 1)).sum())
    seg_teste = time.time() - t_teste

    total = tp + tn + fp + fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    resumo_mon = mon.summary()
    cm = {"tag": a.tag, "init_checkpoint": str(a.init_checkpoint),
          "TP": tp, "TN": tn, "FP": fp, "FN": fn, "total_pontos": total,
          "precision": round(precision, 6), "recall": round(recall, 6),
          "f1": round(f1, 6),
          "accuracy": round((tp + tn) / max(total, 1), 6),
          "taxa_anomalia_teste": round((tp + fn) / max(total, 1), 6),
          # ---------- NOVO: tempo total e recursos do dispositivo ----------
          "wall_time_s": round(seg_treino + seg_teste, 1),
          "wall_time_treino_s": round(seg_treino, 1),
          "wall_time_teste_s": round(seg_teste, 1),
          "epocas_executadas": len(historico),
          "val_mode": a.val_mode,
          "cpu_pct_avg": resumo_mon.get("cpu_pct_avg"),
          "load1_avg": resumo_mon.get("load1_avg"),
          "ram_used_gb_avg": resumo_mon.get("ram_used_gb_avg"),
          "ram_used_gb_max": resumo_mon.get("ram_used_gb_max"),
          "cache": cache.resumo()}
    _atomic_json(out / "confusion_matrix.json", cm)
    _atomic_json(out / "metrics.json",
                 {**cm, "melhor_val_treino": best, "historico": historico,
                  "run_monitor": resumo_mon})
    print(json.dumps(cm, indent=2, default=str))

    # heatmap simples da matriz (opcional; só precisa de matplotlib)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        m = np.array([[tn, fp], [fn, tp]], dtype=float)
        figc, axc = plt.subplots(figsize=(4.2, 3.6))
        im = axc.imshow(m, cmap="Blues")
        for (i, j), v in np.ndenumerate(m):
            axc.text(j, i, f"{int(v):,}", ha="center", va="center",
                     color="black" if v < m.max() * 0.6 else "white")
        axc.set_xticks([0, 1], ["Normal", "Anomalia"])
        axc.set_yticks([0, 1], ["Normal", "Anomalia"])
        axc.set_xlabel("Predito"); axc.set_ylabel("Real")
        axc.set_title(f"Matriz de confusão — {a.tag}")
        figc.colorbar(im, shrink=0.8)
        figc.tight_layout()
        figc.savefig(out / "confusion_matrix.png", dpi=160)
        plt.close(figc)
        print(f"[local] matriz salva em {out/'confusion_matrix.png'}")
    except ImportError:
        print("[local] matplotlib ausente — só o JSON foi gerado")


if __name__ == "__main__":
    main()