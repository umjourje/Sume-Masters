"""task.py — Rotinas locais do Flower App: W-LSTMix + CLASSIFICADOR ao final.

Mantém a ESTRUTURA e as convenções do task original (load_config/get_model,
train() e evaluate() como funções puras devolvendo MetricRecord-ready dicts
com num-examples, métricas do paper CVRMSE/NRMSE com fallback, test_loss
como métrica de seleção do servidor):

  * MODELO: HybridWLSTMix — backbone models/W_LSTMix.py ORIGINAL,
    inalterado, + bloco de classificação ao final. O v0 centralizado é
    instanciado na rodada 1 pelo servidor.
  * DADOS DE TREINO: artefatos anti-leak do pipeline (passo 4-5 v2)
    (<data_root>/02_windows/<res>/train/**.pt, com labels_fused).
  * AVALIAÇÃO: shards de TESTE (sem rótulos) com rotulagem EM RUNTIME
    (label_windows_batch + fuse_labels), devolvendo test_loss (conjunto),
    métricas de forecasting do paper E métricas de classificação.

Observabilidade (fed_monitor.RunMonitor) em train() e evaluate():
wall_time_s, CPU%/load/RAM médios, ETA em progress_<tag>.json, loss por
shard em loss_<tag>.jsonl e confusion_matrix_<tag>.json por cliente.

======================= NOVO NESTA VERSÃO =================================

1) PARTIÇÃO POR Pi (`pi`) — CORREÇÃO DE VALIDADE EXPERIMENTAL.

   O train_local_pi.py (centralizado) recebe `--pi N` e restringe o treino
   aos grupos daquele nó via country_map.groups_for_pi(). Este task.py
   NÃO tinha equivalente: fazia rglob("*.pt") sobre todo o data_root.

   Isso é inofensivo quando cada Pi tem uma CÓPIA LOCAL só da sua
   partição, mas é FATAL quando os clientes montam o MESMO storage
   compartilhado (o caso real: /mnt/juliana-truenas/...). Ali, os 5
   clientes leriam o dataset INTEIRO — deixaria de ser Non-IID e viraria
   5 réplicas do centralizado sob FedAvg, sem erro nenhum e com métricas
   plausíveis. Falha silenciosa é a pior espécie.

   Agora: `pi` vem do node-config de cada SuperNode. pi>0 filtra os
   shards pelos grupos "<setor>/<subconjunto>" de country_map (o MESMO
   formato do --group do pipeline). pi=0 (default) mantém o comportamento
   antigo — use apenas se o data_root já for exclusivo do dispositivo.

2) max_windows — knob OPCIONAL de janelas por shard, para depuração
   ultrarrápida. Default 0 (shard inteiro). Pelas medições reais no Pi 5
   (15 shards, GoiEner, ~48-144 s por shard-época) ele NÃO é necessário
   para o smoke test; fica disponível para diagnóstico.

   No TREINO a amostra dentro do shard é uniformemente espaçada
   (preserva diversidade temporal). Na AVALIAÇÃO é um prefixo CONTÍGUO:
   fuse_labels() e a média de probabilidades por ponto dependem da
   SOBREPOSIÇÃO entre janelas vizinhas; um subconjunto esparso deixaria
   a cobertura cheia de buracos e distorceria precision/recall.

Requisito: PIPELINE_DIR no ambiente apontando para a pasta do pipeline
(onde vivem config.py, step6_train.py, country_map.py, fed_monitor.py).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.environ.get("PIPELINE_DIR", "."))
from config import CFG                                       # noqa: E402
from country_map import groups_for_pi                        # noqa: E402
from model_hybrid import HybridWLSTMix                       # noqa: E402
from step4_5_labels_v2 import label_windows_batch, fuse_labels  # noqa: E402
from step6_train import (WindowedPTDataset, LazyWindowedPTDataset,  # noqa: E402
                         run_epoch)
from fed_monitor import RunMonitor                           # noqa: E402

# Métricas do paper, com fallback (convenção do task original)
try:
    from my_utils.metrics import cal_cvrmse, cal_mae, cal_mse, cal_nrmse
    _METRICS_FALLBACK = False
except ImportError:
    _METRICS_FALLBACK = True

    def cal_mse(p, t): return float(np.mean((p - t) ** 2))
    def cal_mae(p, t): return float(np.mean(np.abs(p - t)))
    def cal_cvrmse(p, t):
        return float(np.sqrt(np.mean((p - t) ** 2)) /
                     (np.mean(t) + 1e-8))
    def cal_nrmse(p, t):
        return float(np.sqrt(np.mean((p - t) ** 2)) /
                     (np.ptp(t) + 1e-8))


def load_config() -> dict:
    """Compatibilidade com a assinatura do task original: a config agora é
    o CFG central do pipeline (mesma de treino e clientes, por construção)."""
    return {"cfg": CFG, "metrics_fallback": _METRICS_FALLBACK}


def get_model(cfg: dict, device: torch.device) -> HybridWLSTMix:
    return HybridWLSTMix(device).to(device)


# --------------------------- seleção de shards -----------------------------
def _load_local(split: str, data_root: Path, pi: int = 0) -> list[Path]:
    """Shards do split, restritos à partição do dispositivo quando pi>0.

    O casamento é por SUBSTRING do caminho relativo contra os grupos
    "<setor>/<subconjunto>" de country_map — o mesmo critério do --group
    usado pelos passos 2-3 e 4-5, portanto consistente com o layout que
    o pipeline gravou (02_windows/<res>/<split>/<setor>/<subconj>/...).
    """
    base = Path(data_root) / "02_windows" / CFG.resolution / split
    if not base.exists():
        return []
    pts = sorted(base.rglob("*.pt"))
    if pi <= 0:
        return pts
    grupos = [g.lower().replace("\\", "/") for g in groups_for_pi(int(pi))]
    if not grupos:
        raise RuntimeError(f"country_map não define grupos para pi={pi}")
    keep = []
    for p in pts:
        rel = str(p.relative_to(base)).lower().replace(os.sep, "/")
        if any(g in rel for g in grupos):
            keep.append(p)
    return keep


def _amostrar_shards(pts: list, max_shards: int) -> list:
    """Amostra uniformemente espaçada de shards (mesma técnica do
    _estimar_pos_weight): preserva diversidade temporal/por prédio sem
    concentrar no início da lista ordenada. max_shards<=0 = todos."""
    if max_shards <= 0 or len(pts) <= max_shards:
        return pts
    idx = np.linspace(0, len(pts) - 1, max_shards).astype(int)
    return [pts[i] for i in idx]


def _idx_janelas(n: int, max_windows: int):
    """Índices uniformemente espaçados de janelas DENTRO de um shard.
    Devolve None quando não há nada a cortar (caminho normal, sem cópia
    nem overhead)."""
    if max_windows <= 0 or n <= max_windows:
        return None
    return np.linspace(0, n - 1, max_windows).astype(int)


def _monitor_metrics(summary: dict) -> dict:
    """Achata o summary do RunMonitor em escalares float com prefixo
    mon_ — apenas chaves numéricas (MetricRecord não aceita strings).
    A TensorBoardFedAvg passa a plotá-las por cliente sem mudança;
    na agregação viram média ponderada de recursos (inócuo)."""
    out = {}
    for k in ("wall_time_s", "cpu_pct_avg", "cpu_pct_max", "load1_avg",
              "ram_used_gb_avg", "ram_used_gb_max", "rss_gb_max",
              "units_per_min"):
        v = summary.get(k)
        if v is not None:
            out[f"mon_{k}"] = float(v)
    return out


# ------------------------------- TREINO -----------------------------------
def _contar_janelas(pts, max_windows: int = 0) -> int:
    """Total de janelas SEM carregar os tensores pesados: o
    LazyWindowedPTDataset lê só o cabeçalho (via mmap). Necessário porque
    'num-examples' é o peso do cliente no FedAvg e precisa ser o total
    REALMENTE treinado — daí o min() com max_windows."""
    total = 0
    for p in pts:
        n = len(LazyWindowedPTDataset(p))
        total += n if max_windows <= 0 else min(n, max_windows)
    return total


def _estimar_pos_weight(pts, device, max_shards: int = 8) -> torch.Tensor:
    """pos_weight do BCE estimado a partir de uma AMOSTRA de shards.
    Concatenar os rótulos de TODOS os shards materializaria o dataset
    inteiro em RAM — a mesma causa raiz do estouro que motivou o
    streaming no passo 6. A taxa de anomalia é estável entre shards
    (definida por percentis intra-janela), então a amostra basta."""
    amostra = pts if len(pts) <= max_shards else [
        pts[i] for i in np.linspace(0, len(pts) - 1, max_shards).astype(int)]
    pos = tot = 0.0
    for p in amostra:
        ds = WindowedPTDataset(p)
        pos += float(ds.cls_tg.sum())
        tot += float(ds.cls_tg.numel())
        del ds
    pos = max(pos, 1.0)
    return torch.tensor((tot - pos) / pos, device=device)


def train(model: HybridWLSTMix, data_root: Path, device,
          epochs: int = 1, lr: float = 1e-3,
          tag: str = "run", pi: int = 0,
          max_shards: int = 0, max_windows: int = 0,
          metrics_dir: Path | None = None, round_no: int | None = None) -> dict:
    """Função pura (sem early stopping local — a decisão é do servidor,
    via checkpoint do melhor global na TensorBoardFedAvg).

    STREAMING POR SHARD (não eager): carrega um shard, treina nele,
    descarta. O pico de RAM é o de UM shard, não a soma de todos — mesma
    estratégia validada no passo 6 (_fit_scale), que rodou 116h com RAM
    estável em ~2,2GB.

    A ordem dos shards é embaralhada a cada época (o shuffle interno do
    DataLoader só embaralha DENTRO do shard corrente).

    Comparabilidade com o centralizado: use o MESMO max_shards que o
    train_local_pi.py usou (--max-shards 15 nos runs de referência), para
    que a diferença medida entre arquiteturas não se confunda com
    diferença de orçamento de dados.
    """
    pts = _amostrar_shards(_load_local("train", data_root, pi), max_shards)
    if not pts:
        raise RuntimeError(
            f"Sem shards de treino em {data_root}/02_windows/"
            f"{CFG.resolution}/train para pi={pi}. Se o data_root é um "
            f"storage COMPARTILHADO, 'pi' precisa vir do node-config; se "
            f"é cópia local exclusiva, use pi=0.")

    n_total = _contar_janelas(pts, max_windows)   # peso do FedAvg, sem carregar
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=_estimar_pos_weight(pts, device))
    mse = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=CFG.use_amp and
                                  device.type == "cuda")
    rng = np.random.default_rng(0)

    mdir = Path(metrics_dir) if metrics_dir else Path("metrics")
    mon = RunMonitor(out_dir=mdir, tag=tag,
                     total_units=len(pts) * epochs,   # unidade do ETA: shard
                     progress_every=1)                # poucos shards/rodada
    losses = []
    with mon:
        gstep = 0
        for e in range(epochs):
            soma = peso = 0.0
            for k in rng.permutation(len(pts)):
                ds = WindowedPTDataset(pts[k])          # 1 shard por vez
                if not len(ds):
                    continue
                sub = _idx_janelas(len(ds), max_windows)
                if sub is not None:                     # só se max_windows>0
                    ds = Subset(ds, sub.tolist())
                # num_workers=0 no streaming: o shard JÁ está em RAM, então
                # workers de IPC não trazem ganho — e, sob a estratégia
                # 'file_system', recriar DataLoaders a cada shard vaza
                # memória via /dev/shm (diagnosticado no passo 6).
                loader = DataLoader(ds, batch_size=CFG.batch_size,
                                    shuffle=True, num_workers=0,
                                    pin_memory=device.type == "cuda")
                l = run_epoch(model, loader, mse, bce, device, scaler, opt)
                soma += float(l) * len(ds)              # média ponderada real
                peso += len(ds)
                mon.log_loss(float(l), step=gstep, stage="fit",
                             round=round_no, epoch=e,
                             n_windows=len(ds), shard=int(k))
                mon.tick()                              # ETA por shard
                gstep += 1
                del loader, ds
            losses.append(soma / max(peso, 1.0))

    summary = mon.summary()
    return {
        "train_loss": float(losses[-1]),
        "train_loss_epochs": [float(v) for v in losses],  # curva p/ TB
        "num-examples": n_total,                          # peso do FedAvg
        "n_shards": len(pts),
        **_monitor_metrics(summary),   # wall_time_s, CPU/RAM, unid/min
    }


# ------------------------------ AVALIAÇÃO ----------------------------------
@torch.no_grad()
def evaluate(model: HybridWLSTMix, data_root: Path, device,
             threshold: float = 0.5,
             tag: str = "run", pi: int = 0,
             max_shards: int = 0, max_windows: int = 0,
             metrics_dir: Path | None = None,
             round_no: int | None = None) -> dict:
    """Modelo GLOBAL na partição de TESTE local: rótulos em runtime +
    test_loss conjunto (métrica de seleção do servidor) + forecasting
    (paper) + classificação.

    Grava <metrics_dir>/confusion_matrix_<tag>.json com o esquema já
    usado no projeto + wall_time_s e recursos do dispositivo."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    B, F = CFG.backcast_length, CFG.forecast_length
    pts = _amostrar_shards(_load_local("test", data_root, pi), max_shards)
    if not pts:
        # Sem shards de teste NÃO existe test_loss -> a estratégia do
        # servidor nunca acha selection_metric e best_model_global.pth
        # nunca é gravado. Falhar alto é melhor que falhar em silêncio.
        raise RuntimeError(
            f"Sem shards de TESTE em {data_root}/02_windows/"
            f"{CFG.resolution}/test para pi={pi} — é o teste que produz "
            f"test_loss, a métrica de seleção do checkpoint global.")
    mse = torch.nn.MSELoss()
    bce = torch.nn.BCEWithLogitsLoss()
    losses, Yc, Pc = [], [], []
    yf_pred, yf_true = [], []
    n_windows = 0

    mdir = Path(metrics_dir) if metrics_dir else Path("metrics")
    mon = RunMonitor(out_dir=mdir, tag=f"{tag}_eval",
                     total_units=len(pts), progress_every=1)
    with mon:
        for pt_i, pt in enumerate(pts):
            pack = torch.load(pt, weights_only=False)
            n_all = int(np.asarray(pack["start"]).shape[0])
            k_win = n_all if max_windows <= 0 else min(n_all, max_windows)
            sl0 = slice(0, k_win)                  # prefixo contíguo
            x = np.asarray(pack["x"][sl0], dtype=np.float64)
            trend = np.asarray(pack["trend"][sl0], dtype=np.float64)
            # Rotulagem EM RUNTIME (dados novos) — mesmas funções do pipeline:
            win_labels = label_windows_batch(x, trend)
            b_idx = np.asarray(pack["building_idx"])[sl0]
            starts = np.asarray(pack["start"])[sl0]
            ti = pack["trend_norm"][sl0, :B]
            si = pack["season_norm"][sl0, :B]
            tt = pack["trend_norm"][sl0, B:]
            st_t = pack["season_norm"][sl0, B:]
            probs = []
            for i in range(0, len(ti), CFG.batch_size):
                sl = slice(i, i + CFG.batch_size)
                a, b, c, d = (t.to(device) for t in (ti[sl], si[sl],
                                                     tt[sl], st_t[sl]))
                t_pred, s_pred, logits = model(a, b)
                l_t, l_s = mse(t_pred, c), mse(s_pred, d)
                ssum = l_t + l_s
                l_fore = (l_s / ssum) * l_t + (l_t / ssum) * l_s
                # alvo de classificação do lote (fusão vem depois; aqui usa o
                # rótulo por janela p/ o loss, coerente com dados nunca vistos)
                ct = torch.tensor(win_labels[sl][:, B:],
                                  dtype=torch.float32, device=device)
                loss = l_fore + CFG.lambda_cls * bce(logits, ct)
                losses.append(float(loss))
                probs.append(torch.sigmoid(logits.float()).cpu().numpy())
                yf_pred.append(t_pred.float().cpu().numpy().ravel())
                yf_true.append(c.float().cpu().numpy().ravel())
            probs = np.concatenate(probs)
            n_windows += len(probs)
            mon.log_loss(float(np.mean(losses)), step=pt_i,
                         stage="evaluate", round=round_no,
                         n_windows=int(k_win))
            mon.tick()
            # Classificação pontual com rótulos FUNDIDOS por prédio:
            for bi in np.unique(b_idx):
                m = b_idx == bi
                L = int(starts[m].max()) + B + F
                y_true = fuse_labels(L, starts[m], win_labels[m])
                psum = np.zeros(L); pcnt = np.zeros(L)
                for j, st in zip(np.where(m)[0], starts[m]):
                    psum[st + B: st + B + F] += probs[j]
                    pcnt[st + B: st + B + F] += 1
                cov = pcnt > 0
                Yc.append(y_true[cov]); Pc.append(psum[cov] / pcnt[cov])

    y = np.concatenate(Yc); p = np.concatenate(Pc)
    pred = (p >= threshold).astype(int)
    fp_, ft = np.concatenate(yf_pred), np.concatenate(yf_true)
    summary = mon.summary()

    # ---- confusion_matrix.json por cliente (esquema do projeto + tempo) ----
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    f1v = float(f1_score(y, pred, zero_division=0))
    prec = float(precision_score(y, pred, zero_division=0))
    rec = float(recall_score(y, pred, zero_division=0))
    cm = {
        "tag": tag,
        "pi": int(pi),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "total_pontos": int(y.size),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1v, 6),
        "accuracy": round(float((tp + tn) / max(y.size, 1)), 6),
        "taxa_anomalia_teste": round(float(y.mean()), 6),
        # amostragem usada (para não confundir smoke com run completo)
        "n_shards": len(pts),
        "max_shards": int(max_shards),
        "max_windows": int(max_windows),
        # tempo total e recursos do dispositivo
        "wall_time_s": summary.get("wall_time_s"),
        "run_monitor": summary,
    }
    cm_path = mdir / f"confusion_matrix_{tag}.json"
    tmp = cm_path.with_suffix(".json.tmp")
    import json as _json
    tmp.write_text(_json.dumps(cm, indent=2, ensure_ascii=False))
    os.replace(tmp, cm_path)

    return {
        "test_loss": float(np.mean(losses)),      # métrica de seleção
        "cvrmse": cal_cvrmse(fp_, ft),            # forecasting (paper)
        "nrmse": cal_nrmse(fp_, ft),
        "mae": cal_mae(fp_, ft),
        "mse": cal_mse(fp_, ft),
        "f1": f1v,                                # classif.
        "precision": prec,
        "recall": rec,
        "anomaly_rate": float(y.mean()),
        "metrics_fallback": int(_METRICS_FALLBACK),
        "num-examples": int(n_windows),
        **_monitor_metrics(summary),   # mon_wall_time_s, mon_cpu_pct_avg, …
    }