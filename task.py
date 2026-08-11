"""task.py — Rotinas locais do Flower App: W-LSTMix + CLASSIFICADOR ao final.

FUSÃO: mantém a ESTRUTURA e as convenções do seu task.py original
(load_config/get_model, train() e evaluate() como funções puras devolvendo
MetricRecord-ready dicts com num-examples, métricas do paper CVRMSE/NRMSE
com fallback, test_loss como métrica de seleção do servidor), trocando o
INTERIOR pela lógica consolidada neste chat:

  * MODELO: HybridWLSTMix — backbone models/W_LSTMix.py ORIGINAL,
    inalterado, + bloco de classificação ao final (o mesmo do passo 6;
    o v0 centralizado da máquina grande é instanciado na rodada 1).
  * DADOS DE TREINO: artefatos anti-leak do pipeline
    (<data_root>/02_windows/<res>/train/**.pt, com labels_fused) —
    substitui a decomposição sobre a série inteira do task antigo.
  * AVALIAÇÃO: shards de TESTE (sem rótulos) com rotulagem EM RUNTIME
    (label_windows_batch + fuse_labels), devolvendo test_loss (conjunto),
    métricas de forecasting do paper E métricas de classificação.

Requisito: PIPELINE_DIR no ambiente apontando para a pasta do pipeline.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.environ.get("PIPELINE_DIR", "."))
from config import CFG                                       # noqa: E402
from model_hybrid import HybridWLSTMix                       # noqa: E402
from step4_5_labels import label_windows_batch, fuse_labels  # noqa: E402
from step6_train import (WindowedPTDataset, LazyWindowedPTDataset,  # noqa: E402
                         run_epoch)

# Métricas do paper, com fallback (convenção do seu task original)
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


def _load_local(split: str, data_root: Path):
    base = Path(data_root) / "02_windows" / CFG.resolution / split
    return sorted(base.rglob("*.pt")) if base.exists() else []


# ------------------------------- TREINO -----------------------------------
def _contar_janelas(pts) -> int:
    """Total de janelas SEM carregar os tensores pesados: o
    LazyWindowedPTDataset lê só o cabeçalho (via mmap). Necessário porque
    'num-examples' é o peso do cliente no FedAvg e precisa ser o total
    real, não uma estimativa."""
    return sum(len(LazyWindowedPTDataset(p)) for p in pts)


def _estimar_pos_weight(pts, device, max_shards: int = 8) -> torch.Tensor:
    """pos_weight do BCE estimado a partir de uma AMOSTRA de shards.
    Antes era calculado concatenando os rótulos de TODOS os shards
    (torch.cat), o que materializava o dataset inteiro em RAM — a mesma
    causa raiz do estouro de memória que motivou o streaming no passo 6.
    A taxa de anomalia é estável entre shards (é definida por percentis
    intra-janela), então uma amostra dá a mesma ordem de grandeza."""
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
          epochs: int = 1, lr: float = 1e-3) -> dict:
    """Função pura (sem early stopping local — decisão é do servidor,
    via checkpoint do melhor global na TensorBoardFedAvg).

    STREAMING POR SHARD (não eager): carrega um shard, treina nele,
    descarta. O pico de RAM é o de UM shard (~2GB), não a soma de todos.
    Motivo: o cliente do nó da Espanha tem ~26,3 milhões de janelas
    (~118GB de shards) contra 16GB de RAM no Raspberry Pi — a carga
    eager anterior (ConcatDataset de todos os shards) não caberia.
    É a mesma estratégia validada no passo 6 (_fit_scale), que rodou
    116h com RAM estável em ~2,2GB.

    A ordem dos shards é embaralhada a cada época (o shuffle interno do
    DataLoader só embaralha DENTRO do shard corrente).
    """
    pts = _load_local("train", data_root)
    if not pts:
        raise RuntimeError(f"Sem shards de treino em {data_root}")

    n_total = _contar_janelas(pts)          # peso do FedAvg, sem carregar
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=_estimar_pos_weight(pts, device))
    mse = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=CFG.use_amp and
                                  device.type == "cuda")
    rng = np.random.default_rng(0)

    losses = []
    for _ in range(epochs):
        soma = peso = 0.0
        for k in rng.permutation(len(pts)):
            ds = WindowedPTDataset(pts[k])          # 1 shard por vez
            if not len(ds):
                continue
            # num_workers=0 no streaming: o shard JÁ está em RAM, então
            # workers de IPC não trazem ganho — e, sob a estratégia
            # 'file_system', recriar DataLoaders a cada shard vaza
            # memória via /dev/shm (diagnosticado no passo 6).
            loader = DataLoader(ds, batch_size=CFG.batch_size, shuffle=True,
                                num_workers=0,
                                pin_memory=device.type == "cuda")
            l = run_epoch(model, loader, mse, bce, device, scaler, opt)
            soma += float(l) * len(ds)              # média ponderada real
            peso += len(ds)
            del loader, ds
        losses.append(soma / max(peso, 1.0))

    return {
        "train_loss": float(losses[-1]),
        "train_loss_epochs": [float(v) for v in losses],  # curva p/ TB
        "num-examples": n_total,                          # peso do FedAvg
    }


# ------------------------------ AVALIAÇÃO ----------------------------------
@torch.no_grad()
def evaluate(model: HybridWLSTMix, data_root: Path, device,
             threshold: float = 0.5) -> dict:
    """Modelo GLOBAL na partição de TESTE local: rótulos em runtime +
    test_loss conjunto (métrica de seleção do servidor) + forecasting
    (paper) + classificação."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    B, F = CFG.backcast_length, CFG.forecast_length
    pts = _load_local("test", data_root)
    if not pts:
        return {"num-examples": 0}
    mse = torch.nn.MSELoss()
    bce = torch.nn.BCEWithLogitsLoss()
    losses, Yc, Pc = [], [], []
    yf_pred, yf_true = [], []
    n_windows = 0
    for pt in pts:
        pack = torch.load(pt, weights_only=False)
        x = np.asarray(pack["x"], dtype=np.float64)
        trend = np.asarray(pack["trend"], dtype=np.float64)
        # Rotulagem EM RUNTIME (dados novos) — mesmas funções do pipeline:
        win_labels = label_windows_batch(x, trend)
        b_idx = np.asarray(pack["building_idx"])
        starts = np.asarray(pack["start"])
        ti = pack["trend_norm"][:, :B]
        si = pack["season_norm"][:, :B]
        tt = pack["trend_norm"][:, B:]
        st_t = pack["season_norm"][:, B:]
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
    fp, ft = np.concatenate(yf_pred), np.concatenate(yf_true)
    return {
        "test_loss": float(np.mean(losses)),      # métrica de seleção
        "cvrmse": cal_cvrmse(fp, ft),             # forecasting (paper)
        "nrmse": cal_nrmse(fp, ft),
        "mae": cal_mae(fp, ft),
        "mse": cal_mse(fp, ft),
        "f1": float(f1_score(y, pred, zero_division=0)),          # classif.
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "anomaly_rate": float(y.mean()),
        "metrics_fallback": int(_METRICS_FALLBACK),
        "num-examples": int(n_windows),
    }