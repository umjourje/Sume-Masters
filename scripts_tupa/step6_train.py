"""
PASSO 6 — Treino do modelo híbrido (backbone W-LSTMix + bloco de
classificação ao final) e teste com rotulagem em tempo real.

MODOS (flag --mode):
  train  : lê os shards .pt já prontos (02_windows, com labels_fused do
           passo 4-5) e treina com validação rolling-origin. NENHUMA
           rotulagem acontece aqui — os dados já estão prontos em disco.
  test   : recebe dados CRUS no formato do dataset puro (timestamp +
           medida), e SÓ AQUI os passos 3-5 rodam em tempo de execução
           (funções vetorizadas importadas dos passos 2-3/4-5) para gerar
           os rótulos dos dados novos e medir o modelo.

Arquitetura: models/W_LSTMix.py ORIGINAL, inalterado; a única adição é a
cabeça de classificação (model_hybrid.HybridWLSTMix) — um logit por
timestep do horizonte, treinada com perda conjunta forecast + BCE.

Velocidade (máquina grande):
  * TF32 + cudnn.benchmark;
  * AMP (mixed precision) com GradScaler (CFG.use_amp);
  * DataLoader com CFG.loader_workers, pin_memory, persistent_workers e
    prefetch — a GPU não espera o disco.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset

from config import CFG
from perf_log import RunLogger, _fmt_dur
from model_hybrid import HybridWLSTMix
from step2_3_windows_wavelet_v2 import (make_windows, decompose_windows_batch,
                                     standardize_batch)
from step4_5_labels_v2 import label_windows_batch, fuse_labels


def _speed_setup():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


# ============================== TREINO =====================================
class WindowedPTDataset(Dataset):
    """Carga EAGER: mantém os tensores do shard vivos na RAM. Rápido para
    acesso, mas o custo de memória é a soma de TODOS os shards carregados —
    inviável para o sintético inteiro (bilhões de janelas)."""
    def __init__(self, pt_path):
        pack = torch.load(pt_path, weights_only=False)
        B = CFG.backcast_length
        self.trend_in = pack["trend_norm"][:, :B]
        self.season_in = pack["season_norm"][:, :B]
        self.trend_tg = pack["trend_norm"][:, B:]
        self.season_tg = pack["season_norm"][:, B:]
        self.cls_tg = pack["labels_fused"][:, B:].float()
        self.building_idx = pack["building_idx"]
        self.starts = pack["start"]
        self.group = pt_path.stem                 # inclui .wNN.partKKK: único

    def __len__(self):
        return self.trend_in.shape[0]

    def __getitem__(self, i):
        return {"trend_input": self.trend_in[i],
                "season_input": self.season_in[i],
                "trend_target": self.trend_tg[i],
                "season_target": self.season_tg[i],
                "cls_target": self.cls_tg[i]}


class LazyWindowedPTDataset(Dataset):
    """Carga LAZY: lê apenas o CABEÇALHO na construção (nº de janelas,
    building_idx, starts — baratos) e mantém os tensores pesados fora da
    RAM até serem pedidos, com um cache de 1 shard. Adequado quando a soma
    dos shards não cabe em memória; o pico de RAM é o de um shard por
    worker do DataLoader. As colunas 'group'/'building_idx'/'starts' ficam
    disponíveis para o rolling-origin sem carregar os pesos."""
    _cache_path = None
    _cache = None

    def __init__(self, pt_path):
        self.path = pt_path
        self.group = pt_path.stem
        # Só metadados: carrega uma vez para dimensionar e indexar folds.
        pack = torch.load(pt_path, weights_only=False, mmap=True)
        self._n = int(pack["trend_norm"].shape[0])
        self.building_idx = pack["building_idx"].clone()
        self.starts = pack["start"].clone()
        del pack

    def __len__(self):
        return self._n

    def _load(self):
        # cache de 1 shard por PROCESSO (compartilhado entre instâncias):
        if LazyWindowedPTDataset._cache_path != self.path:
            LazyWindowedPTDataset._cache = torch.load(
                self.path, weights_only=False)
            LazyWindowedPTDataset._cache_path = self.path
        return LazyWindowedPTDataset._cache

    def __getitem__(self, i):
        pack = self._load()
        B = CFG.backcast_length
        return {"trend_input": pack["trend_norm"][i, :B],
                "season_input": pack["season_norm"][i, :B],
                "trend_target": pack["trend_norm"][i, B:],
                "season_target": pack["season_norm"][i, B:],
                "cls_target": pack["labels_fused"][i, B:].float()}


def _windows_root_of(out_root) -> Path:
    """windows_root (02_windows/<res>) para um OUT_ROOT arbitrário."""
    return Path(out_root) / "02_windows" / CFG.resolution


def load_split(split: str, out_root=None, logger=None) -> ConcatDataset:
    """Carrega os shards de um split, com PROGRESSO visível. out_root=None
    usa o dataset corrente (CFG.out_root). O tipo de dataset (eager/lazy)
    é escolhido por CFG.lazy_loading — lazy evita o estouro de RAM ao
    carregar fontes grandes (ex.: o sintético inteiro)."""
    import time as _t
    base = (_windows_root_of(out_root) if out_root is not None
            else CFG.windows_root) / split
    pts = sorted(base.rglob("*.pt")) if base.exists() else []
    if not pts:
        raise RuntimeError(f"Nenhum .pt em {base} — rode os passos 1-5.")
    cls = (LazyWindowedPTDataset if getattr(CFG, "lazy_loading", False)
           else WindowedPTDataset)
    msg = (f"[step6] carregando {len(pts)} shards de {base} "
           f"({'LAZY' if cls is LazyWindowedPTDataset else 'EAGER'})...")
    (logger.term(msg) if logger else print(msg, flush=True))
    ds, t0, n_win = [], _t.time(), 0
    for k, p in enumerate(pts, 1):
        d = cls(p)
        ds.append(d)
        n_win += len(d)
        # LINHA CRUCIAL (visibilidade): progresso + snapshot de recursos a
        # cada bloco — sem isso o carregamento é totalmente às cegas.
        if logger and (k % 50 == 0 or k == len(pts)):
            logger.term(f"[step6]   {k}/{len(pts)} shards | "
                        f"{n_win:,} janelas | {_fmt_dur(_t.time() - t0)}")
            logger.snapshot(f"carregando {out_root or 'corrente'} "
                            f"shard {k}/{len(pts)}")
    return ConcatDataset(ds)


def _resolve_scopes(data_scope: str):
    """Resolve (fonte_sintética, fonte_real) em OUT_ROOTs concretos.

    Convenção: CFG.out_root é o dataset 'corrente' (aponte-o via .env para
    o SINTÉTICO ao treinar v0 com sintético); CFG.out_root_real
    (OUT_ROOT_REAL, opcional) é o REAL. Para --data-scope real, se
    out_root_real não estiver definido, assume-se que o próprio OUT_ROOT
    já é o real."""
    synth = CFG.out_root
    real = CFG.out_root_real or CFG.out_root
    if data_scope == "synthetic":
        return [("synthetic", synth)]
    if data_scope == "real":
        return [("real", real)]
    if data_scope == "both":
        if CFG.out_root_real is None:
            raise RuntimeError(
                "--data-scope both exige OUT_ROOT_REAL no .env (sintético em "
                "OUT_ROOT, real em OUT_ROOT_REAL).")
        return [("synthetic", synth), ("real", real)]
    raise ValueError(data_scope)


def rolling_origin_folds(full: ConcatDataset):
    from collections import defaultdict
    B, F = CFG.backcast_length, CFG.forecast_length
    day = CFG.val_horizon_steps
    per_building = defaultdict(list)
    offset = 0
    for ds in full.datasets:
        for i in range(len(ds)):
            key = f"{ds.group}:{int(ds.building_idx[i])}"
            per_building[key].append((offset + i, int(ds.starts[i])))
        offset += len(ds)
    folds = []
    for j in range(CFG.n_rolling_folds):
        tr, va = [], []
        for key, wins in per_building.items():
            series_end = max(st for _, st in wins) + B + F
            c0 = int(series_end * CFG.initial_train_frac)
            c0 -= c0 % CFG.stride
            cutoff = c0 + j * day
            for gi, st in wins:
                end = st + B + F
                if CFG.rolling_mode == "sliding":
                    in_train = end <= cutoff and st >= cutoff - CFG.train_span_steps
                else:
                    in_train = end <= cutoff
                if in_train:
                    tr.append(gi)
                elif st + B >= cutoff and end <= cutoff + day:
                    va.append(gi)
        if tr and va:
            folds.append((tr, va))
    if not folds:
        raise RuntimeError("Nenhum fold viável — ajuste initial_train_frac.")
    return folds


def _loader(ds, shuffle):
    return DataLoader(ds, batch_size=CFG.batch_size, shuffle=shuffle,
                      num_workers=CFG.loader_workers, pin_memory=True,
                      persistent_workers=CFG.loader_workers > 0,
                      prefetch_factor=4 if CFG.loader_workers > 0 else None)


def run_epoch(model, loader, mse, bce, device, scaler, optimizer=None):
    training = optimizer is not None
    model.train(training)
    tot, n = 0.0, 0
    amp = CFG.use_amp and device.type == "cuda"
    with torch.set_grad_enabled(training):
        for batch in loader:
            ti = batch["trend_input"].to(device, non_blocking=True)
            si = batch["season_input"].to(device, non_blocking=True)
            tt = batch["trend_target"].to(device, non_blocking=True)
            st = batch["season_target"].to(device, non_blocking=True)
            ct = batch["cls_target"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                t_pred, s_pred, logits = model(ti, si)
                l_t, l_s = mse(t_pred, tt), mse(s_pred, st)
                ssum = l_t + l_s
                l_fore = (l_s / ssum) * l_t + (l_t / ssum) * l_s
                loss = l_fore + CFG.lambda_cls * bce(logits, ct)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            tot += loss.item() * ti.size(0)
            n += ti.size(0)
    return tot / max(n, 1)


def _fit(full, model, device, logger, out_dir, lr, tag_stage):
    """Executa o rolling-origin sobre um ConcatDataset `full`, treinando
    `model` (que pode já vir pré-treinado). Retorna as perdas por fold.
    Fatorado de train() para permitir ESTÁGIOS (pré-treino -> fine-tuning)."""
    folds = rolling_origin_folds(full)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                  model.parameters()), lr=lr)
    scaler = torch.amp.GradScaler(enabled=CFG.use_amp and
                                  device.type == "cuda")
    fold_losses, history = [], []
    for j, (tr_idx, va_idx) in enumerate(folds):
        tl = _loader(Subset(full, tr_idx), True)
        vl = _loader(Subset(full, va_idx), False)
        sample = tr_idx[:200_000]
        ys = torch.stack([full[i]["cls_target"] for i in sample]).flatten()
        pos = ys.sum().clamp(min=1.0)
        bce = torch.nn.BCEWithLogitsLoss(
            pos_weight=((ys.numel() - pos) / pos).to(device))
        mse = torch.nn.MSELoss()
        best, wait = float("inf"), 0
        for ep in range(CFG.epochs_per_fold):
            t_e = time.time()
            tr = run_epoch(model, tl, mse, bce, device, scaler, opt)
            va = run_epoch(model, vl, mse, bce, device, scaler)
            history.append({"stage": tag_stage, "fold": j, "epoch": ep,
                            "train": tr, "val": va})
            logger.term(f"[step6:{tag_stage}] fold {j+1}/{len(folds)} "
                        f"ep {ep+1}: train={tr:.4f} val={va:.4f} "
                        f"({_fmt_dur(time.time() - t_e)})")
            logger.snapshot(f"{tag_stage}_f{j}_e{ep}")
            if va < best:
                best, wait = va, 0
                torch.save(model.state_dict(),
                           out_dir / f"best_{tag_stage}_fold{j}.pth")
            else:
                wait += 1
                if wait >= CFG.patience:
                    break
        fold_losses.append(best)
        model.load_state_dict(torch.load(
            out_dir / f"best_{tag_stage}_fold{j}.pth", map_location=device))
    return fold_losses, history


def train(pretrained_path=None, freeze_backbone=False, tag="rolling",
          data_scope="synthetic", combine="pretrain"):
    """Treina um v0.

    --data-scope:
        synthetic : só o dataset corrente (OUT_ROOT)
        real      : só o real (OUT_ROOT_REAL, ou OUT_ROOT se ausente)
        both      : sintético + real, combinados conforme --combine
    --combine (só relevante p/ both):
        pretrain  : treina no sintético e DEPOIS faz fine-tuning no real
                    (dois estágios; recomendado p/ a comparação metodológica)
        pool      : junta os shards das duas fontes num só treino
        reweight  : pool, mas equalizando o nº de janelas das duas fontes
    """
    logger = RunLogger("step6_train")
    _speed_setup()
    torch.manual_seed(CFG.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.term(f"[step6] device={device} amp={CFG.use_amp} "
                f"scope={data_scope} combine={combine} "
                f"batch={CFG.batch_size}")
    scopes = _resolve_scopes(data_scope)
    out_dir = CFG.models_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    model = HybridWLSTMix(device, freeze_backbone, pretrained_path).to(device)

    # ---- caminho de DOIS ESTÁGIOS: pré-treino no sintético + fine-tune real
    if data_scope == "both" and combine == "pretrain":
        synth = load_split("train", dict(scopes)["synthetic"], logger)
        logger.term(f"[step6] estágio 1/2 (pré-treino sintético): "
                    f"{sum(len(d) for d in synth.datasets):,} janelas")
        fl1, h1 = _fit(synth, model, device, logger, out_dir,
                       CFG.learning_rate, "pretrain")
        real = load_split("train", dict(scopes)["real"], logger)
        logger.term(f"[step6] estágio 2/2 (fine-tuning real): "
                    f"{sum(len(d) for d in real.datasets):,} janelas")
        # fine-tuning costuma usar lr menor; CFG.learning_rate/10 é um padrão
        fl2, h2 = _fit(real, model, device, logger, out_dir,
                       CFG.learning_rate / 10, "finetune")
        fold_losses, history = fl2, h1 + h2
    else:
        # ---- caminho de ESTÁGIO ÚNICO (synthetic | real | both:pool/reweight)
        datasets = []
        for name, root in scopes:
            d = load_split("train", root, logger)
            datasets.append((name, d))
            logger.term(f"[step6] fonte '{name}': "
                        f"{sum(len(x) for x in d.datasets):,} janelas")
        if combine == "reweight" and len(datasets) > 1:
            # equaliza por subamostragem da fonte maior (determinística)
            import numpy as _np
            sizes = [len(d) for _, d in datasets]
            target = min(sizes)
            balanced = []
            for (_, d), sz in zip(datasets, sizes):
                if sz > target:
                    g = _np.random.default_rng(CFG.seed)
                    idx = sorted(g.choice(sz, target, replace=False).tolist())
                    balanced.append(Subset(d, idx))
                else:
                    balanced.append(d)
            full = ConcatDataset(balanced)
        else:
            full = ConcatDataset([d for _, d in datasets])
        logger.term(f"[step6] treino combinado: "
                    f"{len(full):,} janelas")
        fold_losses, history = _fit(full, model, device, logger, out_dir,
                                    CFG.learning_rate, "rolling")

    torch.save(model.state_dict(), out_dir / "best_model.pth")
    (out_dir / "rolling_results.json").write_text(json.dumps(
        {"data_scope": data_scope, "combine": combine,
         "fold_val_losses": fold_losses,
         "mean": float(np.mean(fold_losses)),
         "std": float(np.std(fold_losses)), "history": history}, indent=2))
    logger.term(f"[step6] FIM treino ({data_scope}/{combine}): "
                f"val {np.mean(fold_losses):.4f} ± {np.std(fold_losses):.4f} "
                f"| modelo em {out_dir / 'best_model.pth'}")
    logger.close(f"train {data_scope}/{combine}")



# =============================== TESTE =====================================
def _load_raw_series(path: Path):
    """Dados novos no formato do dataset puro (timestamp + medida).
    Usa o extrator do passo 1 quando disponível (wide/long/single)."""
    try:
        from step1_split import iter_building_series_file
        for bname, bdf in iter_building_series_file(path):
            if "timestamp" in bdf.columns:
                bdf = bdf.sort_values("timestamp")
            yield bname, np.nan_to_num(bdf["energy"].to_numpy(np.float64))
        return
    except ImportError:
        df = (pd.read_parquet(path) if path.suffix == ".parquet"
              else pd.read_csv(path))
        tcol = next((c for c in df.columns
                     if c.lower() in ("timestamp", "datetime", "date")), None)
        if tcol:
            df = df.sort_values(tcol)
        vcol = next(c for c in df.columns if c != tcol)
        yield path.stem, np.nan_to_num(df[vcol].to_numpy(np.float64))


@torch.no_grad()
def test(data: Path, model_path=None, threshold: float = 0.5):
    """Rotulagem EM TEMPO REAL (passos 3-5 vetorizados) só aqui, para
    dados novos, seguida da avaliação do modelo."""
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, roc_auc_score)
    logger = RunLogger("step6_test")
    _speed_setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridWLSTMix(device).to(device)
    model_path = model_path or (CFG.models_root / "rolling" / "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.term(f"[step6:test] modelo={model_path} device={device}")

    B, F = CFG.backcast_length, CFG.forecast_length
    files = ([data] if data.is_file()
             else sorted(p for p in data.rglob("*")
                         if p.suffix in (".parquet", ".csv")))
    Y, P = [], []
    for fp in files:
        for bname, s in _load_raw_series(fp):
            starts, wins = make_windows(s)
            if wins.shape[0] == 0:
                continue
            # PASSOS 3-5 EM RUNTIME (vetorizados, mesmas funções do treino):
            trend, season = decompose_windows_batch(wins)
            win_labels = label_windows_batch(wins, trend)
            y_true = fuse_labels(len(s), starts, win_labels)
            t_n, *_ = standardize_batch(trend)
            s_n, *_ = standardize_batch(season)
            ti = torch.tensor(t_n[:, :B]).to(device)
            si = torch.tensor(s_n[:, :B]).to(device)
            probs = []
            for i in range(0, len(ti), CFG.batch_size):
                with torch.autocast(device_type=device.type,
                                    enabled=CFG.use_amp and
                                    device.type == "cuda"):
                    _, _, logits = model(ti[i:i+CFG.batch_size],
                                         si[i:i+CFG.batch_size])
                probs.append(torch.sigmoid(logits.float()).cpu().numpy())
            probs = np.concatenate(probs)
            prob_sum = np.zeros(len(s)); prob_cnt = np.zeros(len(s))
            for j, st in enumerate(starts):
                prob_sum[st+B: st+B+F] += probs[j]
                prob_cnt[st+B: st+B+F] += 1
            cov = prob_cnt > 0
            Y.append(y_true[cov])
            P.append(prob_sum[cov] / prob_cnt[cov])
            logger.building(f"{fp.stem}:{bname}",
                            f"janelas={wins.shape[0]} "
                            f"taxa_rotulada={y_true.mean():.4f}")
    y = np.concatenate(Y); p = np.concatenate(P)
    pred = (p >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auroc": (float(roc_auc_score(y, p))
                  if y.min() != y.max() else None),
        "anomaly_rate_true": float(y.mean()), "n_points": int(len(y)),
    }
    (CFG.models_root / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2))
    logger.term(f"[step6:test] {json.dumps(metrics)}")
    logger.close("test")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "test"], default="train")
    ap.add_argument("--data", type=Path, default=None,
                    help="(test) arquivo ou pasta com dados CRUS "
                         "(timestamp + medida)")
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--pretrained", type=Path, default=None,
                    help="(train) pesos do W-LSTMix p/ inicializar o backbone")
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--data-scope", choices=["synthetic", "real", "both"],
                    default="synthetic",
                    help="fonte(s) de treino: só sintético, só real, ou ambos")
    ap.add_argument("--combine", choices=["pretrain", "pool", "reweight"],
                    default="pretrain",
                    help="(both) como combinar: pretrain=pré-treino sintético "
                         "+ fine-tuning real; pool=junta tudo; "
                         "reweight=pool equalizado")
    ap.add_argument("--tag", type=str, default="rolling",
                    help="subpasta de saída em 04_models/ (ex.: v0_both, "
                         "v0_real) — separe os dois v0 da comparação")
    a = ap.parse_args()
    if a.mode == "train":
        train(pretrained_path=a.pretrained,
              freeze_backbone=a.freeze_backbone,
              tag=a.tag, data_scope=a.data_scope, combine=a.combine)
    else:
        if a.data is None:
            ap.error("--mode test exige --data")
        test(a.data, model_path=a.model)