"""client_app.py — ClientApp (Message API): treino/avaliação locais do
HybridWLSTMix sobre os artefatos do pipeline visíveis pelo dispositivo.

Cada SuperNode declara sua PARTIÇÃO e o diretório de métricas:

    flower-supernode ... --node-config \
      "data-root='/mnt/juliana-truenas/EnergyBench-Anomaly' pi=1 \
       metrics-dir='/home/pi/tupa_metrics'"

Por que `pi` é obrigatório aqui: os 5 Pis montam o MESMO storage
compartilhado (TrueNAS). Sem o filtro por partição, os 5 clientes leriam
o dataset inteiro e o experimento deixaria de ser Non-IID — viraria 5
réplicas do centralizado sob FedAvg, silenciosamente. O `pi` é o mesmo
índice do `--pi` do train_local_pi.py e resolve para os grupos de
country_map.PI_BUCKETS.

Só omita `pi` (ou use pi=0) se o data-root daquele dispositivo já contiver
EXCLUSIVAMENTE a partição dele.

Do run_config vêm local-epochs, lr, tag, max-shards e max-windows — o
smoke test é apenas

    flwr run . raspberry-deployment --run-config \
      'num-server-rounds=1 local-epochs=1 max-shards=2 tag="smoke"'

sem nenhuma mudança de código entre smoke e treino completo. Para
comparabilidade com o centralizado, use no run completo o mesmo
max-shards do train_local_pi.py (15 nos runs de referência).

Os artefatos do fed_monitor (loss_<tag>.jsonl, progress_<tag>.json,
summary_<tag>.json, confusion_matrix_<tag>.json) ficam em metrics-dir.
"""
from pathlib import Path

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

import task

app = ClientApp()


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _shared_cfg(context: Context):
    """Chaves comuns a train/evaluate, com defaults seguros."""
    tag = str(context.run_config.get("tag", "run"))
    max_shards = int(context.run_config.get("max-shards", 0))
    max_windows = int(context.run_config.get("max-windows", 0))
    pi = int(context.node_config.get("pi", 0))
    metrics_dir = Path(str(context.node_config.get(
        "metrics-dir", Path.home() / "tupa_metrics")))
    data_root = Path(str(context.node_config["data-root"]))
    return tag, pi, max_shards, max_windows, metrics_dir, data_root


def _round_no(msg: Message):
    """Nº da rodada, se o runtime o expuser nos metadados (best-effort)."""
    try:
        return int(getattr(msg.metadata, "group_id", None) or 0) or None
    except (TypeError, ValueError):
        return None


@app.train()
def train(msg: Message, context: Context) -> Message:
    device = _device()
    model = task.get_model(task.load_config(), device)
    # pesos globais da rodada -> modelo local
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    tag, pi, max_shards, max_windows, metrics_dir, data_root = _shared_cfg(context)
    metrics = task.train(
        model, data_root, device,
        epochs=int(context.run_config.get("local-epochs", 1)),
        lr=float(context.run_config.get("lr", 1e-3)),
        tag=tag, pi=pi, max_shards=max_shards, max_windows=max_windows,
        metrics_dir=metrics_dir, round_no=_round_no(msg))
    reply = RecordDict({"arrays": ArrayRecord(model.state_dict()),
                        "metrics": MetricRecord(metrics)})
    return Message(content=reply, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    device = _device()
    model = task.get_model(task.load_config(), device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.eval()
    tag, pi, max_shards, max_windows, metrics_dir, data_root = _shared_cfg(context)
    metrics = task.evaluate(
        model, data_root, device,
        tag=tag, pi=pi, max_shards=max_shards, max_windows=max_windows,
        metrics_dir=metrics_dir, round_no=_round_no(msg))
    return Message(content=RecordDict({"metrics": MetricRecord(metrics)}),
                   reply_to=msg)