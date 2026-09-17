"""strategy.py — Estratégia customizada: FedAvg + TensorBoard + checkpoint do melhor modelo.

Substitui, no mundo novo do Flower (Message API), a antiga SaveModelStrategy
do repositório Tupã, acrescentando:

  1. Registro por rodada no TensorBoard do SERVIDOR:
       - métricas agregadas de treino e avaliação (train/*, evaluate/*);
       - métricas POR CLIENTE (clients/<id>/*), sem agregação;
       - as curvas de épocas locais (train_loss_epochs, train_bce_loss_epochs)
         de cada cliente, projetadas num eixo de passos global — resolvendo
         o ponto sinalizado anteriormente: essas listas NÃO entram na média
         do FedAvg, são consumidas aqui e removidas antes da agregação.
  2. Checkpoint do MELHOR modelo global (menor métrica de avaliação
     agregada, por padrão test_loss) — o análogo federado correto do
     early stopping/best_model.pth do train.py original do W-LSTMix.

ATENÇÃO (assinaturas): os métodos aggregate_train/aggregate_evaluate e os
atributos de Message/metadata seguem o padrão documentado nos tutoriais
oficiais da série "Customize a Flower Strategy" (Flower >= 1.21), mas as
assinaturas exatas podem variar entre versões menores. Confirme contra o
template gerado por `flwr new` na SUA versão instalada antes de rodar.
"""

from __future__ import annotations

from logging import INFO
from typing import Iterable, Optional

import torch
from flwr.app import ArrayRecord, Message, MetricRecord
from flwr.common import log
from flwr.serverapp.strategy import FedAvg
from torch.utils.tensorboard import SummaryWriter

# ATENÇÃO (achado durante o run de patience): `logging.getLogger("qualquer
# nome próprio").info(...)` NUNCA aparece no terminal do `flwr run
# --stream`. Só o logger chamado literalmente "flwr" tem um ConsoleHandler
# conectado (flwr.common.logger.FLOWER_LOGGER) — é dele que vêm as linhas
# "INFO :      configure_train: ..." que você já vê. Por isso as mensagens
# desta estratégia (inclusive as JÁ EXISTENTES antes desta correção, como
# o antigo "Rodada %d: novo melhor...") nunca apareceram — não é bug do
# patience, é logging morto desde a primeira versão. Confirmado lendo o
# código-fonte de flwr.common.logger:
# https://flower.ai/docs/framework/_modules/flwr/common/logger.html
# Fix: usar flwr.common.log (= logging.getLogger("flwr").log) em vez de
# logging.getLogger(__name__).

# Chaves que NÃO devem ser agregadas pelo FedAvg (listas/curvas locais).
# CORREÇÃO: train_bce_loss_epochs entrou aqui junto com train_loss_epochs
# (patch A do BCE isolado) — sem isso, o FedAvg tentaria fazer média
# ponderada de uma LISTA como se fosse escalar, o mesmo tipo de erro que
# já corrigimos para numpy.float32 em MetricRecord, só que do lado do
# servidor em vez do cliente.
NON_AGGREGATABLE_KEYS = ("train_loss_epochs", "train_bce_loss_epochs")


def _client_id(msg: Message) -> str:
    """Identifica o cliente remetente para fins de rotulagem no TensorBoard.

    Nota: o nome exato do campo de origem nos metadados da Message deve ser
    confirmado na sua versão (ex.: msg.metadata.src_node_id). O fallback
    abaixo evita quebra caso o atributo mude de nome.
    """
    meta = getattr(msg, "metadata", None)
    for attr in ("src_node_id", "node_id", "source_node_id"):
        value = getattr(meta, attr, None)
        if value is not None:
            return str(value)
    return "desconhecido"


class TensorBoardFedAvg(FedAvg):
    """FedAvg com registro em TensorBoard e checkpoint do melhor modelo global."""

    def __init__(
        self,
        *args,
        log_dir: str = "tb_logs/server",
        checkpoint_path: str = "best_model_global.pth",
        selection_metric: str = "test_loss",   # métrica agregada de avaliação
        lower_is_better: bool = True,
        local_epochs: int = 1,                 # p/ eixo global da curva de épocas
        min_delta: float = 0.0,                # NOVO: tolerância de ruído p/ "melhorou"
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.checkpoint_path = checkpoint_path
        self.selection_metric = selection_metric
        self.lower_is_better = lower_is_better
        self.local_epochs = local_epochs
        self.min_delta = min_delta
        self._best: Optional[float] = None
        self._latest_arrays: Optional[ArrayRecord] = None  # p/ salvar no melhor round
        # NOVO: contador de rodadas consecutivas SEM melhora de
        # selection_metric, exposto para o laço manual de server_app.py
        # decidir patience — mantido AQUI (não em server_app.py) para não
        # duplicar a lógica de "melhorou ou não" (que já mora aqui, junto
        # de _best/selection_metric/lower_is_better/min_delta). Se algum
        # dia mudar o critério de melhora, muda num lugar só.
        self.rounds_since_improvement: int = 0

    # ------------------------------------------------------------------
    # TREINO: log por cliente + remoção de chaves não agregáveis + agregação
    # ------------------------------------------------------------------
    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        replies = list(replies)

        for msg in replies:
            if not msg.has_content():
                continue  # respostas com erro são tratadas pelo FedAvg
            metrics: MetricRecord = msg.content["metrics"]
            cid = _client_id(msg)

            # (i) Curvas de épocas locais -> eixo de passos global, por
            # cliente. Genérico sobre NON_AGGREGATABLE_KEYS: cobre tanto
            # train_loss_epochs quanto train_bce_loss_epochs (e qualquer
            # outra curva que venha a ser adicionada) sem duplicar o loop.
            for curve_key in NON_AGGREGATABLE_KEYS:
                epochs_curve = metrics.get(curve_key)
                if epochs_curve is None:
                    continue
                # "train_bce_loss_epochs" -> "train_bce_loss_epoch" no TB
                tb_name = (curve_key[:-1] if curve_key.endswith("s")
                          else curve_key)
                for ep_idx, loss_val in enumerate(list(epochs_curve)):
                    global_step = (server_round - 1) * self.local_epochs + ep_idx
                    self.writer.add_scalar(
                        f"clients/{cid}/{tb_name}", float(loss_val), global_step
                    )

            # (ii) Remove ANTES da agregação: listas não devem entrar na
            # média do FedAvg. CORRIGIDO: agora incondicional (antes só
            # rodava dentro do "if epochs_curve is not None", então uma
            # curva ausente numa rodada deixava a(s) outra(s) chave(s)
            # vazar para a agregação sem proteção).
            for key in NON_AGGREGATABLE_KEYS:
                metrics.pop(key, None)

            # (iii) Escalares por cliente (sem agregação), indexados pela rodada
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(
                        f"clients/{cid}/{key}", float(value), server_round
                    )

        # (iv) Agregação padrão do FedAvg (ponderada por num-examples)
        arrays, agg_metrics = super().aggregate_train(server_round, replies)

        # Guarda referência aos pesos agregados desta rodada para o checkpoint
        self._latest_arrays = arrays

        if agg_metrics is not None:
            for key, value in agg_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"train/{key}", float(value), server_round)
        self.writer.flush()
        return arrays, agg_metrics

    # ------------------------------------------------------------------
    # AVALIAÇÃO: log agregado + por cliente + checkpoint do melhor modelo
    # ------------------------------------------------------------------
    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
        replies = list(replies)

        for msg in replies:
            if not msg.has_content():
                continue
            cid = _client_id(msg)
            for key, value in msg.content["metrics"].items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(
                        f"clients/{cid}/eval_{key}", float(value), server_round
                    )

        agg_metrics = super().aggregate_evaluate(server_round, replies)

        if agg_metrics is not None:
            for key, value in agg_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"evaluate/{key}", float(value), server_round)

            # Checkpoint do melhor modelo global (equivalente federado do
            # best_model.pth do W-LSTMix, decidido pelo SERVIDOR)
            current = agg_metrics.get(self.selection_metric)
            if current is not None and self._latest_arrays is not None:
                current = float(current)
                # min_delta: mesma semântica do --min-delta do
                # train_local_pi.py — evita que ruído de 4ª/5ª casa
                # decimal seja contado como "melhora" e zere a paciência
                # indefinidamente sem progresso real.
                improved = (
                    self._best is None
                    or (self.lower_is_better and current < self._best - self.min_delta)
                    or (not self.lower_is_better and current > self._best + self.min_delta)
                )
                if improved:
                    self._best = current
                    self.rounds_since_improvement = 0
                    # Escrita ATÔMICA (padrão do pipeline): .tmp + rename —
                    # um kill no meio nunca corrompe o melhor checkpoint.
                    tmp = str(self.checkpoint_path) + ".tmp"
                    torch.save(
                        self._latest_arrays.to_torch_state_dict(), tmp)
                    import os as _os
                    _os.replace(tmp, self.checkpoint_path)
                    log(
                        INFO,
                        "Rodada %d: novo melhor %s=%.6f — checkpoint salvo em %s",
                        server_round, self.selection_metric, current,
                        self.checkpoint_path,
                    )
                    self.writer.add_scalar(
                        f"evaluate/best_{self.selection_metric}", current, server_round
                    )
                else:
                    self.rounds_since_improvement += 1
                    log(
                        INFO,
                        "Rodada %d: %s=%.6f não melhorou o melhor (%.6f) — "
                        "%d rodada(s) sem melhora",
                        server_round, self.selection_metric, current,
                        self._best, self.rounds_since_improvement,
                    )

        self.writer.flush()
        return agg_metrics