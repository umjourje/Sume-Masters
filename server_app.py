"""server_app.py — ServerApp Flower (Message API) para o W-LSTMix federado.

FUSÃO: mantém a SUA versão (fractions=1.0, TensorBoardFedAvg com checkpoint
do melhor global por selection_metric, salvamento de final + best, pacote
do app) + o bloco do "v0" decidido neste chat: o modelo da rodada 1 NÃO é
aleatório — são os pesos do treino no REAL (passo 6, --data-scope real,
tag v0_real), carregados com strict=True.

NOVO NESTA VERSÃO (2): early stopping federado de verdade (patience sobre
test_loss), via run-config 'patience' e 'min-delta'. Isso exigiu trocar
`strategy.start(...)` por um laço manual round-a-round — `.start()` é um
`for` fechado da biblioteca, sem gancho de interrupção. O laço abaixo é
uma cópia FIEL da implementação oficial de `Strategy.start()` (mesma
ordem de chamadas, mesmas variáveis), conferida contra o código-fonte em:
https://flower.ai/docs/framework/_modules/flwr/serverapp/strategy/strategy.html
(flwr 1.37 no momento da conferência). A ÚNICA adição é a checagem de
`strategy.rounds_since_improvement` (definida em strategy.py) depois do
`aggregate_evaluate` de cada rodada.

⚠️ MANUTENÇÃO: isso é código duplicado da biblioteca, não escala sozinho
com upgrades do flwr. Se atualizar a versão instalada, reconfira este
laço contra o `start()` da nova versão antes de rodar — mudanças na
biblioteca (ex.: um novo argumento em `configure_train`, um novo campo
agregado) não chegam aqui automaticamente.

patience=0 (padrão) desliga o early stopping — comportamento IDÊNTICO ao
anterior (roda as num-server-rounds inteiras). Nenhum run-config anterior
que não passe 'patience' muda de comportamento.

NOVO NA VERSÃO ANTERIOR: cronômetro ponta a ponta do run no servidor,
gravado em run_summary_<tag>.json (escrita atômica). Junto com os
summary_<tag>.json que cada Pi grava via fed_monitor, permite medir o
overhead de comunicação+agregação por rodada:
    overhead ≈ (wall_servidor / R) − max_i(wall_cliente_i / R)
— insumo do --agg-overhead-s do smoke_report.py.

Execução (máquina servidora) — SEM TLS, por decisão explícita: rede local
fechada e controlada, TLS fica para trabalho futuro (não muda os
resultados nesse cenário):

    flower-superlink --insecure

    flwr run . raspberry-deployment

⚠️ NÃO EXECUTADO/VERIFICADO neste ambiente: confirme a flag exata de modo
inseguro (`--insecure`) contra `flower-superlink --help` na SUA versão
instalada do Flower antes de rodar — o nome/comportamento pode variar
entre versões menores. O mesmo vale para `flower-supernode` em cada Pi.

Se decidir adicionar TLS no futuro, troque para:
    flower-superlink \\
        --ssl-ca-certfile certificates/ca.crt \\
        --ssl-certfile certificates/server.pem \\
        --ssl-keyfile certificates/server.key
e restaure `root-certificates = "certificates/ca.crt"` no pyproject.toml
(em vez de `insecure = true`).

TensorBoard: tensorboard --logdir tb_logs/server
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

import task                                   # mesmo diretório do app
from strategy import TensorBoardFedAvg

log = logging.getLogger("wlstmix.server")

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds = int(context.run_config.get("num-server-rounds", 5))
    local_epochs = int(context.run_config.get("local-epochs", 1))
    tag = str(context.run_config.get("tag", "run"))
    # --- NOVO: early stopping federado ---
    # patience=0 (padrão) desliga — roda as num_rounds inteiras, igual
    # antes. patience=N: para depois de N rodadas seguidas sem melhora de
    # test_loss (além de min_delta). round-timeout-s: repassado a cada
    # send_and_receive; default igual ao da biblioteca (3600s = 1h por
    # fase por rodada). Na calibração (TAG=calib, MAX_SHARDS=15, os
    # mesmos do run real), o Pi mais lento (raspserver01/Espanha) ficou
    # em ~18,5 min/rodada — folgado dentro de 3600s. Se mudar
    # LOCAL_EPOCHS ou o conjunto de shards, confira de novo antes de
    # assumir que o default ainda é suficiente.
    patience = int(context.run_config.get("patience", 0))
    min_delta = float(context.run_config.get("min-delta", 0.0))
    round_timeout_s = float(context.run_config.get("round-timeout-s", 3600.0))

    # Modelo global inicial — MESMA config usada pelos clientes
    cfg = task.load_config()
    device = torch.device("cpu")
    model = task.get_model(cfg, device)

    # LINHA CRUCIAL (v0): pesos do treino centralizado no sintético
    # instanciados na rodada 1. strict=True: divergência de arquitetura
    # entre v0 e os clientes falha AQUI, não na rodada 3.
    v0 = str(context.run_config.get("v0-path", ""))
    if v0 and Path(v0).exists():
        model.load_state_dict(torch.load(v0, map_location=device),
                              strict=True)
        log.info("v0 carregado de %s", v0)
    else:
        log.warning("v0-path %r não encontrado — iniciando de pesos "
                    "ALEATÓRIOS (ok só para ensaio).", v0)

    arrays = ArrayRecord(model.state_dict())

    strategy = TensorBoardFedAvg(
        fraction_train=1.0,        # com poucos Pis, use todos a cada rodada
        fraction_evaluate=1.0,
        log_dir="tb_logs/server",
        checkpoint_path="best_model_global.pth",
        selection_metric="test_loss",   # ou "nrmse"/"cvrmse"/"f1"
        lower_is_better=True,
        local_epochs=local_epochs,
        min_delta=min_delta,
    )

    t0 = time.time()                          # <- tempo total do run

    # ------------------------------------------------------------------
    # LAÇO MANUAL round-a-round — substitui strategy.start(). Cópia FIEL
    # da ordem de chamadas do Strategy.start() oficial (ver docstring do
    # módulo para a fonte conferida): configure_train -> send_and_receive
    # -> aggregate_train -> configure_evaluate -> send_and_receive ->
    # aggregate_evaluate. A ÚNICA adição real é o bloco de patience no
    # fim do laço. train_config/evaluate_config vazios (ConfigRecord()):
    # mesmo default do start() quando não fornecidos.
    # ------------------------------------------------------------------
    train_config = ConfigRecord()
    evaluate_config = ConfigRecord()
    eval_history: list[dict] = []      # NOVO: curva round->métricas, p/ inspeção sem TensorBoard
    rounds_executed = 0
    stopped_early = False

    try:
        for current_round in range(1, num_rounds + 1):
            log.info("[ROUND %d/%d]", current_round, num_rounds)
            rounds_executed = current_round

            # ---- treino (ClientApp-side) ----
            train_replies = grid.send_and_receive(
                messages=strategy.configure_train(
                    current_round, arrays, train_config, grid),
                timeout=round_timeout_s,
            )
            agg_arrays, agg_train_metrics = strategy.aggregate_train(
                current_round, train_replies)
            if agg_arrays is not None:
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log.info("\t└──> Aggregated train MetricRecord: %s", agg_train_metrics)

            # ---- avaliação (ClientApp-side) ----
            evaluate_replies = grid.send_and_receive(
                messages=strategy.configure_evaluate(
                    current_round, arrays, evaluate_config, grid),
                timeout=round_timeout_s,
            )
            agg_evaluate_metrics = strategy.aggregate_evaluate(
                current_round, evaluate_replies)
            if agg_evaluate_metrics is not None:
                log.info("\t└──> Aggregated evaluate MetricRecord: %s", agg_evaluate_metrics)
                eval_history.append({
                    "round": current_round,
                    **{k: float(v) for k, v in agg_evaluate_metrics.items()
                       if isinstance(v, (int, float))},
                })

            # ---- patience (NOVO — único trecho que não vem do start() original) ----
            if patience > 0 and strategy.rounds_since_improvement >= patience:
                log.info(
                    "Early stopping na rodada %d/%d: %d rodada(s) seguidas "
                    "sem melhora de test_loss (paciência=%d, min-delta=%s).",
                    current_round, num_rounds,
                    strategy.rounds_since_improvement, patience, min_delta,
                )
                stopped_early = True
                break
    except KeyboardInterrupt:
        log.warning(
            "Interrompido manualmente na rodada %d/%d — salvando o que já "
            "foi treinado até aqui (final_model_global.pth) e o "
            "best_model_global.pth do melhor checkpoint já gravado.",
            rounds_executed, num_rounds)
        stopped_early = True

    wall = time.time() - t0

    # Modelo da ÚLTIMA rodada (o MELHOR já foi salvo pela estratégia)
    tmp = Path("final_model_global.pth.tmp")
    torch.save(arrays.to_torch_state_dict(), tmp)
    tmp.replace("final_model_global.pth")

    # Resumo do run no servidor (escrita atômica, padrão do projeto)
    summary = {
        "tag": tag,
        "wall_time_s": wall,
        "wall_time_per_round_s": wall / max(rounds_executed, 1),
        "num_rounds_configured": num_rounds,
        "rounds_executed": rounds_executed,
        "stopped_early": stopped_early,
        "patience": patience,
        "min_delta": min_delta,
        "local_epochs": local_epochs,
        "v0_path": v0,
        "eval_history": eval_history,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    sp = Path(f"run_summary_{tag}.json")
    sp_tmp = sp.with_suffix(".json.tmp")
    sp_tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    sp_tmp.replace(sp)

    log.info("Execução concluída em %.1f s (%.1f s/rodada, %d/%d rodadas%s): "
             "final_model_global.pth (última rodada), "
             "best_model_global.pth (melhor rodada) e %s salvos.",
             wall, summary["wall_time_per_round_s"], rounds_executed, num_rounds,
             " — parou cedo" if stopped_early else "", sp)