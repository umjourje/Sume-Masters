#!/usr/bin/env bash
# =============================================================================
# smoke_test_fed.sh — Smoke test de rodada(s) federada(s) — VERSÃO ALINHADA
# aos arquivos reais do projeto (pyproject "raspberry-deployment",
# run_config com tag/max-shards, node-config com data-root/metrics-dir).
#
# Infra: 1 servidor + 5 Raspberry Pi, mesma rede local, SEM TLS (decisão
# registrada). Flower >= 1.21 (SuperLink + SuperNode). flwr testado nos
# dispositivos: 1.33.0 — confirme flags com `flower-superlink --help`.
#
# O smoke valida: [1] 5 SuperNodes conectam e R rodadas completam;
# [2] v0 carrega com strict=True; [3] best_model_global.pth é gravado;
# [4] cada Pi grava loss_smoke.jsonl / progress_smoke.json /
#     summary_smoke.json / confusion_matrix_smoke.json (com wall_time_s);
# [5] smoke_report.py consolida e extrapola o ETA do treino completo.
# =============================================================================
set -euo pipefail

# ------------------------- AJUSTE AQUI ---------------------------------------
SERVER_IP="${SERVER_IP:-192.168.0.10}"      # IP fixo do servidor
PIS=("pi1" "pi2" "pi3" "pi4" "pi5")         # hostnames/aliases SSH dos Pis
APP_DIR="${APP_DIR:-$HOME/Tupa-Masters/scripts_tupa}"   # dir do pyproject.toml
PIPELINE_DIR="${PIPELINE_DIR:-$APP_DIR}"    # exigido pelo task.py nos Pis
FEDERATION="raspberry-deployment"            # nome real no pyproject.toml
TAG="smoke"
DATA_ROOT_PI="${DATA_ROOT_PI:-/home/pi/tupa_data}"      # partição em cada Pi
METRICS_DIR_PI="${METRICS_DIR_PI:-/home/pi/tupa_metrics}"
V0_PATH="${V0_PATH:-}"                       # ex.: .../04_models/v0_real/best_model.pth
# Smoke: 1 rodada, 1 época local, 2 shards por cliente — minutos, não horas.
RUN_CONFIG="num-server-rounds=1 local-epochs=1 max-shards=2 tag=\"$TAG\""
[ -n "$V0_PATH" ] && RUN_CONFIG="$RUN_CONFIG v0-path=\"$V0_PATH\""
# -----------------------------------------------------------------------------

usage() {
  cat <<EOF
Uso: $0 {localsim|superlink|supernodes|run|collect|report|all}

  localsim    (servidor) ensaio com 5 supernós SIMULADOS (federação default)
              — valide isto ANTES de tocar nos Pis (passo já previsto no plano)
  superlink   (servidor) sobe o SuperLink em modo --insecure
  supernodes  (servidor) sobe via SSH um SuperNode em cada Pi
  run         (servidor) dispara o run federado com config de smoke
  collect     (servidor) traz por rsync os artefatos do fed_monitor dos Pis
  report      (servidor) consolida e extrapola ETA (smoke_report.py)
  all         supernodes -> run -> collect -> report (SuperLink já ativo)
EOF
  exit 1
}

localsim() {
  cd "$APP_DIR"
  echo "[smoke] ensaio local-sim (5 supernós simulados): $RUN_CONFIG"
  PIPELINE_DIR="$PIPELINE_DIR" flwr run . --run-config "$RUN_CONFIG" --stream
}

superlink() {
  echo "[smoke] subindo SuperLink em ${SERVER_IP} (sem TLS)…"
  # Portas padrão do Flower: 9092 (fleet/SuperNodes), 9093 (exec/`flwr run`)
  # — o pyproject aponta address=servidor:9093.
  flower-superlink --insecure
}

supernodes() {
  for pi in "${PIS[@]}"; do
    echo "[smoke] iniciando SuperNode em ${pi}…"
    ssh "$pi" "mkdir -p ${METRICS_DIR_PI}; cd ${APP_DIR} && \
      PIPELINE_DIR=${PIPELINE_DIR} nohup flower-supernode --insecure \
        --superlink ${SERVER_IP}:9092 \
        --node-config \"data-root='${DATA_ROOT_PI}' metrics-dir='${METRICS_DIR_PI}'\" \
        > supernode_${TAG}.log 2>&1 & echo PID=\$!"
  done
  echo "[smoke] aguarde ~10 s e confira no log do SuperLink se os 5 nós registraram."
}

run() {
  cd "$APP_DIR"
  echo "[smoke] disparando: flwr run . ${FEDERATION} --run-config '${RUN_CONFIG}'"
  T0=$(date +%s)
  flwr run . "$FEDERATION" --run-config "$RUN_CONFIG" --stream
  T1=$(date +%s)
  echo "[smoke] run concluído em $((T1-T0)) s (ponta a ponta, servidor)."
  # o server_app também grava run_summary_${TAG}.json com o wall interno
}

collect() {
  cd "$APP_DIR"
  mkdir -p "metrics_${TAG}"
  for pi in "${PIS[@]}"; do
    mkdir -p "metrics_${TAG}/${pi}"
    rsync -av "${pi}:${METRICS_DIR_PI}/" "metrics_${TAG}/${pi}/" || \
      echo "[aviso] rsync falhou para ${pi}"
  done
  echo "[smoke] artefatos em metrics_${TAG}/<pi>/ (inclui confusion_matrix_${TAG}.json)"
}

report() {
  cd "$APP_DIR"
  # --full-units: total de SHARDS do treino completo por Pi (mesma unidade
  # do RunMonitor.tick no task.py). Conte com:
  #   ssh piN "find ${DATA_ROOT_PI}/02_windows -path '*/train/*.pt' | wc -l"
  python3 scripts_tupa/smoke_report.py \
    --summaries "metrics_${TAG}/*/summary_${TAG}.json" \
    --json-out "smoke_report_${TAG}.json" "$@"
  python3 scripts_tupa/plot_loss.py \
    --inputs "metrics_${TAG}/*/loss_${TAG}*.jsonl" \
    --out "plots_${TAG}" --fmt svg pdf --smooth 5
}

case "${1:-}" in
  localsim)   localsim ;;
  superlink)  superlink ;;
  supernodes) supernodes ;;
  run)        run ;;
  collect)    collect ;;
  report)     shift; report "$@" ;;
  all)        supernodes; sleep 12; run; collect; report ;;
  *)          usage ;;
esac
