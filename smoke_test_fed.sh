#!/usr/bin/env bash
# =============================================================================
# smoke_test_fed.sh — Smoke test da rodada federada
#
# Infra: 1 servidor (agregador) + 5 Raspberry Pi 5 clientes, IPs fixos, mesma
# rede local, SEM TLS (decisão registrada). Flower >= 1.21 (SuperLink +
# SuperNode); flwr 1.33.0 nos dispositivos — confirme as flags com
# `flower-superlink --help` e `flower-supernode --help` antes de rodar.
#
# DADOS: storage COMPARTILHADO (TrueNAS) montado nos Pis. A partição de cada
# cliente NÃO vem do caminho, vem do `pi=N` no --node-config, que o task.py
# resolve via country_map.PI_BUCKETS (o mesmo índice do --pi do
# train_local_pi.py). Sem isso, os 5 clientes treinariam no dataset inteiro.
#
# TEMPO ESPERADO DO SMOKE: pelas medições do centralizado no Pi 5
# (GoiEner, 15 shards: 21522s e 15075s, razão 10:7 => 48-144 s por
# shard-época), MAX_SHARDS=2 com 1 época e 1 rodada custa ~5-15 min por
# cliente. Não é necessário limitar janelas por shard.
#
# O smoke valida: [1] 5 SuperNodes conectam e a rodada completa;
# [2] v0 carrega com strict=True; [3] best_model_global.pth é gravado;
# [4] cada Pi grava loss/progress/summary/confusion_matrix com wall_time_s;
# [5] smoke_report.py consolida e extrapola o ETA do treino completo.
#
# PYTHONS: servidor e Pis usam venvs DIFERENTES, em caminhos DIFERENTES.
# Nunca crave um caminho de python solto no meio do script — use sempre
# $PYTHON_SERVER (roda local, no servidor) ou $PYTHON_PI (roda via ssh,
# dentro dos Pis). Ver bloco de variáveis abaixo.
# =============================================================================
set -euo pipefail

# ------------------------- AJUSTE AQUI ---------------------------------------
SERVER_IP="${SERVER_IP:-172.28.254.64}"      # IP fixo do agregador
PIS=("pi1" "pi2" "pi3" "pi4" "pi5")         # aliases SSH, na ORDEM dos índices
APP_DIR="${APP_DIR:-/mnt/juliana-truenas/Sume-Masters/}"     # dir do pyproject (servidor)
APP_DIR_PI="${APP_DIR_PI:-/home/jpiaz/source/Sume-Masters/}"
PIPELINE_DIR_PI="${PIPELINE_DIR_PI:-$APP_DIR_PI}"
WLSTMIX_DIR_PI="${WLSTMIX_DIR_PI:-/home/jpiaz/source/Sume-Masters/models}"     # contém models/W_LSTMix.py
FEDERATION="raspberry-deployment"

# --- Interpretes Python — SEMPRE explicitados, servidor != Pis -------------
# PYTHON_SERVER: venv usado LOCALMENTE no servidor (agregador), em preflight()
# (checagem do v0 com strict=True) e em report() (smoke_report.py/plot_loss.py).
PYTHON_SERVER="${PYTHON_SERVER:-/mnt/juliana-truenas/314-env/bin/python3}"
# PYTHON_PI: venv usado via SSH DENTRO de cada Raspberry Pi (mesmo venv nos 5,
# pois o ambiente é padronizado neles). Usado em _contar_shards_pi() e em
# qualquer outro comando remoto que rode Python nos clientes.
PYTHON_PI="${PYTHON_PI:-/home/jpiaz/source/312-env/bin/python3}"

# Raiz dos dados no mount compartilhado (o "$REAL" dos seus comandos).
# task.py procura <DATA_ROOT>/02_windows/<CFG.resolution>/{train,test}/**.pt
DATA_ROOT="${DATA_ROOT:-/mnt/juliana-truenas/EnergyBench-Anomaly}"
METRICS_DIR_PI="${METRICS_DIR_PI:-/home/jpiaz/sume_metrics}"

# Checkpoint v0 — caminho LOCAL DO SERVIDOR (é ele que carrega o v0).
# OBS: os "{ }" que existiam aqui antes eram bug — entravam como parte literal
# do path (dava pra ver no log: "v0 ... {/mnt/.../best_model.pth}") e fariam
# o torch.load falhar com FileNotFoundError mesmo depois de resolver o
# ModuleNotFoundError.
SYNTH="/mnt/juliana-truenas/Synth-EnergyBench-Anomaly"
V0REAL="$SYNTH/04_models/v0_real/best_model.pth"
V0BOTH="$SYNTH/04_models/v0_final/best_model.pth"
# Both Real+Sintético
V0_PATH="${V0_PATH:-$V0BOTH}"

TAG="${TAG:-smoke}"
ROUNDS="${ROUNDS:-1}"
MAX_SHARDS="${MAX_SHARDS:-2}"     # smoke=2; run completo=15 (= centralizado)
MAX_WINDOWS="${MAX_WINDOWS:-0}"   # 0 = shard inteiro (só depuração usa >0)
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"

RUN_CONFIG="num-server-rounds=$ROUNDS local-epochs=$LOCAL_EPOCHS \
max-shards=$MAX_SHARDS max-windows=$MAX_WINDOWS tag=\"$TAG\""
[ -n "$V0_PATH" ] && RUN_CONFIG="$RUN_CONFIG v0-path=\"$V0_PATH\""
# -----------------------------------------------------------------------------

usage() {
  cat <<EOF
Uso: $0 {preflight|superlink|supernodes|run|collect|report|all}

  preflight   (servidor) confere mount, shards train/test JÁ FILTRADOS por
              pi, env vars, import do task.py e o v0 com strict=True
  superlink   (servidor) sobe o SuperLink em modo --insecure
  supernodes  (servidor) sobe via SSH um SuperNode em cada Pi, com pi=N
  run         (servidor) dispara o run federado
  collect     (servidor) traz por rsync os artefatos do fed_monitor
  report      (servidor) consolida e extrapola ETA (smoke_report.py)
  all         preflight -> supernodes -> run -> collect -> report
              (SuperLink já ativo em outro terminal)

Variáveis úteis: SERVER_IP DATA_ROOT V0_PATH TAG ROUNDS MAX_SHARDS
                 LOCAL_EPOCHS APP_DIR_PI WLSTMIX_DIR_PI
                 PYTHON_SERVER (interpretador local, servidor)
                 PYTHON_PI     (interpretador via ssh, dentro dos Pis)
EOF
  exit 1
}

# Conta os shards que o task.py REALMENTE verá para um dado pi — usa a
# própria função do task.py, não um find aproximado, para que o preflight
# valide o mesmo critério de filtragem que a execução vai usar.
_contar_shards_pi() {
  local pi_alias="$1" idx="$2"
  ssh "$pi_alias" "cd ${APP_DIR_PI} && PIPELINE_DIR=${PIPELINE_DIR_PI} \
    WLSTMIX_DIR=${WLSTMIX_DIR_PI} ${PYTHON_PI} -c \"
import task
from pathlib import Path
r = Path('${DATA_ROOT}')
tr = task._load_local('train', r, ${idx})
te = task._load_local('test',  r, ${idx})
print(f'train={len(tr)} test={len(te)} res={task.CFG.resolution}')
\""
}

preflight() {
  local falhou=0 idx=0
  echo "[preflight] DATA_ROOT=${DATA_ROOT}"
  echo "[preflight] PYTHON_SERVER=${PYTHON_SERVER}"
  echo "[preflight] PYTHON_PI=${PYTHON_PI}"
  for pi in "${PIS[@]}"; do
    idx=$((idx + 1))
    echo -n "  ${pi} (pi=${idx}): "
    if out=$(_contar_shards_pi "$pi" "$idx" 2>&1); then
      echo "$out"
      case "$out" in
        *"train=0 "*) echo "    [ERRO] nenhum shard de TREINO após o filtro "\
          "de partição — confira o mount e country_map.PI_BUCKETS[$idx]";
          falhou=1 ;;
      esac
      case "$out" in
        *"test=0"*) echo "    [ERRO] nenhum shard de TESTE — sem ele não há "\
          "test_loss e best_model_global.pth NUNCA é gravado"; falhou=1 ;;
      esac
    else
      echo "FALHOU"; echo "$out" | sed 's/^/    /'; falhou=1
    fi
  done

  if [ -n "$V0_PATH" ]; then
    echo "[preflight] v0 com strict=True (servidor): $V0_PATH"
    # Checagem defensiva ANTES de chamar o Python: se task.py não estiver em
    # APP_DIR, o erro abaixo deixa isso óbvio em vez de um ModuleNotFoundError
    # genérico (causa mais comum: APP_DIR apontando pra pasta errada — ex.
    # "scripts" quando o código está em "scripts_tupa").
    if [ ! -f "${APP_DIR}/task.py" ]; then
      echo "    [ERRO] ${APP_DIR}/task.py não existe."
      echo "    Confira se APP_DIR='${APP_DIR}' é mesmo a pasta com o"
      echo "    pyproject.toml + task.py no SERVIDOR (pode ser diferente de"
      echo "    APP_DIR_PI='${APP_DIR_PI}', usado só nos Pis)."
      falhou=1
    elif (cd "$APP_DIR" && "$PYTHON_SERVER" -c "
import torch, task
m = task.get_model(task.load_config(), torch.device('cpu'))
m.load_state_dict(torch.load('$V0_PATH', map_location='cpu'), strict=True)
print('  v0 OK')"); then
      :
    else
      echo "  [ERRO] v0 incompatível com HybridWLSTMix (ou PYTHON_SERVER='${PYTHON_SERVER}' incorreto)"
      falhou=1
    fi
  else
    echo "[preflight] V0_PATH vazio — o run partiria de pesos ALEATÓRIOS."
  fi

  [ "$falhou" -eq 0 ] || { echo "[preflight] FALHOU — corrija antes de rodar."; exit 1; }
  echo "[preflight] tudo OK."
}

superlink() {
  echo "[smoke] subindo SuperLink em ${SERVER_IP} (sem TLS)…"
  # 9092 = fleet (SuperNodes); 9093 = exec (`flwr run`, no pyproject).
  #
  # IMPORTANTE (achado no smoke real, exit code 608): por padrão, o
  # SuperLink LIGA a instalação automática de dependências do ServerApp em
  # tempo de execução. Isso ignora completamente o venv ativo neste shell
  # (o mesmo $PYTHON_SERVER já validado no preflight()) e recria, do zero,
  # um ambiente isolado em $FLWR_HOME/runtime-envs/<run_id> via `uv sync`
  # — no nosso caso baixando a stack CUDA inteira do torch (~2.5GB) e
  # estourando o disco (/home/juliana.piaz fica no `/`, que está a 85%).
  #
  # Como o ambiente já foi testado e aprovado no preflight (import task.py
  # + torch + strict=True do v0), não faz sentido reinstalar nada em
  # tempo de execução: desativamos essa reinstalação e o ServerApp passa
  # a rodar no MESMO Python já ativo aqui.
  #
  # Fonte (Flower 1.33, oficial):
  # https://flower.ai/docs/framework/1.33/en/how-to-install-app-dependencies-at-runtime.html
  # https://flower.ai/docs/framework/1.33/en/ref-exit-codes/608.html
  #
  # ⚠️ Isso só tem efeito se o SuperLink NÃO estiver rodando com
  # `--isolation=process` (não é o caso aqui — este script sobe o
  # SuperExec embutido, chamando só `flower-superlink`). Se um dia vocês
  # passarem a rodar `flower-superexec` separado, o flag equivalente vai
  # nesse comando, não aqui — confirme contra `flower-superexec --help`
  # antes de assumir que ainda se aplica.
  export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION="${FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION:-1}"
  flower-superlink --insecure
}

supernodes() {
  local idx=0
  for pi in "${PIS[@]}"; do
    idx=$((idx + 1))
    echo "[smoke] iniciando SuperNode em ${pi} com pi=${idx}…"
    ssh "$pi" "mkdir -p ${METRICS_DIR_PI}; cd ${APP_DIR_PI} && \
      PIPELINE_DIR=${PIPELINE_DIR_PI} WLSTMIX_DIR=${WLSTMIX_DIR_PI} \
      setsid nohup flower-supernode --insecure \
        --superlink ${SERVER_IP}:9092 \
        --node-config \"data-root='${DATA_ROOT}' pi=${idx} metrics-dir='${METRICS_DIR_PI}'\" \
        > supernode_${TAG}.log 2>&1 < /dev/null & echo PID=\$!"
  done
  echo "[smoke] aguarde ~10 s e confira no log do SuperLink se os 5 nós registraram."
}

run() {
  cd "$APP_DIR"
  echo "[smoke] flwr run . ${FEDERATION} --run-config '${RUN_CONFIG}'"
  T0=$(date +%s)
  flwr run . "$FEDERATION" --run-config "$RUN_CONFIG" --stream
  T1=$(date +%s)
  echo "[smoke] run concluído em $((T1-T0)) s (ponta a ponta, servidor)."
  echo "[smoke] overhead de agregação ≈ (esse valor − max wall_time_s dos "
  echo "        Pis) / ${ROUNDS} — use como --agg-overhead-s no report."
}

collect() {
  cd "$APP_DIR"
  mkdir -p "metrics_${TAG}"
  for pi in "${PIS[@]}"; do
    mkdir -p "metrics_${TAG}/${pi}"
    rsync -av "${pi}:${METRICS_DIR_PI}/" "metrics_${TAG}/${pi}/" || \
      echo "[aviso] rsync falhou para ${pi}"
  done
  echo "[smoke] artefatos em metrics_${TAG}/<pi>/"
}

report() {
  cd "$APP_DIR"
  # --full-units: nº de SHARDS de treino do run COMPLETO por host — mesma
  # unidade do RunMonitor.tick(). Com MAX_SHARDS=15 no run completo, é
  # simplesmente 15 para todos os hosts (o teto vira o total efetivo).
  # (antes rodava com o python dos Pis por engano — corrigido para PYTHON_SERVER)
  "$PYTHON_SERVER" scripts/smoke_report.py \
    --summaries "metrics_${TAG}/*/summary_${TAG}.json" \
    --json-out "scripts/smoke_report_${TAG}.json" "$@"
  if [ -f plot_loss.py ]; then
    "$PYTHON_SERVER" plot_loss.py --inputs "metrics_${TAG}/*/loss_${TAG}*.jsonl" \
      --out "plots_${TAG}" --fmt svg pdf --smooth 5
  else
    echo "[aviso] plot_loss.py ausente — gráficos não gerados."
  fi
}

case "${1:-}" in
  preflight)  preflight ;;
  superlink)  superlink ;;
  supernodes) supernodes ;;
  run)        run ;;
  collect)    collect ;;
  report)     shift; report "$@" ;;
  all)        preflight; supernodes; sleep 12; run; collect; report ;;
  *)          usage ;;
esac