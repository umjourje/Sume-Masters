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

# Opções de ssh centralizadas: -n evita o hang clássico de ssh+comando
# backgrounded (ver supernodes()); ConnectTimeout/ServerAlive* transformam
# uma conexão travada (rede instável, sshd sobrecarregado, etc.) numa
# FALHA RÁPIDA e visível em vez de um hang indefinido que só termina com
# Ctrl+C manual — foi o padrão observado no smoke test (trava num Pi
# diferente a cada tentativa, sinal de instabilidade, não bug fixo).
# ⚠️ Os valores (10s/5s/3) são um ponto de partida razoável, não medidos
# na sua rede — ajuste SSH_TIMEOUT_S se a rede real dos Pis for mais lenta
# que isso em condições normais (para não confundir "lento" com "travado").
SSH_TIMEOUT_S="${SSH_TIMEOUT_S:-10}"
SSH_OPTS=(-n -o "ConnectTimeout=${SSH_TIMEOUT_S}" -o "ServerAliveInterval=5" -o "ServerAliveCountMax=3")
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
# Binário do flower-supernode, MESMO venv de PYTHON_PI (console-scripts de
# um venv ficam no mesmo bin/ do python3 dele). Caminho completo de
# propósito: sessões SSH não-interativas costumam NÃO carregar o PATH que
# ativa esse venv (pulam .bashrc), então um `flower-supernode` "pelado"
# falha com "No such file or directory" mesmo com o venv certo instalado —
# foi exatamente essa a causa raiz confirmada no smoke test.
SUPERNODE_BIN_PI="${SUPERNODE_BIN_PI:-$(dirname "$PYTHON_PI")/flower-supernode}"

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
Uso: $0 {preflight|superlink|supernodes|stop|manual|status|watch|run|collect|report|all}

  preflight   (servidor) confere mount, shards train/test JÁ FILTRADOS por
              pi, env vars, import do task.py e o v0 com strict=True
  superlink   (servidor) sobe o SuperLink em modo --insecure
  supernodes  (servidor) via SSH: primeiro MATA SuperNodes remanescentes em
              TODOS os Pis (equivalente a 'stop', automático agora — é
              exatamente isso que evita o "Port ... 9094 is already in
              use"), depois sobe um SuperNode novo em cada um, com pi=N.
              Se o SSH falhar em algum Pi, os demais continuam sendo
              tentados (antes, um único Pi instável derrubava o script
              INTEIRO por causa do 'set -e', deixando os Pis já subidos
              órfãos — a causa raiz do bug relatado).
  stop        (servidor) mata flower-supernode/flower-superexec em TODOS
              os Pis. Chamado automaticamente no início de 'supernodes',
              mas pode rodar sozinho a qualquer momento.
  manual      NÃO executa nada — só imprime o comando exato para colar em
              cada terminal (1 do servidor + 1 por Pi), sem passar pela
              orquestração via SSH/nohup/setsid. Use quando quiser ver o
              boot de cada nó ao vivo, na tela do próprio Pi.
  status      (servidor) via SSH: para cada Pi, mostra porta 909x em uso,
              processos flower vivos e as últimas linhas do log — uma
              foto rápida de onde cada nó está travado, sem precisar
              abrir terminal em cada Pi.
  watch <pi>  (servidor) abre um 'watch' interativo DENTRO do Pi indicado
              (porta + processos + log, atualizado a cada 2s). Ex.:
              $0 watch pi3
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
  ssh "${SSH_OPTS[@]}" "$pi_alias" "cd ${APP_DIR_PI} && PIPELINE_DIR=${PIPELINE_DIR_PI} \
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

  echo "[preflight] SUPERNODE_BIN_PI=${SUPERNODE_BIN_PI}"
  idx=0
  for pi in "${PIS[@]}"; do
    idx=$((idx + 1))
    if ssh "${SSH_OPTS[@]}" "$pi" "test -x '${SUPERNODE_BIN_PI}'" 2>/dev/null; then
      : # OK, silencioso — não precisa poluir a saída para o caso feliz
    else
      echo "    [ERRO] ${pi}: ${SUPERNODE_BIN_PI} não existe ou não é"
      echo "    executável nesse Pi. Confira com:"
      echo "      ssh ${pi} \"ls -la \$(dirname ${SUPERNODE_BIN_PI})\""
      echo "    Se o venv existir mas o binário não, o flwr provavelmente"
      echo "    não está instalado nesse venv — confira com:"
      echo "      ssh ${pi} \"${PYTHON_PI} -m pip show flwr\""
      falhou=1
    fi
    # flower-supernode roda em --isolation=subprocess por padrão e lança
    # sozinho um `flower-superexec` (achado via PATH, não caminho
    # completo) — se esse binário irmão não existir, o SuperNode morre
    # DEPOIS de logar "Starting Flower SuperNode", o que só aparece no
    # log do próprio Pi, nunca no preflight (até agora). Checar aqui
    # antes evita repetir o ciclo "sobe 5, morre 5, lê 5 logs".
    local superexec_bin
    superexec_bin="$(dirname "${SUPERNODE_BIN_PI}")/flower-superexec"
    if ! ssh "${SSH_OPTS[@]}" "$pi" "test -x '${superexec_bin}'" 2>/dev/null; then
      echo "    [ERRO] ${pi}: ${superexec_bin} não existe. O"
      echo "    flower-supernode SOBE e morre logo em seguida tentando"
      echo "    lançar esse binário internamente (isolation=subprocess,"
      echo "    o padrão) — mesma família de bug do PATH, um nível mais"
      echo "    fundo. Confira a instalação do flwr nesse venv."
      falhou=1
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
  # CORREÇÃO (causa raiz do "Port ... 9094 is already in use" nos 5 Pis):
  # o `setsid nohup` abaixo existe de propósito para o SuperNode sobreviver
  # a uma queda do SSH — só que isso também significa que, se o script
  # morrer (Ctrl+C local, ou o `set -e` do topo abortando por causa de UM
  # Pi instável), os SuperNodes já subidos ficam ÓRFÃOS, segurando a porta
  # 9094 naquele Pi para sempre. Repita esse ciclo manualmente algumas
  # vezes e os 5 Pis acabam com um órfão cada — exatamente o smoke test
  # que falhou. Por isso agora SEMPRE limpamos antes de subir de novo.
  echo "[smoke] limpando SuperNodes remanescentes em todos os Pis antes de subir novos"
  echo "        (evita o 'Port ... 9094 is already in use' visto no smoke)…"
  stop_supernodes
  echo

  local idx=0
  local -a alias_pid=()   # "pi:PID" ou "pi:SSH_FAILED", para checar sobrevivência abaixo
  for pi in "${PIS[@]}"; do
    idx=$((idx + 1))
    echo "[smoke] iniciando SuperNode em ${pi} com pi=${idx}…"
    # NOTA: este PID é do processo backgrounded DENTRO do SSH remoto — ele
    # existe mesmo que o `flower-supernode` real morra logo em seguida
    # (comando não encontrado, erro de import, porta inalcançável...).
    # Por isso NUNCA confie só nele: a checagem de sobrevivência abaixo é
    # que realmente prova alguma coisa.
    #
    # PATH=...:$PATH (remoto, por isso o \$ escapado): o flower-supernode
    # roda por padrão em --isolation=subprocess e, internamente, faz
    # subprocess.Popen(['flower-superexec', ...]) SEM caminho completo —
    # ele conta com o PATH do próprio processo pai para achar esse
    # binário irmão no mesmo venv. Como a sessão SSH não-interativa não
    # carrega esse PATH (mesma causa raiz do bug anterior, um nível mais
    # fundo), sem isso o SuperNode morre tentando subir seu próprio
    # SuperExec interno — confirmado no traceback do smoke test
    # ("FileNotFoundError: ... 'flower-superexec'").
    # Fonte: https://flower.ai/docs/framework/1.33/en/ref-flower-network-communication.html
    #
    # SSH_OPTS (-n + ConnectTimeout/ServerAlive*): -n evita o hang clássico
    # de ssh+comando backgrounded (o processo remoto sobe e registra
    # normalmente — o SuperLink confirma isso no log — só o ssh local é
    # que não retornava o controle). Os timeouts de keepalive cobrem o
    # caso OUTRO observado depois: instabilidade de rede/SSH que trava a
    # conexão em pontos variáveis (Pi diferente a cada tentativa) — agora
    # isso vira falha rápida (${SSH_TIMEOUT_S}s) em vez de hang
    # indefinido só resolvido com Ctrl+C manual.
    #
    # IMPORTANTE (corrigido): antes, se este `ssh` falhasse (timeout, host
    # fora do ar), o `set -e` do topo do script matava o script INTEIRO
    # aqui, sem sequer tentar os Pis seguintes — e os já subidos ficavam
    # órfãos (ver comentário no topo da função). O `if/else` abaixo isola
    # essa falha: o loop continua para os próximos Pis.
    local pid
    if pid=$(ssh "${SSH_OPTS[@]}" "$pi" "mkdir -p ${METRICS_DIR_PI}; cd ${APP_DIR_PI} && \
      PIPELINE_DIR=${PIPELINE_DIR_PI} WLSTMIX_DIR=${WLSTMIX_DIR_PI} \
      PATH=$(dirname "${SUPERNODE_BIN_PI}"):\$PATH \
      setsid nohup ${SUPERNODE_BIN_PI} --insecure \
        --superlink ${SERVER_IP}:9092 \
        --node-config \"data-root='${DATA_ROOT}' pi=${idx} metrics-dir='${METRICS_DIR_PI}'\" \
        > supernode_${TAG}.log 2>&1 < /dev/null & echo \$!"); then
      echo "  PID=${pid}"
      alias_pid+=("${pi}:${pid}")
    else
      echo "  [ERRO] ${pi}: SSH falhou ao disparar o SuperNode nesse Pi (timeout, host" \
           "fora do ar etc. — ver mensagem do ssh acima, se houver)."
      alias_pid+=("${pi}:SSH_FAILED")
    fi
  done

  echo "[smoke] aguardando 5 s para checar se os processos sobreviveram ao boot…"
  sleep 5
  local falhou=0
  for entry in "${alias_pid[@]}"; do
    local pi="${entry%%:*}" pid="${entry##*:}"
    if [ "$pid" = "SSH_FAILED" ]; then
      echo "  [ERRO] ${pi}: nem chegou a subir — falha de SSH (ver acima)."
      falhou=1
      continue
    fi
    if ssh "${SSH_OPTS[@]}" "$pi" "kill -0 ${pid} 2>/dev/null"; then
      echo "  ${pi}: PID ${pid} vivo após 5s — OK"
    else
      echo "  [ERRO] ${pi}: PID ${pid} MORREU logo após subir. Últimas linhas de supernode_${TAG}.log:"
      local logtail
      logtail=$(ssh "${SSH_OPTS[@]}" "$pi" "tail -n 20 ${APP_DIR_PI}/supernode_${TAG}.log" 2>&1)
      echo "$logtail" | sed 's/^/      /'
      # Diagnóstico automático — classifica pela assinatura já vista no
      # smoke test real (ver comentários acima), para não precisar reler
      # o log a olho toda vez.
      case "$logtail" in
        *"already in use"*)
          echo "    >>> DIAGNÓSTICO: porta ClientAppIO (9094) ainda ocupada nesse Pi."
          echo "        Rode: $0 status   (mostra quem está segurando a porta)"
          echo "        ou:   $0 stop     (mata SuperNode/SuperExec remanescentes)" ;;
        *"flower-superexec"*)
          echo "    >>> DIAGNÓSTICO: flower-superexec não encontrado no PATH remoto —" \
               "confira a instalação do flwr no venv desse Pi (${PYTHON_PI})." ;;
        *"Connection refused"* | *"failed to connect"* | *"UNAVAILABLE"*)
          echo "    >>> DIAGNÓSTICO: não alcançou o SuperLink em ${SERVER_IP}:9092 —" \
               "confira se 'superlink' está de pé e se a rede permite esse Pi chegar lá." ;;
        *"ModuleNotFoundError"* | *"ImportError"*)
          echo "    >>> DIAGNÓSTICO: erro de import no venv desse Pi (PYTHON_PI=${PYTHON_PI})." ;;
        *)
          echo "    >>> DIAGNÓSTICO: assinatura não reconhecida automaticamente — leia o" \
               "trecho de log acima." ;;
      esac
      falhou=1
    fi
  done
  if [ "$falhou" -ne 0 ]; then
    echo "[smoke] AVISO: pelo menos 1 SuperNode morreu no boot — corrija antes de prosseguir para 'run'."
  fi
  echo "[smoke] processos vivos != registrados no SuperLink — confira também o log do SuperLink (terminal 1),"
  echo "        ou rode '$0 status' para ver porta+processo+log dos 5 Pis de uma vez."
}

stop_supernodes() {
  local idx=0
  for pi in "${PIS[@]}"; do
    idx=$((idx + 1))
    echo "[smoke] verificando/encerrando flower-supernode em ${pi}…"
    # -n: mesmo motivo do resto do script (evita hang de ssh+background).
    # SIGTERM primeiro (permite desconectar do SuperLink de forma limpa),
    # espera 2s, e só então SIGKILL no que sobrar — sem isso, um processo
    # preso pode nunca soltar a porta 9094 (ClientAppIo API) para a
    # próxima tentativa.
    # IMPORTANTE (corrigido): antes, um SSH que falhasse aqui (Pi fora do
    # ar, por exemplo) derrubava o script inteiro via `set -e`. Agora o
    # `|| echo ...` isola a falha e o loop segue para os próximos Pis.
    ssh "${SSH_OPTS[@]}" "$pi" "
      pids=\$(pgrep -f 'flower-supernode|flower-superexec' || true)
      if [ -n \"\$pids\" ]; then
        echo \"  encontrados: \$pids\"
        kill \$pids 2>/dev/null || true
        sleep 2
        pids2=\$(pgrep -f 'flower-supernode|flower-superexec' || true)
        [ -n \"\$pids2\" ] && kill -9 \$pids2 2>/dev/null || true
        echo '  encerrado.'
      else
        echo '  nenhum processo encontrado.'
      fi
      restante=\$(ss -tulnp 2>/dev/null | grep ':9094 ' || true)
      if [ -n \"\$restante\" ]; then
        echo \"  [AVISO] porta 9094 AINDA ocupada nesse Pi após o kill:\"
        echo \"    \$restante\"
        echo '    (pode ser outro processo, não do Flower — confira manualmente com sudo lsof -i :9094)'
      fi
    " || echo "  [ERRO] não foi possível conectar via SSH em ${pi} para limpar — verifique esse Pi manualmente."
  done
  echo "[smoke] pronto — confira com: '$0 status', ou manualmente:"
  echo "        for p in ${PIS[*]}; do ssh \$p pgrep -af flower-supernode; done"
}

# manual(): NÃO orquestra nada via SSH — só imprime o comando pronto para
# colar em cada terminal (1 do servidor + 1 por Pi). Use quando quiser ver
# o boot de cada nó AO VIVO na própria tela dele, sem nohup/setsid no meio
# escondendo erro, e sem depender do smoke test para propagar falha de SSH.
manual() {
  cat <<EOF
=====================================================================
MODO MANUAL — um comando por terminal, sem orquestração via SSH/nohup.
Abra 6 terminais (1 servidor + 5 Pis, ex.: 'ssh pi1', 'ssh pi2', ...) e
cole o bloco correspondente em cada um. Rodando em primeiro plano,
qualquer erro (porta em uso, import, SuperLink inalcançável) aparece
na hora, na tela daquele nó — Ctrl+C mata o processo de verdade, sem
deixar órfão (diferente do modo orquestrado 'supernodes', que usa
setsid+nohup de propósito para sobreviver a uma queda de SSH).

Antes de rodar, garanta que não há SuperNode antigo vivo: rode
'$0 stop' a partir do servidor, ou 'pgrep -af flower-supernode'
direto em cada Pi.
=====================================================================

--- Terminal do SERVIDOR (SuperLink) ---
export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1
flower-superlink --insecure

EOF
  local idx=0
  for pi in "${PIS[@]}"; do
    idx=$((idx + 1))
    cat <<EOF
--- Terminal do ${pi} (pi=${idx}) ---
mkdir -p ${METRICS_DIR_PI}
cd ${APP_DIR_PI}
export PIPELINE_DIR=${PIPELINE_DIR_PI}
export WLSTMIX_DIR=${WLSTMIX_DIR_PI}
export PATH="$(dirname "${SUPERNODE_BIN_PI}"):\$PATH"
${SUPERNODE_BIN_PI} --insecure \\
  --superlink ${SERVER_IP}:9092 \\
  --node-config "data-root='${DATA_ROOT}' pi=${idx} metrics-dir='${METRICS_DIR_PI}'"

EOF
  done
  echo "Depois que os 5 registrarem (confira no log do SuperLink), dispare o"
  echo "'run' normalmente a partir do servidor: $0 run"
}

# status(): foto rápida (porta + processo + log) dos 5 Pis de uma vez,
# via SSH curto e não-interativo — não precisa abrir 5 terminais com watch.
status() {
  echo "[status] Verificando os 5 Pis (portas 909x, processos flower, log)…"
  local idx=0
  for pi in "${PIS[@]}"; do
    idx=$((idx + 1))
    echo "----- ${pi} (pi=${idx}) -----"
    if out=$(ssh "${SSH_OPTS[@]}" "$pi" "
      echo '[portas 909x]'
      ss -tulnp 2>/dev/null | grep -E ':909[0-9]' || echo '  (nenhuma escutando)'
      echo '[processos flower]'
      pgrep -af 'flower-supernode|flower-superexec' || echo '  (nenhum processo)'
      echo '[últimas linhas: supernode_${TAG}.log]'
      tail -n 6 '${APP_DIR_PI}/supernode_${TAG}.log' 2>/dev/null || echo '  (log ainda não existe)'
    " 2>&1); then
      echo "$out" | sed 's/^/  /'
    else
      echo "  [ERRO] não foi possível conectar via SSH em ${pi}."
    fi
    echo
  done
  echo "[status] Processo vivo != registrado na Fleet API — confirme também no log do SuperLink (terminal 1)."
}

# watch_pi(): abre um 'watch' interativo DENTRO do Pi indicado (porta +
# processos + log, atualizado a cada 2s) — o pedido de "deixar um
# terminal aberto olhando a conexão". Usa opções de ssh próprias (com -t
# para alocar pty; SSH_OPTS tem -n, que é incompatível com terminal
# interativo, por isso não é reaproveitado aqui).
watch_pi() {
  local pi="${1:-}"
  if [ -z "$pi" ]; then
    echo "Uso: $0 watch <alias-do-pi>   (ex.: $0 watch pi3)"
    exit 1
  fi
  ssh -t -o "ConnectTimeout=${SSH_TIMEOUT_S}" "$pi" "watch -n 2 '
    echo \"--- portas 909x ---\"; ss -tulnp 2>/dev/null | grep -E \":909[0-9]\" || echo \"(nenhuma)\"
    echo; echo \"--- processos flower ---\"; pgrep -af \"flower-supernode|flower-superexec\" || echo \"(nenhum)\"
    echo; echo \"--- últimas linhas do log ---\"; tail -n 8 ${APP_DIR_PI}/supernode_${TAG}.log 2>/dev/null
  '"
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
  stop)       stop_supernodes ;;
  manual)     manual ;;
  status)     status ;;
  watch)      shift; watch_pi "${1:-}" ;;
  run)        run ;;
  collect)    collect ;;
  report)     shift; report "$@" ;;
  all)        preflight; supernodes; sleep 12; run; collect; report ;;
  *)          usage ;;
esac
