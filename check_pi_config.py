#!/usr/bin/env python3
"""check_pi_env.py — Verificação ISOLADA de ambiente para o cliente
federado (Raspberry Pi). NÃO depende de nenhum arquivo do pipeline
(config.py, PIPELINE_DIR, etc.) — copie este único arquivo para o Pi e
rode:

    python3 check_pi_env.py

Para checar os 5 Pis de uma vez a partir da máquina servidora (ajuste
usuário/IPs/caminho):

    for ip in 192.168.1.101 192.168.1.102 192.168.1.103 \
              192.168.1.104 192.168.1.105; do
        echo "=== $ip ==="
        scp check_pi_env.py pi@$ip:/tmp/
        ssh pi@$ip "python3 /tmp/check_pi_env.py"
    done

O QUE É TESTADO (e por quê): a lista abaixo é exatamente o que o
CLIENTE FEDERADO (task.py/client_app.py) importa, direta ou
transitivamente — nada a mais. Notas importantes, descobertas ao
rastrear a árvore de imports real do projeto:
  * duckdb NÃO entra aqui: é importado só dentro de uma função usada
    pelos passos 1-3 (preparação de dados na máquina grande), nunca
    pelo cliente. Não precisa instalar no Pi.
  * pywavelets (pywt) e pandas SÃO exigidos mesmo sem uso funcional
    direto no cliente: o módulo step6_train.py importa, no topo do
    arquivo, o step2_3_windows_wavelet.py (que usa pywt/pandas) — o
    Python executa esses imports ao carregar task.py, mesmo que o
    cliente nunca chame as funções que os usam de fato.
  * scikit-learn É usado de verdade, dentro de evaluate() (F1/precisão/
    revocação).
Saída: tabela PASS/FAIL por pacote + um teste funcional mínimo de cada
um (não basta importar — em ARM, builds incompatíveis às vezes importam
mas falham na primeira operação real). Código de saída: 0 se tudo
passou, 1 caso contrário (para uso em scripts de automação).
"""
from __future__ import annotations
import platform
import sys
import time

RESULTS: list[tuple[str, bool, str]] = []   # (nome, ok, detalhe)


def _check(name: str, fn):
    t0 = time.time()
    try:
        detail = fn()
        RESULTS.append((name, True, f"{detail} ({(time.time()-t0)*1000:.0f}ms)"))
    except Exception as e:
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))


def check_python():
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 10
    if not ok:
        raise RuntimeError(f"Python {v.major}.{v.minor} — recomendado >=3.10")
    return f"Python {platform.python_version()} em {platform.machine()}"


def check_numpy():
    import numpy as np
    a = np.arange(1000, dtype=np.float64)
    assert float(a.sum()) == 499500.0
    return f"numpy {np.__version__}"


def check_pandas():
    import pandas as pd
    df = pd.DataFrame({"x": [1, 2, 3]})
    assert int(df["x"].sum()) == 6
    return f"pandas {pd.__version__}"


def check_pywt():
    import pywt
    import numpy as np
    x = np.random.default_rng(0).normal(size=192)
    coeffs = pywt.wavedec(x, "db4", level=5)
    rec = pywt.waverec(coeffs, "db4")[:len(x)]
    assert rec.shape[0] == 192
    return f"PyWavelets {pywt.__version__} (round-trip db4 OK)"


def check_sklearn():
    from sklearn.metrics import f1_score
    score = f1_score([0, 1, 1, 0], [0, 1, 0, 0])
    assert 0.0 <= score <= 1.0
    import sklearn
    return f"scikit-learn {sklearn.__version__} (F1={score:.2f})"


def check_torch():
    import torch
    import torch.nn as nn
    cuda = torch.cuda.is_available()
    # sanidade numérica básica
    x = torch.randn(4, 8)
    y = (x @ x.T).sum().item()
    assert isinstance(y, float)

    # micro-benchmark: forward+backward numa camada do tamanho real do
    # projeto (backcast=168, hidden=256), para ter uma noção de tempo
    # de treino local no hardware do Pi.
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(168, 256), nn.GELU(), nn.Linear(256, 24))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xb = torch.randn(64, 168)
    yb = torch.randn(64, 24)
    n_iter = 20
    t0 = time.time()
    for _ in range(n_iter):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(xb), yb)
        loss.backward()
        opt.step()
    ms_per_iter = (time.time() - t0) * 1000 / n_iter
    return (f"torch {torch.__version__} | CUDA={cuda} (esperado False "
           f"num Pi) | {ms_per_iter:.1f}ms/iteração (lote=64, camada "
           f"~real) — referência p/ estimar tempo de época local")


def check_flwr():
    import flwr
    return f"flwr {flwr.__version__}"


def main():
    checks = [
        ("Python", check_python),
        ("numpy", check_numpy),
        ("pandas", check_pandas),
        ("pywavelets", check_pywt),
        ("scikit-learn", check_sklearn),
        ("torch", check_torch),
        ("flwr", check_flwr),
    ]
    print(f"=== check_pi_env — {platform.node()} ===\n")
    for name, fn in checks:
        _check(name, fn)

    width = max(len(n) for n, _, _ in RESULTS) + 2
    all_ok = True
    for name, ok, detail in RESULTS:
        status = "OK  " if ok else "FAIL"
        all_ok &= ok
        print(f"[{status}] {name:<{width}} {detail}")

    print(f"\n{'TUDO OK — ambiente pronto para o cliente federado.' if all_ok else 'HÁ FALHAS — resolva antes de rodar o client_app neste Pi.'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()