"""fed_monitor.py — Instrumentação leve para os clientes federados (Raspberry Pi).

Fornece, sem nenhuma dependência além da stdlib:

  1. Tempo total de execução (wall time) do bloco monitorado;
  2. Load médio de CPU (via /proc/stat, delta entre amostras) e uso médio
     de RAM do sistema + RSS do processo (via /proc/meminfo e /proc/self/status);
  3. ETA no mesmo espírito do step6_train.py: unidades/min + projeção,
     persistido em progress.json com escrita atômica (.tmp + rename);
  4. Log da loss por passo de treinamento em JSONL (uma linha por passo),
     consumível pelo plot_loss.py.

Integração mínima (client_app.py / task.py), ~4 linhas:

    from fed_monitor import RunMonitor

    mon = RunMonitor(out_dir=metrics_dir, tag=f"round{r}", total_units=n_batches)
    with mon:
        for step, (xb, yb) in enumerate(loader):
            loss = train_step(...)
            mon.log_loss(loss, step=step, stage="fit", round=r, epoch=e)
            mon.tick(1)                      # avança o ETA em 1 unidade
    summary = mon.summary()                  # dict pronto para o MetricRecord
    # ... e/ou mesclar no confusion_matrix.json existente:
    mon.merge_into_json(confusion_matrix_path)

O `summary` inclui `wall_time_s` — resolvendo a ausência de tempo total no
confusion_matrix.json — e as médias de CPU/RAM do dispositivo durante o run.

Compatibilidade: Linux (aarch64 incluso), Python >= 3.9. Testado sem psutil.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional


# --------------------------------------------------------------------------
# Leitura de /proc (Linux)
# --------------------------------------------------------------------------

def _read_cpu_jiffies() -> Optional[tuple[int, int]]:
    """Retorna (busy, total) jiffies agregados da primeira linha de /proc/stat."""
    try:
        with open("/proc/stat") as f:
            parts = f.readline().split()
        vals = [int(v) for v in parts[1:]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
        total = sum(vals)
        return total - idle, total
    except (OSError, ValueError, IndexError):
        return None


def _read_meminfo_gb() -> Optional[tuple[float, float]]:
    """Retorna (mem_total_gb, mem_available_gb) de /proc/meminfo."""
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                info[key] = int(rest.split()[0])  # kB
        return info["MemTotal"] / 1048576.0, info["MemAvailable"] / 1048576.0
    except (OSError, ValueError, KeyError):
        return None


def _read_rss_gb() -> Optional[float]:
    """RSS do processo corrente em GB, via /proc/self/status (VmRSS)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1048576.0
    except (OSError, ValueError):
        pass
    return None


def _read_loadavg1() -> Optional[float]:
    try:
        with open("/proc/loadavg") as f:
            return float(f.read().split()[0])
    except (OSError, ValueError):
        return None


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Amostrador de recursos em background
# --------------------------------------------------------------------------

class _ResourceSampler(threading.Thread):
    """Thread daemon que amostra CPU/RAM a cada `interval_s` segundos."""

    def __init__(self, interval_s: float = 5.0):
        super().__init__(daemon=True, name="fed-monitor-sampler")
        self.interval_s = interval_s
        self._stop = threading.Event()
        self.samples: list[dict[str, float]] = []
        self._prev_jiffies = _read_cpu_jiffies()

    def run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample_once()

    def _sample_once(self) -> None:
        sample: dict[str, float] = {"ts": time.time()}

        cur = _read_cpu_jiffies()
        if cur is not None and self._prev_jiffies is not None:
            dbusy = cur[0] - self._prev_jiffies[0]
            dtotal = cur[1] - self._prev_jiffies[1]
            if dtotal > 0:
                sample["cpu_pct"] = 100.0 * dbusy / dtotal
        self._prev_jiffies = cur

        load1 = _read_loadavg1()
        if load1 is not None:
            sample["load1"] = load1

        mem = _read_meminfo_gb()
        if mem is not None:
            sample["ram_total_gb"] = mem[0]
            sample["ram_used_gb"] = mem[0] - mem[1]
            sample["ram_available_gb"] = mem[1]

        rss = _read_rss_gb()
        if rss is not None:
            sample["rss_gb"] = rss

        self.samples.append(sample)

    def stop(self) -> None:
        self._stop.set()
        # amostra final para runs mais curtos que o intervalo
        self._sample_once()


# --------------------------------------------------------------------------
# Monitor principal
# --------------------------------------------------------------------------

class RunMonitor:
    """Cronômetro + amostrador de recursos + ETA + logger de loss.

    Parameters
    ----------
    out_dir : diretório onde serão gravados loss_<tag>.jsonl,
              progress_<tag>.json e summary_<tag>.json.
    tag     : identificador do run (ex.: "round3", "smoke").
    total_units : total de unidades de trabalho para o ETA (batches,
              janelas ou shards — a escolha é sua, contanto que `tick`
              use a mesma unidade). Se None, o ETA fica desabilitado.
    sample_interval_s : período de amostragem de CPU/RAM.
    progress_every : grava progress.json a cada N ticks (default 25,
              espelhando o comportamento do step6_train.py).
    """

    def __init__(
        self,
        out_dir: str | os.PathLike,
        tag: str = "run",
        total_units: Optional[int] = None,
        sample_interval_s: float = 5.0,
        progress_every: int = 25,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tag = tag
        self.total_units = total_units
        self.progress_every = max(1, progress_every)

        self._sampler = _ResourceSampler(sample_interval_s)
        self._t0: Optional[float] = None
        self._t1: Optional[float] = None
        self._done_units = 0
        self._loss_path = self.out_dir / f"loss_{tag}.jsonl"
        self._loss_fh = None

    # ---- ciclo de vida -----------------------------------------------------

    def __enter__(self) -> "RunMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        self._t0 = time.time()
        self._loss_fh = open(self._loss_path, "a", buffering=1)
        self._sampler.start()

    def stop(self) -> None:
        self._t1 = time.time()
        self._sampler.stop()
        if self._loss_fh is not None:
            self._loss_fh.close()
            self._loss_fh = None
        _atomic_write_json(self.out_dir / f"summary_{self.tag}.json", self.summary())

    # ---- loss --------------------------------------------------------------

    def log_loss(self, loss: float, *, step: int, stage: str = "fit",
                 round: Optional[int] = None, epoch: Optional[int] = None,
                 **extra: Any) -> None:
        """Uma linha JSONL por passo: {ts, stage, round, epoch, step, loss, ...}."""
        if self._loss_fh is None:
            raise RuntimeError("RunMonitor não iniciado (use start() ou `with`).")
        rec: dict[str, Any] = {
            "ts": time.time(), "stage": stage, "step": int(step),
            "loss": float(loss),
        }
        if round is not None:
            rec["round"] = int(round)
        if epoch is not None:
            rec["epoch"] = int(epoch)
        rec.update(extra)
        self._loss_fh.write(json.dumps(rec) + "\n")

    # ---- ETA ---------------------------------------------------------------

    def tick(self, n: int = 1) -> None:
        """Avança `n` unidades de trabalho e, periodicamente, grava progress.json."""
        self._done_units += n
        if self._done_units % self.progress_every == 0:
            self.write_progress()

    def eta(self) -> dict[str, Optional[float]]:
        """Taxa (unid/min) e ETA (s), no formato do step6_train/analyze_log."""
        if self._t0 is None:
            return {"units_per_min": None, "eta_s": None}
        elapsed = (self._t1 or time.time()) - self._t0
        if elapsed <= 0 or self._done_units == 0:
            return {"units_per_min": None, "eta_s": None}
        rate = self._done_units / (elapsed / 60.0)
        eta_s = None
        if self.total_units:
            remaining = max(0, self.total_units - self._done_units)
            eta_s = remaining / rate * 60.0
        return {"units_per_min": rate, "eta_s": eta_s}

    def write_progress(self) -> None:
        eta = self.eta()
        _atomic_write_json(self.out_dir / f"progress_{self.tag}.json", {
            "tag": self.tag,
            "done_units": self._done_units,
            "total_units": self.total_units,
            "units_per_min": eta["units_per_min"],
            "eta_s": eta["eta_s"],
            "eta_h": None if eta["eta_s"] is None else eta["eta_s"] / 3600.0,
            "elapsed_s": None if self._t0 is None else time.time() - self._t0,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    # ---- resumo ------------------------------------------------------------

    @staticmethod
    def _avg(samples: list[dict[str, float]], key: str) -> Optional[float]:
        vals = [s[key] for s in samples if key in s]
        return sum(vals) / len(vals) if vals else None

    @staticmethod
    def _max(samples: list[dict[str, float]], key: str) -> Optional[float]:
        vals = [s[key] for s in samples if key in s]
        return max(vals) if vals else None

    def summary(self) -> dict[str, Any]:
        s = self._sampler.samples
        t_end = self._t1 or time.time()
        eta = self.eta()
        return {
            "tag": self.tag,
            "hostname": os.uname().nodename,
            "started_at": self._t0,
            "wall_time_s": None if self._t0 is None else t_end - self._t0,
            "done_units": self._done_units,
            "total_units": self.total_units,
            "units_per_min": eta["units_per_min"],
            "eta_s_at_end": eta["eta_s"],
            "cpu_pct_avg": self._avg(s, "cpu_pct"),
            "cpu_pct_max": self._max(s, "cpu_pct"),
            "load1_avg": self._avg(s, "load1"),
            "ram_used_gb_avg": self._avg(s, "ram_used_gb"),
            "ram_used_gb_max": self._max(s, "ram_used_gb"),
            "ram_total_gb": self._avg(s, "ram_total_gb"),
            "rss_gb_avg": self._avg(s, "rss_gb"),
            "rss_gb_max": self._max(s, "rss_gb"),
            "n_samples": len(s),
        }

    def merge_into_json(self, path: str | os.PathLike,
                        key: str = "run_monitor") -> None:
        """Mescla o summary num JSON existente (ex.: confusion_matrix.json).

        Se o arquivo não existir, cria-o contendo apenas {key: summary}.
        Escrita atômica.
        """
        p = Path(path)
        try:
            data = json.loads(p.read_text()) if p.exists() else {}
            if not isinstance(data, dict):
                data = {"_original": data}
        except (OSError, json.JSONDecodeError):
            data = {}
        data[key] = self.summary()
        _atomic_write_json(p, data)


# --------------------------------------------------------------------------
# Auto-teste rápido (não requer torch): python fed_monitor.py
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import math
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        mon = RunMonitor(td, tag="selftest", total_units=200,
                         sample_interval_s=0.5, progress_every=50)
        with mon:
            for step in range(200):
                # loss sintética decrescente com ruído
                loss = 1.0 / (1 + 0.05 * step) + 0.02 * math.sin(step)
                mon.log_loss(loss, step=step, stage="fit", round=1, epoch=1)
                mon.tick()
                time.sleep(0.01)
        out = mon.summary()
        assert out["wall_time_s"] and out["wall_time_s"] > 1.5
        assert out["done_units"] == 200
        assert (Path(td) / "loss_selftest.jsonl").exists()
        assert (Path(td) / "summary_selftest.json").exists()
        assert (Path(td) / "progress_selftest.json").exists()
        print("[fed_monitor] selftest OK")
        print(json.dumps(out, indent=2))