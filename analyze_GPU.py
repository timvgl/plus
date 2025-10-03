import subprocess
import threading
import time
from typing import Dict, List, Optional

class GPUUtilMonitor:
    """
    Asynchroner GPU-Auslastungs-Monitor.
    - Start mit start() (oder via 'with' Kontextmanager)
    - Stop mit stop()
    - max_utilization() liefert das Maximum (gesamt oder je Gerät)

    Nutzt bevorzugt NVML (pynvml). Fällt sonst auf 'nvidia-smi' zurück.
    """
    def __init__(self, devices: Optional[List[int]] = None, interval: float = 0.25):
        self.devices = devices          # None => alle
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._max_by_dev: Dict[int, int] = {}
        self._backend = None  # "nvml" oder "smi"

        # Backend ermitteln
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._backend = "nvml"
            self._device_count = pynvml.nvmlDeviceGetCount()
        except Exception:
            self._pynvml = None
            self._backend = "smi"
            # Prüfen, ob nvidia-smi verfügbar ist
            try:
                subprocess.run(["nvidia-smi", "-L"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            except Exception as e:
                raise RuntimeError("Weder pynvml noch nvidia-smi verfügbar. Kann GPU-Utilization nicht messen.") from e

            # Gerätezahl via nvidia-smi erkennen
            out = subprocess.run(["nvidia-smi", "-L"], check=True, stdout=subprocess.PIPE, text=True).stdout.strip()
            self._device_count = sum(1 for line in out.splitlines() if line.strip())

        # Standard: alle Geräte, falls nicht angegeben
        if self.devices is None:
            self.devices = list(range(self._device_count))

        # Max-Map initialisieren
        for d in self.devices:
            self._max_by_dev[d] = 0

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def max_utilization(self) -> int:
        """Maximale GPU-Utilization (in %) über alle überwachten Geräte."""
        with self._lock:
            return max(self._max_by_dev.values()) if self._max_by_dev else 0

    def max_utilization_by_device(self) -> Dict[int, int]:
        """Maximale GPU-Utilization (in %) je Gerät."""
        with self._lock:
            return dict(self._max_by_dev)

    # ---------- Intern ----------

    def _run(self):
        if self._backend == "nvml":
            self._loop_nvml()
        else:
            self._loop_smi()

    def _loop_nvml(self):
        p = self._pynvml
        handles = {i: p.nvmlDeviceGetHandleByIndex(i) for i in self.devices}
        while not self._stop.is_set():
            for i, h in handles.items():
                try:
                    util = p.nvmlDeviceGetUtilizationRates(h).gpu  # Prozent
                except Exception:
                    util = 0
                with self._lock:
                    if util > self._max_by_dev[i]:
                        self._max_by_dev[i] = util
            self._sleep_interval()

    def _loop_smi(self):
        # Wir pollen einmal pro Intervall: ein Aufruf gibt Utilization ALLER GPUs zurück
        base_cmd = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu",
            "--format=csv,noheader,nounits"
        ]
        while not self._stop.is_set():
            try:
                out = subprocess.run(base_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True).stdout
                lines = [l.strip() for l in out.splitlines() if l.strip()]
                sample: Dict[int, int] = {}
                for line in lines:
                    # Format: "0, 27"
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) >= 2:
                        idx = int(parts[0])
                        util = int(parts[1])
                        sample[idx] = util
                with self._lock:
                    for d in self.devices:
                        if d in sample and sample[d] > self._max_by_dev[d]:
                            self._max_by_dev[d] = sample[d]
            except Exception:
                # still try next cycle
                pass
            self._sleep_interval()

    def _sleep_interval(self):
        # kleine Schleife für reaktiveres Stoppen
        t_end = time.time() + self.interval
        while not self._stop.is_set() and time.time() < t_end:
            time.sleep(min(0.05, self.interval))

