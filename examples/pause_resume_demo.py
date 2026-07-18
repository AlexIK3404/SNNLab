"""
Demonstrates in-process pause/resume control with a worker thread.

Демонстрирует pause/resume в одном процессе через worker thread.
"""

from __future__ import annotations

import threading
import time

from snnlab.runtime.control import RunControl


def worker(control: RunControl) -> None:
    for sample in range(10):
        time.sleep(0.1)
        print(f"finished sample {sample}")
        if control.stop_requested:
            return
        if control.pause_requested:
            print("safe-point pause")
            control.wait_if_paused()


control = RunControl()
thread = threading.Thread(target=worker, args=(control,), daemon=True)
thread.start()
time.sleep(0.25)
control.request_pause()
time.sleep(0.4)
control.resume()
thread.join()
