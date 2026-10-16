"""Runtime acceptance checks and measured GPU memory (includes PhysX/Kit)."""
import subprocess
import threading
import torch

class MemoryMonitor:
    def __init__(self,index=0):
        self.peak_mib=0.;self.index=index;self.stop_event=threading.Event()
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
    def _run(self):
        while not self.stop_event.is_set():
            try:
                out=subprocess.check_output(['nvidia-smi',f'--id={self.index}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=3)
                self.peak_mib=max(self.peak_mib,float(out.strip()))
            except (OSError,ValueError,subprocess.SubprocessError):pass
            self.stop_event.wait(.5)
    def stop(self):
        self.stop_event.set();self.thread.join(timeout=4)
        return self.peak_mib
