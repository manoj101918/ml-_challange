"""Progress lines with a time estimate (ETA) for the long loops.

Plain log lines instead of a tqdm bar: a bar redraws itself with carriage returns, which turns into one long garbled
line in log files (the Colab runs `tee` their output to Google Drive). One line every ~`every` seconds is readable
everywhere:  "  features: entity chunks: 6/17 (35%) | 4m10s elapsed, ~7m40s left"
"""

import time


def _fmt(seconds):
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"
    return f"{seconds // 60}m{seconds % 60:02d}s"


class Progress:
    def __init__(self, total, label, log=print, every=60):
        self.total, self.label, self.log, self.every = max(int(total), 1), label, log, every
        self.start = self.last = time.time()
        self.done = 0

    def update(self, n=1):
        self.done += n
        now = time.time()
        if now - self.last >= self.every or self.done >= self.total:
            self.last = now
            elapsed = now - self.start
            left = elapsed / self.done * max(self.total - self.done, 0) if self.done else 0
            self.log(f"  {self.label}: {self.done:,}/{self.total:,} ({min(self.done / self.total, 1):.0%}) | "
                     f"{_fmt(elapsed)} elapsed, ~{_fmt(left)} left")
