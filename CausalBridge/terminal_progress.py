"""Compact parent-side stderr progress, with bounded redirected output."""
import logging
import shutil
import sys
import unicodedata
from contextlib import ExitStack

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


def _width(text):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


class TerminalProgress:
    def __init__(self, total, desc, *, stream=None, logger=None):
        self.total = total
        self.desc = desc
        self.stream = sys.stderr if stream is None else stream
        self.logger = logger
        self.completed = 0
        self.bar = None

    def _layout(self):
        assert self.bar is not None
        columns = max(1, shutil.get_terminal_size(fallback=(80, 24)).columns - 1)
        count_width = len(f"{self.total}/{self.total}")
        desc = self.desc
        # Preserve the count on small terminals; omit the label before the bar.
        if columns < _width(desc) + count_width + 6:
            desc = ""
        if columns <= count_width + 1:
            fmt = "{n_fmt}/{total_fmt}"
        else:
            fmt = "{desc}: {bar} {n_fmt}/{total_fmt}" if desc else "{bar} {n_fmt}/{total_fmt}"
        self.bar.desc = desc
        self.bar.bar_format = fmt
        return columns, None

    def __enter__(self):
        self.stack = ExitStack()
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        if self.tty:
            loggers = []
            current = self.logger or logging.getLogger()
            while current is not None:
                loggers.append(current)
                if not current.propagate:
                    break
                current = current.parent
            self.stack.enter_context(logging_redirect_tqdm(loggers=loggers))
            self.bar = tqdm(total=self.total, desc=self.desc, file=self.stream,
                            dynamic_ncols=True, leave=True, mininterval=.15,
                            bar_format="{desc}: {bar} {n_fmt}/{total_fmt}",
                            ascii=True, disable=False, delay=float("inf"))
            self.bar.dynamic_ncols = lambda _: self._layout()
            self.bar.delay = 0
            self.bar.refresh()
        else:
            self.stream.write(f"{self.desc}: 0/{self.total}\n")
            self.stream.flush()
        return self

    def update(self, completed):
        if not self.completed <= completed <= self.total:
            raise ValueError("progress count must be monotonic and within total")
        delta = completed - self.completed
        self.completed = completed
        if self.bar is not None:
            self.bar.update(delta)

    def __exit__(self, exc_type, exc, tb):
        try:
            success = exc_type is None and self.completed == self.total
            if self.bar is not None:
                if not success:
                    self.bar.leave = False
                self.bar.close()
            if not self.tty or not success:
                suffix = "" if success else "（中断）"
                self.stream.write(f"{self.desc}{suffix}: {self.completed}/{self.total}\n")
                self.stream.flush()
        finally:
            self.stack.close()


class SimulationProgress:
    """One parent-owned row per cohort; workers never write terminal cursors."""

    def __init__(self, jobs, *, stream=None):
        self.stream = sys.stderr if stream is None else stream
        self.rows = {key: dict(label=label, phase="等待", completed=0, total=None, bar=None)
                     for key, label in jobs}

    def __enter__(self):
        self.stack = ExitStack()
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        if self.tty:
            self.stack.enter_context(logging_redirect_tqdm())
            for position, row in enumerate(self.rows.values()):
                row["bar"] = tqdm(total=None, desc=f"{row['label']} 等待", position=position,
                    file=self.stream, dynamic_ncols=True, leave=True, ascii=True,
                    mininterval=.15, bar_format="{desc}: {bar} {n_fmt}/{total_fmt}")
        return self

    def update(self, key, phase, completed=None, total=None):
        row = self.rows[key]
        if row["phase"] in ("完成", "失败"):
            return
        first = row["phase"] == "等待"
        row["phase"] = phase
        bar = row["bar"]
        if total is not None:
            if total < 0 or completed is None or not 0 <= completed <= total:
                raise ValueError("invalid simulation progress")
            if bar is not None and (row["total"] != total or completed < row["completed"]):
                bar.reset(total=total)
            row.update(completed=completed, total=total)
        if bar is not None:
            bar.set_description_str(f"{row['label']} {phase}", refresh=False)
            if total is not None:
                bar.total = total
                bar.update(completed - bar.n)
            bar.refresh()
        elif first:
            self.stream.write(f"Step 8 {row['label']}: 开始\n")
            self.stream.flush()

    def finish(self, key, *, failed=False):
        row = self.rows[key]
        count = row["completed"] if failed else row["total"]
        phase = "失败" if failed else "完成"
        self.update(key, phase, count, row["total"])
        if not self.tty:
            count = f"{row['completed']}/{row['total']}" if row["total"] is not None else "--/--"
            self.stream.write(f"Step 8 {row['label']}: {phase} {count}\n")
            self.stream.flush()

    def __exit__(self, exc_type, exc, tb):
        try:
            for row in self.rows.values():
                if row["bar"] is not None:
                    row["bar"].close()
        finally:
            self.stack.close()
