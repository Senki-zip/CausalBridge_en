"""Visible progress contracts, independent of the fitting implementation."""
import io
import logging
import os
import unicodedata
from concurrent.futures import Future, as_completed

import pytest

from CausalBridge.terminal_progress import TerminalProgress


class Terminal(io.StringIO):
    def isatty(self):
        return True


def frames(stream):
    return [line.rstrip() for line in stream.getvalue().replace("\n", "\r").split("\r") if line.strip()]


def width(text):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


@pytest.mark.parametrize("columns", [80, 32, 18, 10, 1])
def test_tty_rightmost_count_resize_and_completion(monkeypatch, columns):
    monkeypatch.setattr("shutil.get_terminal_size", lambda **kwargs: os.terminal_size((columns, 24)))
    stream = Terminal()
    with TerminalProgress(2971, "Step 5 拟合", stream=stream) as progress:
        progress.update(123)
        # Render an intermediate frame without wall-clock sleeps.
        progress.bar.refresh()
        progress.update(2971)
    rendered = frames(stream)
    assert rendered[0].endswith("0/2971")
    assert any(line.endswith("123/2971") for line in rendered)
    assert rendered[-1].endswith("2971/2971")
    assert stream.getvalue().endswith("\n")
    assert all(width(line) <= max(columns - 1, len("2971/2971")) for line in rendered)
    assert not any(token in stream.getvalue() for token in ("elapsed", "ETA", "%", "workers", "relations", "\x1b"))


def test_redirected_output_bounded_and_stdout_untouched(capsys):
    stream = io.StringIO()
    with TerminalProgress(1000, "Step 6 拟合", stream=stream) as progress:
        for count in range(1, 1001):
            progress.update(count)
    assert stream.getvalue() == "Step 6 拟合: 0/1000\nStep 6 拟合: 1000/1000\n"
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("tty", [False, True])
def test_failure_preserves_actual_count_and_newline(tty):
    stream = Terminal() if tty else io.StringIO()
    with pytest.raises(RuntimeError, match="fit failed"):
        with TerminalProgress(4, "Step 6 拟合", stream=stream) as progress:
            progress.update(2)
            raise RuntimeError("fit failed")
    assert "4/4" not in stream.getvalue()
    assert stream.getvalue().endswith("Step 6 拟合（中断）: 2/4\n")
    assert "\x1b" not in stream.getvalue()


def test_out_of_order_targets_count_completion_not_target_index():
    tasks = {name: Future() for name in ("a", "b", "c")}
    stream = Terminal()
    with TerminalProgress(3, "Step 5 拟合", stream=stream) as progress:
        for name in ("c", "a", "b"):
            tasks[name].set_result(name)
            result = next(as_completed([tasks[name]])).result()
            assert result == name
            progress.update(progress.completed + 1)
            progress.bar.refresh()
    assert [line.rsplit(" ", 1)[-1] for line in frames(stream)] == ["0/3", "1/3", "2/3", "3/3", "3/3"]


def test_rapid_updates_are_throttled():
    stream = Terminal()
    with TerminalProgress(1000, "Step 5 拟合", stream=stream) as progress:
        for count in range(1, 1001):
            progress.update(count)
    assert len(frames(stream)) <= 3
    assert frames(stream)[-1].endswith("1000/1000")


def test_parent_parallel_progress_includes_skips_without_detail_info(monkeypatch, caplog, capsys):
    from types import SimpleNamespace
    from CausalBridge import saturating_transfer as transfer

    class Pool:
        def __init__(self, **kwargs):
            pass

        def submit(self, function, job):
            future = Future()
            future.set_result((job, {}))
            return future

        def shutdown(self, **kwargs):
            pass

    def reverse_completion(pending, **kwargs):
        future = max(pending, key=lambda f: pending[f][0])
        return {future}, set(pending) - {future}

    monkeypatch.setattr(transfer, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(transfer, "wait", reverse_completion)
    jobs = [(name, 1, name, None) for name in ("a", "b", "c", "d")]
    jobs.insert(0, ("skip", 1, None, {"status": "insufficient_fit_rows"}))
    with caplog.at_level(logging.INFO, logger=transfer.__name__):
        result = transfer._execute_target_jobs(iter(jobs), SimpleNamespace(weights=[], fingerprint="test"),
                                               SimpleNamespace(to_dict=lambda: {}), 4, 5, 5, transfer.GENE_TYPE)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Step 5 拟合: 0/5\nStep 5 拟合: 5/5\n"
    assert list(result) == ["a", "b", "c", "d", "skip"]
    assert result["skip"] == (None, {"status": "insufficient_fit_rows"})
    messages = [r.getMessage() for r in caplog.records if r.name == transfer.__name__]
    assert len(messages) == 2
    assert "complete: completed=5/5" in messages[-1]
    assert not any("first target ready" in message for message in messages)


def test_warning_clears_bar_and_restores_console_handler(monkeypatch):
    stream = Terminal()
    monkeypatch.setattr("sys.stderr", stream)
    logger = logging.getLogger("terminal-progress-test")
    handler = logging.StreamHandler(stream)
    original_handlers, original_propagate, original_level = logger.handlers, logger.propagate, logger.level
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    try:
        with TerminalProgress(2, "Step 5 拟合", logger=logger) as progress:
            logger.warning("warning stays visible")
            logger.error("error stays visible")
            progress.update(2)
        assert logger.handlers == [handler]
        assert "warning stays visible\n" in stream.getvalue()
        assert "error stays visible\n" in stream.getvalue()
        assert frames(stream)[-1].endswith("2/2")
    finally:
        logger.handlers, logger.propagate, logger.level = original_handlers, original_propagate, original_level
