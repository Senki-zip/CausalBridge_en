"""Parallel Step 8 output is rendered only in its parent process."""
import io
import logging
import queue
import json

import pytest

from CausalBridge.terminal_progress import SimulationProgress
from CausalBridge.simulation_queue import _SimulationLogHandler, _drain_progress


class Terminal(io.StringIO):
    def isatty(self):
        return True


@pytest.mark.parametrize("tty", [False, True])
def test_independent_cohort_rows_and_replay(tty):
    stream = Terminal() if tty else io.StringIO()
    with SimulationProgress([("a", "cluster_0"), ("b", "cluster_1")], stream=stream) as progress:
        progress.update("a", "加载断点")
        progress.update("b", "加载断点")
        progress.update("a", "传播第1轮", 20, 41)
        progress.update("b", "传播第1轮", 10, 50)
        assert progress.rows["a"]["completed"] == 20
        assert progress.rows["b"]["completed"] == 10
        progress.update("a", "传播第2轮", 0, 41)
        assert progress.rows["a"]["completed"] == 0
        progress.update("a", "传播第2轮", 41, 41)
        progress.finish("a")
        progress.update("a", "传播第1轮", 35, 41)  # delayed IPC cannot undo completion
        assert progress.rows["a"]["phase"] == "完成"
        progress.finish("b", failed=True)
        assert progress.rows["b"]["completed"] == 10
    output = stream.getvalue()
    assert "cluster_0" in output and "cluster_1" in output
    assert "41/41" in output and "10/50" in output
    assert "50/50" not in output
    if not tty:
        assert len(output.splitlines()) == 4
        assert "\x1b" not in output


def test_worker_logs_become_scoped_events_and_keep_warnings(caplog):
    events = queue.Queue()
    handler = _SimulationLogHandler(events, "manifest", "cluster_4")
    stream = io.StringIO()
    messages = [
        (logging.INFO, "Step 8 校验 peak→gene 模型库并加载"),
        (logging.INFO, "Step 8 传播 [MYB]: bin=41，活跃模型=100，锁定组=0"),
        (logging.INFO, "Step 8 传播 bin 20/41，活跃模型=100，耗时 10.0s"),
        (logging.WARNING, "unsupported input"),
        (logging.INFO, "Step 8 重建细胞端点 [MYB]"),
        (logging.INFO, "扰动完成 [cluster_4/MYB]: 扰动基因=79"),
    ]
    with caplog.at_level(logging.INFO), SimulationProgress([("manifest", "cluster_4")], stream=stream) as progress:
        for level, message in messages:
            handler.handle(logging.LogRecord("CausalBridge.test", level, __file__, 1, message, (), None))
        _drain_progress(events, progress)
        assert progress.rows["manifest"]["completed"] == 41
        progress.finish("manifest")
    assert "[cluster_4] unsupported input" in caplog.text
    assert "[cluster_4] 扰动完成" in caplog.text
    assert "传播 bin 20/41" not in caplog.text
    assert stream.getvalue() == "Step 8 cluster_4: 开始\nStep 8 cluster_4: 完成 41/41\n"


def test_failure_before_propagation_does_not_report_full_bar():
    stream = io.StringIO()
    with SimulationProgress([("a", "cluster_0")], stream=stream) as progress:
        progress.update("a", "加载断点")
        progress.finish("a", failed=True)
    assert "失败 --/--" in stream.getvalue()
    assert "1/1" not in stream.getvalue()


def test_spawn_failure_keeps_cluster_and_real_completion_count(tmp_path, capsys):
    from CausalBridge import simulation_queue
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(dict(schema=simulation_queue.SCHEMA,
        scope="cluster_broken", context="missing-context.pkl")))
    with pytest.raises(FileNotFoundError):
        list(simulation_queue.run_simulation_jobs([str(path)], workers=1))
    captured = capsys.readouterr()
    assert "cluster_broken: 失败 --/--" in captured.err
    assert "完成" not in captured.err
