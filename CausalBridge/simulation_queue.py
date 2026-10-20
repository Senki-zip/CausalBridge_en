"""Disk-backed Step 8 workers for trusted, invocation-local pipeline snapshots."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import hashlib
import json
import logging
import multiprocessing
import os
from pathlib import Path
import pickle
import tempfile
import queue
import re

logger = logging.getLogger(__name__)
SCHEMA = "causalbridge-simulation-inputs-v1"
_WORKER_PROGRESS = None


class _SimulationLogHandler(logging.Handler):
    """Translate existing runtime logs without changing numerical functions."""
    def __init__(self, events, job, scope):
        super().__init__(logging.INFO)
        self.events, self.job, self.scope = events, job, scope
        self.passes = 0
        self.total = None

    def emit(self, record):
        message = self.format(record)
        event = dict(job=self.job, scope=self.scope)
        start = re.search(r"Step 8 传播 \[.*?\]: bin=(\d+)", message)
        count = re.search(r"Step 8 传播 bin (\d+)/(\d+)", message)
        if start:
            self.passes += 1
            self.total = int(start[1])
            self.events.put(dict(event, kind="progress", phase=f"传播第{self.passes}轮",
                                 completed=0, total=int(start[1])))
        elif count:
            self.events.put(dict(event, kind="progress", phase=f"传播第{self.passes}轮",
                                 completed=int(count[1]), total=int(count[2])))
        elif record.levelno >= logging.WARNING or message.startswith(("扰动完成", "变化关系")):
            self.events.put(dict(event, kind="log", level=record.levelno, message=message))
        elif "Step 8" in message:
            phases = (("重建细胞端点", "重建端点"), ("构建通路", "构建通路"),
                      ("WT 质量核验", "核验 WT"), ("WT 快照", "构建 WT"),
                      ("TF→peak", "加载 TF→peak"), ("peak→gene", "加载 peak→gene"),
                      ("模型库快照", "准备模型"))
            phase = next((value for token, value in phases if token in message), None)
            if phase:
                counter = dict(completed=self.total, total=self.total) if "重建细胞端点" in message and self.total is not None else {}
                self.events.put(dict(event, kind="progress", phase=phase, **counter))


def _simulation_worker_init(events):
    global _WORKER_PROGRESS
    _WORKER_PROGRESS = events


@contextmanager
def _worker_progress(manifest_path):
    if _WORKER_PROGRESS is None:
        yield
        return
    scope = json.loads(Path(manifest_path).read_text())["scope"]
    handler = _SimulationLogHandler(_WORKER_PROGRESS, str(manifest_path), scope)
    package_logger = logging.getLogger("CausalBridge")
    saved = (package_logger.handlers[:], package_logger.propagate, package_logger.level)
    # Spawned workers normally have no handlers. Route package logs to the
    # parent even if a caller configured a package console handler at import.
    package_logger.handlers = [handler]
    package_logger.propagate = False
    package_logger.setLevel(logging.INFO)
    _WORKER_PROGRESS.put(dict(kind="progress", job=str(manifest_path), scope=scope, phase="加载断点"))
    try:
        yield
    finally:
        package_logger.handlers, package_logger.propagate, package_logger.level = saved


def _drain_progress(events, progress):
    while True:
        try:
            event = events.get_nowait()
        except queue.Empty:
            break
        if event["kind"] == "log":
            logger.log(event["level"], "[%s] %s", event["scope"], event["message"])
        else:
            progress.update(event["job"], event["phase"], event.get("completed"), event.get("total"))


def execution_options(config):
    options = config.get("execution", {})
    if not isinstance(options, dict):
        raise ValueError("execution must be a mapping")
    enabled = options.get("defer_cluster_simulation", False)
    workers = options.get("simulation_workers", 2)
    if not isinstance(enabled, bool):
        raise ValueError("execution.defer_cluster_simulation must be boolean")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("execution.simulation_workers must be a positive integer")
    return enabled, workers


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_artifact(path, writer):
    """Publish a complete new artifact atomically without replacing a file."""
    with tempfile.NamedTemporaryFile(dir=Path(path).parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink()


def _write_pickle(path, value):
    _write_artifact(path, lambda handle: pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL))


def save_simulation_job(inputs, checkpoint_paths, checkpoint_metadata, upstream):
    """Preserve exact processed inputs; reference already saved fitted models."""
    inputs = dict(inputs)
    # AnnData views retain their whole parent in _adata_ref when pickled.
    # Materialize only the cohort, preserving its matrices/axes/metadata.
    for name in ("rna_adata", "atac_adata"):
        if getattr(inputs[name], "is_view", False):
            inputs[name] = inputs[name].copy()
    scope = inputs.get("suffix") or "global"
    safe_scope = "".join(char if char.isalnum() or char in "._-" else "_" for char in scope)
    if safe_scope != scope or len(safe_scope) > 100:
        safe_scope = safe_scope[:80] + "_" + hashlib.sha256(scope.encode()).hexdigest()[:12]
    root = Path(inputs["config"]["output"]["dir"]) / "simulation_inputs" / safe_scope
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="job_", dir=root)).resolve()
    references = {}
    for stage, path in checkpoint_paths.items():
        path = Path(path).resolve(strict=True)
        references[stage] = {"path": str(path), "sha256": _sha256(path)}
    context = dict(schema=SCHEMA, inputs=inputs,
                   checkpoint_metadata=checkpoint_metadata, upstream=upstream)
    _write_pickle(directory / "context.pkl", context)
    manifest = dict(schema=SCHEMA, scope=scope,
                    context="context.pkl", checkpoints=references)
    _write_artifact(directory / "manifest.json",
                    lambda handle: handle.write(json.dumps(manifest, indent=2).encode("utf-8")))
    logger.info("Saved simulation inputs for %s: %s", manifest["scope"], directory)
    return str(directory / "manifest.json")


def load_simulation_inputs(manifest_path):
    """Strict restoration: incompatible artifacts never trigger training."""
    from .run import (_validate_adaptive_checkpoint, _validate_step5_checkpoint,
                      _validate_step6_checkpoint)
    from .saturating_transfer import validate_registry
    from .atac_normalization import expected_registry_schema
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported simulation manifest schema")
    with open(manifest_path.parent / manifest["context"], "rb") as handle:
        context = pickle.load(handle)
    if context.get("schema") != SCHEMA:
        raise ValueError("Unsupported simulation context schema")
    inputs = context["inputs"]
    checkpoints = {}
    for stage in ("step4_granger", "step5_atac_to_rna", "step6_rna_to_atac"):
        reference = manifest["checkpoints"][stage]
        path = reference["path"]
        with open(path, "rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != reference["sha256"]:
                raise RuntimeError(f"Simulation checkpoint changed: {path}")
            handle.seek(0)
            checkpoint = pickle.load(handle)
        _validate_adaptive_checkpoint(checkpoint, context["checkpoint_metadata"],
                                      stage, context["upstream"].get(stage))
        checkpoints[stage] = checkpoint
    step5 = checkpoints["step5_atac_to_rna"]
    step6 = checkpoints["step6_rna_to_atac"]
    _validate_step5_checkpoint(step5, config=inputs["_cfg"])
    _validate_step6_checkpoint(step6["tf_weights"])
    validate_registry(step6["tf_model_registry"], expected_type="saturating_hybrid_tf_peak",
                      expected_schema=expected_registry_schema("saturating_hybrid_tf_peak", inputs["_cfg"]))
    inputs.update(granger_results=checkpoints["step4_granger"]["granger_results"],
                  causal_edges=step5["causal_edges"],
                  transfer_functions=step5["transfer_functions"],
                  transfer_models=step5["transfer_models"],
                  tf_weights=step6["tf_weights"], tf_model_registry=step6["tf_model_registry"])
    return inputs


def simulate_job(manifest_path):
    """Spawn entry point: load a completed model and execute simulation only."""
    from threadpoolctl import threadpool_limits
    from .run import _finish_modeling_for_subset
    with _worker_progress(manifest_path), threadpool_limits(limits=1):
        inputs = load_simulation_inputs(manifest_path)
        logger.info("Step 8 simulation started: %s (pid=%s)", inputs.get("suffix") or "global", os.getpid())
        result = _finish_modeling_for_subset(**inputs)
        output = Path(manifest_path).parent / "result.pkl"
        if _WORKER_PROGRESS is not None:
            _WORKER_PROGRESS.put(dict(kind="progress", job=str(manifest_path), phase="保存结果"))
        from .result_store import DiskResults
        stored = DiskResults.create(output.parent / "result_fields", result)
        del result, inputs
        _write_pickle(output, stored)
    return str(output)


@contextmanager
def _worker_environment():
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update({name: "1" for name in names})
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_simulation_jobs(manifests, workers=2):
    """Yield ordered disk results with the configured simulation concurrency."""
    _, workers = execution_options({"execution": {"simulation_workers": workers}})
    if not manifests:
        return
    logger.info("All cohort training complete; starting %s simulations with %s worker(s). "
                "Each worker loads one full cohort; memory depends on cohort size.",
                len(manifests), min(workers, len(manifests)))
    from .terminal_progress import SimulationProgress
    manifests = list(map(str, manifests))
    labels = [(path, json.loads(Path(path).read_text())["scope"]) for path in manifests]
    context = multiprocessing.get_context("spawn")
    events = context.Queue()
    try:
        with _worker_environment(), SimulationProgress(labels) as progress, ProcessPoolExecutor(
                max_workers=min(workers, len(manifests)), mp_context=context,
                initializer=_simulation_worker_init, initargs=(events,)) as executor:
            pending = {executor.submit(simulate_job, path): index for index, path in enumerate(manifests)}
            results, next_result = {}, 0
            while pending:
                ready, _ = wait(pending, timeout=.15, return_when=FIRST_COMPLETED)
                _drain_progress(events, progress)
                for future in ready:
                    index = pending.pop(future)
                    try:
                        results[index] = future.result()
                    except BaseException:
                        progress.finish(manifests[index], failed=True)
                        for remaining, other_index in list(pending.items()):
                            if remaining.cancel():
                                pending.pop(remaining)
                                progress.finish(manifests[other_index], failed=True)
                        # The pool waits for already running jobs on exit.
                        # Keep draining telemetry meanwhile so worker queue
                        # feeder threads cannot block shutdown on a full pipe.
                        while pending:
                            finished, _ = wait(pending, timeout=.15, return_when=FIRST_COMPLETED)
                            _drain_progress(events, progress)
                            for remaining in finished:
                                other_index = pending.pop(remaining)
                                progress.finish(manifests[other_index], failed=remaining.exception() is not None)
                        raise
                    progress.finish(manifests[index])
                while next_result in results:
                    with open(results.pop(next_result), "rb") as handle:
                        yield pickle.load(handle)
                    next_result += 1
            _drain_progress(events, progress)
    finally:
        events.close()
        events.join_thread()
