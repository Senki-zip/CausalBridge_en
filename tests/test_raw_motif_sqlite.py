"""SQLite migration, indexed queries, bounded writes and concurrency regressions."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing
import pickle
import sqlite3
import subprocess
import sys

import pytest

from CausalBridge import kinetics as k
from CausalBridge import _raw_motif_sqlite as sql
from test_raw_motif_memory import resources, record


def opened(path):
    motifs, descriptor = resources()
    return sql.Store(path, descriptor, k._raw_motif_validator(descriptor, motifs))


def process_write(path, start):
    with opened(path) as store:
        store.update(record(i) for i in range(start, start + 200))
    return True


def test_migration_exact_hash_readonly_and_warm_never_reads_legacy(tmp_path, monkeypatch):
    motifs, descriptor = resources()
    original = dict(record(i) for i in range(1000))
    original[record(2)[0]] = []
    path = tmp_path / "raw.pkl"
    encoded = pickle.dumps(dict(descriptor, records=original), protocol=4)
    path.write_bytes(encoded)
    before = sql.stamp(path)
    with opened(path) as store:
        assert dict(store.get()) == original
        assert store.db.execute("SELECT value FROM metadata WHERE name='legacy_sha256'").fetchone()[0] == hashlib.sha256(encoded).hexdigest()
    assert sql.stamp(path) == before and path.read_bytes() == encoded
    monkeypatch.setattr(sql, "read_records", lambda *a, **kw: pytest.fail("warm scanned legacy"))
    selected = [record(i)[0] for i in (999, 2, 1)]
    with opened(path) as store:
        assert list(store.get(selected)) == [(key, original[key]) for key in selected]
        queries = []
        store.db.set_trace_callback(queries.append)
        assert list(store.get([selected[0]]))
        assert not any("ORDER BY rowid" in query for query in queries)
        plan = store.db.execute("EXPLAIN QUERY PLAN WITH requested(p,c,s,e) AS (VALUES (?,?,?,?)) "
            "SELECT r.hits FROM requested q CROSS JOIN records r WHERE r.scan=? "
            "AND r.peak=q.p AND r.chrom=q.c AND r.start=q.s AND r.end=q.e",
            (*selected[0], descriptor["scan_key"])).fetchall()
        assert any("SEARCH r USING INDEX" in row[-1] and "end=?" in row[-1] for row in plan)


def test_bounded_generator_and_batch_transactions(tmp_path):
    with opened(tmp_path / "fresh.pkl") as store:
        queries = []
        store.db.set_trace_callback(queries.append)
        def pairs():
            for i in range(3000):
                assert len(store.pending) < 256
                yield record(i)
        store.update(pairs())
        assert len(store) == 3000
        assert len(store.pending) == 3000 % 256
        store.flush()
        assert sum(query == "BEGIN IMMEDIATE" for query in queries) == 12
        assert store.db.execute("SELECT count(*) FROM records").fetchone()[0] == 3000


def test_spawn_concurrent_migration_and_disjoint_union(tmp_path):
    _, descriptor = resources()
    path = tmp_path / "legacy.pkl"
    path.write_bytes(pickle.dumps(dict(descriptor, records=dict(record(i) for i in range(100))), protocol=4))
    before = path.read_bytes()
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(process_write, path, start) for start in (100, 300)]
        assert all(future.result(timeout=60) for future in futures)
    with opened(path) as store:
        assert dict(store.get()) == dict(record(i) for i in range(500))
    assert path.read_bytes() == before


def test_conflict_rolls_back_batch_and_interrupted_pending_not_published(tmp_path):
    path = tmp_path / "raw.pkl"
    with opened(path) as store:
        store.update([record(0)])
    conflict = record(0)
    conflict[1][0]["max_score"] = -123.
    with pytest.raises(ValueError, match="Conflicting"):
        with opened(path) as store:
            store.update([record(1), conflict])
    with pytest.raises(RuntimeError):
        with opened(path) as store:
            store.update([record(2)])
            raise RuntimeError("interrupted")
    with opened(path) as store:
        assert dict(store.get()) == dict([record(0)])


def test_interrupted_migration_never_installs_and_retries(tmp_path, monkeypatch):
    _, descriptor = resources()
    path = tmp_path / "raw.pkl"
    path.write_bytes(pickle.dumps(dict(descriptor, records=dict(record(i) for i in range(1000))), protocol=4))
    original = sql.read_records
    def interrupted(handle, consume, **kwargs):
        def callback(key, hits):
            consume(key, hits)
            if key == record(800)[0]:
                raise RuntimeError("interrupted migration")
        return original(handle, callback, **kwargs)
    monkeypatch.setattr(sql, "read_records", interrupted)
    with pytest.raises(RuntimeError):
        with opened(path): pass
    assert not path.with_suffix(".sqlite").exists() and not list(tmp_path.glob("*.tmp*"))
    monkeypatch.setattr(sql, "read_records", original)
    with opened(path) as store:
        assert len(dict(store.get())) == 1000


def test_changed_legacy_rejected_and_descriptor_mismatch(tmp_path, monkeypatch):
    _, descriptor = resources()
    path = tmp_path / "raw.pkl"
    path.write_bytes(pickle.dumps(dict(descriptor, records=dict([record(0)])), protocol=4))
    original = sql.read_records
    def changed(handle, consume, **kwargs):
        result = original(handle, consume, **kwargs)
        with path.open("ab") as output: output.write(b"changed")
        return result
    monkeypatch.setattr(sql, "read_records", changed)
    with pytest.raises(ValueError, match="changed during migration"):
        with opened(path): pass
    assert not path.with_suffix(".sqlite").exists()
    monkeypatch.setattr(sql, "read_records", original)
    with opened(path): pass
    with sqlite3.connect(path.with_suffix(".sqlite")) as db:
        db.execute("UPDATE metadata SET value='{}' WHERE name='descriptor'")
    with pytest.raises(ValueError, match="schema/identity mismatch"):
        with opened(path): pass


def test_fresh_migration_and_warm_under_128_mib_address_limit(tmp_path):
    _, descriptor = resources()
    path = tmp_path / "legacy.pkl"
    path.write_bytes(pickle.dumps(dict(descriptor, records=dict(record(i) for i in range(20000))), protocol=4))
    script = r'''
import resource, sys, types
resource.setrlimit(resource.RLIMIT_AS, (128*1024**2, 128*1024**2))
package = types.ModuleType('CausalBridge')
package.__path__ = [sys.argv[1]]
sys.modules['CausalBridge'] = package
from CausalBridge._raw_motif_sqlite import Store
from pathlib import Path
descriptor = {'scan_key':'test','identity':{'threshold':1.0}}
def validate(key,hits):
    assert isinstance(key,tuple) and len(key)==4 and len(hits)==12
path = Path(sys.argv[2])
with Store(path,descriptor,validate) as store:
    assert len(list(store.get([('chr1:19999-20009','chr1',19999,20009)])))==1
with Store(path,descriptor,validate) as store:
    assert len(list(store.get([('chr1:0-10','chr1',0,10)])))==1
fresh = path.with_name('fresh.pkl')
with Store(fresh,descriptor,validate) as store:
    def records():
        for i in range(20000):
            yield (str(i),'chr1',i,i+10), [dict(motif_id='m'+str(j),p_value=0.1,max_score=float(i+j)) for j in range(12)]
    store.update(records())
    assert len(store)==20000
print('128 MiB limit: legacy migration, warm subset and fresh streaming writes passed')
'''
    result = subprocess.run([sys.executable, "-c", script, str(k.Path(k.__file__).parent), str(path)],
                            check=True, capture_output=True, text=True)
    assert "passed" in result.stdout
