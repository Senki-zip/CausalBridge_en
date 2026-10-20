"""Indexed raw motif storage. Legacy pickle files are migration inputs only."""
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import tempfile

from ._raw_motif_io import read_records

logger = logging.getLogger(__name__)
VERSION = 1


class _FingerprintReader:
    """Hash the first existing pickle pass, without another file-sized read."""
    def __init__(self, handle):
        self.handle = handle
        self.hash = hashlib.sha256()
        self.digest = None

    def read(self, size=-1):
        data = self.handle.read(size)
        if self.digest is None:
            self.hash.update(data)
        return data

    def readline(self, size=-1):
        data = self.handle.readline(size)
        if self.digest is None:
            self.hash.update(data)
        return data

    def tell(self):
        return self.handle.tell()

    def seek(self, offset, whence=0):
        if self.digest is None:
            while True:
                data = self.handle.read(1024*1024)
                if not data:
                    break
                self.hash.update(data)
            self.digest = self.hash.hexdigest()
        return self.handle.seek(offset, whence)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stamp(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


class Store:
    """One reference descriptor per database, one ordered JSON payload per peak."""
    def __init__(self, legacy, descriptor, validate):
        self.legacy = Path(legacy)
        self.path = self.legacy.with_suffix(".sqlite")
        self.descriptor = descriptor
        self.validate = validate
        self.identity = canonical({k: descriptor[k] for k in ("scan_key", "identity")})
        self.count = 0
        self.db = None
        self.pending = []
        self.pending_bytes = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".sqlite.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not self.path.exists():
                self._install()
        self.db = self._connect(self.path)
        try:
            metadata = dict(self.db.execute("SELECT name,value FROM metadata"))
            if (metadata.get("version") != str(VERSION) or
                    metadata.get("descriptor") != self.identity or metadata.get("ready") != "1"):
                raise ValueError("SQLite raw motif cache schema/identity mismatch")
            self.db.execute("PRAGMA journal_mode=WAL")
        except BaseException:
            self.db.close()
            raise
        return self

    @staticmethod
    def _connect(path):
        db = sqlite3.connect(str(path), timeout=120)
        db.execute("PRAGMA cache_size=-4096")
        db.execute("PRAGMA busy_timeout=120000")
        return db

    def __exit__(self, *args):
        if self.db is not None:
            try:
                if args[0] is None:
                    self.flush()
            finally:
                self.db.close()

    def _install(self):
        fd, name = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent)
        os.close(fd)
        temporary = Path(name)
        db = self._connect(temporary)
        try:
            db.execute("CREATE TABLE metadata(name TEXT PRIMARY KEY,value TEXT NOT NULL)")
            db.execute("CREATE TABLE records(scan TEXT NOT NULL,peak TEXT NOT NULL,chrom TEXT NOT NULL,"
                       "start INTEGER NOT NULL,end INTEGER NOT NULL,hits TEXT NOT NULL,"
                       "PRIMARY KEY(scan,peak,chrom,start,end))")
            db.execute("INSERT INTO metadata VALUES('version',?)", (str(VERSION),))
            db.execute("INSERT INTO metadata VALUES('descriptor',?)", (self.identity,))
            db.commit()
            previous = self.db
            self.db = db
            try:
                if self.legacy.exists():
                    before = stamp(self.legacy)
                    logger.info("One-time raw motif SQLite migration: %s (%.2f GiB; two streaming passes)",
                                self.legacy, before[2] / 1024**3)
                    batch = []
                    batch_bytes = 0
                    def consume(key, hits):
                        nonlocal batch_bytes
                        encoded = self._encode(key, hits)
                        batch.append(encoded)
                        batch_bytes += len(encoded[-1].encode("utf-8"))
                        if len(batch) >= 256 or batch_bytes >= 4*1024**2:
                            self._write(batch)
                            batch.clear()
                            batch_bytes = 0
                    with self.legacy.open("rb") as handle:
                        fingerprint = _FingerprintReader(handle)
                        payload, _, _ = read_records(fingerprint, consume, progress=lambda phase, pos:
                            logger.info("Raw motif migration %s: %.2f/%.2f GiB", phase,
                                        pos / 1024**3, before[2] / 1024**3))
                    if (payload.get("schema") != "peak-motif-raw-v1" or
                            payload.get("scan_key") != self.descriptor["scan_key"] or
                            payload.get("identity") != self.descriptor["identity"]):
                        raise ValueError("Legacy raw motif cache schema/identity mismatch")
                    self._write(batch)
                    if stamp(self.legacy) != before:
                        raise ValueError("Legacy raw motif cache changed during migration; retry")
                    db.execute("INSERT INTO metadata VALUES('legacy_stat',?)", (canonical(before),))
                    db.execute("INSERT INTO metadata VALUES('legacy_sha256',?)", (fingerprint.digest,))
                db.execute("INSERT INTO metadata VALUES('ready','1')")
                db.commit()
            finally:
                self.db = previous
            db.close()
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            logger.info("Raw motif SQLite cache ready: %s", self.path)
        finally:
            db.close()
            temporary.unlink(missing_ok=True)

    def _encode(self, key, hits):
        self.validate(key, hits)
        return (self.descriptor["scan_key"], *key, json.dumps(hits, separators=(",", ":"), allow_nan=False))

    def _write(self, batch):
        if not batch:
            return
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            for row in batch:
                old = self.db.execute("SELECT hits FROM records WHERE scan=? AND peak=? AND chrom=? AND start=? AND end=?", row[:5]).fetchone()
                if old is not None and json.loads(old[0]) != json.loads(row[5]):
                    raise ValueError("Conflicting raw motif hits for identical reference and peak")
                self.db.execute("INSERT OR IGNORE INTO records VALUES(?,?,?,?,?,?)", row)

    def update(self, pairs):
        for key, hits in pairs:
            row = self._encode(key, hits)
            self.pending.append(row)
            self.pending_bytes += len(row[-1].encode("utf-8"))
            self.count += 1
            if len(self.pending) >= 256 or self.pending_bytes >= 4*1024**2:
                self.flush()

    def flush(self):
        self._write(self.pending)
        self.pending.clear()
        self.pending_bytes = 0

    def __len__(self):
        return self.count

    def get(self, requested=None):
        self.flush()
        if requested is None:
            rows = self.db.execute("SELECT peak,chrom,start,end,hits FROM records WHERE scan=? ORDER BY rowid", (self.descriptor["scan_key"],))
            for *key, text in rows:
                key, hits = tuple(key), json.loads(text)
                self.validate(key, hits)
                yield key, hits
            return
        iterator = iter(requested)
        from itertools import islice
        while True:
            keys = list(islice(iterator, 150))
            if not keys:
                break
            placeholders = ",".join("(?,?,?,?)" for _ in keys)
            rows = self.db.execute("WITH requested(p,c,s,e) AS (VALUES " + placeholders + ") "
                "SELECT r.peak,r.chrom,r.start,r.end,r.hits FROM requested q CROSS JOIN records r "
                "WHERE r.scan=? AND r.peak=q.p AND r.chrom=q.c AND r.start=q.s AND r.end=q.e",
                [*[x for key in keys for x in key], self.descriptor["scan_key"]])
            found = {tuple(row[:4]): row[4] for row in rows}
            for key in keys:
                if key in found:
                    hits = json.loads(found[key])
                    self.validate(key, hits)
                    yield key, hits
