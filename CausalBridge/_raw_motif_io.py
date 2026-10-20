"""Stream the primitive raw-motif pickle schema without retaining its hit graph.

Old caches are ordinary protocol-4 dictionaries. Their millions of hit dicts
are memoized even though they are never referenced again. A cheap opcode pass
identifies actual references; the second pass retains only those memo entries
and consumes records in pickle's bounded SETITEMS batches. New writes use no
memo for individual records. Files remain readable with ordinary pickle.load.
"""
import pickle
import pickletools
import struct
import sqlite3
import tempfile
import time


class DiskRecords:
    """Ordered temporary scan results; only a worker batch stays in RAM."""
    def __enter__(self):
        self.directory = tempfile.TemporaryDirectory(prefix="causalbridge-motif-")
        self.db = sqlite3.connect(self.directory.name + "/records.sqlite")
        self.db.execute("PRAGMA cache_size=-2048")
        self.db.execute("PRAGMA journal_mode=OFF")
        self.db.execute("CREATE TABLE records (key BLOB PRIMARY KEY, hits BLOB)")
        self.count = 0
        return self

    def __exit__(self, *exc):
        self.db.close()
        self.directory.cleanup()

    def __len__(self):
        return self.count

    def update(self, batch):
        for key, hits in batch:
            self.db.execute("INSERT INTO records VALUES (?, ?)",
                            (pickle.dumps(key, protocol=4), pickle.dumps(hits, protocol=4)))
            self.count += 1
        self.db.commit()

    def items(self):
        for key, hits in self.db.execute("SELECT key, hits FROM records ORDER BY rowid"):
            yield pickle.loads(key), pickle.loads(hits)


class Records(dict):
    valid = False
    records_memo = None
    stop_offset = None


class _Memo(dict):
    def __init__(self, references):
        super().__init__()
        self.references = references
        self.count = 0

    def __len__(self):
        return self.count

    def __setitem__(self, index, value):
        self.count = max(self.count, index + 1)
        if index in self.references:
            super().__setitem__(index, value)


class _Sink(dict):
    def __init__(self, consume):
        self.consume = consume

    def __setitem__(self, key, value):
        self.consume(key, value)


class _Reader(pickle._Unpickler):
    dispatch = pickle._Unpickler.dispatch.copy()

    def __init__(self, handle, references, consume):
        super().__init__(handle)
        self.memo = _Memo(references)
        self.consume = consume
        self.root = None
        self.records = None
        self.records_memo = None

    def load_empty_dictionary(self):
        is_records = (self.stack and self.stack[-1] == "records" and
            ((len(self.metastack) == 1 and self.metastack[0][-1] is self.root) or
             (not self.metastack and len(self.stack) > 1 and self.stack[-2] is self.root)))
        if is_records:
            if self.records is not None:
                raise ValueError("duplicate raw motif records")
            value = self.records = _Sink(self.consume)
            # The writer always memoizes this dictionary immediately. Explicit
            # BINPUT and MEMOIZE both use the next sequential index.
            self.records_memo = len(self.memo)
        else:
            value = {}
            if self.root is None:
                self.root = value
        self.append(value)

    dispatch[pickle.EMPTY_DICT[0]] = load_empty_dictionary

    def find_class(self, module, name):
        raise ValueError("raw motif caches contain primitives only")


def read_records(handle, consume, *, progress=None):
    """Validate/read a cache, feeding one complete peak at a time to consume.

    The reference set contains shared strings and metadata in caches produced
    by pickle.dump; unreferenced mutable hit objects never enter the memo.
    This also preserves genuine aliases in historical primitive pickles.
    """
    references = set()
    stop_offset = None
    last_report = time.monotonic()
    # Do not permit constructors, extensions, persistent IDs, or out-of-band
    # buffers in this primitive-only format.
    forbidden = {"GLOBAL", "STACK_GLOBAL", "REDUCE", "BUILD", "INST", "OBJ",
                 "NEWOBJ", "NEWOBJ_EX", "EXT1", "EXT2", "EXT4", "PERSID",
                 "BINPERSID", "NEXT_BUFFER", "READONLY_BUFFER"}
    for opcode, argument, position in pickletools.genops(handle):
        if opcode.name in forbidden:
            raise ValueError("raw motif caches contain primitives only")
        if opcode.name in {"GET", "BINGET", "LONG_BINGET"}:
            references.add(argument)
        if opcode.name == "STOP":
            stop_offset = position
        if progress is not None and opcode.name == "FRAME" and time.monotonic() - last_report >= 30:
            progress("index", position)
            last_report = time.monotonic()
    handle.seek(0)
    def consume_with_progress(key, hits):
        nonlocal last_report
        consume(key, hits)
        if progress is not None and time.monotonic() - last_report >= 30:
            progress("validate", handle.tell())
            last_report = time.monotonic()
    reader = _Reader(handle, references, consume_with_progress)
    payload = reader.load()
    if (not isinstance(payload, dict) or reader.records is None or
            payload.get("records") is not reader.records):
        raise ValueError("invalid raw motif record dictionary")
    if reader.records_memo not in reader.memo.references:
        # Ordinary old caches do not GET records; its inferred index still
        # needs to be a real memo entry before an incremental write can use it.
        # Check this in a header-only opcode pass, with no record allocations.
        handle.seek(0)
        index = 0
        found = False
        for opcode, argument, _ in pickletools.genops(handle):
            if opcode.name in {"MEMOIZE", "BINPUT", "LONG_BINPUT", "PUT"}:
                assigned = index if opcode.name == "MEMOIZE" else argument
                index = max(index, assigned + 1)
                if assigned == reader.records_memo:
                    found = True
                    break
        if not found:
            raise ValueError("raw motif records dictionary was not memoized")
    return payload, reader.records_memo, stop_offset


def _fragment(handle, value):
    """Emit a standalone primitive pickle expression without its final STOP."""
    writer = pickle.Pickler(handle, protocol=4)
    # The schema is an acyclic tree of primitives, so memoizing every hit has
    # no semantic purpose and would otherwise grow with the full cache size.
    writer.fast = True
    writer.dump(value)
    handle.seek(-1, 1)
    handle.truncate()


def write_records(handle, metadata, additions, *, previous=None, records_memo=None,
                  stop_offset=None):
    """Write the same dictionary schema using constant serialization workspace.

    For an existing validated cache, copy its pickle up to STOP, reference its
    records dictionary, and apply SETITEMs. This preserves all unrelated hits
    and aliases without reloading or repickling the existing graph.
    """
    if previous is None:
        handle.write(b"\x80\x04}q\x00(")  # root dict, memo 0, MARK
        for key, value in metadata.items():
            _fragment(handle, key)
            _fragment(handle, value)
        _fragment(handle, "records")
        handle.write(b"}q\x01")  # records dict, memo 1
    else:
        with previous.open("rb") as source:
            remaining = stop_offset
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("raw motif cache changed while copying")
                handle.write(chunk)
                remaining -= len(chunk)
        handle.write(b"h" + bytes([records_memo]) if records_memo < 256 else
                     b"j" + struct.pack("<I", records_memo))
    for key, hits in additions.items():
        _fragment(handle, key)
        _fragment(handle, hits)
        handle.write(b"s")  # SETITEM on records
    handle.write(b"u." if previous is None else b"0.")  # SETITEMS root / POP records; STOP
