"""Per-field result storage: explicit disk ownership without a RAM object cache."""
from collections.abc import MutableMapping, Mapping
from pathlib import Path
import json
import os
import pickle
import tempfile


class DiskResults(MutableMapping):
    """Dictionary-like results; each access loads only the requested field.

    Values are snapshots. To change a loaded value, assign it back explicitly.
    Stores are durable run artifacts, not fit checkpoints or automatic resumes.
    """
    def __init__(self, directory):
        self.directory = str(Path(directory).resolve())
        self._reload()

    def _reload(self):
        self._fields = json.loads((Path(self.directory) / 'index.json').read_text())

    @classmethod
    def create(cls, parent, values=None):
        parent = Path(parent)
        parent.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix='result_', dir=parent))
        (directory / 'index.json').write_text('{}')
        result = cls(directory)
        if values is not None:
            result.update(values)
        return result

    def __getitem__(self, key):
        with (Path(self.directory) / self._fields[key]).open('rb') as handle:
            return pickle.load(handle)

    def __setitem__(self, key, value):
        # New immutable field file, then atomic index publication. Old values
        # remain untouched until the replacement is completely persisted.
        previous = self._fields.get(key)
        filename = None
        try:
            with tempfile.NamedTemporaryFile(prefix='field_', suffix='.pkl', dir=self.directory, delete=False) as handle:
                filename = Path(handle.name).name
                pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            self._fields[key] = filename
            self._publish()
        except BaseException:
            if previous is None:
                self._fields.pop(key, None)
            else:
                self._fields[key] = previous
            if filename is not None:
                (Path(self.directory) / filename).unlink(missing_ok=True)
            raise
        if previous is not None:
            (Path(self.directory) / previous).unlink(missing_ok=True)

    def _publish(self):
        with tempfile.NamedTemporaryFile(mode='w', prefix='index_', dir=self.directory, delete=False) as handle:
            json.dump(self._fields, handle)
            temporary = Path(handle.name)
        try:
            temporary.replace(Path(self.directory) / 'index.json')
        finally:
            temporary.unlink(missing_ok=True)

    def __delitem__(self, key):
        filename = self._fields.pop(key)
        self._publish()
        (Path(self.directory) / filename).unlink()

    def __iter__(self):
        return iter(self._fields)

    def __len__(self):
        return len(self._fields)

    def __contains__(self, key):
        return key in self._fields

    def __getstate__(self):
        return {'directory': self.directory}

    def __setstate__(self, state):
        self.directory = state['directory']
        self._reload()


class ExportResults(Mapping):
    """Filter and decorate one field at a time, without eager dict copies."""
    def __init__(self, source, excluded=(), presenter=None):
        self.source = source
        self.excluded = frozenset(excluded)
        self.presenter = presenter

    def __iter__(self):
        return (key for key in self.source if key not in self.excluded)

    def __len__(self):
        return sum(1 for _ in self)

    def __getitem__(self, key):
        if key in self.excluded:
            raise KeyError(key)
        value = self.source[key]
        return self.presenter(key, value) if self.presenter else value

    def __contains__(self, key):
        return key not in self.excluded and key in self.source
