#!/usr/bin/env python3
"""Run an ordered YAML task list using isolated pipeline subprocesses."""
import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml

PROJECT = Path(__file__).resolve().parent


def merge(base, updates):
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def path_value(value, directory):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return str((directory / path).resolve() if not path.is_absolute() else path.resolve())


def resolve_config(config, directory):
    config = copy.deepcopy(config)
    for section, fields in {'input': ('rna_h5ad', 'atac_h5ad', 'annotation_gtf',
                                      'genome_fasta', 'target_genes'),
                            'output': ('dir', 'cache_dir', 'checkpoint_dir')}.items():
        for key in fields:
            value = config.get(section, {}).get(key)
            if value is not None:
                config[section][key] = path_value(value, directory)
    return config


def read_yaml(path):
    result = yaml.safe_load(path.read_text())
    if not isinstance(result, dict):
        raise ValueError(f'{path}: expected a YAML mapping')
    return result


def targets_text(values):
    if not isinstance(values, list) or not values:
        raise ValueError('Each task needs a nonempty knockout_genes list')
    lines, seen = [], set()
    for value in values:
        if isinstance(value, str):
            gene, kind = value, 'TF'
        elif isinstance(value, dict) and set(value).issubset({'gene', 'type'}):
            gene, kind = value.get('gene'), value.get('type', 'TF')
        else:
            raise ValueError('knockout_genes entries must be gene strings or {gene: ..., type: TF/TARGET}')
        if not isinstance(gene, str) or not gene or any(c.isspace() for c in gene) or gene.startswith('#') or gene.upper().startswith('ALL'):
            raise ValueError(f'Invalid explicit knockout gene: {gene!r}')
        if not isinstance(kind, str) or kind.upper() not in ('TF', 'TARGET'):
            raise ValueError(f'Invalid gene type: {kind!r}')
        if gene in seen:
            raise ValueError(f'Duplicate knockout gene: {gene}')
        seen.add(gene)
        lines.append(f'{gene}\t{kind.upper()}\n')
    return ''.join(lines)


def prepare_tasks(manifest_path, order=None):
    spec = read_yaml(manifest_path)
    base_path = Path(path_value(spec.get('base_config', 'config.yaml'), manifest_path.parent))
    base = resolve_config(read_yaml(base_path), base_path.parent)
    defaults = spec.get('defaults', {})
    tasks = spec.get('tasks')
    if not isinstance(defaults, dict) or not isinstance(tasks, list) or not tasks:
        raise ValueError('defaults must be a mapping; tasks must be a nonempty list')
    names, plans = set(), []
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError('Each task must be a mapping')
        name = task.get('name')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in names or name in ('.', '..'):
            raise ValueError(f'Task names must be unique safe directory names: {name!r}')
        names.add(name)
        options = merge(defaults, task)
        allowed = {'name', 'base_config', 'input', 'output', 'knockout_genes', 'joint_knockout', 'overrides'}
        if set(options) - allowed:
            raise ValueError(f'{name}: unknown task fields: {sorted(set(options)-allowed)}')
        selected_base = base
        if 'base_config' in options:
            path = Path(path_value(options['base_config'], manifest_path.parent))
            selected_base = resolve_config(read_yaml(path), path.parent)
        overrides = options.get('overrides', {})
        if not isinstance(overrides, dict):
            raise ValueError(f'{name}: overrides must be a mapping')
        update = merge(overrides, {k:options[k] for k in ('input', 'output') if k in options})
        config = merge(selected_base, resolve_config(update, manifest_path.parent))
        if 'joint_knockout' in options:
            if not isinstance(options['joint_knockout'], bool):
                raise ValueError(f'{name}: joint_knockout must be true or false')
            config.setdefault('perturbation', {})['joint_knockout'] = options['joint_knockout']
        config.setdefault('perturbation', {})['knockout_type'] = 'gene'
        text = targets_text(options.get('knockout_genes'))
        for key in ('rna_h5ad', 'atac_h5ad'):
            path = config.get('input', {}).get(key)
            if not path or not Path(path).is_file():
                raise ValueError(f'{name}: input.{key} is not a file: {path}')
        # Require deliberate destinations instead of silently inheriting a
        # different experiment's output/checkpoints from the base config.
        for key in ('dir', 'cache_dir'):
            if not update.get('output', {}).get(key):
                raise ValueError(f'{name}: specify output.{key} in task/defaults')
        if 'checkpoint_dir' not in update.get('output', {}):
            config['output']['checkpoint_dir'] = None
        plans.append(dict(name=name, config=config, targets=text))
    if order:
        if len(order) != len(set(order)) or set(order)-names:
            raise ValueError('--order contains duplicate or unknown names')
        lookup = {p['name']:p for p in plans}
        plans = [lookup[name] for name in order]
    outputs = [p['config']['output']['dir'] for p in plans]
    if len(outputs) != len(set(outputs)):
        raise ValueError('Tasks must have distinct output.dir values; cache_dir may be shared')
    return plans


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--order', nargs='+', help='Run only these task names, in this order')
    parser.add_argument('--python', default=sys.executable, help='Pipeline Python interpreter')
    parser.add_argument('--dry-run', action='store_true', help='Validate task plan without running or writing results')
    parser.add_argument('--continue-on-error', action='store_true')
    args = parser.parse_args(argv)
    try:
        plans = prepare_tasks(args.manifest.resolve(), args.order)
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    for i, plan in enumerate(plans, 1):
        cfg = plan['config']
        print(f"[{i}/{len(plans)}] {plan['name']} KO={plan['targets'].strip().replace(chr(10), ', ')}\n"
              f"  RNA={cfg['input']['rna_h5ad']}\n  ATAC={cfg['input']['atac_h5ad']}\n"
              f"  output={cfg['output']['dir']}\n  cache={cfg['output']['cache_dir']}", flush=True)
    if args.dry_run:
        return 0
    batch_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    failed = False
    for plan in plans:
        cfg = copy.deepcopy(plan['config'])
        output = Path(cfg['output']['dir'])
        folder = output / 'batch_runs' / batch_id / plan['name']
        folder.mkdir(parents=True, exist_ok=False)
        target_file = folder / 'target_genes.txt'
        target_file.write_text(plan['targets'])
        cfg['input']['target_genes'] = str(target_file)
        config_file = folder / 'config.yaml'
        config_file.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        command = [args.python, '-u', str(PROJECT / 'run_example.py'), str(config_file)]
        record = dict(task=plan['name'], command=command, started_at=datetime.now().astimezone().isoformat(), status='running')
        status_file = folder / 'status.json'
        status_file.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        print(f"Starting {plan['name']}; log: {folder / 'pipeline.log'}", flush=True)
        with (folder / 'pipeline.log').open('ab', buffering=0) as log:
            process = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            record['pid'] = process.pid
            status_file.write_text(json.dumps(record, ensure_ascii=False, indent=2))
            try:
                code = process.wait()
            except KeyboardInterrupt:
                if process.poll() is None:
                    process.terminate()
                code = process.wait()
                record.update(status='interrupted', returncode=code, finished_at=datetime.now().astimezone().isoformat())
                status_file.write_text(json.dumps(record, ensure_ascii=False, indent=2))
                return 130
        record.update(status='success' if code == 0 else 'failed', returncode=code,
                      finished_at=datetime.now().astimezone().isoformat())
        status_file.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        print(f"{plan['name']}: {record['status']} (exit={code})", flush=True)
        if code != 0:
            failed = True
            if not args.continue_on_error:
                break
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
