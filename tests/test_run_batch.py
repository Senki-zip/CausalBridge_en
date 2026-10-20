"""Batch ordering, isolated configuration, failure behavior and dry-run safety."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml

spec = importlib.util.spec_from_file_location('batch_runner', Path(__file__).resolve().parents[1] / 'run_batch.py')
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def fixture(tmp_path):
    (tmp_path/'rna').touch(); (tmp_path/'atac').touch()
    base = dict(input=dict(rna_h5ad='rna',atac_h5ad='atac',annotation_gtf='genes.gtf'),
                output=dict(dir='old_output',cache_dir='old_cache',checkpoint_dir='old_checkpoints'),
                perturbation=dict(ko_strength=1.,joint_knockout=False), execution=dict(simulation_workers=4))
    (tmp_path/'base.yaml').write_text(yaml.safe_dump(base))
    manifest=dict(base_config='base.yaml',defaults=dict(knockout_genes=['Zic2'],joint_knockout=True),tasks=[
        dict(name=name,output=dict(dir=name,cache_dir='shared_cache')) for name in ['one','two','three']])
    path=tmp_path/'tasks.yaml';path.write_text(yaml.safe_dump(manifest,sort_keys=False))
    return path,manifest


def test_merge_order_paths_and_targets(tmp_path):
    path,manifest=fixture(tmp_path)
    manifest['tasks'][1].update(knockout_genes=[dict(gene='Runx1',type='TF'),dict(gene='Other',type='target')],
                               overrides=dict(execution=dict(simulation_workers=2)))
    path.write_text(yaml.safe_dump(manifest))
    plans=batch.prepare_tasks(path,['two','one'])
    assert [p['name'] for p in plans]==['two','one']
    cfg=plans[0]['config']
    assert cfg['input']['rna_h5ad']==str(tmp_path/'rna')
    assert cfg['input']['annotation_gtf']==str(tmp_path/'genes.gtf')
    assert cfg['output']['checkpoint_dir'] is None
    assert cfg['execution']['simulation_workers']==2
    assert cfg['perturbation']['joint_knockout'] is True
    assert plans[0]['targets']=='Runx1\tTF\nOther\tTARGET\n'
    assert yaml.safe_load((tmp_path/'base.yaml').read_text())['output']['checkpoint_dir']=='old_checkpoints'


def test_dry_run_creates_no_outputs(tmp_path):
    path,_=fixture(tmp_path)
    assert batch.main([str(path),'--dry-run'])==0
    assert not (tmp_path/'one').exists()


@pytest.mark.parametrize('continued,expected',[(False,['one','two']),(True,['one','two','three'])])
def test_real_subprocess_serial_order_and_failures(tmp_path,continued,expected,monkeypatch):
    path,_=fixture(tmp_path)
    runner=tmp_path/'fake_runner.py'
    runner.write_text('''
import sys,json
from pathlib import Path
import yaml
cfg=yaml.safe_load(Path(sys.argv[1]).read_text())
name=Path(cfg['output']['dir']).name
targets=Path(cfg['input']['target_genes']).read_text()
assert targets=='Zic2\\tTF\\n'
assert cfg['output']['checkpoint_dir'] is None
with open(Path(cfg['output']['dir']).parent/'order.txt','a') as f:f.write(name+'\\n')
print('task '+name)
sys.exit(7 if name=='two' else 0)
''')
    # Use the real Python process boundary, with a tiny stand-in pipeline.
    original=batch.subprocess.Popen
    def launch(command,**kwargs):
        return original([sys.executable,str(runner),command[-1]],**kwargs)
    monkeypatch.setattr(batch.subprocess,'Popen',launch)
    assert batch.main([str(path)]+(['--continue-on-error'] if continued else []))==1
    assert (tmp_path/'order.txt').read_text().splitlines()==expected
    for name in expected:
        folder=next((tmp_path/name/'batch_runs').glob('*/'+name))
        status=json.loads((folder/'status.json').read_text())
        assert status['returncode']==(7 if name=='two' else 0)
        assert (folder/'pipeline.log').read_text().strip()=='task '+name


def test_reject_output_collision_and_unknown_order(tmp_path):
    path,manifest=fixture(tmp_path)
    with pytest.raises(ValueError,match='unknown'):
        batch.prepare_tasks(path,['missing'])
    manifest['tasks'][1]['output']['dir']='one'
    path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(ValueError,match='distinct'):
        batch.prepare_tasks(path)
