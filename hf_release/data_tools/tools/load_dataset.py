"""Resolve portable release paths for the Mira-Scene checkout loaders."""
import fcntl,gzip,hashlib,importlib,json,os,sys
from pathlib import Path
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
REPO_ROOT = Path(__file__).resolve().parents[3]
LOADER_ROOT = REPO_ROOT / 'UniDataset' / 'src'
if not (LOADER_ROOT / 'UniDataset').is_dir():
 raise RuntimeError('Keep hf_release/data_tools inside the Mira-Scene checkout; UniDataset/src is missing')
sys.path.insert(0, str(LOADER_ROOT))

def load_dataset(root,subset,cache_dir=None):
 root=Path(root).resolve()
 if subset not in ('objaverse_outpaint','3dfront'):raise ValueError('Unknown subset: '+subset)
 cache=Path(cache_dir).resolve() if cache_dir else root/'.runtime'/subset
 cache.mkdir(parents=True,exist_ok=True)
 config_path=root/'configs/datasets.json'
 if not config_path.exists():config_path=Path(__file__).resolve().parent.parent/'configs/datasets.json'
 cfg=json.loads(config_path.read_text())[subset];params=dict(cfg['params']);params['repeat']=1
 if subset=='objaverse_outpaint':
  source=root/subset/'metadata.jsonl.gz'
  digest=hashlib.sha256(source.read_bytes()+str(root).encode()).hexdigest()
  resolved=cache/('resolved_summary_'+digest+'.json')
  with (cache/'summary.lock').open('w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   if not resolved.exists():
    temp=resolved.with_suffix('.partial')
    with gzip.open(source,'rt') as inp,temp.open('w') as out:
     out.write('{"entries":[');first=True
     for line in inp:
      e=json.loads(line);mesh=Path(e['mesh_path'])
      if mesh.is_absolute() or '..' in mesh.parts:raise ValueError('Unsafe mesh path')
      e['mesh_path']=str(root/mesh)
      if not first:out.write(',')
      json.dump(e,out);first=False
     out.write(']}')
    temp.replace(resolved)
  params.update(summary_json=str(resolved),valid_scenes_dir=str(root/subset/'valid_scenes'))
 else:
  params.update({k:str(root/subset/v) for k,v in [('renderings_root','renderings'),('view_samples_dir','view_samples'),('poses_dir','poses'),('model_data_dir','models'),('preprocess_json_path','preprocess_train.json')]})
  params['error_log_path']=str(cache/'errors.log')
 params['voxel_cache_dir']=str(cache/'voxels')
 module,cls=cfg['target'].rsplit('.',1)
 return getattr(importlib.import_module(module),cls)(**params)
