#!/usr/bin/env python3
"""Extract a small, dependency-complete training subset from verified archives."""
import argparse,gzip,hashlib,json,shutil,tarfile,time
from pathlib import Path,PurePosixPath

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def safe(root,name):
 p=PurePosixPath(name)
 if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe path')
 dest=root/str(p)
 if not dest.resolve().is_relative_to(root.resolve()):raise ValueError('Escaping path')
 return dest

def main():
 a=argparse.ArgumentParser();a.add_argument('package',type=Path);a.add_argument('destination',type=Path);a.add_argument('--views-per-subset',type=int,default=8);args=a.parse_args()
 source=args.package.resolve();dest=args.destination.resolve();n=args.views_per_subset
 if n<1:raise ValueError('views-per-subset must be positive')
 if json.loads((source/'manifests/integrity_summary.json').read_text())['status']!='PASS':raise ValueError('Package verification incomplete')
 if dest.exists() and any(dest.iterdir()):raise ValueError('Use an empty destination')
 dest.mkdir(parents=True,exist_ok=True);start=time.time()
 obj=[]
 with gzip.open(source/'objaverse_outpaint/metadata.jsonl.gz','rt') as f:
  for line in f:
   obj.append(json.loads(line))
   if len(obj)==n:break
 front=json.loads((source/'3dfront/preprocess_train.json').read_text());views=front['results'][:n]
 sm={s['scene_id']:s for s in front['scene_results']};scene_ids={v['scene_id'] for v in views}
 required=set(e['mesh_path'] for e in obj)
 prefixes={f"objaverse_outpaint/valid_scenes/{e['obj_id']}_{str(e['view']).zfill(3)}/" for e in obj}
 for v in views:
  sid=v['scene_id'];uid=v.get('id') or v['unique_id']
  required.update([f"3dfront/renderings/{sid}/{v['view_relpath']}",f"3dfront/view_samples/{uid}/depth.npy",f"3dfront/poses/{sid}_scene_state.json"])
  for i in v['valid_object_indices']:
   k=sm[sid]['obj_keys'][i].split('|');mid=k[1] or k[0];required.add(f'3dfront/models/{mid}/raw_model.obj')
 selected={};shards=[]
 for rec in json.loads((source/'manifests/shards.json').read_text()):
  manifest=safe(source,rec['members'])
  if sha(manifest)!=rec['members_sha256']:raise ValueError('Member manifest corrupted')
  wanted={}
  with gzip.open(manifest,'rt') as f:
   for line in f:
    row=json.loads(line);name=row['archive'];prefix=name.rsplit('/',1)[0]+'/'
    if name in required or prefix in prefixes:wanted[name]=row;selected[name]=row
  if wanted:shards.append((rec,wanted))
 missing=required-set(selected)
 if missing:raise ValueError('Missing dependencies: '+str(sorted(missing)))
 for prefix in prefixes:
  names={x[len(prefix):] for x in selected if x.startswith(prefix)}
  if not {'scene.png','mask.png'}<=names or not ({'depth.npy','depth.exr'}&names):raise ValueError('Incomplete view: '+prefix)
 for rec,wanted in shards:
  archive=safe(source,rec['path']);print('VERIFY_ARCHIVE',rec['path'],'needed_files',len(wanted),flush=True)
  if sha(archive)!=rec['sha256']:raise ValueError('Archive corrupted')
  pending=set(wanted)
  with tarfile.open(archive,mode='r|gz') as tar:
   for info in tar:
    if info.name not in pending:continue
    row=wanted[info.name]
    if not info.isfile() or info.size!=row['size']:raise ValueError('Unexpected archive member')
    path=safe(dest,info.name);path.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256()
    with tar.extractfile(info) as inp,path.open('xb') as out:
     for b in iter(lambda:inp.read(8<<20),b''):h.update(b);out.write(b)
    if h.hexdigest()!=row['sha256']:raise ValueError('Extracted content mismatch')
    pending.remove(info.name)
    if not pending:break
  if pending:raise ValueError('Archive lacks selected members')
 with gzip.open(dest/'objaverse_outpaint/metadata.jsonl.gz','wt') as f:
  for e in obj:f.write(json.dumps(e)+'\n')
 (dest/'3dfront/preprocess_train.json').write_text(json.dumps({'scene_results':[s for s in front['scene_results'] if s['scene_id'] in scene_ids],'results':views})+'\n')
 report=dict(status='EXTRACTION_VERIFIED',views={'objaverse_outpaint':len(obj),'3dfront':len(views)},payload_files=len(selected),payload_bytes=sum(x['size'] for x in selected.values()),archives_read=[x['path'] for x,_ in shards],files=list(selected.values()),elapsed_seconds=time.time()-start)
 (dest/'SAMPLE_MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n')
 print('SAMPLE_READY',str(dest),report['views'],report['payload_bytes'],flush=True)
if __name__=='__main__':main()
