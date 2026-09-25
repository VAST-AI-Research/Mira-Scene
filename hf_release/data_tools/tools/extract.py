#!/usr/bin/env python3
"""Verify and extract complete subsets, with resumable per-file validation."""
import argparse,gzip,hashlib,json,shutil,tarfile
from pathlib import Path,PurePosixPath

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def safe(root,name):
 p=PurePosixPath(name)
 if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe archive path')
 target=root/str(p)
 if not target.resolve().is_relative_to(root.resolve()):raise ValueError('Path escapes destination')
 return target

def extract(source,destination,subsets):
 source=Path(source).resolve();dest=Path(destination).resolve()
 if json.loads((source/'manifests/integrity_summary.json').read_text())['status']!='PASS':raise ValueError('Data archive verification incomplete')
 manifest=source/'manifests/shards.json';identity=sha(manifest)
 dest.mkdir(parents=True,exist_ok=True);state=dest/'.extract_state.json'
 if state.exists():
  if json.loads(state.read_text())['manifest_sha256']!=identity:raise ValueError('Destination belongs to another release')
 elif any(dest.iterdir()):raise ValueError('Use an empty destination')
 else:state.write_text(json.dumps({'manifest_sha256':identity}))
 for shard in json.loads(manifest.read_text()):
  if shard['path'].split('/')[0] not in subsets:continue
  archive=safe(source,shard['path']);members=safe(source,shard['members'])
  if sha(archive)!=shard['sha256'] or sha(members)!=shard['members_sha256']:raise ValueError('Download checksum mismatch')
  with gzip.open(members,'rt') as f:rows=[json.loads(line) for line in f]
  count=0
  with tarfile.open(archive,mode='r|gz') as tar:
   for info in tar:
    if count>=len(rows):raise ValueError('Unexpected archive member')
    row=rows[count];count+=1
    if not info.isfile() or info.name!=row['archive'] or info.size!=row['size']:raise ValueError('Member mismatch')
    target=safe(dest,info.name)
    if target.exists():
     if target.is_symlink() or not target.is_file() or sha(target)!=row['sha256']:raise ValueError('Existing file differs: '+str(target))
     continue
    target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_name(target.name+'.extract-partial')
    if temp.is_symlink():raise ValueError('Unexpected symlink')
    h=hashlib.sha256()
    with tar.extractfile(info) as inp,temp.open('wb') as out:
     for b in iter(lambda:inp.read(8<<20),b''):h.update(b);out.write(b)
    if h.hexdigest()!=row['sha256']:raise ValueError('Member content mismatch')
    temp.replace(target)
  if count!=len(rows):raise ValueError('Incomplete archive')
  print('VERIFIED',shard['path'],flush=True)
 for subset in subsets:
  for name in ('metadata.jsonl.gz','preprocess_train.json'):
   p=source/subset/name
   if p.exists():(dest/subset).mkdir(exist_ok=True);shutil.copyfile(p,dest/subset/name)
 shutil.copyfile(source/'README.md',dest/'README.md')
 print('EXTRACTION_VERIFIED',str(dest),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');p.add_argument('--subset',choices=['all','objaverse_outpaint','3dfront'],default='all');a=p.parse_args()
 extract(a.source,a.destination,['objaverse_outpaint','3dfront'] if a.subset=='all' else [a.subset])
