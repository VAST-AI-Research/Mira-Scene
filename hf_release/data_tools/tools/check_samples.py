#!/usr/bin/env python3
"""Strict CPU sample smoke; never fall back to a different sample on failure."""
import argparse,os,random
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
from load_dataset import load_dataset
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--subset',choices=['objaverse_outpaint','3dfront'],required=True);p.add_argument('--count',type=int,default=3);a=p.parse_args()
torch.set_num_threads(2);ds=load_dataset(a.root,a.subset)
for i in range(min(a.count,len(ds))):
 random.seed(42);np.random.seed(42);torch.manual_seed(42);sample=ds._get_item(i)
 tensors={k:v for k,v in sample.items() if torch.is_tensor(v)}
 assert tensors and all(torch.isfinite(v).all() for v in tensors.values())
 print(i,{k:list(v.shape) for k,v in tensors.items()})
print('PASS',a.subset)
