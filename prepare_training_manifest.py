"""Convert external labeled inputs into a training manifest without copying training data.
Supply an explicit split table; this tool does not invent new train/validation memberships.
"""
from pathlib import Path
import argparse,sys
import numpy as np
import pandas as pd
import h5py
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from lvclr.model import RAW_NAMES

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('data',type=Path,help='2D3D: directory of phase-ID .npy slices; LRHR: HDF5 with X, y, names.')
    ap.add_argument('--targets',type=Path,help='2D3D normalized target CSV indexed by volume filename.')
    ap.add_argument('--splits',type=Path,required=True,help='CSV with name, split, and (LRHR) job_group. Use saved memberships.')
    ap.add_argument('--output',type=Path,default=Path('training_manifest.csv'))
    args=ap.parse_args();split=pd.read_csv(args.splits).set_index('name')
    if split.index.duplicated().any():ap.error('Duplicate names in split table.')
    rows=[]
    if args.data.is_dir():
        if not args.targets:ap.error('2D image inputs require --targets.')
        targets=pd.read_csv(args.targets,index_col=0);targets.index=targets.index.str.removesuffix('.npy')
        for p in sorted(args.data.glob('*.npy')):
            if p.name not in split.index:continue
            key=p.stem.rsplit('_',1)[0]
            if key not in targets.index:ap.error(f'No target for {p.name}; expected <volume>_x/y/z.npy naming.')
            rows.append({'input':str(p.resolve()),**split.loc[p.name].to_dict(),**targets.loc[key,RAW_NAMES].to_dict()})
    else:
        with h5py.File(args.data,'r') as h:
            for i,name in enumerate(h['names']):
                name=name.decode() if isinstance(name,bytes) else str(name)
                if name not in split.index:continue
                rows.append({'input':str(args.data.resolve()),'h5_index':i,**split.loc[name].to_dict(),**dict(zip(RAW_NAMES,h['y'][i]))})
    if len(rows)!=len(split):ap.error(f'Matched {len(rows)} inputs to {len(split)} split entries; fix missing names before training.')
    args.output.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(args.output,index=False)
    print('Saved',len(rows),'rows:',args.output)
if __name__=='__main__':main()
