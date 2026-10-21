"""Prepare inference inputs from a segmented reference volume (phase labels 1,2,3)."""
from pathlib import Path
import argparse,json,numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('volume',type=Path,help='.npy, or .npz containing phase_labels')
    ap.add_argument('--model',type=Path,default=ROOT/'models/lrhr_0.5um_A')
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args();c=json.loads((args.model/'config.json').read_text())
    if args.volume.suffix=='.npz':
        with np.load(args.volume,allow_pickle=False) as z:v=z['phase_labels']
    else:v=np.load(args.volume,allow_pickle=False)
    if list(v.shape)!=c['high_resolution_volume_shape'] or not np.isin(v,[1,2,3]).all():ap.error('Volume shape or phase IDs do not match the selected model.')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if c['family']=='2D3D':
        for axis,label in enumerate(['z','y','x']):
            p=args.output_dir/(args.volume.stem+'_'+label+'.npy')
            if p.exists():ap.error(f'Refusing to overwrite {p}')
            np.save(p,np.take(v,v.shape[axis]//2,axis=axis));print(p)
    else:
        factor=c['resolution_um']/c['high_resolution_voxel_um'];f=int(round(factor))
        if not np.isclose(f,factor):ap.error('This helper supports only integer voxel-size ratios.')
        n=c['input_shape'][1];cropped=v[:n*f,:n*f,:n*f]
        levels=[0,13500,43500,55500 if c['contrast']=='A' else 76500]
        a=np.take(np.asarray(levels,dtype=np.float64),cropped)
        a=a.reshape(n,f,n,f,n,f).mean(axis=(1,3,5)).astype(np.float32).astype(np.uint32)
        p=args.output_dir/(args.volume.stem+'.npy')
        if p.exists():ap.error(f'Refusing to overwrite {p}')
        np.save(p,a);print(p)
if __name__=='__main__':main()
