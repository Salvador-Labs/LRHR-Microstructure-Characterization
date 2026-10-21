"""Predict physical microstructure characteristics from unlabeled .npy inputs."""
from pathlib import Path
import sys,argparse,json
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from lvclr.model import load_bundle,predict_arrays,OUTPUT_NAMES,RAW_NAMES
ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('inputs',type=Path,nargs='+',help='One-observation .npy files or directories containing them.')
    ap.add_argument('--model',type=Path,default=ROOT/'models/lrhr_0.5um_A',help='Model bundle directory.')
    ap.add_argument('--output',type=Path,default=Path('predictions.csv'))
    ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu')
    ap.add_argument('--normalized',action='store_true',help='LRHR only: inputs have already been scaled to [0,1].')
    ap.add_argument('--checkpoint',type=Path,help='Optional fine-tuned Lightning checkpoint compatible with this bundle.')
    args=ap.parse_args()
    if args.batch_size<1:ap.error('Batch size must be positive.')
    paths=[]
    for p in args.inputs:paths.extend(sorted(p.glob('*.npy')) if p.is_dir() else [p])
    if not paths:ap.error('No .npy inputs found.')
    model,config=load_bundle(args.model,args.device,args.checkpoint)
    if args.normalized and config['family']=='2D3D':ap.error('--normalized applies only to LRHR inputs.')
    raw,physical=predict_arrays(model,[np.load(p,allow_pickle=False) for p in paths],config,args.model,args.normalized,args.batch_size)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(physical,columns=OUTPUT_NAMES);df.insert(0,'input',[str(p) for p in paths]);df.to_csv(args.output,index=False)
    norm=args.output.with_name(args.output.stem+'_normalized.csv')
    df=pd.DataFrame(raw,columns=RAW_NAMES);df.insert(0,'input',[str(p) for p in paths]);df.to_csv(norm,index=False)
    print(f'Saved {len(paths)} predictions to {args.output}; normalized outputs to {norm}')
if __name__=='__main__':main()
