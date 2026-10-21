"""Evaluate the bundled, pre-normalized 0.5 um anode HDF5 example."""
from pathlib import Path
import sys,argparse
import h5py,numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from lvclr.model import load_bundle,predict_arrays,decode_outputs,OUTPUT_NAMES
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--data',type=Path,default=ROOT/'examples/test_subvolumes_0.5um_A_normalized.h5')
 p.add_argument('--output-dir',type=Path,default=Path('example_evaluation'))
 args=p.parse_args();bundle=ROOT/'models/lrhr_0.5um_A'
 with h5py.File(args.data,'r') as h:
  if h.attrs.get('model_id')!='lrhr_0.5um_A':p.error('Expected the bundled normalized 0.5 um anode example.')
  x=h['X'][:];y=h['y'][:];names=h['names'].asstr()[:]
 model,cfg=load_bundle(bundle)
 raw,pred=predict_arrays(model,list(x),cfg,bundle,normalized=True)
 truth=decode_outputs(y,bundle)
 args.output_dir.mkdir(parents=True,exist_ok=True)
 for filename,arr in [('predictions',pred),('ground_truth',truth)]:
  pd.DataFrame(arr,index=pd.Index(names,name='input'),columns=OUTPUT_NAMES).to_csv(args.output_dir/(filename+'.csv'))
 error=abs(pred-truth)
 metrics=pd.DataFrame({'property':OUTPUT_NAMES,'MAE':error.mean(axis=0),'RMSE':np.sqrt(np.mean((pred-truth)**2,axis=0)),'maximum_absolute_error':error.max(axis=0)})
 metrics.to_csv(args.output_dir/'errors.csv',index=False)
 print(metrics.to_string(index=False));print('These related examples overlap historical training data; this is not an independent validation benchmark.')
if __name__=='__main__':main()
