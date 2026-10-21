"""Check bundled examples against saved deterministic inference outputs on this machine."""
from pathlib import Path
import sys,hashlib,json,argparse
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from lvclr.model import load_bundle,predict_arrays,RAW_NAMES,OUTPUT_NAMES
ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu')
    args=ap.parse_args();torch.set_num_threads(2)
    for p in (ROOT/'models').glob('*/config.json'):
        c=json.loads(p.read_text())
        for filename,key in [('model.ckpt','checkpoint_sha256'),('normalization.csv','normalization_sha256')]:
            actual=hashlib.sha256((p.parent/filename).read_bytes()).hexdigest()
            if actual!=c[key]:raise RuntimeError(f'Checksum mismatch: {p.parent/filename}')
    bundle=ROOT/'models/lrhr_0.5um_A';model,c=load_bundle(bundle,args.device)
    expected=pd.read_csv(ROOT/'examples/expected_predictions_normalized.csv')
    paths=[ROOT/'examples/inputs'/name for name in expected.input]
    raw,physical=predict_arrays(model,[np.load(p,allow_pickle=False) for p in paths],c,bundle)
    np.testing.assert_allclose(raw,expected[RAW_NAMES].to_numpy(),rtol=2e-4,atol=2e-5)
    expected_phys=pd.read_csv(ROOT/'examples/expected_predictions.csv')
    np.testing.assert_allclose(physical,expected_phys[OUTPUT_NAMES].to_numpy(),rtol=2e-4,atol=2e-5)
    print(f'PASS: checkpoint/normalization hashes and {len(paths)} example predictions.')
    print('These examples verify software execution, not independent scientific generalization.')
if __name__=='__main__':main()
