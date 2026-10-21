"""Train or resume a model using an explicit split manifest and a published configuration."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from lvclr.model import PublishedModel,prepare_input,RAW_NAMES
ROOT=Path(__file__).resolve().parent

class ManifestDataset(Dataset):
    def __init__(self,rows,base,config,seed):
        import h5py
        self.config=config;self.arrays=[];self.targets=[]
        for _,row in rows.iterrows():
            path=Path(row['input']);path=path if path.is_absolute() else base/path
            if path.suffix in ['.h5','.hdf5']:
                with h5py.File(path,'r') as h:a=h['X'][int(row['h5_index'])]
            else:a=np.load(path,allow_pickle=False)
            self.arrays.append(prepare_input(a,config))
            self.targets.append(torch.tensor(row[RAW_NAMES].to_numpy(dtype=np.float32)))
        if config['family']=='LRHR' and config['training']['augmentation_factor']>1:
            # Fixed augmented dataset, analogous to the archived preprocessing;
            # explicitly seeded here so repeated runs are reproducible.
            rng=np.random.default_rng(seed);count=len(self.arrays)
            for _ in range((config['training']['augmentation_factor']-1)*count):
                i=int(rng.integers(count));x=self.arrays[i].clone()
                transformed=False
                while not transformed:
                    if rng.binomial(1,.5):
                        axes=rng.permutation(3)
                        while np.array_equal(axes,np.arange(3)):axes=rng.permutation(3)
                        x=x.permute(0,*list(axes+1));transformed=True
                    dims=[a+1 for a in range(3) if rng.binomial(1,.5)]
                    if dims:x=x.flip(dims);transformed=True
                self.arrays.append(x.contiguous());self.targets.append(self.targets[i])
    def __len__(self):return len(self.arrays)
    def __getitem__(self,i):
        x=self.arrays[i]
        if self.config['family']=='2D3D':
            # Historical pipeline applied random flips to the shared dataset,
            # including the validation subset.
            if torch.rand(())<.5:x=x.flip(-1)
            if torch.rand(())<.5:x=x.flip(-2)
        return x,self.targets[i]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('manifest',type=Path,help='CSV with input, split=train/val, job_group, and 15 normalized target columns.')
    ap.add_argument('--config',type=Path,default=ROOT/'models/lrhr_0.5um_A/config.json')
    ap.add_argument('--output-dir',type=Path,default=Path('runs'))
    ap.add_argument('--epochs',type=int,help='Default: selected model config (2D3D 1000, LRHR 20000).')
    ap.add_argument('--batch-size',type=int,help='Default: selected config, 32.')
    ap.add_argument('--learning-rate',type=float,help='Default: selected config, 1e-4, constant.')
    ap.add_argument('--seed',type=int,help='Default: selected config (2D3D 100, LRHR 42).')
    ap.add_argument('--accelerator',choices=['auto','cpu','gpu','mps'],default='auto')
    ap.add_argument('--devices',type=int,default=1)
    ap.add_argument('--workers',type=int,default=0)
    ap.add_argument('--resume',type=Path,help='Resume a checkpoint made by this release, including optimizer state.')
    ap.add_argument('--fast-dev-run',action='store_true',help='One train/validation batch; smoke test only.')
    args=ap.parse_args();c=json.loads(args.config.read_text());tc=c['training']
    for arg,key in [('epochs','epochs'),('batch_size','batch_size'),('learning_rate','learning_rate'),('seed','seed')]:
        if getattr(args,arg) is not None:tc[key]=getattr(args,arg)
    if min(tc['epochs'],tc['batch_size'],tc['learning_rate'])<=0:ap.error('Epochs, batch size and learning rate must be positive.')
    df=pd.read_csv(args.manifest);required={'input','split',*RAW_NAMES}
    if not required.issubset(df):ap.error(f'Missing manifest columns: {sorted(required-set(df))}')
    if not set(df.split)<= {'train','val'} or set(df.split)!= {'train','val'}:ap.error('Manifest must contain nonempty train and val splits only.')
    if not np.isfinite(df[RAW_NAMES].to_numpy(float)).all():ap.error('Targets must be finite normalized values.')
    identity=['input']+(['h5_index'] if 'h5_index' in df else [])
    if df.groupby(identity,dropna=False).split.nunique().max()>1:ap.error('An input occurs in both splits.')
    if c['family']=='LRHR':
        if 'job_group' not in df or df.job_group.isna().any():ap.error('LRHR manifests require job_group to prevent related-volume split leakage.')
        if df.groupby('job_group').split.nunique().max()>1:ap.error('An LRHR job group occurs in both splits.')
    pl.seed_everything(tc['seed'],workers=True)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'effective_config.json').write_text(json.dumps(c,indent=2))
    df.to_csv(args.output_dir/'split_manifest.csv',index=False)
    train=ManifestDataset(df[df.split=='train'],args.manifest.resolve().parent,c,tc['seed'])
    val=ManifestDataset(df[df.split=='val'],args.manifest.resolve().parent,c,tc['seed']+1)
    # Explicit override saves optimizer-compatible config in new checkpoints.
    model=PublishedModel(model_config=c)
    if args.resume:
        checkpoint=torch.load(args.resume,map_location='cpu',weights_only=True)
        old=checkpoint.get('hyper_parameters',{}).get('model_config')
        if old is None:ap.error('Resume requires a checkpoint from this release. Original paper checkpoints are inference assets.')
        if old['input_shape']!=c['input_shape'] or old['architecture']!=c['architecture']:ap.error('Resume architecture/input shape differs from selected config.')
    callback=ModelCheckpoint(monitor='val_loss',mode='min',save_top_k=1,save_last=True,filename='best-{epoch:04d}-{val_loss:.6f}')
    trainer=pl.Trainer(default_root_dir=args.output_dir,max_epochs=tc['epochs'],accelerator=args.accelerator,devices=args.devices,
        callbacks=[callback],fast_dev_run=args.fast_dev_run,deterministic=True,logger=pl.loggers.CSVLogger(args.output_dir,name='training'))
    trainer.fit(model,DataLoader(train,batch_size=tc['batch_size'],shuffle=True,num_workers=args.workers),
                DataLoader(val,batch_size=tc['batch_size'],num_workers=args.workers),ckpt_path=str(args.resume) if args.resume else None)
if __name__=='__main__':main()
