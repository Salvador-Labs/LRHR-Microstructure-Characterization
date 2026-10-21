"""Checkpoint-compatible Lightning model with no training-data dependency."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from .architecture import CNNModel

OUTPUT_NAMES=['phi1','phi2','phi3','tau1','tau2','tau3','d1_um','d2_um','d3_um',
              'rho_c_um-2','rho_c_over_rho_t','rho_t_um-2','S12_um-1','S13_um-1','S23_um-1']
RAW_NAMES=['f1','f2','f3','tort1','tort2','tort3','davg1','davg2','davg3','tpbconn','tpbconnfrac','tpbtot','sa12','sa13','sa23']

class PublishedModel(pl.LightningModule):
    def __init__(self, model_config=None, **legacy_hyperparameters):
        super().__init__()
        if model_config is None:
            raise ValueError('Original checkpoints require model_config from the accompanying config.json; use load_bundle().')
        self.model_config=dict(model_config)
        c=self.model_config
        if c['family']=='2D3D':
            self.model=CNNModel(c['architecture'],in_chan=c['input_shape'][0],outparams=15,imsize=c['input_shape'][1:],soft=False,advar=False)
        else:
            self.model=CNNModel(img_shape=c['input_shape'],out_shape=15,arcstr=c['architecture'],reg=None)
        self.save_hyperparameters({'model_config':c})

    def forward(self,x):
        return self.model(x)

    def training_step(self,batch,batch_idx):
        x,y=batch;loss=torch.nn.functional.mse_loss(self(x),y)
        self.log('train_loss',loss,on_step=False,on_epoch=True,batch_size=len(x))
        return loss

    def validation_step(self,batch,batch_idx):
        x,y=batch;loss=torch.nn.functional.mse_loss(self(x),y)
        self.log('val_loss',loss,on_step=False,on_epoch=True,batch_size=len(x))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.model_config['training']['learning_rate'])


def load_bundle(bundle, device='cpu', checkpoint=None):
    """Load the unchanged original Lightning checkpoint plus explicit input metadata."""
    bundle=Path(bundle)
    config=json.loads((bundle/'config.json').read_text())
    path=Path(checkpoint) if checkpoint else bundle/config['checkpoint_file']
    model=PublishedModel.load_from_checkpoint(str(path),map_location='cpu',model_config=config,strict=True)
    return model.to(device).eval(),config


def prepare_input(array,config,normalized=False):
    """One observation: phase-label image or raw single-channel XCT volume."""
    a=np.asarray(array)
    if not np.isfinite(a).all():raise ValueError('Input contains non-finite values.')
    if config['family']=='2D3D':
        expected=tuple(config['input_shape'][1:])
        if a.shape!=expected:raise ValueError(f'Expected segmented image {expected}, got {a.shape}.')
        if not np.isin(a,[1,2,3]).all():raise ValueError('Segmented images must contain phase IDs 1, 2, 3, not grayscale intensities.')
        a=np.stack([a==label for label in [1,2,3]]).astype(np.float32)
    else:
        expected=tuple(config['input_shape'][1:])
        if a.shape==(1,*expected):a=a[0]
        if a.shape!=expected:raise ValueError(f'Expected raw XCT volume {expected}, got {a.shape}.')
        # Cast before subtraction: raw data can be unsigned integers.
        a=a.astype(np.float32)
        if normalized:
            if a.min()<0 or a.max()>1:raise ValueError('Normalized input must be in [0,1].')
        else:
            low,high=(13500.,55500.) if config['contrast']=='A' else (13500.,76500.)
            if a.min()<low or a.max()>high:
                raise ValueError(f'Raw synthetic {config["contrast"]} contrast must be in [{low:g},{high:g}]. Use --normalized only for pre-normalized [0,1] input.')
            a=(a-low)/(high-low)
        a=a[None,...]
    return torch.from_numpy(np.ascontiguousarray(a,dtype=np.float32))


def decode_outputs(raw,bundle):
    """Inverse target scaling, including the physical-unit TPB convention of the paper."""
    bundle=Path(bundle);cfg=json.loads((bundle/'config.json').read_text())
    conv=pd.read_csv(bundle/cfg['normalization_file'],index_col=0)
    a=np.asarray(raw,dtype=np.float64)
    if a.shape[-1]!=15:raise ValueError('Expected 15 normalized outputs.')
    z=a*conv.loc['STD',RAW_NAMES].to_numpy()+conv.loc['Mean',RAW_NAMES].to_numpy()
    z=np.array(z,copy=True)
    z[...,3:6]=np.exp(z[...,3:6])+1
    idx=[6,7,8,9,11,12,13,14]
    z[...,idx]=np.exp(z[...,idx])
    z[...,10]=1-np.exp(z[...,10])
    v=cfg['high_resolution_voxel_um']
    z[...,6:9]*=v
    z[...,[9,11]]*=2/(v*v)
    z[...,12:15]/=v
    return z


def predict_arrays(model,arrays,config,bundle,normalized=False,batch_size=32):
    raw=[]
    with torch.inference_mode():
        for start in range(0,len(arrays),batch_size):
            x=torch.stack([prepare_input(a,config,normalized) for a in arrays[start:start+batch_size]])
            raw.append(model(x.to(model.device)).cpu().numpy())
    if not raw:raise ValueError('No input observations.')
    raw=np.concatenate(raw)
    return raw,decode_outputs(raw,bundle)
