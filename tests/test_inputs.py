from pathlib import Path
import json, sys
import numpy as np
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from lvclr.model import prepare_input
CONFIG=json.loads((ROOT/'models/lrhr_0.5um_A/config.json').read_text())

class InputTests(unittest.TestCase):
    def test_encoding_and_shape(self):
        shape=CONFIG['input_shape'][1:]
        if CONFIG['family']=='2D3D':
            a=np.full(shape,2,dtype=np.uint8)
            x=prepare_input(a,CONFIG)
            assert tuple(x.shape)==tuple(CONFIG['input_shape'])
            np.testing.assert_allclose(x.numpy(),2/3)
        else:
            raw=np.full(shape,34500,dtype=np.uint32)
            np.testing.assert_allclose(prepare_input(raw,CONFIG).numpy(),prepare_input(np.full(shape,.5),CONFIG,normalized=True).numpy())
    
    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):prepare_input(np.zeros((2,2)),CONFIG)
        with self.assertRaises(ValueError):prepare_input(np.full(CONFIG['input_shape'][1:],np.nan),CONFIG)
