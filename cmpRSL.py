import os
import os.path as op
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import BZ
import h5py
plt.style.use('seaborn-dark-palette')

Base = BZ.bzBase()

def load_py_res(save_tag=0):
    import pickle
    path_load = op.join(Base.path_Brsl, 'bayes_model_solution')
    dst_h5    = op.join(path_load, f'py_exp{save_tag}.h5')
    arrnames  = 'MU NU PI_2 DELTA_2 SIGMA_2 TAU_2 PHI B L R Y_O Y TGDATA N K D'.split()
    dct_arrs  = {}
    with h5py.File(dst_h5, 'r') as h5:
        for arrn in arrnames:
            try:
                arr = h5[arrn][:]
            except:
                arr = h5[arrn] # scalars
            dct_arrs[arrn] = arr

    dst_dct   = f'{op.splitext(dst_h5)[0]}.dct'
    with open(dst_dct, 'rb') as fh:
        dct_hp    = pickle.load(fh)
    return dct_arrs, dct_hp

class cmpPyMat(BZ.bzBase):
    def __init__(self):
        super().__init__()
        self.arrnames = 'MU NU PI_2 DELTA_2 SIGMA_2 TAU_2 PHI B L R Y_O Y TGDATA N K D'.split()
        self.dct_arrs, self.dct_hp = load_py_res()
        self.plot_py()

    def plot_py(self):
        y = self.dct_arrs['Y'] # simulation x year x tide gauge?
        [print (y[sim, :, :].mean()) for sim in range(y.shape[0])]

if __name__ == '__main__':
    cmpPyMat()
