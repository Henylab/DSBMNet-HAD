import numpy as np
import scipy.io as sio

# Salinas
def load_hsi(datalabel):
    if datalabel == 'Salinas':
        data = sio.loadmat('dataset/Salinas_new.mat')
        Nr = 120
        Nc = 120
        M_vca = data['E']
        Map_gt = data['map']
        Y = data['data'].astype(np.float32)
        return Y, M_vca, Map_gt, Nr, Nc
