import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import random
import torchvision.transforms as transforms
from evaluate import calculate_auc
from load_datasets import load_hsi
from model.model_AD import DSBMNet
import os
from pathlib import Path


SEED=2345
batchsize = 1
torch.manual_seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
datalabels = ['Salinas']
datalabel = datalabels[0]
Y, M_vca, Map_gt, Nr, Nc = load_hsi(datalabel)
N = Nr * Nc
B = Y.shape[0]
R = M_vca.shape[1]
K=32
Y = (Y - Y.min()) / (Y.max() - Y.min())
Y = torch.from_numpy(Y)
Y = torch.reshape(Y, (B, Nr, Nc))
M_init = torch.from_numpy(M_vca).unsqueeze(2).unsqueeze(3).float()

folder_path = "./weight/" + datalabel
os.makedirs(folder_path, exist_ok=True)

if datalabel == 'Salinas':
    learning_rate =0.001
    weight_decay_param = 0.0005
    beta, gamma = 1000, 0.01
    IterationNumber = 500

#loss_func = nn.MSELoss(reduction='mean')
loss_func = nn.L1Loss(reduction='mean')
def normalization_max_min(x):
    x_n = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x_n

def ResultFunction(x, z):
    x = x.detach().cpu().numpy()
    z = z.detach().cpu().numpy()
    residual = np.linalg.norm(x - z, ord=2, axis=0, keepdims=True)
    residual_np = normalization_max_min(residual ** 2)
    return residual_np

def SAD(num_bands, inp, target):
    try:
        input_norm = torch.sqrt(torch.bmm(inp.view(-1, 1, num_bands),
                                              inp.view(-1,num_bands, 1)))
        target_norm = torch.sqrt(torch.bmm(target.view(-1, 1, num_bands),
                                               target.view(-1, num_bands, 1)))
        summation = torch.bmm(inp.view(-1, 1, num_bands), target.view(-1, num_bands, 1))
        angle = torch.acos(summation / (input_norm * target_norm))
    except ValueError:
        return 0.0
    return angle

class load_data(torch.utils.data.Dataset):
    def __init__(self, img, transform=None):
        self.img = img.float()
        self.transform = transform
    def __getitem__(self, idx):
        return self.img
    def __len__(self):
        return 1

def train():
    net = DSBMNet(Nc, Nr, B, K, R).cuda()
    net.apply(net.weights_init)
    model_dict = net.state_dict()
    model_dict['Decoder.0.weight'] = M_init.cuda()
    net.load_state_dict(model_dict)
    train_y = load_data(img=Y, transform=transforms.ToTensor())
    train_y = torch.utils.data.DataLoader(dataset=train_y, batch_size=batchsize, shuffle=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay_param)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
    train_loss = []
    print("Start training!")
    for epoch in range(IterationNumber):
        for i, (y) in enumerate(train_y):
            y = y.cuda()
            x1, _, y_hat = net(y)
            ResultMap_path = Path("Result_map") / datalabel
            ResultMap_path.mkdir(parents=True, exist_ok=True)
            loss_re =  beta * loss_func(y_hat, y)
            loss_sad = SAD(B,y_hat.view(1, B, -1).transpose(1, 2),y.view(1, B, -1).transpose(1, 2))
            loss_sad = gamma * torch.sum(loss_sad).float()
            Loss = loss_re + loss_sad
            train_loss.append(Loss.cpu().data.numpy())
            optimizer.zero_grad()
            Loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
        if epoch % 10 == 0:
           print("Epoch:", epoch,"| Loss: %.4f" % Loss.cpu().data.numpy(),)
           torch.save(net.state_dict(), os.path.join(
                folder_path, 'epoch_' + str(epoch) + '_' + str(datalabel) + '.pt'))

def select_best(Y, M_vca):
    opt_epoch = 0
    max_auc = 0
    net = DSBMNet(Nc, Nr, B, K, R).cuda()
    for e in range(0, IterationNumber , 10):
        net.load_state_dict(torch.load(os.path.join(
            folder_path, 'epoch_' + str(e) + '_' + str(datalabel) + '.pt')))
        net.eval()
        with torch.no_grad():
             x1,A_est, Y_hat = net(torch.unsqueeze(Y, 0).cuda())
        Y = Y.cuda()
        y1 = torch.squeeze(Y)
        y1 = torch.reshape(y1, (B, N))
        y_hat1 = torch.squeeze(Y_hat)
        y_hat1 = torch.reshape(y_hat1, (B, N))
        E_est = ResultFunction(y1, y_hat1)
        Det_map = E_est.reshape(Nr, Nc)
        auc = calculate_auc(Map_gt.reshape(Nr * Nc, 1), Det_map.reshape(Nr * Nc, 1))
        if auc > max_auc:
             max_auc = auc
             opt_epoch = e
    return max_auc, opt_epoch

if __name__=="__main__":
    train()
    max_auc, opt_epoch = select_best(Y, M_vca)
    net = DSBMNet(Nc, Nr, B, K, R).cuda()
    net.load_state_dict(torch.load(os.path.join(
        folder_path,'epoch_' + str(opt_epoch) + '_' + str(datalabel) + '.pt')))
    net.eval()
    with torch.no_grad():
        x1, A_est, Y_hat = net(torch.unsqueeze(Y, 0).cuda())
    Y = Y.cuda()
    y1 = torch.squeeze(Y)
    y_est = y1.cpu().numpy().T
    y1 = torch.reshape(y1, (B, N))
    y_hat1 = torch.squeeze(Y_hat)
    y_hat1 = torch.reshape(y_hat1, (B, N))
    E_est = ResultFunction(y1, y_hat1)
    Det_map = E_est.reshape(Nr, Nc)
    data1 = {
        'det_map': Det_map.reshape(Nr*Nc,1),
        'GT': Map_gt.reshape(Nr*Nc,1)
    }
    PD_PF_auc1 = calculate_auc(Map_gt.reshape(Nr*Nc,1), Det_map.reshape(Nr*Nc,1))
    min_val = Det_map.min()
    max_val = Det_map.max()
    Det_map1 = (Det_map - min_val) / (max_val - min_val)
    print('AUC_PD_PF: ', PD_PF_auc1)
    ResultMap_path = Path("Result_map") / datalabel
    ResultMap_path.mkdir(parents=True, exist_ok=True)
    sio.savemat(os.path.join(ResultMap_path, str(datalabel) + '_PD_PF_auc_' + str(PD_PF_auc1) + '.mat'), data1)
