from torch import nn
import torch
from model.DSSA import Spe, Spa
import torch.nn.functional as FN
from torch.nn import init


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(FN.softplus(x)))
        return x
def MutilGaussian_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
def gaussian_init(m):
    if isinstance(m, nn.Conv2d):
        init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def inverse_hyperbolic_transform(tensor):
    return torch.acosh(tensor)

def hyperbolic_transform(tensor):
    return torch.cosh(tensor)

class DSBMNet(nn.Module):
    def __init__(self, w, h, B, K, R):
        super(DSBMNet, self).__init__()
        self.R = R
        self.w= w
        self.h = h
        self.K = K
        self.wlocal = nn.Parameter(torch.randn(1, requires_grad=True))
        self.wglobal = nn.Parameter(torch.randn(1, requires_grad=True))

        self.SDRM = nn.Sequential(
            nn.Conv2d(B, K, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(K, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.MCNN = nn.Sequential(
            MDCNN(K, K, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4)),
            nn.Conv2d(K, K, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(K, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.Encoder1 = nn.Sequential(
            nn.Conv2d(K, K // 2, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=8),
            nn.BatchNorm2d(K // 2, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
        )

        self.Encoder2 = nn.Sequential(
            nn.Conv2d(K // 2, K // 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4),
            nn.BatchNorm2d(K // 4, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
        )

        self.Latent = nn.Sequential(
            nn.Conv2d(K // 4, K // 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(K // 4, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
        )

        self.Decoder2 = nn.Sequential(
            nn.Conv2d(K // 4, K // 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4),
            nn.BatchNorm2d(K // 2, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
        )

        self.Decoder1 = nn.Sequential(
            nn.Conv2d(K // 2, K, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=8),
            nn.BatchNorm2d(K, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
        )
        self.spa_atten_i = SCSA()
        self.spa_atten_d2 = SCSA()
        self.spa_atten_d1 = SCSA()
        self.SpaDR = Spa(dim=K, num_heads=2, ffn_dim=K)
        self.SpeDR = Spe(dim=K, num_heads=2, ffn_dim=K)
        self.ConstraintModule = nn.Sequential(
                nn.Conv2d(K, R, kernel_size=(1,1), stride=(1, 1), padding=(0,0)),
                nn.Softmax(dim=1),
         )

        self.Decoder =nn.Sequential(
            nn.Conv2d(R, B, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(inplace=True),
        )


    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
# pre-processing
        x1 = self.SDRM(x)
# multi-direction
        Y2=self.MCNN(x1)
# local
        x_e1 = self.Encoder1(Y2)
        x_e2 = self.Encoder2(x_e1)
        x_i = self.Latent(x_e2)
        x_i_atten = self.spa_atten_i(x_i + x_e2)
        x_d2 = self.Decoder2(x_i_atten)
        x_d2_atten = self.spa_atten_d2(x_d2 + x_e1)
        x_d1 = self.Decoder1(x_d2_atten)
        Flocla = x_d1
# global
        Y3 = self.SpaDR(Y2)
        Fglobal = self.SpeDR(Y3)
# fusing
        F = self.wlocal * Flocla + self.wglobal * Fglobal
        abu_est = self.ConstraintModule(F)
        x_hat = self.Decoder(abu_est)
        return x1, abu_est, x_hat


