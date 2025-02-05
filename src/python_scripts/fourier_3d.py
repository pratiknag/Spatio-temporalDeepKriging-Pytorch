import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io

torch.manual_seed(0)
np.random.seed(0)

activation = F.relu

################################################################
# 3d fourier layers
################################################################

def compl_mul3d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    op = partial(torch.einsum, "bixyz,ioxyz->boxyz")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)

class SpectralConv3d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d_fast, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        # self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        # self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        # self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")  # Using rfftn for real-to-complex FFT
        x_ft = torch.stack([x_ft.real,x_ft.imag], dim = -1)
        
        # # Prepare output tensor for relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-3)// 2 +1, x.size(-2)// 2 + 1, x.size(-1) // 2 + 1,2, device=x.device)
        # out_ft = torch.zeros_like(x_ft, device=x.device)
        # print("I am here")
        # Multiply relevant Fourier modes (adjust indexing for complex tensors)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        # out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
        #     compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        # out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
        #     compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        # out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
        #     compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
        # print("I am here2")
        # Convert back to physical space
        out_ft = torch.view_as_complex(out_ft)

        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)), norm="ortho")  # Using irfftn for inverse FFT
        return x

class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(SimpleBlock2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.fc0 = nn.Linear(4, self.width)

        self.conv0 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        # Trainable parameter for x_
        self.x_ = nn.Parameter(torch.tensor(0.5))  # initialize at some value (e.g., 0.5)
        
        # Final output layer for x_
        self.fc_x_ = nn.Linear(128, 1)

    def forward(self, x):
        # print("I am here 1")
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        # print("I am here 2")
        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn0(x1 + x2)
        x = F.relu(x)
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn1(x1 + x2)
        x = F.relu(x)
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn2(x1 + x2)
        x = F.relu(x)
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn3(x1 + x2)
        

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x1 = self.fc2(x)

         # x_ output (with trainable parameter, transformed via softplus)
        x_ = torch.nn.functional.softplus(self.fc_x_(x)) * self.x_
        
        # Multiply x by 10
        #x = x * 10
        
        # x2 = x1 + x_
        x2 = x1 + x_
        
        # x3 = x1 - x_
        x3 = x1 - x_
        return x1,x2,x3

class Net3d(nn.Module):
    def __init__(self, modes1,modes2,modes3, width):
        super(Net3d, self).__init__()

        self.conv1 = SimpleBlock2d(modes1, modes2, modes3, width)


    def forward(self, x):
        x1,x2,x3 = self.conv1(x)
        return x1.squeeze(),x2.squeeze(),x3.squeeze()


    def count_params(self):
        c = 0
        for p in self.parameters():
            print(p)
            c += reduce(operator.mul, list(p.size()))
        return c

################################################################
# configs
################################################################

# TRAIN_PATH = 'data/ns_data_V1000_N1000_train.mat'
# TEST_PATH = 'data/ns_data_V1000_N1000_train_2.mat'
# TRAIN_PATH = 'data/ns_data_V1000_N5000.mat'
# TEST_PATH = 'data/ns_data_V1000_N5000.mat'
data = np.load("datasets/median_data.npy")
#data = np.load("generated_1d_data_Burger_FNO.npy")
print(data.shape)

ntrain = 3974
ntest = 20

modes1 = 32
modes2 = 32
modes3 = 2
S = 64
width = 20

batch_size = 64
batch_size2 = batch_size

epochs = 100
learning_rate = 0.0025
scheduler_step = 100
scheduler_gamma = 0.5

print(epochs, learning_rate, scheduler_step, scheduler_gamma)


runtime = np.zeros(2, )
t1 = default_timer()

T_in = 3
T = 6

################################################################
# load data
################################################################

train_data = data[:ntrain,:,:,:]
test_data = data[ntrain:,:,:,:]

print(train_data[:,:,:,:T_in].shape)
print(test_data.shape)


train_a = torch.tensor(train_data[:,:,:,:T_in].reshape(ntrain,S,S,T_in,1),dtype=torch.float)
# train_a = train_a.repeat([1,1,1,T_in,1])
test_a = torch.tensor(test_data[:,:,:,:T_in].reshape(ntest,S,S,T_in,1),dtype=torch.float)
np.save("true_forecast-input.npy", test_a)
train_u = torch.tensor(train_data[:,:,:,T_in:],dtype=torch.float)
test_u = torch.tensor(test_data[:,:,:,T_in:],dtype=torch.float)
print(test_u.shape)
a_normalizer = UnitGaussianNormalizer(train_a)
train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

y_normalizer = UnitGaussianNormalizer(train_u)
train_u = y_normalizer.encode(train_u)
# pad locations (x,y,t)
gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T_in, 1])
gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T_in, 1])
gridt = torch.tensor(np.linspace(0, 1, T_in+1)[1:], dtype=torch.float)
gridt = gridt.reshape(1, 1, 1, T_in, 1).repeat([1, S, S, 1, 1])

train_a = torch.cat((gridx.repeat([ntrain,1,1,1,1]), gridy.repeat([ntrain,1,1,1,1]),
                       gridt.repeat([ntrain,1,1,1,1]), train_a), dim=-1)
test_a = torch.cat((gridx.repeat([ntest,1,1,1,1]), gridy.repeat([ntest,1,1,1,1]),
                       gridt.repeat([ntest,1,1,1,1]), test_a), dim=-1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')

################################################################
# training and evaluation
################################################################
model = Net3d(modes1,modes2,modes3, width).cuda()
# model = torch.load('model/ns_fourier_V100_N1000_ep100_m8_w20')

# print(model.count_params())
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)


myloss = LpLoss(size_average=False)
y_normalizer.cuda()
best_val_loss = float('inf')
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        x1,x2,x3 = model(x)
        mse = F.mse_loss(x1, y, reduction='mean')
        # mse.backward()
        y = y_normalizer.decode(y)
        x1 = y_normalizer.decode(x1)
        x2 = y_normalizer.decode(x2)
        x3 = y_normalizer.decode(x3)
        size = y.shape[0]
        l2 = myloss(x1.view(size, -1), x2.view(size, -1),x3.view(size, -1),y.view(size, -1))
        l2.backward()

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            x1,x2,x3 = model(x)
            # mse.backward()
            x1 = y_normalizer.decode(x1)
            x2 = y_normalizer.decode(x2)
            x3 = y_normalizer.decode(x3)
            size = y.shape[0]
            test_l2 += myloss(x1.view(size, -1), x2.view(size, -1),x3.view(size, -1),y.view(size, -1)).item()
            

    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(ep, t2-t1, train_mse, train_l2, test_l2)
    # torch.save(model, path_model)
    # Early stopping
    if train_l2 < best_val_loss:
        best_val_loss = train_l2
        epochs_without_improvement = 0
        # Save the model
        torch.save(model.state_dict(), 'models/model_forecasting-FNO.pth')
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= 30:
            print(f"Early stopping at epoch {epochs+1}")
            break


pred_med = torch.zeros(test_u.shape)
pred_lb = torch.zeros(test_u.shape)
pred_ub = torch.zeros(test_u.shape)
print(pred_med.shape)
index = 0
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.cuda(), y.cuda()

        x1,x2,x3 = model(x)
        x1 = y_normalizer.decode(x1)
        x2 = y_normalizer.decode(x2)
        x3 = y_normalizer.decode(x3)
        pred_med[index] = x1
        pred_lb[index] = x3
        pred_ub[index] = x2
        size = 1
        test_l2 += myloss(x1.view(size, -1), x2.view(size, -1),x3.view(size, -1),y.view(size, -1)).item()
        print(index, test_l2)
        index = index + 1
print("average loss is : {}".format(test_l2/index))
pred_med = pred_med.numpy()
pred_lb = pred_lb.numpy()
pred_ub = pred_ub.numpy()
np.save("datasets/pred_forecast-FNO-med.npy",pred_med)
np.save("datasets/pred_forecast-FNO-lb.npy",pred_lb)
np.save("datasets/pred_forecast-FNO-ub.npy",pred_ub)
np.save("true_forecast.npy", test_u)





