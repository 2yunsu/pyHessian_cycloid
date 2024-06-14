import numpy as np
import torch 
from torchvision import datasets, transforms
from utils import *
from pyhessian import hessian # Hessian computation
from density_plot import get_esd_plot # ESD plot
from pytorchcv.model_provider import get_model as ptcv_get_model # model

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb #added
from regression import * #added
import copy #added

# enable cuda devices
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

"""
regression.py 를 통해 Cycloid Loss landscape를 만드는 Loss function을 구현하려고 했으나
regression.py의 CyclLossY가 결국은 MAE Loss와 같은 꼴이 나와서 실패.
"""

# get dataset 
# train_loader, test_loader = getData()

# for illustrate, we only use one batch to do the tutorial
# for inputs, targets in train_loader:
#     break

inputs = x
targets = target

# we use cuda to make the computation fast
model = model.cuda()
inputs, targets = inputs.cuda(), targets.cuda()

# create the hessian computation module
hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=True)

# get the top1, top2 eigenvectors
top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()

# This is a simple function, that will allow us to perturb the model paramters and get the result
def get_params(model_orig,  model_perb, direction, alpha):
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        m_perb.data = m_orig.data + alpha * d
    return model_perb

# lambda is a small scalar that we use to perturb the model parameters along the eigenvectors 
lams1 = np.linspace(-0.5, 0.5, 21).astype(np.float32)

loss_list = []

# create a copy of the model
model_perb1 = copy.deepcopy(model)
model_perb1.eval()
model_perb1 = model_perb1.cuda()

for lam1 in lams1:
    model_perb1 = get_params(model, model_perb1, top_eigenvector[0], lam1)
    loss_list.append(criterion(model_perb1(inputs), targets).item())   

loss_list = np.array(loss_list)

# plot the loss landscape
plt.plot(lams1, loss_list, label='Loss')
plt.ylabel('Loss')
plt.xlabel('Perturbation')
plt.title('Loss landscape perturbed based on top Hessian~ eigenvector')

#landscape.view_init(elev=15, azim=75)
plt.savefig('1D_test_regression.png', dpi=300)