import numpy as np
import torch 
from torchvision import datasets, transforms
from utils import * # get the dataset
from pyhessian import hessian # Hessian computation
from density_plot import get_esd_plot # ESD plot
from pytorchcv.model_provider import get_model as ptcv_get_model # model

import matplotlib.pyplot as plt
import pdb

# enable cuda devices
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# get the model 
model = ptcv_get_model("resnet20_cifar10", pretrained=True)
model2 = ptcv_get_model("sepreresnet20_cifar10", pretrained=True)
# change the model to eval mode to disable running stats upate
model.eval()
model2.eval()

# create loss function
criterion = torch.nn.CrossEntropyLoss()

# get dataset 
train_loader, test_loader = getData()

# for illustrate, we only use one batch to do the tutorial
for inputs, targets in train_loader:
    break

# we use cuda to make the computation fast
model = model.cuda()
model2 = model2.cuda()
inputs, targets = inputs.cuda(), targets.cuda()

# create the hessian computation module
hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=True)
hessian_comp2 = hessian(model2, criterion, data=(inputs, targets), cuda=True)

# get the top1, top2 eigenvectors
top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)
top_eigenvalues2, top_eigenvector2 = hessian_comp2.eigenvalues(top_n=2)

# This is a simple function, that will allow us to perturb the model paramters and get the result
def get_params(model_orig,  model_perb, direction, alpha):
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        m_perb.data = m_orig.data + alpha * d
    return model_perb

# lambda is a small scalar that we use to perturb the model parameters along the eigenvectors 
lams1 = np.linspace(-0.5, 0.5, 31).astype(np.float32)
lams2 = np.linspace(-0.5, 0.5, 31).astype(np.float32)

loss_list = []
loss_list2 = []

# create a copy of the model
model_perb1 = ptcv_get_model("resnet20_cifar10", pretrained=True)
model_perb1.eval()
model_perb1 = model_perb1.cuda()

model_perb2 = ptcv_get_model("resnet20_cifar10", pretrained=True)
model_perb2.eval()
model_perb2 = model_perb2.cuda()

model2_perb1 = ptcv_get_model("sepreresnet20_cifar10", pretrained=True)
model2_perb1.eval()
model2_perb1 = model2_perb1.cuda()

model2_perb2 = ptcv_get_model("sepreresnet20_cifar10", pretrained=True)
model2_perb2.eval()
model2_perb2 = model2_perb2.cuda()


for lam1 in lams1:
    for lam2 in lams2:
        model_perb1 = get_params(model, model_perb1, top_eigenvector[0], lam1)
        model_perb2 = get_params(model_perb1, model_perb2, top_eigenvector[1], lam2)
        loss_list.append((lam1, lam2, criterion(model_perb2(inputs), targets).item()))   

        model2_perb1 = get_params(model2, model2_perb1, top_eigenvector2[0], lam1)
        model2_perb2 = get_params(model2_perb1, model2_perb2, top_eigenvector2[1], lam2)
        loss_list2.append((lam1, lam2, criterion(model2_perb2(inputs), targets).item()))   

loss_list = np.array(loss_list)
loss_list2 = np.array(loss_list2)

fig = plt.figure()
landscape = fig.add_subplot(111, projection='3d')  # Use add_subplot to create a 3D subplot
landscape.plot_trisurf(loss_list[:,0], loss_list[:,1], loss_list[:,2],alpha=0.8, cmap='viridis')
                       #cmap=cm.autumn, #cmamp = 'hot')
landscape.plot_trisurf(loss_list2[:,0], loss_list2[:,1], loss_list2[:,2],alpha=0.8, cmap='hot')
                       #cmap=cm.autumn, #cmamp = 'hot')

landscape.set_title('Loss Landscape')
landscape.set_xlabel('ε_1')
landscape.set_ylabel('ε_2')
landscape.set_zlabel('Loss')

#z_min = min(min(loss_list[:,2]),min(loss_list2[:,2]))
#landscape.set_zlim(z_min, z_min+13)
landscape.view_init(elev=30, azim=45)
landscape.dist = 6
plt.savefig('2D_Compare.png')