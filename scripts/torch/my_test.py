#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8). pp
    1788-1800. 2019. 

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
import random
import argparse
import time
import numpy as np
import torch
from dataloader import UKB_SAX_IMG
from torch.utils import data
import pandas as pd 
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt 

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8
def compute_color(u, v):
	"""
	compute optical flow color map
	:param u: horizontal optical flow
	:param v: vertical optical flow
	:return:
	"""

	height, width = u.shape
	img = np.zeros((height, width, 3))

	NAN_idx = np.isnan(u) | np.isnan(v)
	u[NAN_idx] = v[NAN_idx] = 0

	colorwheel = make_color_wheel()
	ncols = np.size(colorwheel, 0)

	rad = np.sqrt(u ** 2 + v ** 2)

	a = np.arctan2(-v, -u) / np.pi

	fk = (a + 1) / 2 * (ncols - 1) + 1

	k0 = np.floor(fk).astype(int)

	k1 = k0 + 1
	k1[k1 == ncols + 1] = 1
	f = fk - k0

	for i in range(0, np.size(colorwheel, 1)):
		tmp = colorwheel[:, i]
		col0 = tmp[k0 - 1] / 255
		col1 = tmp[k1 - 1] / 255
		col = (1 - f) * col0 + f * col1

		idx = rad <= 1
		col[idx] = 1 - rad[idx] * (1 - col[idx])
		notidx = np.logical_not(idx)

		col[notidx] *= 0.75
		img[:, :, i] = np.uint8(np.floor(255 * col * (1 - NAN_idx)))

	return img


def make_color_wheel():
	"""
	Generate color wheel according Middlebury color code
	:return: Color wheel
	"""
	RY = 15
	YG = 6
	GC = 4
	CB = 11
	BM = 13
	MR = 6

	ncols = RY + YG + GC + CB + BM + MR

	colorwheel = np.zeros([ncols, 3])

	col = 0

	# RY
	colorwheel[0:RY, 0] = 255
	colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
	col += RY

	# YG
	colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
	colorwheel[col:col + YG, 1] = 255
	col += YG

	# GC
	colorwheel[col:col + GC, 1] = 255
	colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
	col += GC

	# CB
	colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
	colorwheel[col:col + CB, 2] = 255
	col += CB

	# BM
	colorwheel[col:col + BM, 2] = 255
	colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
	col += + BM

	# MR
	colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
	colorwheel[col:col + MR, 0] = 255

	return colorwheel

def flow2img(flow_data):
    '''
    convert optical flow into color image
    :param flow_data: Numpy 2,H,W
    :return: color image
    '''
    u = flow_data[0,...]
    v = flow_data[1,...]

    UNKNOW_FLOW_THRESHOLD = 1e7
    pr1 = abs(u) > UNKNOW_FLOW_THRESHOLD
    pr2 = abs(v) > UNKNOW_FLOW_THRESHOLD
    idx_unknown = (pr1 | pr2)
    u[idx_unknown] = v[idx_unknown] = 0

    # get max value in each direction
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxu = max(maxu, np.max(u))
    maxv = max(maxv, np.max(v))
    minu = min(minu, np.min(u))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))
    u = u / maxrad + np.finfo(float).eps
    v = v / maxrad + np.finfo(float).eps

    img = compute_color(u, v)
    # print(img)

    idx = np.repeat(idx_unknown[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

def jacobian_determinant(disp):
    """
    Input: displacement field with size [C,H,W] (numpy)
    Output: jacombian determinant [H,W]
    """
    # check inputs
    if type(disp).__module__ != 'numpy':
        dt = disp.cpu().detach().numpy()
    else:
        dt = disp
    _,H,W = disp.shape
    
    x,y = np.meshgrid(np.linspace(0,W-1,W),np.linspace(H-1,0,H))
    grids = np.stack([x,y],axis=0)
    # compute gradients
    J = np.gradient(dt + grids )

    dfdx = J[0]
    dfdy = J[1]
    return dfdx[0,...] * dfdy[1,...] - dfdy[0,...] * dfdx[1,...]

def save_dvfs(flow, dir:str,L:int = 9):
    # *Get Shape of flow 
    assert len(flow.shape) == 4 and flow.size(1) == 2, 'The flow shape is {}. Should be T,C,H,W instead.'.format(flow.shape)
    T,C,H,W = flow.shape
    # *convert tensor to numpy array 
    flow = flow.cpu().detach().numpy()
    # *TCHW -> THW3
    dvfs_ctn = np.empty((T,H,W,3),dtype=np.uint8)
    
    for t in range(1,T):
        dvfs_ctn[t,...] = flow2img(flow[t,...])
    img_grid = np.concatenate([dvfs_ctn[t,...] for t in range(10-L,10)],axis=1)
    plt.imshow(img_grid)
    plt.axis('off')
    plt.savefig(dir, bbox_inches='tight',pad_inches = 0, dpi=1200)

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
# parser.add_argument('--img-list', required=True, help='line-seperated list of training files')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--model-dir', default='models',
                    help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-model', help='optional model file to initialize with', default='/usr/not-backed-up/scnb/0050.pt')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

# loss hyperparameters
parser.add_argument('--image-loss', default='mse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                    help='weight of deformation loss (default: 0.01)')
args = parser.parse_args()

bidir = args.bidir
data_dir = '/usr/not-backed-up/scnb/data/masks_sax_5k/'
data_dicom = '/usr/not-backed-up/scnb/data/dicom_lsax_5k/'
model_dir = '/usr/not-backed-up/scnb/'
dvc = 'cuda:1'
# load and prepare training data
dataset = UKB_SAX_IMG(root_gt= data_dir,root_dicom = data_dicom,mode='test',num_of_frames=10)
print('Len(dataset): {}'.format(len(dataset)))
num_of_tests = len(dataset)
test_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=4)
 

def grad(y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
        # dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        d = torch.mean(dx) + torch.mean(dy) 
        g = d / 2.0

        return g
# no need to append an extra feature axis if data is multichannel
# add_feat_axis = not args.multichannel

# if args.atlas:
#     # scan-to-atlas generator
#     atlas = vxm.py.utils.load_volfile(args.atlas, np_var='vol',
#                                       add_batch_axis=True, add_feat_axis=add_feat_axis)
#     generator = vxm.generators.scan_to_atlas(train_files, atlas,
#                                              batch_size=args.batch_size, bidir=args.bidir,
#                                              add_feat_axis=add_feat_axis)
# else:
#     # scan-to-scan generator
#     generator = vxm.generators.scan_to_scan(
#         train_files, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)

# # extract shape from sampled input
inshape = (128,128)
# # prepare model folder
# model_dir = args.model_dir
# os.makedirs(model_dir, exist_ok=True)

# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = dvc
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# assert np.mod(args.batch_size, nb_gpus) == 0, \
#     'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_devices)

# enabling cudnn determinism appears to speed up training by a lot
# torch.backends.cudnn.deterministic = not args.cudnn_nondet

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]


model = vxm.networks.VxmDense.load(args.load_model, device)

# prepare the model for training and send to device
model.to(device)
model.eval()


# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# need two image loss functions if bidirectional
if bidir:
    losses = [image_loss_func, image_loss_func]
    weights = [0.5, 0.5]
else:
    losses = [image_loss_func]
    weights = [1]

# prepare deformation loss
losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
weights += [args.weight]

mse_loss = torch.empty(9,num_of_tests)
njd = torch.empty(9,num_of_tests)
gts = torch.empty(9,1,128,128)
preds = torch.empty(9,1,128,128)
dvfs = torch.empty(9,2,128,128)
# training loops
for epoch in range(args.initial_epoch, args.epochs):

    # if epoch % 20 == 0:
    #     model.save(os.path.join(model_dir, '%04d.pt' % epoch))

    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []
    for ind, items in enumerate(test_loader):

        step_start_time = time.time()

        # generate inputs (and true outputs) and convert them to tensors
        movings, fixeds = items
        movings = movings.to(device)
        fixeds = fixeds.to(device)
        for t in range(movings.size(1)):
            
            inputs = movings[:,t,...]
            y_true = fixeds[:,t,...]

            
            

            # run inputs through the model to produce a warped image and flow field
            y_pred = model(inputs,y_true)

            if ind == 0:
                gts[t,...] = y_true[0]
                preds[t,...] = y_pred[0][0]
                print(y_pred[1].shape)
            # print(y_pred[1].shape)
            # calculate total loss
            loss = 0
            loss_list = []
            for n, loss_function in enumerate(losses):
                if n == 1:
                    curr_loss = grad(y_pred=y_pred[n]) * weights[n]
                    njd[t,ind] = (jacobian_determinant(y_pred[n][0]) < 0.).sum()
                    loss_list.append(curr_loss.item())
                    loss += curr_loss
                else:
                    curr_loss = loss_function(y_true, y_pred[n])
                    mse_loss[t,ind] = curr_loss.item()
                    loss_list.append(curr_loss.item())
                    loss += curr_loss
        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())
        # get compute time
        epoch_step_time.append(time.time() - step_start_time)

    # print epoch info
    epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
    time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
    print(' - '.join((epoch_info, time_info, loss_info)), flush=True)
results = {}
results['RMSE_mean'] = torch.mean(torch.sqrt(mse_loss),dim=-1).numpy()
results['RMSE_std'] = torch.std(torch.sqrt(mse_loss),dim=-1).numpy()
results['NJD_mean'] = torch.mean(njd,dim=-1).numpy()
results['NJD_std'] = torch.std(njd,dim=-1).numpy()
save_image(make_grid(gts,nrow=9,padding=0),"./gts_vxm.png")
save_image(make_grid(preds,nrow=9,padding=0),"./preds_vxm.png")
save_dvfs(flow=df_seqs[0], dir='{}/df_x{}.png'.format(rp,ind),L=9)
pd.DataFrame.to_csv(pd.DataFrame(data=results),path_or_buf=f'./results_vxm.csv',header=True,index=True)
# final model save
# model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))
