#!/usr/bin/env python
import os
import argparse
import time
import numpy as np
import torch
from dataloader import UKB_SAX_IMG
from torch.utils import data
import pandas as pd 
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt 
import torch.nn as nn
import nibabel as nib
from matplotlib.collections import LineCollection
import torch.nn.functional as F 
import pickle
from tqdm import tqdm

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8
import pystrum.pynd.ndutils as nd
from mpl_toolkits.axes_grid1 import make_axes_locatable
def save_segs(img,dir,L:int,cmap='viridis', nc:int = 4):
    plt.rcParams["savefig.bbox"] = 'tight'

    # print(img.shape)
    B,T,C,H,W = img.shape
    # img = img[:,:,0:3,...]
    if type(img).__module__ != 'numpy':
        img_print = img.cpu().detach()
    else:
        img_print = img
    if C == 3:
        tmp = torch.ones(B,T,1,H,W)
        parts = torch.sum(img_print,dim=2,keepdim=True) > 0.
        bg = tmp - (parts*tmp)
        img_print = torch.cat([img_print,bg],dim=2)
    img_print = torch.argmax(img_print, dim=2,keepdim=True).permute(0,1,3,4,2)
    img_grid = torch.cat([img_print[0,t,...] for t in range(L)],dim=1)
    plt.imsave(dir,img_grid[...,0].numpy())
class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)

        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            # new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
        
def apply_stn(stn,dt,target): 
    C,_,_ = target.shape 
    output = torch.empty_like(target)
    for i in range(C):
        output[i,...] = stn(target[None,i:i+1,...],dt)
    return output  

def save_dvfs(flow, dir:str,L:int = 9):
    # *Get Shape of flow 
    assert len(flow.shape) == 4 and flow.size(1) == 2, 'The flow shape is {}. Should be T,C,H,W instead.'.format(flow.shape)
    flow = F.interpolate(input=flow,scale_factor=0.5,mode='bilinear')
    T,C,H,W = flow.shape
    fig, ax = plt.subplots()

    # *convert tensor to numpy array 
    flow = flow.cpu().detach().numpy()
    # *TCHW -> THW3
    dvfs_ctn = np.empty((T,H,W,3),dtype=np.uint8)
    # *form grid 
    grid_c,grid_r = np.meshgrid(np.linspace(0,W*L-1,W*L),np.linspace(0,H-1,H))

    # print(grid_c)
    for t in range(T):
        dvfs_ctn[t,...] = flow2img(flow[t,...])
    flow_grid = np.concatenate([dvfs_ctn[t,...] for t in range(L)],axis=1)
    img_grid = np.concatenate([flow[t,...] for t in range(L)],axis=2)

    ax.imshow(flow_grid)
    # plot_grid(grid_c,grid_r, ax=ax,  color="lightgrey")
    plot_grid(grid_c+img_grid[1], grid_r+img_grid[0], ax=ax, color="red")
    # plt.imshow(img_grid)
    # plt.show()
    plt.axis('off')
    plt.savefig(dir, bbox_inches='tight',pad_inches = 0, dpi=1200)

def save_jbd(flow,dir:str):
    # *Get Shape of flow 
    assert len(flow.shape) == 4 and flow.size(1) == 2, 'The flow shape is {}. Should be T,C,H,W instead.'.format(flow.shape)
    T,C,H,W = flow.shape
    jbd_maps = np.empty((T,H,W),dtype=np.float)
    fig = plt.figure()
    ax = plt.gca()

    for t in range(T):
        jbd_maps[t,...] = jacobian_determinant(flow[t,...])
    jcd_grid = np.concatenate([jbd_maps[t,...] for t in range(T)],axis=1)
    jcd_grid = np.clip(jcd_grid,-1,7)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.05)
    im = ax.imshow(jcd_grid,cmap='jet')
    im.set_clim(-1,7)
    ax.axis('off')
    fig.colorbar(im,cax=cax)
    plt.savefig(dir, bbox_inches='tight',pad_inches = 0, dpi=1200)

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
def plot_grid(x,y, ax=None, **kwargs):
    ax = ax or plt.gca()
    segs1 = np.stack((x,y), axis=2)
    segs2 = segs1.transpose(1,0,2)
    ax.add_collection(LineCollection(segs1, linewidths=0.1, **kwargs))
    ax.add_collection(LineCollection(segs2, linewidths=0.1, **kwargs))
    ax.autoscale()


def draw_deform_grids(f,root='./d',h=128,w=128):
    '''
    grid--image_grid used to show deform field
    type: numpy ndarray, shape: (h, w, 2), value range: (-1, 1)
    '''
    df = f.cpu().detach().numpy()

    fig, ax = plt.subplots(figsize=(6.4,6.4))

    grid_x,grid_y = np.meshgrid(np.linspace(0,h,h),np.linspace(0,w,w))
    plot_grid(grid_x,grid_y, ax=ax,  color="lightgrey")

    plot_grid(grid_x+np.flip(df[0],0), grid_y+np.flip(df[1],0), ax=ax)
    plt.axis('off')
    fig.savefig('{}.png'.format(root),bbox_inches='tight')
    plt.close(fig)

def jacobian_determinant(disp):
    """
    Input: displacement field with size [C,H,W] (numpy)
    Output: jacombian determinant [H,W]
    """
    # check inputs
    if disp.size(0) in (2,3):
        disp = disp.permute(1,2,0)
    if type(disp).__module__ != 'numpy':
        dt = disp.cpu().detach().numpy()
    else:
        dt = disp
    volshape = dt.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(dt + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

# def save_dvfs(flow, dir:str,L:int = 9):
#     # *Get Shape of flow 
#     assert len(flow.shape) == 4 and flow.size(1) == 2, 'The flow shape is {}. Should be T,C,H,W instead.'.format(flow.shape)
#     T,C,H,W = flow.shape
#     # *convert tensor to numpy array 
#     flow = flow.cpu().detach().numpy()
#     # *TCHW -> THW3
#     dvfs_ctn = np.empty((T,H,W,3),dtype=np.uint8)
    
#     for t in range(T):
#         dvfs_ctn[t,...] = flow2img(flow[t,...])
#     img_grid = np.concatenate([dvfs_ctn[t,...] for t in range(L)],axis=1)
#     plt.imshow(img_grid)
#     plt.axis('off')
#     plt.savefig(dir, bbox_inches='tight',pad_inches = 0, dpi=1200)

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
parser.add_argument('--load-model', help='optional model file to initialize with', default='/usr/not-backed-up/scnb/vxmorph_0020.pt')
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

inshape = (128,128)


# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = dvc

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]


model = vxm.networks.VxmDense.load(args.load_model, device)
stn = SpatialTransformer(size=(128,128)).to(device)
# prepare the model for training and send to device
model.to(device)
model.eval()

# fixed_imgs = {'img':np.empty((num_of_tests,9,1,128,128)).astype(np.float32),'info':[]}
# moving_imgs = {'img':np.empty((num_of_tests,9,1,128,128)).astype(np.float32),'info':[]}
moved_imgs = np.empty((num_of_tests,9,1,128,128)).astype(np.float32)
dvfs = np.empty((num_of_tests,9,2,128,128)).astype(np.float32)
warped_masks = np.empty((num_of_tests,9,4,128,128)).astype(np.float32)
# gt_masks = np.empty((num_of_tests,2,4,128,128)).astype(np.float32)
inference_time = np.empty((num_of_tests,9))
with tqdm(total=len(test_loader), desc=f'Testing...', unit='seqs') as pbar:
    for ind, items in enumerate(test_loader):
        # *generate inputs (and true outputs) and convert them to tensors
        movings, fixeds, seg, info = items
        # # *Save fixed images 
        # fixed_imgs['img'][ind,...] = fixeds[0]
        # fixed_imgs['info'].append(info[2][1:])
        # # *Save moving images
        # moving_imgs['img'][ind,...] = movings[0]
        # moving_imgs['info'].append(info[2][0])
        # # *Save ground-truth masks
        # gt_masks[ind,...] = seg[0]

        movings = movings.to(device)
        fixeds = fixeds.to(device)
        seg_ed = seg[0,0,...].to(device)

        for t in range(movings.size(1)):
            
            inputs = movings[:,t,...]
            y_true = fixeds[:,t,...]

            step_start_time = time.time()
            # *run inputs through the model to produce a warped image and flow field
            y_pred = model(inputs,y_true)
            step_end_time = time.time()
            # *Save inference time 
            inference_time[ind,t] = step_end_time - step_start_time
            # *Save dvfs
            dvfs[ind,t,...] = y_pred[1][0].detach().cpu().numpy()
            # *Save warpped seg mask 
            warped_masks[ind,t,...] = apply_stn(stn,y_pred[1],seg_ed).detach().cpu().numpy()
            # *Save moved image 
            moved_imgs[ind,t,...] = y_pred[0][0].detach().cpu().numpy()
        pbar.update(1)

dump_results = { 'moved_imgs':moved_imgs, 'deformations':dvfs, 'warpped_masks':warped_masks, 'inference_time': inference_time}
with open("vxm_results", 'wb') as f:
    pickle.dump(dump_results, f)

print("==============")
print("Well Done!")
print("==============")



        #     loss = 0
        #     loss_list = []
        #     for n, loss_function in enumerate(losses):
        #         if n == 1:
        #             curr_loss = grad(y_pred=y_pred[n]) * weights[n]
        #             jbd_tmp = jacobian_determinant(y_pred[n][0])
        #             njd_tmp = ( jbd_tmp < 0).sum()
        #             njd[t,ind] = njd_tmp
        #             to_pd.append({'NJD':njd_tmp.item(),'TP':t+1, 'model':'VolxelMorph'})
        #             to_pd.append({'GJD':np.mean(np.abs(np.gradient(jbd_tmp))),'TP':t+1, 'model':'VolxelMorph'})
        #             loss_list.append(curr_loss.item())
        #             loss += curr_loss
        #         else:
        #             curr_loss = nn.MSELoss()(y_true, y_pred[n])
        #             mse_loss[t,ind] = curr_loss.item()
        #             # nmi_tmp[t,ind] = NMI(y_true,y_pred[n]).item()
        #             to_pd.append({'RMSE':torch.sqrt(curr_loss).item(),'TP':t+1, 'model':'VolxelMorph'})
        #             loss_list.append(curr_loss.item())
        #             loss += curr_loss
        # if ind == 3:
        #     save_dvfs(flow=dvfs,dir='./dvfs')
        #     save_jbd(flow=dvfs, dir='./jbd')
        #     save_segs(img=segs,dir='./segs.png',L=9)

        # epoch_loss.append(loss_list)
        # epoch_total_loss.append(loss.item())
        # # get compute time
        # epoch_step_time.append(time.time() - step_start_time)

    # print epoch info
#     epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
#     time_info = '%.4f sec/step' % np.mean(epoch_step_time)
#     losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
#     loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
#     print(' - '.join((epoch_info, time_info, loss_info)), flush=True)
# results = {}
# results['RMSE_mean'] = torch.mean(torch.sqrt(mse_loss),dim=-1).numpy()
# results['RMSE_std'] = torch.std(torch.sqrt(mse_loss),dim=-1).numpy()
# results['NJD_mean'] = torch.mean(njd,dim=-1).numpy()
# results['NJD_std'] = torch.std(njd,dim=-1).numpy()
# save_image(make_grid(gts,nrow=9,padding=0),"./gts_vxm.png")
# save_image(make_grid(preds,nrow=9,padding=0),"./preds_vxm.png")
# pd.DataFrame.to_csv(pd.DataFrame(data=to_pd),path_or_buf=f'./rmse_vxm.csv',header=True,index=True)
# pd.DataFrame.to_csv(pd.DataFrame(data=results),path_or_buf=f'./results_vxm.csv',header=True,index=True)
# final model save
# model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))
