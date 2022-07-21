import numpy as np 
import pystrum.pynd.ndutils as nd
from medpy import metric

def jacobian_determinant(disp):
    """
    Input: displacement field with size [C,H,W] (numpy)
    Output: jacombian determinant [H,W]
    """
    # check inputs
    if disp.shape[0] in (2,3):
        disp = np.transpose(disp,[1,2,0])

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


def HD(pred,ref,connectivity:int=3):
    assert ref.shape == pred.shape, 'Make sure two inputs are in same shapes bro :)' 
    pred_np = (255*pred).astype(np.uint8)
    ref_np = (255*ref).astype(np.uint8)
    hd_val = metric.binary.hd(ref_np,pred_np,connectivity=connectivity)
    return(hd_val)


def DSC(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0
    i_flat = prediction.reshape(-1)
    t_flat = target.reshape(-1)

    intersection = (i_flat * t_flat).sum()

    return ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))