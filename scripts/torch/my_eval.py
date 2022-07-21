import pickle
import logging 
import time
import pandas as pd
from metrices import jacobian_determinant, HD, DSC
from dataloader import UKB_SAX_IMG
from tqdm import tqdm
import numpy as np 
import matplotlib.pyplot as plt


data_dir = '/usr/not-backed-up/scnb/data/masks_sax_5k/'
data_dicom = '/usr/not-backed-up/scnb/data/dicom_lsax_5k/'
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'''
    Loading evaluation matrix...
    ''')
    load_time = time.time()
    with open("vxm_results", 'rb') as f:
        val_mat = pickle.load(f) 
    logging.info(f'''
    Load sussessfully, take {time.time()-load_time} sec.
    Validation matrix keys: 
    {val_mat.keys()}
    Number of patients: {len(val_mat['moved_imgs'])/6}
    ''')
    # *Load test dataset 
    dataset = UKB_SAX_IMG(root_gt= data_dir,root_dicom = data_dicom,mode='test',num_of_frames=10)
    
    logging.info(f'''
    Initial test loader DONE.
    Number of test: {len(dataset)}
    ''')
    num_of_tests = len(dataset)
    to_pd = []
    with tqdm(total=num_of_tests, desc=f'Testing...', unit='seqs') as pbar:
        for ind, item in enumerate(dataset):
            T = item[0].size(0)
            slice_id = item[-1][-1][-1].split('/')[-1].split('.')[0]
            for t in range(T):
                # *RMSE - time steps
                rmse = np.sqrt(np.mean((val_mat['moved_imgs'][ind][t,...] - item[1][t,...].numpy())**2,axis=(1,2)))[0]
                to_pd.append({'RMSE':rmse,'TP':t+1, 'model':'VoxelMorph', 'pid':item[-1][0], 'slice_id':slice_id})
                
                # *Jacobian determinant map
                jd = jacobian_determinant(val_mat['deformations'][ind][t,...])

                # *NJD- time steps 
                njd = ( jd < 0).sum()
                to_pd.append({'NJD':njd,'TP':t+1, 'model':'VoxelMorph', 'pid':item[-1][0], 'slice_id':slice_id})

                # *Average Gradient of JD map
                gjd = np.mean(np.abs(np.gradient(jd)))
                to_pd.append({'GJD':gjd,'TP':t+1, 'model':'VoxelMorph', 'pid':item[-1][0], 'slice_id':slice_id})
                
                if t == 4: # *ES
                    # fig,ax = plt.subplots(2)
                    # ax[0].imshow(np.transpose(val_mat['warpped_masks'][ind][t,0:3,...],[1,2,0]))
                    # ax[1].imshow(np.transpose(item[2][-1,0:3,...],[1,2,0]))
                    # # ax[2].imshow(item[0][0,0,...])
                    # # ax[3].imshow(item[0][5,0,...])
                    # plt.show()
                    for j, p in enumerate(['rvendo','lvendo','lvmyo']):
                        # *DICE at ES - cardiac parts
                        dsc = DSC(val_mat['warpped_masks'][ind][t,j,...],item[2][-1,j,...].numpy())
                        to_pd.append({'DSC':dsc,'parts':p,'TP':t+1, 'model':'VoxelMorph', 'pid':item[-1][0], 'slice_id':slice_id})

                        # *HD at ES - cardiac parts
                        hd = HD(val_mat['warpped_masks'][ind][t,j,...],item[2][-1,j,...].numpy())
                        to_pd.append({'HD':hd,'parts':p,'TP':t+1, 'model':'VoxelMorph', 'pid':item[-1][0], 'slice_id':slice_id})

                        # logging.info(f'''
                        # RMSE: {rmse}
                        # NJD: {njd}
                        # AGJD: {gjd}
                        # DICE ES: {dsc}
                        # HD ES: {hd}
                        # ''')

                # fig,ax = plt.subplots(4)
                # ax[0].imshow(val_mat['moved_imgs'][ind][3,0,...])
                # ax[1].imshow(item[1][3,0,...])
                # ax[2].imshow(item[0][0,0,...])
                # ax[3].imshow(item[0][5,0,...])
                # plt.show()

            # break
            pbar.update(1)
    pd.DataFrame.to_csv(pd.DataFrame(data=to_pd),path_or_buf=f'./results_vxm.csv',header=True,index=True)


