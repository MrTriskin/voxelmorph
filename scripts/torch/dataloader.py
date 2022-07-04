import os
import re
import cv2
import pydicom as dicom 
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils import data
from torchvision import transforms as T
import torch.nn.functional as F
import imageio
def get_indexs(seg_tps:list,tg:int):
    ed = int(re.findall(r'\d+',seg_tps[0])[0])
    es = int(re.findall(r'\d+',seg_tps[1])[0])
    tp = int(tg / 2 )
    gap = int((es - ed) // tp) 
    indexs = [ed,es]
    for i in range(1,tp):
        indexs.append(int(ed + i*gap))
        indexs.append(int(min(es + i*gap,50 - 2 * (tp - i))))
    indexs.sort()
    return indexs

def getROI(rvendo,lvepi):
    x = imageio.imread(rvendo)
    
    tmp = np.where(x == np.max(x))
    r_max = np.max(tmp[0])
    c_min = np.min(tmp[1])
    x = imageio.imread(lvepi)
    tmp = np.where(x == np.max(x))
    c_max = np.max(tmp[1])
    r_min = np.min(tmp[0])
    
    return(r_min, r_max, c_min, c_max)

def fill_seg(seg):
    cnts = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.fillPoly(seg, pts =cnts[0], color=(255,255,255))
    return seg

class Norm_intensity(object):
    def __call__(self, pic, thres_roi=10.0):
        """ Normalise the image intensity by the mean and standard deviation """
        val_l = np.percentile(pic, thres_roi)
        roi = (pic >= val_l)
        mu, sigma = torch.mean(pic[roi]), torch.std(pic[roi])
        eps = 1e-6
        image2 = (pic - mu) / (sigma + eps)
        image2 = (image2 - torch.min(image2)) / (torch.max(image2) - torch.min(image2) + eps) 
        return image2

    def __repr__(self):
        return(self.__class__.__name__ + '()')
class Rescale_Intensity(object):
    def __call__(self, image, thres=(1.0, 99.0)):
        """ Rescale the image intensity to the range of [0, 1] """
        image = np.array(image)
        val_l, val_h = np.percentile(image, thres)
        image2 = image
        image2[image < val_l] = val_l
        image2[image > val_h] = val_h
        image2 = ((image2.astype(np.float32) - val_l) / (val_h - val_l)) 
        return image2

    def __repr__(self):
        return(self.__class__.__name__ + '()')
class UKB_SAX_IMG(data.Dataset):
    def __init__(self,root_gt:str,root_dicom:str,mode='train',num_of_frames:int=6):
        super(UKB_SAX_IMG,self).__init__()
        self.root_gt = root_gt
        self.root_dicom = root_dicom
        self.num_of_frames = num_of_frames
        self.resize_to = (128,128)
        self.no_seg = []
        self.with_seg = []
        self.sequence_list = []
        
        # *Check for pre-saved id list 
        patient_ids = np.load('./patient_ids.npy')
        print('Loaded patient id list from local file...')
        all_patients = []
        for c,id in enumerate(patient_ids) :
            # *Get gt frame number
            # *REF gt_dir/image&mask/time/masks/RVendo/LVendo/LVepi
            gt_dir = "{}{}/image&mask/".format(self.root_gt,id)
            seg_tps = os.listdir(gt_dir)
            # *under cine, image is DICOM 
            # *REF img_dir/id/time00x/SAX/0005
            img_dir = "{}{}/".format(self.root_dicom,id)
            if os.path.isdir(img_dir):
                if len(seg_tps) == 2 and len(os.listdir(img_dir)) >= 50:
                    patient_id = id 
                    seg = True 
                    indexs = get_indexs(seg_tps,self.num_of_frames)
                    instance_dict = {'id':patient_id, 'has_seg':seg, 'img_dir':img_dir, 'gt_dir':gt_dir, 'seg_tps':seg_tps, 'indexs':indexs}
                    all_patients.append(instance_dict)   
        print(f'num of patient: {len(all_patients)}')
        # *centre crop instead
        [r_min, r_max, c_min, c_max] = [43, 171, 7, 135]
        for patient in all_patients:
            max_lenth_dicom = len(os.listdir("{}{}/SAX".format(patient['img_dir'],patient['seg_tps'][0])))-1
            bbx_list = ['{}{}/masks/{}/{}.png'.format(patient['gt_dir'],j, k,"%04d" %(5))  for j in [patient['seg_tps'][0],patient['seg_tps'][-1]] for k in ['RVendo','LVendo','LVepi'] ]
            # r_min, r_max, c_min, c_max = getROI(bbx_list[0],bbx_list[2])
            # avg += [r_min,r_max,c_min,c_max]
            # r_min -= 20 
            # c_min -= 20
            # assert r_min>=0 and c_min>=0
            # r_max += 20
            # c_max += 20
            
            for i in range(2,8):
                tmp_img_list = ['{}time{}/SAX/{}'.format(patient['img_dir'],"%03d" %(j),"%04d" %(max_lenth_dicom-i+1)) for j in patient['indexs']]
                tmp_seg_list = ['{}{}/masks/{}/{}.png'.format(patient['gt_dir'],j, k,"%04d" %(i))  for j in [patient['seg_tps'][0],patient['seg_tps'][-1]] for k in ['RVendo','LVendo','LVepi'] ]
                self.sequence_list.append({'imgs':tmp_img_list,'masks':tmp_seg_list, 'pid':patient['id'], 'bbx':[r_min,r_max,c_min,c_max]})     
        if mode == 'train':
            self.loader_ready = self.sequence_list[0:3000*6]
        if mode == 'test':
            self.loader_ready = self.sequence_list[3000*6:3000*6+600*6]
                
    def make_seg_gt(self,segs):
        # *INPUT: 6*H*W 
        # *OUTPUT: 2*4*H*W 
        _,H,W = segs.shape
        # *background = ones - rvendo - lvepi
        br_ed = torch.ones(1,H,W,dtype=torch.long) - segs[0,...] - segs[2,...]
        br_es = torch.ones(1,H,W,dtype=torch.long) - segs[3,...] - segs[5,...]
        # *subtract lvendo from lvepi to form lvepi segs 
        segs[2,...] = segs[2,...] - segs[1,...]
        segs[-1,...] = segs[-1,...] - segs[-2,...]
        # *saperate segs for ed and es phases
        ed_seg = segs[0:3,...]
        ed_seg = torch.cat([segs[0:3,...],br_ed],dim=0)
        es_seg = segs[3:6,...]
        es_seg = torch.cat([segs[3:6,...],br_es],dim=0)
        out = torch.stack([ed_seg,es_seg],dim=0)
        out = torch.clamp(out,min=0.,max=1.)
        return out
        # make label maps for cross entropy. dtype should be int64
        # background -> 3, rvendo -> 0, lvendo -> 1, lvepi -> 2
    def __getitem__(self,index):
        # init Transform 
        Transform = {
            'train': T.Compose([
                T.ToPILImage(),
                T.Resize(self.resize_to),
                Rescale_Intensity(),
                T.ToTensor(),
            ]),
            'seg': T.Compose([
                T.ToPILImage(),
                T.Resize(self.resize_to),
                T.ToTensor(),
            ]),
            'vali': T.Compose([
                T.ToPILImage(),
                T.Resize(self.resize_to),
                Rescale_Intensity(),
                T.ToTensor(),
            ]),
        }
        Transform_train = Transform['train']
        Transform_seg = Transform['seg']
        sample = self.loader_ready[index]
        img = torch.cat([Transform_train(np.float32(dicom.read_file(i).pixel_array[sample['bbx'][0]:sample['bbx'][1],sample['bbx'][2]:sample['bbx'][3]])) for i in sample['imgs']],0)
        seg = torch.cat([Transform_seg(np.float32(fill_seg(imageio.imread(i))[sample['bbx'][0]:sample['bbx'][1],sample['bbx'][2]:sample['bbx'][3]]/255.)) for i in sample['masks']],0)
        seg = self.make_seg_gt(seg)
        moving = img[0:1,...].repeat(9,1,1,1)
        fixed = img[1:,...].unsqueeze(1)
        info = [sample['pid'], sample['bbx'], sample['imgs'], sample['masks']]

        return moving, fixed, seg, info

    def __len__(self):
        return len(self.loader_ready)




def get_loader(root_gt:str,root_dicom :str,axis='lax',mode='train'):
    if axis == 'lax':
        print('DATA ROOT: {}'.format(root_dicom))
        # dataset = UKB_LAX_4CH(root=root,mode=mode)
    if axis == 'sax':
        dataset = UKB_SAX_IMG(root_gt= root_gt,root_dicom = root_dicom,mode=mode,num_of_frames=10)
        print('Len(dataset): {}'.format(len(dataset)))
    if mode == 'train':
        data_loader = data.DataLoader(dataset=dataset, batch_size=16, shuffle=True, num_workers=4)
    else:
        data_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=4)

    return data_loader

if __name__ == '__main__':
    data_dir = '/usr/not-backed-up/scnb/data/masks_sax_5k/'
    data_dicom = '/usr/not-backed-up/scnb/data/dicom_lsax_5k/'
    train_loader = get_loader(root_gt=data_dir,root_dicom = data_dicom, axis='sax',mode='train')
    for ind, item in enumerate(train_loader):
        print(f'{ind+1}/{len(train_loader)}')
        print(item[0].shape)
        print(item[1].shape)

        # print(item[1][0:1,...].shape)>
        # print(torch.min(item[1][0:1,0,3,...]))
        # print(torch.max(item[1][0:1,0,3,...]))
        # print(item[1].shape)
        # plot = torch.sum(item[1][0,0,0:3,...],dim=0)
        # plt.imshow(plot)
        # plt.show()
        # save_segs(img=item[1][0:1,...],id='gt_x_{}'.format(ind), cmap='coolwarm')
        pass

# data_dir = '/usr/not-backed-up/scnb/data/masks_sax_5k/'
# data_dicom = '/usr/not-backed-up/scnb/data/dicom_lsax_5k/'
# train_loader = get_loader(root_gt=data_dir,root_dicom = data_dicom, axis='sax',mode='train')
# for ind, item in enumerate(train_loader):
#     # print(item[1][0:1,...].shape)>
#     # print(torch.min(item[1][0:1,0,3,...]))
#     # print(torch.max(item[1][0:1,0,3,...]))
#     print(item[0].shape)
#     # plot = torch.sum(item[1][0,0,0:3,...],dim=0)
#     # plt.imshow(plot)
#     # plt.show()
#     # save_segs(img=item[1][0:1,...],id='gt_x_{}'.format(ind), cmap='coolwarm')
#     break
