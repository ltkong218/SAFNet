import os
from glob import glob
import numpy as np
import cv2
from torch.utils.data import Dataset


def read_ldr(img_path):
    img = np.asarray(cv2.imread(img_path, -1)[:, :, ::-1])
    img = (img / 2 ** 16).clip(0, 1).astype(np.float32)
    return img

def read_hdr(img_path):
    img = np.asarray(cv2.imread(img_path, -1)[:, :, ::-1]).astype(np.float32)
    return img

def read_expos(txt_path):
    expos = np.power(2, np.loadtxt(txt_path) - min(np.loadtxt(txt_path))).astype(np.float32)
    return expos


class SIGGRAPH17_Test_Dataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        sequences = sorted(os.listdir(dataset_dir))
        print(len(sequences))
        self.img0_list = []
        self.img1_list = []
        self.img2_list = []
        self.gt_list = []
        self.expos_list = []
        for seq in sequences:
            ldr_list = sorted(glob(os.path.join(dataset_dir, seq, '*.tif')))
            assert len(ldr_list) == 3
            self.img0_list.append(ldr_list[0])
            self.img1_list.append(ldr_list[1])
            self.img2_list.append(ldr_list[2])
            hdr_list = sorted(glob(os.path.join(dataset_dir, seq, '*.hdr')))
            assert len(hdr_list) == 1
            self.gt_list.append(hdr_list[0])
            expo_list = sorted(glob(os.path.join(dataset_dir, seq, '*.txt')))
            assert len(expo_list) == 1
            self.expos_list.append(expo_list[0])

    def __len__(self):
        return len(self.gt_list)
    
    def __getitem__(self, idx):
        img0 = read_ldr(self.img0_list[idx])
        img1 = read_ldr(self.img1_list[idx])
        img2 = read_ldr(self.img2_list[idx])
        gt = read_hdr(self.gt_list[idx])
        expos = read_expos(self.expos_list[idx])

        imgs_ldr = [img0.copy(), img1.copy(), img2.copy()]
        imgs_lin = []
        for i in range(3):
            imgs_lin.append((imgs_ldr[i] ** 2.2) / expos[i])
        return imgs_lin, imgs_ldr, expos, gt.copy()


class Challenge123_Test_Dataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        sequences = sorted(os.listdir(dataset_dir))
        sequences = sorted(list(set(s[:3] for s in sequences)))
        print(len(sequences))
        self.img0_list = []
        self.img1_list = []
        self.img2_list = []
        self.gt_list = []
        self.expos_list = []
        for seq in sequences:
            self.img0_list.append(os.path.join(dataset_dir, seq + '_1/ldr_img_1.tif'))
            self.img1_list.append(os.path.join(dataset_dir, seq + '_2/ldr_img_2.tif'))
            self.img2_list.append(os.path.join(dataset_dir, seq + '_3/ldr_img_3.tif'))
            self.gt_list.append(os.path.join(dataset_dir, seq + '_2/hdr_img.hdr'))
            self.expos_list.append(os.path.join(dataset_dir, seq + '_2/exposure.txt'))

    def __len__(self):
        return len(self.gt_list)
    
    def __getitem__(self, idx):
        img0 = read_ldr(self.img0_list[idx])
        img1 = read_ldr(self.img1_list[idx])
        img2 = read_ldr(self.img2_list[idx])
        gt = read_hdr(self.gt_list[idx])
        expos = read_expos(self.expos_list[idx])

        imgs_ldr = [img0.copy(), img1.copy(), img2.copy()]
        imgs_lin = []
        for i in range(3):
            imgs_lin.append((imgs_ldr[i] ** 2.2) / expos[i])
        return imgs_lin, imgs_ldr, expos, gt.copy()
