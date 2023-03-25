import argparse

import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
import time
import cv2
import torch
import glob
import json
import mmcv
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count, Manager
from mmdet.apis import inference_detector, init_detector, show_result
from math import pi
import numpy as np
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_img_dir', type=str, help='the dir of input images')
    parser.add_argument('--out', type=str, help='the dir for result images')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--mean_teacher', action='store_true', help='test the mean teacher pth')
    args = parser.parse_args()
    return args



class NFOV():
    def __init__(self, height=2750, width=5500):
        self.FOV = [0.45, 0.45]
        self.PI = pi
        self.PI_2 = pi * 0.5
        self.PI2 = pi * 2.0
        self.height = height
        self.width = width
        self.screen_points = self._get_screen_img()

    def _get_coord_rad(self, isCenterPt, center_point=None):
        return (center_point * 2 - 1) * np.array([self.PI, self.PI_2]) \
            if isCenterPt \
            else \
            (self.screen_points * 2 - 1) * np.array([self.PI, self.PI_2]) * (
                np.ones(self.screen_points.shape) * self.FOV)

    def _get_screen_img(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, self.width), np.linspace(0, 1, self.height))
        return np.array([xx.ravel(), yy.ravel()]).T

    def _calcSphericaltoGnomonic(self, convertedScreenCoord):
        x = convertedScreenCoord.T[0]
        y = convertedScreenCoord.T[1]

        rou = np.sqrt(x ** 2 + y ** 2)
        c = np.arctan(rou)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        lat = np.arcsin(cos_c * np.sin(self.cp[1]) + (y * sin_c * np.cos(self.cp[1])) / rou)
        lon = self.cp[0] + np.arctan2(x * sin_c, rou * np.cos(self.cp[1]) * cos_c - y * np.sin(self.cp[1]) * sin_c)

        lat = (lat / self.PI_2 + 1.) * 0.5
        lon = (lon / self.PI + 1.) * 0.5

        return np.array([lon, lat]).T

    def _bilinear_interpolation(self, screen_coord):
        uf = np.mod(screen_coord.T[0],1) * self.frame_width  # long - width
        vf = np.mod(screen_coord.T[1],1) * self.frame_height  # lat - height

        x0 = np.floor(uf).astype(int)  # coord of pixel to bottom left
        y0 = np.floor(vf).astype(int)
        x2 = np.add(x0, np.ones(uf.shape).astype(int))  # coords of pixel to top right
        y2 = np.add(y0, np.ones(vf.shape).astype(int))

        base_y0 = np.multiply(y0, self.frame_width)
        base_y2 = np.multiply(y2, self.frame_width)

        A_idx = np.add(base_y0, x0)
        B_idx = np.add(base_y2, x0)
        C_idx = np.add(base_y0, x2)
        D_idx = np.add(base_y2, x2)

        flat_img = np.reshape(self.frame, [-1, self.frame_channel])

        A = np.take(flat_img, A_idx, axis=0)
        B = np.take(flat_img, B_idx, axis=0)
        C = np.take(flat_img, C_idx, axis=0)
        D = np.take(flat_img, D_idx, axis=0)

        wa = np.multiply(x2 - uf, y2 - vf)
        wb = np.multiply(x2 - uf, vf - y0)
        wc = np.multiply(uf - x0, y2 - vf)
        wd = np.multiply(uf - x0, vf - y0)

        # interpolate
        AA = np.multiply(A, np.array([wa, wa, wa]).T)
        BB = np.multiply(B, np.array([wb, wb, wb]).T)
        CC = np.multiply(C, np.array([wc, wc, wc]).T)
        DD = np.multiply(D, np.array([wd, wd, wd]).T)
        nfov = np.reshape(np.round(AA + BB + CC + DD).astype(np.uint8), [self.height, self.width, 3])
        #import matplotlib.pyplot as plt
        #plt.imshow(nfov)
        #plt.show()
        return nfov

    def toNFOV(self, frame, center_point):
        self.frame = frame
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.frame_channel = frame.shape[2]

        self.cp = self._get_coord_rad(center_point=center_point, isCenterPt=True)
        convertedScreenCoord = self._get_coord_rad(isCenterPt=False)
        spericalCoord = self._calcSphericaltoGnomonic(convertedScreenCoord)
        return self._bilinear_interpolation(spericalCoord)

def mock_detector(model, image_name, image):
    results = inference_detector(model, image)
    mock_detections = []
    #print(image_name)
    for box in results[0][0]:
        if float(box[4]) > .01:
            box = (image_name,float(box[0]),float(box[1]),float(box[2]),float(box[3]),'person',float(box[4]))
            mock_detections.append(box)
    return mock_detections
    
def crop_and_send(model,image_name,outdir):
    results = [0]*32
    nfov = NFOV()
    center_point1 = np.array([.75, .5])  # camera center point (valid range [0,1])
    center_point2 = np.array([.25, .5])  # camera center point (valid range [0,1])

    img = cv2.imread(image_name)
    a = nfov.toNFOV(img, center_point1)
    b = nfov.toNFOV(img, center_point2)    

    results[0] = mock_detector(model,image_name[:-4]+"_l.jpg",a)
    results[1] = mock_detector(model,image_name[:-4]+"_r.jpg",b)

    img_fname = image_name.split('/')[-1][:-4]
    img_dest = os.path.join(outdir,img_fname)
    
    _a = Image.fromarray(a)
    print('Saving {} to {}...'.format(img_fname,img_dest),flush=True)
    _a.save(img_dest+"_l.jpg")
    _b = Image.fromarray(np.uint8(b))                                                    
    _b.save(img_dest+"_r.jpg")

    #st = cv2.imwrite(os.path.join(outdir,image_name[:-4]+"_l.jpg"),a)
    #st = cv2.imwrite(os.path.join(outdir,image_name[:-4]+"_r.jpg"),b)

    print('{} processed'.format(img_fname),flush=True)    

    #return (results,img_dest,_a,_b)
    return results

def create_base_dir(dest):
    basedir = os.path.dirname(dest)
    if not os.path.exists(basedir):
        os.makedirs(basedir)

result_list = []
img_list = []
def log_result(results):
    #result,img_name,a,b = results
    result = results

    #print(img_name)
    #a.save(img_name+"_l.jpg")
    #b.save(img_name+"_r.jpg")
    #img_list.append((img_name,a,b))

    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    for i in range(len(result)):
        if result[i]:
            for j in range(len(result[i])):
                result_list.append(result[i][j])

def run_detector_on_dataset():
    
    args = parse_args()
    
    input_dir = args.input_img_dir
    #output_dir = args.output_dir
    print('Input directory: {}'.format(input_dir))
    eval_imgs = glob.glob(os.path.join(input_dir, '*.jpg'))
    print('Number of images found: {}'.format(len(eval_imgs)))
    
    model = init_detector(args.config, args.checkpoint, device=torch.device('cuda:0'))

    pool = ThreadPool(cpu_count())  
    
    out_dir = '/'.join(args.out.split('/')[:-1])
    folder = input_dir.split('/')[-1].replace('_Output','')
    
    #print(os.getcwd())
    #print(folder)
    
    if folder not in os.listdir(out_dir):
        os.mkdir(os.path.join(out_dir,folder))
    out_dir = os.path.join(out_dir,folder)
    print('Output directory: {}'.format(out_dir))

    for im in eval_imgs:
        pool.apply_async(crop_and_send,args= (model,im,out_dir,),callback=log_result)
        #result = crop_and_send(model,im)
        
    pool.close()
    pool.join()
    
    mmcv.dump(result_list, args.out)

    #for img_name,a,b in img_list:
    #    a.save(img_name+"_l.jpg")
    #    b.save(img_name+"_r.jpg")
        

def run_detector_on_dataset_single_proc():

    args = parse_args()

    input_dir = args.input_img_dir
    #output_dir = args.output_dir
    print('Input directory: {}'.format(input_dir))
    eval_imgs = glob.glob(os.path.join(input_dir, '*.jpg'))
    print('Number of images found: {}'.format(len(eval_imgs)))

    model = init_detector(args.config, args.checkpoint, device=torch.device('cuda:0'))

    out_dir = '/'.join(args.out.split('/')[:-1])
    folder = input_dir.split('/')[-1].replace('_Output','')

    #print(os.getcwd())
    #print(folder)

    if folder not in os.listdir(out_dir):
        os.mkdir(os.path.join(out_dir,folder))
    out_dir = os.path.join(out_dir,folder)
    print('Output directory: {}'.format(out_dir))

    for im in eval_imgs:
        result = crop_and_send(model,im,out_dir)
    
        for i in range(len(result)):
            if result[i]:
                for j in range(len(result[i])):
                    result_list.append(result[i][j])

    mmcv.dump(result_list, args.out)

if __name__ == '__main__':
    run_detector_on_dataset()
