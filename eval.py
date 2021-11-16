import tensorflow as tf
from tensorflow.keras.models import Model
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from encoder_decoder_Architectures import ViT_b16_US_decoder, ViT_s16_US_decoder2,  ViT_s16_deconv_decoder, ViT_t16_US_decoder2, ViT_t16_deconv_decoder, ViT_t16_DS_decoder

dataset='NYUV2'
errors = []
#load the pre-trained model 
model = ViT_b16_US_decoder(dataset=dataset)
# model = ViT_s16_US_decoder2(dataset='NYUV2')
# model = ViT_s16_deconv_decoder(dataset='NYUV2')
# model = ViT_t16_US_decoder2(dataset='NYUV2')
# model = ViT_t16_deconv_decoder(dataset='NYUV2')
# model = ViT_t16_DS_decoder(dataset='NYUV2')

model.summary()

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

imgs = []
depths = []
anno_val = []

images_path = 'E:/NYU depthv2/images/'
depth_path = 'E:/NYU depthv2/depth2/'

with open("test.txt", "r") as f:
    val_data = f.readlines()
for data in val_data:
    anno_val.append(data)

for file in anno_val:
    #print(file)
    if  dataset=='NYUV2':
        depth = np.load(depth_path+file.split('\n')[0]+'.npy')*10.0
        depth = cv2.resize(depth, (448, 448))
        plt.imsave('E:/NYU depthv2/results_trans/gt/'+file.split('\n')[0]+'.png',depth, cmap= 'magma')
        gt = depth
        #end = time.time()
        img = cv2.imread(images_path+file.split('\n')[0]+'.png')
        img = cv2.resize(img, (448, 448))
        img = img / 255.
        img= np.expand_dims(img, axis=0)
        pred1 = model.predict(img)
        #now = time.time()
        #print(now - end)
        pred = pred1[0,:,:,0]
        mask = np.logical_and(pred, gt)
        errors.append(compute_errors(gt[mask], pred[mask]))
    elif dataset='CS':
        depth_png = cv2.imread(file, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_png = cv2.resize(depth_png, dsize=(train_w , train_h), interpolation=cv2.INTER_NEAREST)
        depth_png[depth_png > 0] = (depth_png[depth_png > 0] - 1) / 256
        gt = depth_png
        #start = time.time()
        img = cv2.imread(images_path+file.split('/')[-3]+'/'+file.split('/')[-2]+'/'+file.split('/')[-1].replace('disparity.png','leftImg8bit.png'))
        img = cv2.resize(img, (train_w , train_h))
        img = img / 255.
        img= np.expand_dims(img, axis=0)
        pred1 = model.predict(img)
        #now = time.time()
        #print(now-start)
        pred = pred1[0,:,:,0]
        mask = np.logical_and(pred, gt)
        errors.append(compute_errors(gt[mask], pred[mask]))
     else:
        print('not supported dataset')
        break
    
mean_errors = np.array(errors).mean(0)  
print(mean_errors)
