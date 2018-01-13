# coding: utf-8
from PIL import Image
import os
import numpy as np
import cv2

''' Some times, the boxes file has been removed accidenly,
    but the original images has been saved. We can compare
    the two versions (w/o mosaic) of images to get the bound box
'''


# Step1, Find the images, without bound box
files = ['akari_01_boxes.txt', 'ai_uehara_01_boxes.txt']
files_have_boxes = {}

for _ in files:
    fobj = open(_, 'r')
    fs = [line.strip().split(":")[0].strip() for line in fobj]
    fs = [os.path.basename(f) for f in fs]
    for f in fs:
        files_have_boxes[f] = 1 
    fobj.close()
os.mkdir('diff_images')

f_without_box = [f for f in os.listdir("mosaic_images/") if f not in files_have_boxes]


#  Step2, compare the images, find bound box for each file that has no bound box
boxes = {}
for f in f_without_box:
    raw_img = Image.open(os.path.join('raw_images', f))
    mosaic_img = Image.open(os.path.join('mosaic_images', f))
    
    raw_img = np.asarray(raw_img)
    mosaic_img = np.asarray(mosaic_img)
    diff_img = mosaic_img - raw_img
    diff_y, diff_x = np.nonzero(diff_img[:,:,0])
    first_non_zero = (diff_x.min(), diff_y.min())
    last_non_zero = (diff_x.max(), diff_y.max())
    try:
        box = [first_non_zero[0], first_non_zero[1],\
                last_non_zero[0]-first_non_zero[0], \
                last_non_zero[1] -first_non_zero[1]]
        #print(box)
        boxes[f] = box
        cv2.rectangle(diff_img, first_non_zero, last_non_zero, (255, 0,0), 2)
        diff_img = Image.fromarray(diff_img, 'RGB')
        diff_img.save(os.path.join('diff_images', f))
    except Exception, e:
        print e
        continue

# Step3. Save the bound box to a file
fobj = open('akari_01_boxes.txt.missed', 'w')
for f in boxes:
    line = "./data/anote_videos/mosaic_images/{} : {} : 0".format(f, tuple(boxes[f]))
    fobj.writelines(line + os.linesep)
fobj.close()
