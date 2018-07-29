#crop the images produced by vann
import cv2
import sys
import os
import numpy as np
from PIL import Image

from vann import mkdir

def isvalid_box(box):
    return len(box) == 4 and box[2] >= 0 and box[3] >= 0

def parse_annotations(boxes_file):
    anns = {}
    invalid_cnt = 0
    with open(boxes_file, 'r') as fobj:
        for line in fobj:
            f, box, iou = line.strip().split(":")
            f = os.path.basename(f)
            f = f.strip()
            box = eval(box)
            if not isvalid_box(box):
                invalid_cnt += 1
                print ("Invalid line:{}".format(line))
                continue
            #may be negative, but that's fine, we can make it zero
            box = [max(int(_), 0) for _ in box]
            anns[f] = (box, iou)
    print( "Invalid box entry count:{}".format(invalid_cnt))
    return anns

def main():
    boxes_file_dir = sys.argv[1]
    ann_basedir = sys.argv[2]
    output_dir = sys.argv[3]
    print boxes_file_dir, ann_basedir, output_dir

    assert output_dir != ann_basedir, "This would over write the original data"

    mkdir(output_dir)
    mkdir(output_dir, "mosaic_images")
    mkdir(output_dir, "raw_images")
    output_dir, "mosaic_images"

    def process_one_file(boxes_file):
        anns = parse_annotations(boxes_file)

        invalid_file = 0
        for f in anns:
            mosaic_img = os.path.join(ann_basedir, 'mosaic_images', f)
            raw_img = os.path.join(ann_basedir, 'raw_images', f)
            if not os.path.exists(mosaic_img):
                print ("%s no exist" % mosaic_img)
                invalid_file += 1
                continue
            if not os.path.exists(raw_img):
                print ("%s no exist" % raw_img)
                invalid_file += 1
                continue
            mosaic_img = Image.open(mosaic_img)
            raw_img = Image.open(raw_img)
            orig_box = anns[f][0]
            #adapt the box to proper scale ans ratio

            # make it 1:1
            size = min(max(orig_box[3], orig_box[2]), min(mosaic_img.size[:2]) )
            orig_box[0] -= abs(size-orig_box[2])/2
            orig_box[1] -= abs(size-orig_box[3])/2
            orig_box[0] = max(orig_box[0], 0)
            orig_box[1] = max(orig_box[1], 0)
            orig_box[3] = orig_box[2] = size

            #adopt principle is the cropped image not bigger than
            #original image, and no padding is needed
            SIZE = min(400, min(mosaic_img.size[:2]))
            need_resize = True
            assert orig_box[2] == orig_box[3]
            if orig_box[2] < SIZE:
                orig_box[0] -= (SIZE-orig_box[2])/2
                orig_box[1] -= (SIZE-orig_box[2])/2
                orig_box[2] = orig_box[3] = SIZE
                need_resize = False
                orig_box[1] = max(orig_box[1], 0)
                orig_box[0] = max(orig_box[0], 0)

            #move to proper location
            assert orig_box[2] == orig_box[3] and  \
                 orig_box[2] <= mosaic_img.size[0] and \
                 orig_box[3] <= mosaic_img.size[1] 
            if orig_box[0] + orig_box[2] > mosaic_img.size[0]:
                orig_box[0] = mosaic_img.size[0] - orig_box[2]
            if orig_box[1] + orig_box[3] > mosaic_img.size[1]:
                orig_box[1] = mosaic_img.size[1] - orig_box[3]

            box = orig_box
            #crop image
            mosaic_img = mosaic_img.crop((box[0], box[1], box[0]+box[2], box[1]+box[3]))
            raw_img = raw_img.crop((box[0], box[1], box[0]+box[2], box[1]+box[3]))
            #if need_resize:
            #    mosaic_img = mosaic_img.resize((SIZE, SIZE))
            #    raw_img = raw_img.resize((SIZE, SIZE))
            mosaic_img.save(os.path.join(output_dir, 'mosaic_images', f))
            raw_img.save(os.path.join(output_dir, "raw_images", f))

        print ("Numer of files missed: %d" % invalid_file)

    for f in os.listdir(boxes_file_dir):
        boxes_file = os.path.join(boxes_file_dir, f)
        if os.path.isfile(boxes_file):
            process_one_file(boxes_file)
if __name__ == "__main__":
    main()
