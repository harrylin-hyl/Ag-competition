#encoding=utf-8
import os
import openslide
import numpy as np
import cv2
from PIL import Image

IMAGE_SIZE = 512
STRIDE = 512
SAVE_DIR = './round1_test/image_3'
IMAGE_3_W, IMAGE_3_H = 37241,19903
IMAGE_4_W, IMAGE_4_H = 25936,28832

def train_split(wsi_path, label_path):
    index = 0
    patch_prefix = wsi_path.split('.')[0].split('/')[-1]
    wsi_slide = openslide.open_slide(wsi_path)
    label_slide = openslide.open_slide(label_path)
    w,h = wsi_slide.level_dimensions[0]
    assert w == label_slide.level_dimensions[0][0]
    assert h == label_slide.level_dimensions[0][1]
    X_pix = np.arange(0, w//STRIDE)
    Y_pix = np.arange(0, h//STRIDE)
    X = np.tile(X_pix, len(Y_pix))
    Y = np.repeat(Y_pix, len(X_pix))
    for x, y in zip(X, Y):
        label_patch = label_slide.read_region((x * STRIDE, y * STRIDE), 0, (IMAGE_SIZE, IMAGE_SIZE))
        wsi_patch = wsi_slide.read_region((x * STRIDE, y * STRIDE), 0, (IMAGE_SIZE, IMAGE_SIZE))
        label_patch = label_patch.convert('L')
        wsi_patch = wsi_patch.convert('RGB')
        if len(np.where(np.array(label_patch)!=0)[0]) == 0:
            # all is black
            if len(np.where(np.array(wsi_patch)!=0)[0]) == 0:
                continue
            # all is background, save to background
            wsi_patch.save(SAVE_DIR+'/background/'+patch_prefix+'_'+str(x)+'_'+str(y)+'.jpg', 'PNG')
        else:
            wsi_patch.save(SAVE_DIR+'/image/'+patch_prefix+'_'+str(x)+'_'+str(y)+'.jpg', 'PNG')
            label_patch.save(SAVE_DIR+'/label/'+patch_prefix+'_'+str(x)+'_'+str(y)+'.jpg', 'PNG')
        index += 1
        if index % 100 == 0:
            print(index)
        wsi_patch.close()
        label_patch.close()
    print('index:', index)

def test_split(wsi_path):
    index = 0
    patch_prefix = wsi_path.split('/')[-1].split('.')[0]
    wsi_slide = openslide.open_slide(wsi_path)
    w,h = wsi_slide.level_dimensions[0]
    X_pix = np.arange(0, w//STRIDE)
    Y_pix = np.arange(0, h//STRIDE)
    X = np.tile(X_pix, len(Y_pix))
    Y = np.repeat(Y_pix, len(X_pix))
    for x, y in zip(X, Y):
        wsi_patch = wsi_slide.read_region((x * STRIDE, y * STRIDE), 0, (IMAGE_SIZE, IMAGE_SIZE))
        wsi_patch = wsi_patch.convert('RGB')
        # all is black
        if len(np.where(np.array(wsi_patch)!=0)[0]) == 0:
            continue
        else:
            wsi_patch.save(SAVE_DIR+'/'+patch_prefix+'/'+str(x)+'_'+str(y)+'.jpg', 'PNG')
        index += 1
        if index % 100 == 0:
            print(index)
        wsi_patch.close()
    print('index:', index)    


def test_converge(img_dir):
    if img_dir.split('/')[-1] == 'image_3':
        w,h = IMAGE_3_W, IMAGE_3_H
        save_path = './submit/image_3_predict.png'
    elif img_dir.split('/')[-1] == 'image_4':
        w,h = IMAGE_4_W, IMAGE_4_H
        save_path = './submit/image_4_predict.png'
    else:
        print('path erro!')
    img_list = os.listdir(img_dir)
    output = np.zeros((h,w), dtype=np.uint8)
    for img_name in img_list:
        loc_x, loc_y = int(img_name.split('.')[0].split('_')[0]), int(img_name.split('.')[0].split('_')[1])
        pred = np.array(Image.open((os.path.join(img_dir, img_name))))
        output[loc_y*IMAGE_SIZE:(loc_y+1)*IMAGE_SIZE, loc_x*IMAGE_SIZE:(loc_x+1)*IMAGE_SIZE] = pred
    img = Image.fromarray(output, mode='L')
    img.save(save_path)


if __name__ == '__main__':
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 3000000000
    import matplotlib.pyplot as plt
    # wsi_path = './test/image_4.png'

    # test_split(wsi_path)
    img_dir = './output/image_4'
    test_converge(img_dir)
    
