import os
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage as ndi
from skimage import img_as_ubyte
from skimage.morphology import label, watershed, remove_small_objects
from skimage.segmentation import random_walker
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.exposure import equalize_adapthist
import configparser
import cv2

# config related handling
def run_once(func):
    ''' a declare wrapper function to call only once, use @run_once declare keyword '''
    def wrapper(*args, **kwargs):
        if 'result' not in wrapper.__dict__:
            wrapper.result = func(*args, **kwargs)
        return wrapper.result
    return wrapper

@run_once
def read_config():
    conf = configparser.ConfigParser()
    candidates = ['config_default.ini', 'config.ini']
    conf.read(candidates)
    return conf

config = read_config() # keep the line as top as possible

# copy from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139
class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def iou_mean(prediction, target):
    batch_size = target.shape[0]
    metric = []
    for idx in range(batch_size):
        intersection = np.logical_and(target[idx], prediction[idx])
        union = np.logical_or(target[idx], prediction[idx])
        if np.sum(union) == 0:
            iou_score = 1
        else:
            iou_score = np.sum(intersection) / np.sum(union)
        metric.append(iou_score)
    return np.mean(metric)


# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(y):
    dots = np.where(y.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(y, y_c, y_m):
    segmentation = config['post'].getboolean('segmentation')
    remove_objects = config['post'].getboolean('remove_objects')
    min_object_size = config['post'].getint('min_object_size')
    remove_fiber = config['post'].getboolean('filter_fiber')

    if segmentation:
        y, _ = partition_instances(y, y_m, y_c)
    if remove_objects:
        y = remove_small_objects(y, min_size=min_object_size)
    if remove_fiber:
        y = filter_fiber(y)
    idxs = np.unique(y) # sorted, 1st is background (e.g. 0)
    if len(idxs) == 1:
        yield []
    else:
        for idx in idxs[1:]:
            yield rle_encoding(y == idx)

# checkpoint handling
def check_ckpt_dir():
    checkpoint_dir = os.path.join('.', 'checkpoint')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

def ckpt_path(epoch=None):
    check_ckpt_dir()
    current_path = os.path.join('.', 'checkpoint', 'current.json')
    if epoch is None:
        if os.path.exists(current_path):
            with open(current_path) as infile:
                data = json.load(infile)
                epoch = data['epoch']
        else:
            return ''
    else:
        with open(current_path, 'w') as outfile:
            json.dump({
                'epoch': epoch
            }, outfile)
    return os.path.join('.', 'checkpoint', '{}.pkl'.format(epoch))

def is_best_ckpt(epoch, iou_cv):
    check_ckpt_dir()
    best_json = os.path.join('.', 'checkpoint', 'best.json')
    best_iou_cv = 0
    if os.path.exists(best_json):
        with open(best_json) as infile:
            data = json.load(infile)
            best_iou_cv = data['iou_cv']
    cv_threshold = 0.01 # tolerance of degraded CV IoU
    if iou_cv > best_iou_cv - cv_threshold:
        with open(best_json, 'w') as outfile:
            json.dump({
                'epoch': epoch,
                'iou_cv': iou_cv,
            }, outfile)
        return True
    return False

# DataParallel will change model's class name to 'dataparallel' & prefix 'module.' to existing parameters.
# Here the saved checkpoint might or might not be 'DataParallel' model (e.g. might be trained with multi-GPUs or single GPU),
# handle this variation while loading checkpoint.
# Refer to:
#   https://github.com/pytorch/pytorch/issues/4361
#   https://github.com/pytorch/pytorch/issues/3805
#   https://stackoverflow.com/questions/44230907/keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict
def _extract_state_from_dataparallel(checkpoint_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint_dict.items():
        if k.startswith('module.'):
            name = k[7:] # remove 'module.'
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


def save_ckpt(model, optimizer, epoch, iou_cv, ckpt_name):
    def do_save(filepath):
        torch.save({
            'epoch': epoch,
            'name': config['param']['model'],
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filepath)
    # check if best checkpoint
    if is_best_ckpt(epoch, iou_cv):
        filepath = os.path.join('.', ckpt_name, 'best.pkl')
        do_save(filepath)
    # # save checkpoint per n epoch
    # n_ckpt_epoch = config['train'].getint('n_ckpt_epoch')
    # if epoch % n_ckpt_epoch == 0:
    #     # filepath = ckpt_path(epoch)
    #     filepath = os.path.join('.', ckpt_name, '{}.pkl'.format(epoch))
    #     do_save(filepath)


def load_ckpt(model=None, optimizer=None, filepath=None):
    if filepath is None:
        filepath = ckpt_path()
    if not os.path.isfile(filepath):
        return 0
    print("Loading checkpoint '{}'".format(filepath))
    if torch.cuda.is_available():
        # Load all tensors onto previous state
        checkpoint = torch.load(filepath)
    else:
        # Load all tensors onto the CPU
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    epoch = checkpoint['epoch']
    if optimizer:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError as err:
            print('[WARNING]', err)
            print('[WARNING] optimizer not restored from last checkpoint, continue without previous state')

    if model:
        model.load_state_dict(_extract_state_from_dataparallel(checkpoint['model']))
        return epoch
    else:
        # build model based on checkpoint
        from model import build_model
        assert 'name' in checkpoint, "Abort! No model name in checkpoint, use ckpt.py to convert first"
        model_name = checkpoint['name']
        model = build_model(model_name)

        model.load_state_dict(_extract_state_from_dataparallel(checkpoint['model']))
        return model

# Evaluate the average nucleus size.
def mean_blob_size(image, ratio):
    label_image = label(image)
    label_counts = len(np.unique(label_image))
    #Sort Area sizes:
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    total_area = 0
    #To avoild eval_count ==0
    if int(label_counts * ratio)==0:
        eval_count = 1
    else:
        eval_count = int(label_counts * ratio)
    average_area = np.array(areas[:eval_count]).mean()
    size_index = average_area ** 0.5
    return size_index

def add_missed_blobs(full_mask, labeled_mask, edges):
    missed_mask = full_mask & ~(labeled_mask > 0)
    missed_mask = drop_small_blobs(missed_mask, 2) # bodies must be larger than 1-pixel
    if edges is not None:
        missed_markers = label(missed_mask & ~edges)
    else:
        missed_markers = label(missed_mask)
    if missed_markers.max() > 0:
        missed_markers[missed_mask == 0] = -1
        if np.sum(missed_markers > 0) > 0:
            missed_labels = random_walker(missed_mask, missed_markers)
        else:
            missed_labels = np.zeros_like(missed_markers, dtype=np.int32)
        missed_labels[missed_labels <= 0] = 0
        missed_labels = np.where(missed_labels > 0, missed_labels + labeled_mask.max(), 0)
        final_labels = np.add(labeled_mask, missed_labels)
    else:
        final_labels = labeled_mask
    return final_labels

def drop_small_blobs(mask, min_size):
    mask = remove_small_objects(mask, min_size=min_size)
    return mask

def filter_fiber(blobs):
    objects = [(obj.area, obj.eccentricity, obj.label) for obj in regionprops(blobs)]
    objects = sorted(objects, reverse=True) # sorted by area in descending order
    # filter out the largest one which is (1) 5 times larger than 2nd largest one (2) eccentricity > 0.95
    if len(objects) > 1 and objects[0][0] > 5 * objects[1][0] and objects[0][1] > 0.95:
        print('\nfilter suspecious fiber', objects[0])
        blobs = np.where(blobs==objects[0][2], 0, blobs)
    return blobs

def partition_instances(raw_bodies, raw_markers=None, raw_edges=None):
    threshold=config['param'].getfloat('threshold')
    threshold_edge = config['param'].getfloat('threshold_edge')
    threshold_marker = config['param'].getfloat('threshold_mark')
    policy = config['post']['policy']
    min_object_size = config['post'].getint('min_object_size')

    # Random Walker fails for a 1-pixel seed, which is exactly on top of a 1-pixel semantic mask.
    # https://github.com/scikit-image/scikit-image/issues/1875
    # Workaround by eliminating 1-pixel semantic mask first.
    bodies = raw_bodies > threshold
    bodies = drop_small_blobs(bodies, 2) # bodies must be larger than 1-pixel
    markers = None if raw_markers is None else (raw_markers > threshold_marker)
    edges = None if raw_edges is None else (raw_edges > threshold_edge)

    if markers is not None and edges is not None:
        markers = (markers & ~edges) & bodies
        # remove artifacts caused by non-perfect (mask - contour)
        markers = drop_small_blobs(markers, min_object_size)
        markers = label(markers)
    elif markers is not None:
        markers = markers & bodies
        markers = label(markers)
    elif edges is not None:
        # to remedy error-dropped edges around the image border (1 or 2 pixels holes)
        box_bodies = bodies.copy()
        h, w = box_bodies.shape
        box_bodies[0:2, :] = box_bodies[h-2:, :] = box_bodies[:, 0:2] = box_bodies[:, w-2:] = 0
        markers = box_bodies & ~edges
        markers = drop_small_blobs(markers, min_object_size)
        markers = label(markers)
    else:
        threshold=config['param'].getfloat('threshold')
        size_scale=config['post'].getfloat('seg_scale')
        ratio=config['post'].getfloat('seg_ratio')
        size_index = mean_blob_size(bodies, ratio)
        """
        Add noise to fix min_distance bug:
        If multiple peaks in the specified region have identical intensities,
        the coordinates of all such pixels are returned.
        """
        noise = np.random.randn(bodies.shape[0], bodies.shape[1]) * 0.1
        distance = ndi.distance_transform_edt(bodies)+noise
        # 2*min_distance+1 is the minimum distance between two peaks.
        local_maxi = peak_local_max(distance, min_distance=(size_index*size_scale), exclude_border=False,
                                    indices=False, labels=bodies)
        markers = label(local_maxi)

    if policy == 'ws':
        seg_labels = watershed(-ndi.distance_transform_edt(bodies), markers, mask=bodies)
    elif policy == 'rw':
        markers[bodies == 0] = -1
        if np.sum(markers > 0) > 0:
            seg_labels = random_walker(bodies, markers)
        else:
            seg_labels = np.zeros_like(markers, dtype=np.int32)
        seg_labels[seg_labels <= 0] = 0
        markers[markers <= 0] = 0
    else:
        raise NotImplementedError("Policy not implemented")
    # final_labels = add_missed_blobs(bodies, seg_labels, edges)
    return seg_labels, markers

def clahe(x):
    '''
    return PIL image or numpy array
    '''
    is_pil = isinstance(x, Image.Image)
    if is_pil:
        x = np.asarray(x, dtype=np.uint8)
    x = equalize_adapthist(x)
    x = img_as_ubyte(x)
    if is_pil:
        x = Image.fromarray(x)
    return x

def filter_by_group(root, use_filter):
    c = config['dataset']
    csv = c.get('csv_file')
    files = next(os.walk(root))[1]
    files.sort()
    # if no csv file, return real file list
    if not os.path.isfile(csv) or not use_filter:
        return pd.DataFrame({'image_id': files, 'group': 0})
    # read csv and do sanity check with existing files
    df = pd.read_csv(csv)
    assert len(df) > 0
    files = next(os.walk(root))[1]
    df = df.loc[ df['image_id'].isin(files) ]
    print("Number of existed file in csv file:", len(df))
    # filter by group
    group = []
    for g in ['source', 'major_category', 'sub_category']:
        filter = c.get(g)
        if filter is not None:
            group.append(g)
            filter = [e.strip() for e in filter.split(',')]
            # apply filter
            df = df.loc[ df[g].isin(filter) ]
    # verbose check groupby, which will be used as distribution weight
    if len(group) > 0:
        group = df.groupby(group)
        print("Group by white-list:")
        print(group['image_id'].count().reset_index())
        # assign group id to new column 'group'
        df['group'] = group.ngroup()
    else:
        # assign as same group id
        df['group'] = 0
    # final list of valid training data
    print("Number of white-list file in csv file:", len(df))
    return df[['image_id','group']].reset_index(drop=True)

def _drawContours(blobLabel):
    mask=np.zeros((512,512))
    if blobLabel['Value'] == 0:
        contours=list(map(lambda point_str:[int(point_str.split(', ')[0]), int(point_str.split(', ')[1])],blobLabel['Blob']))
        cv2.drawContours(mask, np.array([contours]), -1, 1, -1)
    else:
        contours=list(map(lambda point_str:[int(point_str.split(', ')[0]), int(point_str.split(', ')[1])],blobLabel['Blob']))
        cv2.drawContours(mask, np.array([contours]), -1, 2, -1)
    return mask

def load_json_label(json_file):
    with open(json_file, 'r') as f:
        content = f.read()
        if content.startswith(u'\ufeff'):
            json_dict = content.encode('utf8')[3:].decode('utf8')
    blobLabels = json.loads(json_dict)

    #multi mask  
    masks=map(_drawContours,blobLabels)
    return masks
