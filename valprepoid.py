import pandas as pd
from imageio import imread
import psycopg2
import numpy as np
import csv 
import pickle
import os
from imageio import imread 
import pdb
import copy
import json
import csv

def init_data():
    root_dir = os.path.abspath(os.path.dirname(__file__))
    labdata = pd.read_csv(root_dir + '/data/train/challenge-2019-classes-description-500.csv')   
    labeldict = {}
    labeldescdict = {}
    for rows in labdata.iterrows():
        labeldict[rows[1]['labelname']] = rows[1]['labeldesc']
        labeldescdict[rows[1]['labeldesc']] = rows[1]['labelname']
    labels = labdata.loc[:, ['labeldesc']]                                                                                                                                                       
    labels = list(labels['labeldesc'])                                                                               
    labels.insert(0, '__background__')                                                                                                     
    labels = tuple(labels)
    num_classes = len(labels)                                                                                            
    classdict = dict(zip(labels, range(num_classes)))

    imgdata = pd.read_csv(root_dir + '/data/valoiddata/validation-images-with-rotation.csv')
    imgs = (imgdata.loc[:, ['ImageID']]).drop_duplicates()
    img_names = list(imgs['ImageID'])
    img_size = len(img_names) 
    return img_names, img_size, labels, num_classes, classdict, labeldescdict


def load_oid_annotation(i, img, img_size, classdict):
    print("Loading annotation for image {} of {}".format(i+1, img_size))                                                                
    boxes = np.zeros((1, 4), dtype=np.float32)
    gt_classes = np.zeros((1), dtype=np.int32)
    root_dir = os.path.abspath(os.path.dirname(__file__))  
    img_path = os.path.join(root_dir + '/data/valoiddata/validation', img+'.jpg')
    try:
        img_dim = imread(img_path)
        width, height = img_dim.shape[1], img_dim.shape[0]
    except:
        with open('imgnotfound.csv', 'a') as out:
            csv_out = csv.writer(out)
            csv_out.writerow([img])
        width, height = 0, 0
                            
    flipped = False
    return {'boxes': boxes, 'gt_classes': gt_classes, 'img_id': img, 'image': img_path,  'width': width, 'height': height, 'flipped': flipped}

def gt_oid_roidb(img_names, img_size, classdict):
    root_dir = os.path.abspath(os.path.dirname(__file__))
    cache_file = os.path.join(root_dir + '/data/valoiddata/', 'valoid' + '_gt_roidb.pk')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            roidb = pickle.load(fid)
        print("OID gt roidb loaded from {}".format(cache_file))
        print("Number of images loaded from .pk file {}".format(len(roidb)))
        for i in range(len(roidb)):
            if not os.path.exists(roidb[i]['image']):
                del roidb[i]
        print("Number of images after image path check is {}".format(len(roidb)))
        return roidb
    roidb = [load_oid_annotation(i, img_names[i], img_size, classdict) for i in range(img_size)]
    with open(cache_file, 'wb') as fid:
        pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
    print("Wrote OID gt roidb to {}".format(cache_file))
    return roidb

def rank_roidb_ratio(roidb):
    ratio_large = 2
    ratio_small = 0.5
    ratio_list = []

    for i in range(len(roidb)):
        width = roidb[i]['width']
        height = roidb[i]['height']
        if height != 0:
             ratio = width/float(height)

        if ratio > ratio_large:
            roidb[i]['need_crop'] = 1
            ratio = ratio_large
        elif ratio < ratio_small:
            roidb[i]['need_crop'] = 1
            ratio = ratio_small
        else:
            roidb[i]['need_crop'] = 0

        ratio_list.append(ratio)

    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    return ratio_list[ratio_index], ratio_index

def filter_roidb(roidb):
    print("Before filtering there are {} images".format(len(roidb)))
    i = 0
    while i < len(roidb):
        if len(roidb[i]['boxes']) == 0 or roidb[i]['width'] == 0:
            del roidb[i]
            i -= 1
        i += 1
    print("After filtering there are {} images".format(len(roidb)))
    return roidb

def ret_oid_data():
    img_names, img_size, labels, num_classes, classdict, labeldescdict = init_data()
    gt_roidb = gt_oid_roidb(img_names, img_size, classdict)
    gt_roidb = filter_roidb(gt_roidb)
    ratio_list, ratio_index = rank_roidb_ratio(gt_roidb)
    return labels, num_classes, gt_roidb, ratio_list, ratio_index, labeldescdict

