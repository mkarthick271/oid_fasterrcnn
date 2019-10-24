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
    connection = psycopg2.connect(user='postgres', password='cts-0000', host='127.0.0.1', port='5432', database='oid')
    root_dir = os.path.abspath(os.path.dirname(__file__))
    cursor = connection.cursor()                  
    cursor.execute("select * from datasetbbox250")                                            
    imgbbox = cursor.fetchall()                                                              
    imgbbox = pd.DataFrame(imgbbox)                                                      
    imgbbox.columns=['imageid', 'labeldesc', 'xmin1', 'xmax1', 'ymin1', 'ymax1']                                             

    labdata = pd.read_csv(root_dir + '/data/train/challenge-2019-classes-description-500.csv')   
    labeldict = {}
    for rows in labdata.iterrows():
        labeldict[rows[1]['labelname']] = rows[1]['labeldesc']
    labels = labdata.loc[:, ['labeldesc']]                                                                                                        
           
    labels = list(labels['labeldesc'])                                                                               
    labels.insert(0, '__background__')                                                                                                     
    labels = tuple(labels)
    num_classes = len(labels)
                                                                                                                                                          
    classdict = dict(zip(labels, range(num_classes)))

    imgdata = pd.read_csv(root_dir + '/data/train/dataset250.csv')
    imgs = (imgdata.loc[:, ['imageid']]).drop_duplicates()
    img_names = list(imgs['imageid'])
    img_size = len(img_names) 

    with open(root_dir + '/data/train/challenge-2019-label500-hierarchy.json') as f:
        hierarchy = json.load(f)
    keyed_parent, keyed_child, _ = build_plain_hierarchy(hierarchy, skip_root=True)
    desc_parents = {}
    desc_children = {}
    for rows in labdata.iterrows():
        children = keyed_parent[rows[1]['labelname']]
        parents = keyed_child[rows[1]['labelname']]
        childlist = []
        parentlist = []
        for child in children:
            childlist.append(labeldict[child])
        for parent in parents:
            parentlist.append(labeldict[parent])
        desc_parents[rows[1]['labeldesc']] = parentlist
        desc_children[rows[1]['labeldesc']] = childlist
    keyed_parent = desc_parents
    keyed_child = desc_children
#    with open ('hierparents.csv', 'w') as out:
#        csv_out = csv.writer(out)
#        for dics in sorted(keyed_parent.keys()):
#            csv_out.writerow(([dics], keyed_parent[dics]))
#            print(dics, '\t', keyed_parent[dics])
#    with open ('hierchildren.csv', 'w') as out:
#        csv_out = csv.writer(out)
#        for dics in sorted(keyed_child.keys()):
#            csv_out.writerow(([dics], keyed_child[dics]))
#            print(dics, '\t', keyed_child[dics])

    #print(keyed_parent['Infant bed'])
    #pdb.set_trace()
    return img_names, imgbbox, img_size, labels, num_classes, keyed_parent, keyed_child, classdict

def update_dict(initial_dict, update):
    for key, value_list in update.items():
        if key in initial_dict:
            initial_dict[key].update(value_list)
        else:
            initial_dict[key] = set(value_list)

def build_plain_hierarchy(hierarchy, skip_root=False):
    all_children = set([])
    all_keyed_parent = {}
    all_keyed_child = {}
    if 'Subcategory' in hierarchy:
        for node in hierarchy['Subcategory']:
            keyed_parent, keyed_child, children = build_plain_hierarchy(node)
            # Update is not done through dict.update() since some children have multi-
            # ple parents in the hiearchy.
            update_dict(all_keyed_parent, keyed_parent)
            update_dict(all_keyed_child, keyed_child)
            all_children.update(children)
                                                
    if not skip_root:
        all_keyed_parent[hierarchy['LabelName']] = copy.deepcopy(all_children)
        all_children.add(hierarchy['LabelName'])
        for child, _ in all_keyed_child.items():
            all_keyed_child[child].add(hierarchy['LabelName'])
        all_keyed_child[hierarchy['LabelName']] = set([])

    return all_keyed_parent, all_keyed_child, all_children


def load_oid_annotation(i, img, imgbbox, img_size, keyed_parent, keyed_child, classdict):
    print("Loading annotation for image {} of {}".format(i+1, img_size))
    perimgbbox = pd.DataFrame(imgbbox[(imgbbox.imageid==img)])
    perimgbbox.columns=['imageid', 'labeldesc', 'xmin1', 'xmax1', 'ymin1', 'ymax1']                             
    boxes = np.zeros((perimgbbox.shape[0], 4), dtype=np.float32)
    gt_classes = np.zeros((perimgbbox.shape[0]), dtype=np.int32)
    root_dir = os.path.abspath(os.path.dirname(__file__)) 
    img_path = os.path.join(root_dir + '/data/train/dataset250', img+'.jpg')
    try:
        img_dim = imread(img_path)
        width, height = img_dim.shape[1], img_dim.shape[0]
    except:
        with open('imgnotfound.csv', 'a') as out:
            csv_out = csv.writer(out)
            csv_out.writerow([img])
        width, height = 0, 0

    for ix, row  in enumerate(perimgbbox.iterrows()):
        x1 = row[1]['xmin1'] * width
        y1 = row[1]['ymin1'] * height
        x2 = row[1]['xmax1'] * width
        y2 = row[1]['ymax1'] * height
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = classdict[row[1]['labeldesc']]
        parents = keyed_parent[row[1]['labeldesc']]
        for index in range(len(parents)):
            boxes = np.append(boxes, [[x1, y1, x2, y2]], 0)
            gt_classes = np.append(gt_classes, [classdict[parents[index]]], 0)
                            
    flipped = False
    return {'boxes': boxes, 'gt_classes': gt_classes, 'img_id': img, 'image': img_path,  'width': width, 'height': height, 'flipped': flipped}

def gt_oid_roidb(img_names, imgbbox, img_size, keyed_parent, keyed_child, classdict):
    root_dir = os.path.abspath(os.path.dirname(__file__)) 
    cache_file = os.path.join(root_dir + '/data/train/', 'oid' + '_gt_roidb.pk')
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
    roidb = [load_oid_annotation(i, img_names[i], imgbbox, img_size, keyed_parent, keyed_child, classdict) for i in range(img_size)]
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
    img_names, imgbbox, img_size, labels, num_classes, keyed_parent, keyed_child, classdict = init_data()
    gt_roidb = gt_oid_roidb(img_names, imgbbox, img_size, keyed_parent, keyed_child, classdict)
    gt_roidb = filter_roidb(gt_roidb)
    ratio_list, ratio_index = rank_roidb_ratio(gt_roidb)
    roidb_max_gt = 0
    for i in range(len(gt_roidb)):
        max_gt = gt_roidb[i]['boxes'].shape[0]
        if max_gt > roidb_max_gt:
            roidb_max_gt = max_gt
    return labels, num_classes, gt_roidb, ratio_list, ratio_index, roidb_max_gt

