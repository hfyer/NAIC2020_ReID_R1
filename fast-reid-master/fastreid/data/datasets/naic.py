# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY



@DATASET_REGISTRY.register()
class NAIC_All(ImageDataset):
    """NAIC.
    #NAIC 2020 R1 + NAIC 2019 R1 R2
    """
  
    dataset_dir = ''
    dataset_name = "naic_all"

    def __init__(self,root='/home/reid/PCL/dataset/',**kwargs):
        self.root = root
        self.train_dir_naic20r1 = osp.join(self.root,'naic2020_round1/train')
        # self.train_dir_naic19r1 = osp.join(self.root,'naic2019/round1')   # 和2019 r2存在数据重复 不应该加入 
        self.train_dir_naic19r2 = osp.join(self.root,'naic2019/round2') 
        self.query_dir = osp.join(self.root,'naic2020_round1/image_A/query') 
        self.gallery_dir = osp.join(self.root,'naic2020_round1/image_A/gallery') 
        #use all data
        self.naic_full = True


        query = self.process_dir(self.query_dir)
        gallery = self.process_dir(self.gallery_dir)

        if self.naic_full:

            train_20r1 = self.process_dir(self.train_dir_naic20r1)
            # train_19r1 = self.process_dir(self.train_dir_naic19r1)
            train_19r2 = self.process_dir(self.train_dir_naic19r2)
            # train = self.merge(train_20r1,train_19r1,train_19r2)
            train = self.merge(train_20r1,train_19r2)
        else:
            train = self.process_dir(self.train_dir_naic20r1)
        


        super(NAIC_All, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path):
        data = []
        if 'query' in dir_path or 'gallery' in dir_path:
            files = glob.glob(osp.join(dir_path, '*.png'))
            for fpath in files:
                data.append((fpath,0,0))
            return data
        if '2020' in dir_path:
            img_paths = osp.join(dir_path, 'images')
            labels = open(osp.join(dir_path, 'label.txt')).readlines()
        if '2019' in dir_path:
            img_paths = osp.join(dir_path)
            labels = open(osp.join(dir_path, 'train_list.txt')).readlines()
        #count ids
        count_pids = {}
        for label in labels:
            if " " in label:
                fname, pid = label.split(' ')
            elif ":" in label:
                fname, pid = label.split(':')
            pid = int(pid)
            if pid  not in count_pids.keys():
                count_pids[pid] = 1
            else:
                count_pids[pid] +=1
        #print(len(count_pids))  19658  4768  9968
        #remove pid==1 and relabel
        pid_map={}
        for label in labels:
            if " " in label:
                fname, pid = label.split(' ')
            elif ":" in label:
                fname, pid = label.split(':')
            pid = int(pid)
            if count_pids[pid] > 1:
                if pid not in pid_map.keys():
                    pid_map[pid] = len(pid_map)
                fname = osp.join(img_paths,fname)
                data.append((fname, pid_map[pid], 0))

        return data


    def merge(self,train1,train2):
        pid_set1=set() 
        pid_set2=set() 
      
        pid_setall=set()
        train =[]
        
        for fname,pid,_ in train1:
            train.append((fname,pid,0))
            pid_set1.add(int(pid))
        train1_pid_num=len(pid_set1)
        for fname,pid,_ in train2:
            train.append((fname,pid+train1_pid_num,0))
            pid_set2.add(int(pid))
        train12_pid_num = len(pid_set1) + len(pid_set2)
        print(len(pid_set1),len(pid_set2))
        for _,pid,_ in train:
            pid_setall.add(int(pid))
        print(len(pid_setall))
        return train

#-------------------------------------------------------------------------
@DATASET_REGISTRY.register()
class NAIC_test(ImageDataset):
    """NAIC.
    # B test
    """
  
    dataset_dir = ''
    dataset_name = "naic_test"

    def __init__(self,root='/home/zhangzhengjie/workspace/image_B',**kwargs):
        self.root = root
      
        self.query_dir = osp.join(self.root,'query') 
        print(self.query_dir)
        self.gallery_dir = osp.join(self.root,'gallery') 
   

        query = self.process_dir(self.query_dir)
        gallery = self.process_dir(self.gallery_dir)

        train=[]
        super(NAIC_test, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path):
        data = []
        if 'query' in dir_path or 'gallery' in dir_path:
            files = glob.glob(osp.join(dir_path, '*.png'))
            # print(files)
            for fpath in files:
                data.append((fpath,0,0))
            return data
      