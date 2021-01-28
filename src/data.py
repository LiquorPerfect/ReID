import os
import shutil


query_path='F:/ReID/reid_myself/data/Market-1501-v15.09.15/query'
train_path='F:/ReID/reid_myself/data/Market-1501-v15.09.15/train'

for dir in os.listdir(query_path):
    for jpg in os.listdir(os.path.join(query_path,dir)):
        rm_jpg=os.path.join(train_path,dir+'/'+jpg)
        os.remove(rm_jpg)
        print("hello the world!")
        print("111111111111")