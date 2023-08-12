import cv2
import numpy as np
import json
import os
from copy import deepcopy
import pandas as pd

json_path = '/Users/yangfan/Documents/MMPose/train_valid.json'
new_csv_path = '/Users/yangfan/Documents/MMPose/train_valid.csv'
src_json  = open(json_path,'r')
src_json_file = json.load(src_json)
src_annotations = src_json_file['annotations']
src_image_lists = src_json_file['images']
csv_data = []
for i in range(len(src_annotations)):
    data=[]
    annotation = src_annotations[i]
    image_id = src_image_lists[i]['file_name']
    keypoints = annotation['keypoints']
    keypoints=np.array(keypoints)
    keypoints=keypoints.reshape(-1,3)
    keypoints=keypoints[:,:2]
    keypoints=keypoints.reshape(1,-1).squeeze()
    keypoints=keypoints.tolist()
    spaceing = src_image_lists[i]['spacing']
    W = src_image_lists[i]['width']
    H = src_image_lists[i]['height']
    data.append(image_id)
    data.extend(keypoints)
    data.append(W)
    data.append(H)
    data.append(spaceing)
    csv_data.append(data)
df = pd.DataFrame(csv_data)
df.to_csv(new_csv_path,encoding = 'utf-8-sig',header=False,index=False)

# print('shape:',crop.shape)
# cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),3)
# cv2.imshow('img',image)
# cv2.circle(crop,(int(100-xoffset),int(100-yoffset)),5,(255,0,255),-1)
# cv2.imshow('crop',crop)
# cv2.waitKey()
