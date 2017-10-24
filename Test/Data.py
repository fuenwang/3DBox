import os
import cv2
import numpy as np
from collections import OrderedDict

ROOT = './data'
LABEL_PATH = '%s/labels'%ROOT
FRAME_PATH = '%s/frames'%ROOT

def ReadLabel(ID):
    with open('%s/%s.txt'%(LABEL_PATH, ID), 'r') as f:
        data = OrderedDict()
        for line in f:
            line = line[:-1].split('\t')
            frame_id = line[0]
            track_id = line[1]
            top_x = int(line[3])
            top_y = int(line[4])
            bottom_x = int(line[5])
            bottom_y = int(line[6])

            top = (top_x, top_y)
            bottom = (bottom_x, bottom_y)

            if track_id not in data:
                data[track_id] = OrderedDict()
            data[track_id][frame_id] = OrderedDict((('top', top), ('bottom', bottom)))
    return data

class Data:
    def __init__(self, ID):
        self.labels = ReadLabel(ID)
        frame_path = '%s/%s'%(FRAME_PATH, ID)
        img_lst = [x for x in sorted(os.listdir(frame_path))]
        self.img = OrderedDict()
        for img_name in img_lst:
            img_id = img_name.split('.')[0]
            img_path = '%s/%s'%(frame_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            self.img[img_id] = img.astype(np.float32) / 255
        self.track_lst = sorted(self.labels.keys())
        self.frame_lst = sorted(self.img.keys())

        self.result = OrderedDict()
        for one in self.track_lst:
            self.result[one] = OrderedDict()

        self.now_track = 0
        self.now_frame = 0

    def Next(self):
        track_id = self.track_lst[self.now_track]
        frame_id = self.labels[track_id].keys()[self.now_frame]
        update_info = (track_id, frame_id) 
        top = self.labels[track_id][frame_id]['top']
        bottom = self.labels[track_id][frame_id]['bottom']
        img = self.img[frame_id]

        crop = img[top[1]:bottom[1]+1, top[0]:bottom[0]+1, :]

        if self.now_frame == len(self.labels[track_id].keys()) - 1:
            self.now_frame = 0
            self.now_track += 1
        else:
            self.now_frame += 1

        if self.now_track == len(self.track_lst):
            stop = True
        else:
            stop = False

        crop_resize = cv2.resize(crop, (224, 224), cv2.INTER_CUBIC)
        batch = np.zeros([1, 3, 224, 224], np.float64)
        batch[:, 0, :, :] = crop_resize[:, :, 2]
        batch[:, 1, :, :] = crop_resize[:, :, 1]
        batch[:, 2, :, :] = crop_resize[:, :, 0]
        return batch, stop, crop, update_info
    
    def Update(self, info, orientation, dimension, conf):
        track_id = info[0]
        frame_id = info[1]

        self.result[track_id][frame_id] = {'orientation':orientation, 'dimension':dimension, 'confidence': conf}

    def WriteFile(self, f_path):
        tmp = OrderedDict()
        #print self.result.keys()
        for track_id in self.result:
            for frame_id in self.result[track_id]:
                if frame_id not in tmp:
                    tmp[frame_id] = OrderedDict()
                tmp[frame_id][track_id] = self.result[track_id][frame_id]
        tmp = OrderedDict(sorted(tmp.items()))
        with open(f_path, 'w') as f:
            for frame_id in tmp:
                for track_id in tmp[frame_id]:
                    a = tmp[frame_id][track_id]
                    ori = a['orientation']
                    dim = a['dimension']
                    conf = a['confidence']
                    s = '%s\t%s\t%lf\t%f\t%f\t%f\t%f\n'%(frame_id, track_id, ori, dim[0], dim[1], dim[2], conf)
                    f.write(s)


if __name__ == '__main__':
    data = Data('000001')

    stop = False
    #data.now_track = 3
    #data.now_frame = 9
    while not stop:
        #print (data.now_track, data.now_frame)
        [img, stop, crop, info] = data.Next()
        cv2.namedWindow('GG')
        cv2.imshow('GG', crop)
        #cv2.imshow('GG', big)
        cv2.waitKey(0)
        #exit()











