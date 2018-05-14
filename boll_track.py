#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
boll_track.py [<video_source>]


Keys
----
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import video
from common import anorm2, draw_str
from time import clock
import time
import imutils

import os
from darkflow.net.build import TFNet
import sys
    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
        
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 400,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

class App:
    def __init__(self, video_src):
        self.track_len = 200
        self.detect_interval = 10
        self.tracks = []
        self.bolls = []
        self.cam = video.create_capture(video_src)
        self.frame_idx = 0
    
    def get_centroid(self,x, y, w, h):
        x1 = int(w / 2)
        y1 = int(h / 2)
        cx = x + x1
        cy = y + y1
        return (cx, cy)
    
    def get_bbox_centre(self,x, y, xj, yj):
        cx = x + int((xj-x) / 2)
        cy = y + int((yj-y) / 2)
        return (cx, cy)
    
    def get_bbox(self,x, y, cx, cy):
        xj = int(2*cx + x)
        yj = int(2*cy + y)
        return (x, y,xj,yj)

    def non_max_suppression (self,results,overlap):
        x1 =[]
        y1 = []
        x2 = []
        y2 = []
        score = []
        area = []
        if not results:
            pick = []
        else:
            for j in range(0, (len(results)-1)):
                x1.append(results[j]['topleft']['x'])
                y1.append(results[j]['topleft']['y'])
                x2.append(results[j]['bottomright']['x'])
                y2.append(results[j]['bottomright']['y'])
                score.append(results[j]['confidence'])
                area.append((results[j]['bottomright']['x']-results[j]['topleft']['x']+1)*(results[j]['bottomright']['y']-results[j]['topleft']['y']+1))
            I = np.argsort(score)
            pick = []
            count = 1
            while (I.size!=0):
                last = I.size
                i = I[last-1]
                pick.append(i)
                suppress = [last-1]
                for pos in range(last-1):
                    j = I[pos]
                    xx1 = max(x1[i],x1[j])
                    yy1 = max(y1[i],y1[j])
                    xx2 = min(x2[i],x2[j])
                    yy2 = min(y2[i],y2[j])
                    w = xx2-xx1+1
                    h = yy2-yy1+1
                    if (w>0 and h>0):
                        o = w*h/area[j]
                       # print("Overlap is",o, " at i ", i, " and j ", j)
                        if (o >overlap or (x1[i]>=x1[j] and y1[i]>=y1[j] and x2[i]<=x2[j] and y2[i]<=y2[j])):
                            suppress.append(pos)
                I = np.delete(I,suppress)
                count = count + 1
        return pick

    def non_max_suppression_no_dict (self,results,overlap):
        x1 =[]
        y1 = []
        x2 = []
        y2 = []
        score = []
        area = []
        if not results:
            pick = []
        else:
            for j in range(0, (len(results)-1)):
                contour, centroid, contour_area = results[j]
                x, y, xw, yh = contour
                x1.append(x)
                y1.append(y)
                x2.append(xw)
                y2.append(yh)
                score.append(contour_area)
                area.append((xw-x+1)*(yh-y+1))
            I = np.argsort(score)
            pick = []
            count = 1
            while (I.size!=0):
                last = I.size
                i = I[last-1]
                pick.append(i)
                suppress = [last-1]
                for pos in range(last-1):
                    j = I[pos]
                    xx1 = max(x1[i],x1[j])
                    yy1 = max(y1[i],y1[j])
                    xx2 = min(x2[i],x2[j])
                    yy2 = min(y2[i],y2[j])
                    w = xx2-xx1+1
                    h = yy2-yy1+1
                    if (w>0 and h>0):
                        o = w*h/area[j]
                       # print("Overlap is",o, " at i ", i, " and j ", j)
                        if (o >overlap or (x1[i]>=x1[j] and y1[i]>=y1[j] and x2[i]<=x2[j] and y2[i]<=y2[j])):
                            suppress.append(pos)
                I = np.delete(I,suppress)
                count = count + 1
        return pick
        
    def run(self):
        option = {
        		'model': 'cfg/tiny-yolo-voc-1c.cfg',
        		'load': 2375,
        		'threshold': 0.5,
        		'gpu': 0.7
        	    }
        	    
        tfnet = TFNet(option)
        avg = 0
        while True:
            stime = time.time()
            _ret, frame = self.cam.read()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis = frame.copy()
            colors = [tuple(255 * np.random.rand(3)) for i in range(40)]
            results = tfnet.return_predict(frame)
            cur_match = []
       # print(len(results))
    ##    refinedBoxes = non_max_suppression_slow(results, 0.4) ## overlap
            refinedBoxes = self.non_max_suppression(results, 0.60) 
            print(len(refinedBoxes))

            for i, (color, result) in enumerate(zip(colors, results)):
                if i in refinedBoxes:
                    tl = (result['topleft']['x'], result['topleft']['y'])
                    br = (result['bottomright']['x'], result['bottomright']['y'])
                   # label = result['label']
                    confidence = result['confidence']
                    text = '{}: {:.0f}%'.format(i, confidence * 100)
                             
                 #   frame = cv.rectangle(frame, tl, br, color, 2)
                    frame = cv.putText(frame, text, tl, cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                   # height = 100
                    cur_match.append((result['topleft']['x'], result['topleft']['y'],result['bottomright']['x'], result['bottomright']['y']))
             ##color segmentation to get undetected bolls
                    #----------Bgr----------
                    
                    lower = np.array([210, 210, 120], dtype = "uint16")
                    upper = np.array([255, 255, 255], dtype = "uint16")
                    mask = cv.inRange(frame, lower, upper)
           
                    
                    kernel = np.ones((4,4),np.uint8)
                    mask = cv.erode(mask,kernel,iterations = 2)
                    mask = cv.dilate(mask,kernel,iterations = 5)
                    output = cv.bitwise_and(frame, frame, mask = mask)
                    fg_mask = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
                    
            if imutils.is_cv2():
                        
                (contours, hier) = cv.findContours(fg_mask, cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    	# check to see if we are using OpenCV 3
            elif imutils.is_cv3():
                (_, contours, hier) = cv.findContours(fg_mask, cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    
            matches_ = []
            for cnt in contours:
                if (20<cv.contourArea(cnt)<1000000 ):
                    cv.drawContours(fg_mask,[cnt],0,255,-1)
                    (x,y,w,h) = cv.boundingRect(cnt)
                    centroid = self.get_centroid(x, y, w, h)
                    matches_.append(((x, y, x+w-1, y+h-1), centroid,cv.contourArea(cnt)))
            ##check the false positive of the YOLOv2
           
            refinedMatches_ = self.non_max_suppression_no_dict(matches_, 0.65) 
            print(len(refinedMatches_))
            for (i, match) in enumerate(matches_):
                contour, centroid, contour_area = match
                x, y, xw, yh = contour
                w = xw - x
                h = yh - y
                cx = x+w/2
                cy = y+h/2
                yolocovered = False
                res = False
                for j, ( result) in enumerate(zip(results)):
                    xi = result[0]['topleft']['x']
                    yi = result[0]['topleft']['y']
                    xj = result[0]['bottomright']['x']
                    yj = result[0]['bottomright']['y']
                    res = True
                    if j in refinedBoxes:
                        if ( xi<= cx and yi <= cy and xj >=cx and yj >= cy):
                             yolocovered = True
                             break
                if not (yolocovered) and (res): 
                    if i in refinedMatches_:
                       # frame = cv.rectangle(frame, (x, y), (xw-1, yh-1), (255,0,0), 3)
                        cur_match.append((x, y, xw-1, yh-1))
            vis = frame.copy()            
            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                new_bolls = []
                for tr,bo, (x, y), good_flag in zip(self.tracks,self.bolls, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    for boll_bbox in zip(cur_match):
                        xi,yi,xj,yj = boll_bbox[0]
                        area = (xj-xi+1)*(yj-yi+1)
                        xii,yii,xjj,yjj,area2 = bo[-1]
                        if(area2>area):    ##take the large areas and substitute
                                cx, cy = self.get_bbox_centre(xi,yi,xj,yj)
                                dx = (xjj - xii)/2
                                dy = (yjj - yii)/2
                                if (dx > 50 and dy > 50):
                                    dx = 50
                                    dy = 50
                                xi = cx - dx
                                yi = cy - dy
                                xj = cx + dx
                                yj = cy + dy
                        else:
                                cx, cy = self.get_bbox_centre(xi,yi,xj,yj)
                                dx = (xjj - xii)/2
                                dy = (yjj - yii)/2
                        
                                xi = cx - dx
                                yi = cy - dy
                                xj = cx + dx
                                yj = cy + dy
                                
                        if ( xi<= x and yi <= y and xj >=x and yj >= y):
                           
                            tr.append((x, y))
                            bo.append((xi,yi,xj,yj,area))
                            if len(tr) > self.track_len:
                                del tr[0]
                           
                            new_tracks.append(tr)
                            new_bolls.append(bo)
                            cv.circle(vis, (x, y), 2, (0, 255, 0), -1)
                            break
                self.tracks = new_tracks
                self.bolls = new_bolls
                cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                unique_bolls = []
                for xi,yi,xj,yj,area in [np.int32(bo[-1]) for bo in self.bolls]:
                    unique_bolls.append(((xi,yi,xj,yj), (xi,yi), area))
                
                refinedunique_bolls = self.non_max_suppression_no_dict(unique_bolls, 0.7) 
                for j ,(xi,yi,xj,yj,area) in enumerate([np.int32(bo[-1]) for bo in self.bolls]):
                    if j in refinedunique_bolls:
                        vis = cv.rectangle(vis, (xi, yi), (xj, yj), (255, 0,250), 3)
                    
                draw_str(vis, (20, 20), 'track count: %d' % len(self.bolls))

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1)
                p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        for boll_bbox in zip(cur_match):  
                            xi,yi,xj,yj = boll_bbox[0]
                            area = (xj-xi+1)*(yj-yi+1)
                            if ( xi<= x and yi <= y and xj >=x and yj >= y):
                                self.tracks.append([(x, y)])
                                self.bolls.append([(xi,yi,xj,yj,area)])
                                break

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv.imshow('lk_track', vis)
            ##-------------calculate FPS ----------------------------------------------
          #  avg = (avg*(self.frame_idx-1) + (1 / (time.time() - stime)))/self.frame_idx
          #  print('FPS {:.1f}'.format(avg))
            fps =  (1 / (time.time() - stime))
            print('FPS {:.1f}'.format(fps))
            
            ch = cv.waitKey(1)
            if ch == 27:
                break

def main():
    
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    print(__doc__)
    App(video_src).run()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
