#!/usr/bin/env python

'''

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

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
 #---------------------------boll class-----------------------------------------------------------------------        

IMAGE_DIR_RES = "output"
fourcc = cv.VideoWriter_fourcc(*'MP4V')


class Boll(object):
    def __init__(self, id, position,height,avg_height):
        self.id = id
        self.positions = [position]
        self.heights = [height]
        self.avg_heights = [avg_height]
        self.frames_since_seen = 0
        self.counted = False

    @property
    def last_position(self):
        return self.positions[-1]
    @property
    def last_height(self):
        return self.heights[-1]
    @property
    def last_avg_height(self):
        return self.avg_heights[-1]

    def add_position(self, new_position,new_height,new_avg_height):
        self.positions.append(new_position)
        self.heights.append(new_height)
        self.avg_heights.append(round(new_avg_height,1))
        self.frames_since_seen = 0

    def draw(self, output_image):
        car_colour = TRACKING_COLOURS[self.id % len(TRACKING_COLOURS)]
        #for point in self.positions:
      #  average_height = reduce(lambda x,y: x+y,self.heights)/len(self.heights)
        for (i, point) in enumerate(self.positions):
            cv.circle(output_image, point, 2, car_colour, -1)
            u, v = point
            cv.polylines(output_image, [np.int32(self.positions)], False, car_colour, 1)
            cv.putText(output_image,str(self.avg_heights[i]) ,point, cv.FONT_HERSHEY_SIMPLEX, 0.4,car_colour,1)
            #cv2.putText(output_image,str(average_height),point, cv2.FONT_HERSHEY_SIMPLEX, 0.4,car_colour,1) 
           #cv2.putText(output_image,str(self.heights[i])+","+str(average_height),point, cv2.FONT_HERSHEY_SIMPLEX, 0.4,car_colour,1)
        #    cv2.putText(output_image,str(self.heights[i])+"  ("+str(u)+","+str(v)+")" ,point, cv2.FONT_HERSHEY_SIMPLEX,VideoWriter_fourcc(*'MP4V') 0.5,car_colour,1)
         


# ============================================================================

class App:
    def __init__(self, video_src):
        self.track_len = 80
        self.detect_interval = 5
        self.tracks = []
        self.bolls = []
        self.count_bolls = []
        self.boll_number = 0
        self.boll_count = 0
        self.cam = video.create_capture(video_src)
        video_ = video_src.split('/')[-1]
        self.videoname, self.video_extension = os.path.splitext(video_)
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
        	#	'model': 'cfg/tiny-yolo-voc-1c.cfg',
        	#	'load': 2375,
                'pbLoad': 'built_graph/tiny-yolo-voc-1c.pb',
                'metaLoad': 'built_graph/tiny-yolo-voc-1c.meta',
        		'threshold': 0.5,
        		'gpu': 0.7
        	    }
        	    
        tfnet = TFNet(option)
        avg = 0
        divider = 700 #horizontal line after which we count the bolls
        checkedbolls = 0 #initialized check boll index
        refinedunique_bolls = []
        colors = [tuple(np.random.randint(255, size=3)) for i in range(10000)] #make multiple colors to display bolls and tracklets
        #out = cv.VideoWriter('video_'+self.videoname+'.mp4',fourcc, 12.0, (1280,720))
        bollcnt = 0
        while True:
            stime = time.time()
            _ret, frame = self.cam.read()
            if _ret == False:
                print("Bad frame, video ended")
               # out.release()
                exit(1)
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis = frame.copy()
           
            results = tfnet.return_predict(frame)
            cur_match = []
            print("Image # "+str(self.frame_idx) )
       # print(len(results))
    ##    refinedBoxes = non_max_suppression_slow(results, 0.4) ## overlap
          #  refinedBoxes = self.non_max_suppression(results, 0.55) 
            #print(len(refinedBoxes))
            check_mask = 0
            for i, (color, result) in enumerate(zip(colors, results)):
              #  if i in refinedBoxes:
               
                    cur_match.append((result['topleft']['x'], result['topleft']['y'],result['bottomright']['x'], result['bottomright']['y']))
             
             
             ##color segmentation to get undetected bolls
                    #----------Bgr----------                    
            lower = np.array([190, 190, 150], dtype = "uint16")
            upper = np.array([255, 255, 255], dtype = "uint16")
            mask = cv.inRange(frame, lower, upper)
           
                    
            kernel = np.ones((4,4),np.uint8)
            mask = cv.erode(mask,kernel,iterations = 2)
            mask = cv.dilate(mask,kernel,iterations = 5)
           
            #check_mask = 1
            output = cv.bitwise_and(frame, frame, mask = mask)
                   
            fg_mask = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
            #if check_mask == 0:
               # continue
           # for (i, match) in enumerate(cur_match):
              #  xi, yi, xj, yj = match
             #   fg_mask_ = cv.rectangle(fg_mask, (xi, yi), (xj+1, yj+1), (0,0,0),-1,8)
                
            if imutils.is_cv2():
                        
                (contours, hier) = cv.findContours(fg_mask, cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    	# check to see if we are using OpenCV 3
            elif imutils.is_cv3():
                (_, contours, hier) = cv.findContours(fg_mask, cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    
            matches_ = []
            for cnt in contours:
                if (80<cv.contourArea(cnt)):
                    cv.drawContours(fg_mask,[cnt],0,255,-1)
                    (x,y,w,h) = cv.boundingRect(cnt)
                    centroid = self.get_centroid(x, y, w, h)
                    matches_.append(((x, y, x+w-1, y+h-1), centroid,cv.contourArea(cnt)))
            ##check the false positives of the YOLOv2
           
            refinedMatches_ = self.non_max_suppression_no_dict(matches_, 0.40) 
         
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
                #    if j in refinedBoxes:
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
                new_counts = []
                pts_src = p0.reshape(-1, 2)
                pts_dst = p1.reshape(-1, 2)
                H, status = cv.findHomography(pts_src, pts_dst)
                present_bolls = []
                for boll_bbox in zip(cur_match): 
                    for k, (tr,bo,cnt, (x, y), good_flag) in enumerate(zip(self.tracks,self.bolls,self.count_bolls, p1.reshape(-1, 2), good)):
                        if not good_flag:
                            continue
                        xi,yi,xj,yj = boll_bbox[0]
                        area = (xj-xi+1)*(yj-yi+1)
                        cx, cy = self.get_bbox_centre(xi,yi,xj,yj)
                        xi = cx - 50
                        yi = cy - 50
                        xj = cx + 50
                        yj = cy + 50        
                        if ( xi<= x and yi <= y and xj >=x and yj >= y):
                            
                            xii,yii,xjj,yjj,area2 = bo[-1] # get previous bolls
                            x0, y0 = tr[-1] #get previos tracklet head
                 
                          
                            if k not in present_bolls:
                                tr.append((x, y))
                                bo.append((xi,yi,xj,yj,area))
                                cnt.append((cnt[-1]))
                                if len(tr) > self.track_len:
                                    del tr[0]
                                    del bo[0]
                                    del cnt[0]
                               
                                new_tracks.append(tr)
                                new_bolls.append(bo)
                                present_bolls.append(k)
                                new_counts.append(cnt)
                                cv.circle(vis, (x, y), 2, (0, 255, 0), -1)
                                break
              
             
                #restore the lost tracklets that are greater than 200 and less than 700 in vertical pixel position
                new_tracklets = []
                update_tracklets = []
                tracklets = []
                for k, (tr,bo,cnt) in enumerate(zip(self.tracks,self.bolls,self.count_bolls)):
                    if len(tr) > 2 :
                        xii,yii,xjj,yjj,area2 = bo[-1] # get previous boll bbox position
                        x0, y0 = tr[-1] #get previos tracklet head position
                        for ki, (tri,boo,cntt) in enumerate(zip(self.tracks,self.bolls,self.count_bolls)):
                                   # if ki not in present_bolls:
                                        xiii,yiii,xjjj,yjjj,area3 = boo[-1] # get previous boll bbox position
                                        if len(tri) > 0 :
                                            x0i, y0i = tri[-1] #get previos tracklet head position
                                            #get newer points only
                                            
                                            if ( len(boo) <= 2 and xii<= x0i and yii <= y0i and xjj >=x0i and yjj >= y0i):
                                                tracklets.append((x0i, y0i))
                                                update_tracklets.append(k)
                                                new_tracklets.append(ki)
                                                break
            
                for k, (tr,bo,cnt) in enumerate(zip(self.tracks,self.bolls,self.count_bolls)):   
                   # if k in update_tracklets:
                      #  x0i, y0i = tracklets[update_tracklets.index(k)]
                     
                       # self.tracks[k][-1] = (x0i, y0i)
                     
                    
                    if len(tr) > 0 and k not in present_bolls:
                        xii,yii,xjj,yjj,area2 = bo[-1] # get previous boll bbox position
                        x0, y0 = tr[-1] #get previos tracklet head position
                        
                        
                        
                        if (len(tr) > 7 and y0 >= 360 and y0 <= 700 and len(present_bolls) > 0): ##check if the tracklet is old NOT new
                            
                            dst_pts_1 = (x0,y0,1)
                            dst_pts_2 = (xii,yii,1)
                            dst_pts_3 = (xjj,yjj,1)
                            src_pts_1 = np.matmul(H , dst_pts_1)
                            src_pts_2 = np.matmul(H , dst_pts_2)
                            src_pts_3 = np.matmul(H, dst_pts_3)
                            x = np.int32(src_pts_1[0]/src_pts_1[2])
                            y = np.int32(src_pts_1[1]/src_pts_1[2])
                            xi = np.int32(src_pts_2[0]/src_pts_2[2])
                            yi = np.int32(src_pts_2[1]/src_pts_2[2])
                            xj = np.int32(src_pts_3[0]/src_pts_3[2])
                            yj = np.int32(src_pts_3[1]/src_pts_3[2])
                           
                                        
                                    
                            tr.append((x, y))
                            bo.append((xi,yi,xj,yj,area))
                            cnt.append((cnt[-1]))
                            if (len(tr) > self.track_len):
                               
                                del tr[0]
                                del bo[0]
                                del cnt[0]
                               
                            new_tracks.append(tr)
                            new_bolls.append(bo)
                            new_counts.append(cnt)
                            cv.circle(vis, (x, y), 2, (0, 255, 0), -1)
                       
                    if (k in new_tracklets):
                                
                                del tr[-1]
                                del bo[-1]
                                del cnt[-1]
                                
                         
                self.tracks = new_tracks
                self.bolls = new_bolls
                self.count_bolls = new_counts
                
                for j,(tr,bo,cnt) in enumerate(zip(self.tracks,self.bolls,self.count_bolls)):
                    k = cnt[-1]
                    b,g,r = colors[k]
                    b = int(b)
                    g = int(g)
                    r = int(r)
                    if(len(tr) > 1):       
                        cv.polylines(vis, [np.int32(tr)], False, (b,g,r),2)
                        
                        xc1, yc1 = np.int32(tr[-1])
                        xc2, yc2 = np.int32(tr[-2])
                        if (yc2 <= divider and yc1 > divider and len(tr) > 7):
                            self.boll_count = self.boll_count + 1
                            
                        xi,yi,xj,yj,area = np.int32(bo[-1])
                 
                        cx, cy = self.get_bbox_centre(xi,yi,xj,yj)
                      #  text = '{}'.format(cnt[-1])
                        vis = cv.rectangle(vis, (xi, yi), (xj, yj), (b,g,r), 2)
                       # vis = cv.putText(vis, text, (cx, cy), cv.FONT_HERSHEY_COMPLEX, 1, (160, 0, 200), 2)
                    color
                
            
            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1)
                
                p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                print("--------------------------------------------------")
                if p is not None:
                    for boll_bbox in zip(cur_match): 
                        for x, y in np.float32(p).reshape(-1, 2):  
                            xi,yi,xj,yj = boll_bbox[0]
                            area = (xj-xi+1)*(yj-yi+1)
                            #if (x,y) not in self.tracks:
                            cx, cy = self.get_bbox_centre(xi,yi,xj,yj)
                            xi = cx - 50
                            yi = cy - 50
                            xj = cx + 50
                            yj = cy + 50
                           # if (x,y) not in self.tracks:
                            if ( xi<= x and yi <= y and xj >=x and yj >= y):
                                self.count_bolls.append([(bollcnt)])
                                self.tracks.append([(x, y)])
                                self.bolls.append([(xi,yi,xj,yj,area)])  
                                bollcnt = bollcnt + 1
                                break

            cv.line(vis, (0,360), (1280,360), (245, 8, 0), thickness=2, lineType=8)
            cv.line(vis, (0,700), (1280,700), (10, 255, 90), thickness=2, lineType=8)
            draw_str(vis, (20, 20), 'track count: %d' % self.boll_count)
            avg = (avg*(self.frame_idx) + (1 / (time.time() - stime)))/(self.frame_idx+1)
            fps =  (1 / (time.time() - stime))
            vis = cv.putText(vis, 'FPS {:.1f} AFPS {:.1f}'.format(fps,avg), (900, 20), cv.FONT_HERSHEY_COMPLEX, 1, (25, 0, 200), 2)
            file_name_format = IMAGE_DIR_RES + "/proc_"+self.videoname+"_%04d.jpg"
            file_name = file_name_format % self.frame_idx
            print(file_name)
            cv.imwrite(file_name, vis)
            #cv.imshow('lk_track', vis)
            #Write the frame into the file 'output.avi'
           # out.write(vis)
            
            self.frame_idx += 1
            self.prev_gray = frame_gray
            ##-------------calculate FPS ----------------------------------------------
           # avg = (avg*(self.frame_idx-1) + (1 / (time.time() - stime)))/self.frame_idx
           # print('FPS {:.1f}'.format(avg))
            #fps =  (1 / (time.time() - stime))
            print('FPS {:.1f}'.format(fps))
            
            #if self.frame_idx > 80:
            #    exit(1)
            ch = cv.waitKey(0)
            if ch == 27:
                break

def main():
    
    try:
        video_src = sys.argv[1]
    except:
        print("Format error: boll_track.py [<video_source>]")
        exit(1)

    print(__doc__)
    App(video_src).run()
    cv.destroyAllWindows()
    

if __name__ == '__main__':
    main()
