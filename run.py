from darkflow.defaults import argHandler #Import the default arguments
import os
from darkflow.net.build import TFNet


FLAGS = argHandler()
FLAGS.setDefaults()

FLAGS.demo = "20180306_160047.mp4" # video file to use, or if camera just put "camera"
FLAGS.model = "cfg/tiny-yolo-voc-1c.cfg" # tensorflow model
FLAGS.load = "bin/tiny-yolo-voc.weights" # tensorflow weights
# FLAGS.pbLoad = "tiny-yolo-voc-traffic.pb" # tensorflow model
# FLAGS.metaLoad = "tiny-yolo-voc-traffic.meta" # tensorflow weights
FLAGS.threshold = 0.6 # threshold of decetion confidance (detection if confidance > threshold )
FLAGS.gpu = 0.5 #how much of the GPU to use (between 0 and 1) 0 means use cpu
FLAGS.track = True # wheither to activate tracking or not
FLAGS.trackObj = ["cotton_boll"] # the object to be tracked
#FLAGS.trackObj = ["person"]
FLAGS.saveVideo = True  #whether to save the video or not
FLAGS.BK_MOG = True # activate background substraction using cv2 MOG substraction,
                        #to help in worst case scenarion when YOLO cannor predict(able to detect mouvement, it's not ideal but well)
                        # helps only when number of detection < 3, as it is still better than no detection.
FLAGS.tracker = "sort" # wich algorithm to use for tracking deep_sort/sort (NOTE : deep_sort only trained for people detection )
FLAGS.skip = 0 # how many frames to skipp between each detection to speed up the network
FLAGS.csv = True #whether to write csv file or not(only when tracking is set to True)
FLAGS.display = True # display the tracking or not

tfnet = TFNet(FLAGS)


tfnet.camera()
exit('Demo stopped, exit.')
