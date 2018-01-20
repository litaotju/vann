import cv2
import sys
import os
import time
import threading

import numpy as np
from PIL import Image


try:
    import queue as queue
except ImportError:
    import Queue as queue

WINDOW_NAME = "window"
DEFAULT_MOSAIC_SIZE = 10

# if save the mosaic video
SAVE_VIDEO = False

# the iou threshold, tracker will stop if iou smaller than this
IOU_THRESH = 0

# whether to resize or not before process the video
RESIZE = True

# if RESIZE==TRUE, and width > MAX_WIDTH, then resize the video
# and keep the width/height aspect_ration
MAX_WIDTH = 1080

# max cache as mb
MAX_CACHE_IN_MB = 1024

ALLOWED_TRACKER_TYPE = ["TLD", "KCL"]

def debug(var, name):
    print ("Var:{}, shape{}, mean:{}, min:{}, max{}"\
            .format(name, var.shape, var.mean(), var.min(), var.max()))

def mkdir(d1, d2=None):
    if not os.path.exists(d1):
        os.mkdir(d1)
    d = d1
    if d2:
        d = os.path.join(d1, d2)
        if not os.path.exists(d):
            os.mkdir(d)
    return d

class Mode:
    ANN = 'ANN'
    MOSAIC = 'MOSAIC'
    CROP = 'CROP'

class VideoCapture:

    def __init__(self, input_file):
        self.__cap = cv2.VideoCapture(input_file)
        self.__cache = {}
        self.__cached_num = 0
        
        size = (int(self.get(cv2.CAP_PROP_FRAME_WIDTH)),   
                    int(self.get(cv2.CAP_PROP_FRAME_HEIGHT)))  
        size = self.__get_size(size)
        frame_bytes = float(size[0]*size[1]*3)

        self.MAX_CACHE_FRAME = int(MAX_CACHE_IN_MB*1024*1024/frame_bytes)

    def read(self):
        ''' Find the position of the frame (may be set by user, or just naturely go)
            If the frame has been read before, and in cache, just get it from cache,
            otherwise, read the frame from the raw video capture
            return: ok, (img, boxes)
                ok, bool, indicates whethere ok
                img, np.array, the frame
                boxes, a list of bound box of the frame, None if no boxes for the frame
        '''
        next_frame = self.get(cv2.CAP_PROP_POS_FRAMES)
        #print ("reading frame: {}".format(next_frame))

        MAX_FRAME_NO = self.get(cv2.CAP_PROP_FRAME_COUNT)
        #assert next_frame >= 0 and next_frame <= MAX_FRAME_NO, next_frame
        if next_frame < 0 and next_frame > MAX_FRAME_NO:
            print ("Warning, try to read frame:{}".format(next_frame))

        ok, img , boxes = False, None, None
        if next_frame in self.__cache:
            ok = True
            img, boxes = self.__cache[next_frame]

        if img is None: 
            # Two possibility:
            #  1. frame not in cache at all 
            #  2. frame cached None, could possible be cleaned up
            ok, img = self.__cap.read()
            if ok and img is not None:
                img = self.__preprocess(img)
                self.cache_frame(next_frame, img, boxes)
        else:
            # Frame in cache, and not cleaned up
            # instread of really reading the next frame
            # we only set the next frame
            #   to pretend we have really read a frame from the video
            next_frame = min(MAX_FRAME_NO, next_frame+1)
            self.set(cv2.CAP_PROP_POS_FRAMES, next_frame)

        return ok, (img,  boxes)

    def cache_frame(self, frame_no, img, boxes):
        ''' put the frame `img` with given `frame_no`, and given `boxes` to cache
            so next time, when user need to read the frame, we can given the
            boxes stored in the cache
        '''
        #print("caching frame:{} ".format(frame_no))
        MAX_FRAME_NO = self.get(cv2.CAP_PROP_FRAME_COUNT)
        if frame_no >= 0 and frame_no <= MAX_FRAME_NO:
            self.__cache[frame_no] = (img.copy(), boxes)
            self.__cached_num += 1
            self.__fresh_cache()

    def __fresh_cache(self):
        '''
            fresh the cache, if the cache is full,  throw out older frames
        '''
        if self.__cached_num >= self.MAX_CACHE_FRAME:
            frames_with_cache = [f for f, _ in self.__cache.items() if _[0] is not None] 
            frames_with_cache.sort()
            #TODO: find more proper way to cleaning the caches
            frames_be_cleaning = frames_with_cache[:self.MAX_CACHE_FRAME/2]
            for f in frames_be_cleaning:
                self.__cache[f] = (None, self.__cache[f][1])
                self.__cached_num -= 1

    def set(self, prop, val):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            #print ("setting next frame to:{}.".format(val))
            pass
        self.__cap.set(prop, int(val))

    def get(self, prop):
        return  self.__cap.get(prop)

    def release(self):
        self.__cap.release()


    def __preprocess(self, img):
        '''
           preprocess the given image, before really feed it into next
           Here we resize it
        '''
        assert img is not None
        width = img.shape[1]
        height = img.shape[0]
        size = (width, height)
        new_size = self.__get_size(size)
        if new_size != size:
            im = Image.fromarray(img)
            im = im.resize(new_size)
            img = np.asarray(im, dtype=np.uint8)
        return img

    def __get_size(self, (width, height)):
        '''return the size of resized image, if need to resize
        '''
        new_size = (width, height)
        if RESIZE and width > MAX_WIDTH:
            aspect_ratio = float(width)/height
            new_size = (MAX_WIDTH, int(MAX_WIDTH/aspect_ratio))
        return new_size 

class BoxSaver:
    '''Save the given boxes to a file
    '''
    def __init__(self, output_file):
        self.boxes = {}
        self.fobj = open(output_file, "a")

    def add_box(self, fname, box):
        if box is None:
            return
        assert not self.fobj.closed

        assert len(box) == 2, box
        iou, box = box
        assert len(box) == 4, box # 4 coord and 1 iou

        assert box[2] >= 0 and box[3] >= 0
        #negative consider zero
        box = [max(int(_), 0) for _ in box]

        self.boxes[fname] = box
        line = "{} : {} : {}".format(fname, box, iou)
        self.fobj.writelines(line+os.linesep)
        
    def __del__(self):
        if not self.fobj.closed:
            self.fobj.close()


class ScreenSaver:
    '''Save the given frame of a video to specified dir
    '''

    def __init__(self, output_dir, basename):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.output_dir = output_dir
        self.basename = basename

    def save(self, img, frame_no):
        fname = os.path.join(self.output_dir, \
                        "{}_{}.jpg".format(self.basename, frame_no))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Image.fromarray(img).save(fname)
        return fname


class SamplePairSaver:
    ''' @brief: 
              Save the sample pairs, and boxes of a video screen snapshot images
        @details:
            This saver will run a sepearte daemon thread to save the images/boxes to file, 
            if the constructor was given the `threaded=True`,
            The seperate thread saver will import above 1/3 the fps of the whole program

            Single thread (main thread) saver be used when threaded=False
    '''

    def __init__(self, output_dir, basename, threaded=False):
        self.output_dir = output_dir
        self.basename = basename

        dirA = mkdir(output_dir, "raw_images")
        dirB = mkdir(output_dir, "mosaic_images")

        self.saverA = ScreenSaver(dirA, basename)
        self.saverB = ScreenSaver(dirB, basename)

        fname = os.path.join(self.output_dir, self.basename+"_boxes.txt")
        self.boxSaver = BoxSaver(fname)

        self.__threaded = threaded
        if threaded:
            self.__queue = queue.Queue(100)
            self.__thread = threading.Thread(target=self.worker)
            self.__thread.setDaemon(True)
            self.__thread.start()
    
    def save_images(self, orig_img, mosaic_img, box, frame_no, thread_save):
        #start = time.clock()
        if self.__threaded and thread_save:
            #print "Put frame:{}".format(frame_no)
            try:
                self.__queue.put((orig_img, mosaic_img, box, frame_no), timeout=0.01)
            except queue.Full:
                print "Queue full, abandoning frame:{}".format(frame_no)
            #print "Put used {}s".format(time.clock()-start)
        else:
            self.__save_images(orig_img, mosaic_img, box, frame_no)
            #print "Syn save used {}s".format(time.clock()-start)

    def worker(self):
        while True:
            orig_img, mosaic_img, box, frame_no = self.__queue.get()
            #print "Saving frame:{}".format(frame_no)
            self.__save_images(orig_img, mosaic_img, box, frame_no)

    def __save_images(self, orig_img, mosaic_img, box, frame_no):
        '''Blocking call to save given images/box to file
        '''
        self.saverA.save(orig_img, frame_no)
        fname = self.saverB.save(mosaic_img, frame_no)
        self.boxSaver.add_box(fname, box)

    def finished(self):
        if self.__threaded:
            return self.__queue.empty()
        return True

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    boxInter = (xA, yA, xB, yB)
    if xB < xA or yB < yA: #invalid box
        return 0

    # compute the area of intersection rectangle
    interArea = (xB - xA ) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] ) * (boxA[3] - boxA[1] )
    boxBArea = (boxB[2] - boxB[0] ) * (boxB[3] - boxB[1] )
    assert boxAArea >= 0 and boxBArea >= 0, \
            "boxA:{}, area:{}, boxB:{}, area:{}"\
            .format(boxA, boxAArea, boxB, boxBArea)

    #one of the box is a zero box (boxAArea ==0 or boxBArea==0)
    #or when the two box has no intersection at all(interArea <0)
    if interArea <= 0 or boxAArea == 0 or boxBArea == 0:
        iou = 0.0
        return iou

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    assert iou > 0 and iou <= 1.0, \
            "boxA:{}, area:{}, boxB:{}, area:{}, interBox:{} iou: {}"\
            .format(boxA, boxAArea, boxB, boxBArea, boxInter, iou)
    # print boxAArea, boxBArea, interArea, iou
    return iou

def twoboxes_too_far(box0, box1):
    return False

class State:
    ''' A class holding the current state of the video-player
        The main fields conclude:

            drawing: indicates whether use is drawing the box right now
            start_point: the point where mouse left button down
            end_point: the point where the left button up

            speed: the desired speed spcecified by user, by -> | <- key 
            max_fps: the maximum fps current player can have, it's about performance
            cur_fps: the real fps, based on the performance limit and user intention
            size: the current size of the player

            pause: indicates whether it's in pause 
            terminate: whether it's in terminate state, (q or esc) key

            tracker: current object tracker, which can be none
            iou: the iou of tracker's bound box and user's bound box(compute by start/end point)
            cap: the current opencv VideoCapture (used to get the video frame info
    '''
    def __init__(self, size, speed, cap):
        self.drawing = False
        self.start_point = (-1, -1)
        self.end_point = (-1, -1)
        self.pause = False  
        self.speed = 1.0 # fps = orig_fps * speed
        self.max_fps = -1
        self.cur_fps = -1
        self.size = size  #(h, w)
        self.terminate = False
        self.tracker = None
        self.iou = None
        self.cap = cap # VideoCapture
        self.freeze_box = None
        self.__tracker_type = "KCL"
        self.__tracker_fn = {
                "KCL": cv2.TrackerKCF_create,
                "TLD": cv2.TrackerTLD_create,
                "NULL": lambda : None
                }
        self.mosaic_size = DEFAULT_MOSAIC_SIZE
        self.__max_frame_no = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.mosaiced_frames = np.zeros(self.__max_frame_no, np.int8)

        self.__intervals = 0

        self.save = True #whether or not enable to save results
        self.thread_save = True #whether or not enable multi-thread save

    def clear_bbox(self):
        self.drawing = False
        self.clear_start_point()
        self.clear_end_point()
        self.freeze_box = None

    def clear_start_point(self):
        self.start_point = (-1, -1)

    def clear_end_point(self):
        self.end_point = (-1, -1)

    def has_valid_bbox(self):
        return self.start_point != (-1, -1) and \
                self.end_point != (-1, -1) and  \
                self.start_point[0] != self.end_point[0] and \
                self.start_point[1] != self.end_point[1]

    def update_on_key(self, k):
        # space button
        if k == 32:
            self.pause = not self.pause
            if self.pause:
                self.freeze_box = self.update_tracker()
            else:
                self.freeze_box = None
        # up,  up (mac), '.'
        elif k in (82, 0, ord('.')):
            self.speed *= 2
            self.speed = min(32.0, self.speed)

        #down, down(mac), ','
        elif k in(84, 1, ord(',')):
            self.speed /= 2
            self.speed = max(1.0/32, self.speed)

        #esc or 'q' key
        elif k in (27, ord('q')):
            self.terminate = True
        #left, right
        if k in (81, 83, ord(']'), ord('['), 2, 3):
            self.jump_frame(k)

        #k , t, n
        if k in (107, 116, 110):
            if k == 107: #k
                self.__tracker_type = "KCL"
            if k == 116: #t
                self.__tracker_type = "TLD"
            if k == 110: #n
                self.__tracker_type = "NULL"
            print ("Changing the tracker type to %s" % self.__tracker_type)

        #print k
        if k in (61, 43, 45):
            if k in (61, 43): # =/+
                #print ("Increazing mosaic size...")
                self.mosaic_size += 1
                self.mosaic_size = min(40, self.mosaic_size)
            if k == 45: # -
                #print ("Decreasing mosaic size...")
                self.mosaic_size -= 1
                self.mosaic_size = max(5, self.mosaic_size)
        if k == ord('s'):
            self.save = not self.save

        if k == ord('h'):
            self.thread_save = not self.thread_save
    
    def jump_frame(self, k):
        cap = self.cap
        MAX_FRAME_NO = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        current_frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if k in (83, 3, ord(']')): # right 
            next_frame = current_frame_no + self.cur_fps * self.speed
            next_frame = min(MAX_FRAME_NO, next_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
        elif k in (81, 2, ord('[')): # left
            next_frame = current_frame_no - self.cur_fps * self.speed
            next_frame = max(next_frame, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)

    def get_current_bbox(self):
        if self.has_valid_bbox():
            bound_box = [0, 0, 0, 0]
            bound_box[0] = min(self.start_point[0], self.end_point[0])
            bound_box[1] = min(self.start_point[1], self.end_point[1])
            bound_box[2] = abs(self.start_point[0] - self.end_point[0])
            bound_box[3] = abs(self.start_point[1] - self.end_point[1])
            return True, bound_box
        return False, None

    def init_tracker(self):
        ''' init a tracker based on the current box of user specified
            if not valid box, just do nothing
        '''
        if self.has_valid_bbox():
            # get the initial bounding box, from user specifed
            ok, bbox = self.get_current_bbox()

            try:
                self.tracker = self.__tracker_fn[self.__tracker_type]()
            except AttributeError:
                self.tracker = cv2.Tracker_create(self.__tracker_type)

            print ("Initializing tracker of type:%s" % self.__tracker_type)
            # Initialize tracker with first frame and bounding box
            if self.tracker is not None:
                ok = self.tracker.init(self.img, tuple(bbox))
            
            if not ok:
                print ("Error: Initialize the tracker failed, box:{}".format(bbox))
                self.clear_bbox()
                self.clear_tracker()
            else:
                print ("Okay: Initialize the tracker box:{}".format(bbox))
            _, self.freeze_box =  self.get_current_bbox()

    def update_tracker(self):
        '''update the tracker (if there is one), and return the detected bound box
            if there is no tracker, just return the user specifed bound box
        '''
        ok = False
        bound_box = None
        if self.tracker:
            ok, bound_box = self.tracker.update(self.img)
            if ok:
                # When the tracker did good job
                # compute the iou of tracker's box and user specifed bound box 
                # which was used to initialize the tracker

                _, user_bound_box = self.get_current_bbox()
                # re-arrange the bound box to  pointA->pointB format
                boxA = [
                        user_bound_box[0], user_bound_box[1], 
                        user_bound_box[0] + user_bound_box[2], user_bound_box[1] + user_bound_box[3]
                        ]
                boxB = [ bound_box[0], bound_box[1], 
                            bound_box[0] + bound_box[2], bound_box[1] + bound_box[3] ]

                self.iou =  bb_intersection_over_union(boxA, boxB)

            if not ok or self.iou < IOU_THRESH:
                # When tracking failed, stop the video auto matically.
                # and clear the bboux, wait user to specify another box
                print ("Update the tracker failed, status :{}, iou:{}"\
                        .format(ok, self.iou))
                self.pause = True
                self.clear_bbox()
                self.clear_tracker()

        #only when current has no tracker/or pause, return the user specified box
        #if currently there is an tracker, should always use tracker's box
        #   and if tracker's box is not qualified, just abandan it, and consider no box
        else:
            ok, bound_box = self.get_current_bbox()
            if ok:
                self.iou = 1

        if bound_box is None:
            return bound_box

        #no matter user's spcified or tracker found, we dont consider
        #a line (without height or without width) as a valid box
        if bound_box[2] < 1 or bound_box[3] < 1:
            bound_box = None
        return bound_box

    def clear_tracker(self):
        self.tracker = None
        self.iou = None

    @property
    def tracker_type(self):
        return str(self.__tracker_type)

    def mark_frame_as_mosaic(self, frame_no):
        '''add the current frame no to marked set
        '''
        if frame_no >= 0 and frame_no < self.__max_frame_no:
            self.mosaiced_frames[int(frame_no)] = 1

    @property
    def intervals(self):
        return self.__intervals
   
class Render:
    def __init__(self):
        self.__mode = Mode.MOSAIC #process mode
        pass

    def render(self, state, window_name, current_frame_no):
        #Make a copy here, only process the copied one, so processing the frame
        #when video is on pause, won't mess the state's original frame 
        img = state.img.copy()

        bound_box = None
        if not state.pause:
            bound_box = state.update_tracker()
        else:
            bound_box = state.freeze_box

        ret_img = None
        if bound_box:
            #print("Bounding box on:{}".format(bound_box))
            if self.__mode == Mode.MOSAIC:
                self.mosaic_on_bound_box(img, bound_box, state.mosaic_size)
            elif self.__mode in (Mode.ANN, Mode.CROP):
                cv2.rectangle(img, (int(bound_box[0]), int(bound_box[1])), \
                            (int(bound_box[0] + bound_box[2]),  \
                                int(bound_box[1] + bound_box[3])), \
                            (255, 0, 0), 1)
            if self.__mode == Mode.CROP:
                x_start = int(bound_box[0])
                x_end = min(int(bound_box[0] + bound_box[2]+1), img.shape[1])
                y_start = int(bound_box[1])
                y_end = min(int(bound_box[1] + bound_box[3]+1), img.shape[0])
                ret_img = img.copy()[y_start:y_end, x_start:x_end, :]
                #assert ret_img is not None, ("%d,%d,%d,%d" % (x_start, x_end, y_start, y_end))

        if ret_img is None:
            ret_img = img.copy()

        # Draw the user specifed box only on the copy frame, 
        # This do not affect the returned one 
        if state.has_valid_bbox():
             cv2.rectangle(img, state.start_point, 
                           state.end_point, (0, 255, 0), 1)
        if bound_box:
             cv2.rectangle(img, (int(bound_box[0]), int(bound_box[1])), \
                            (int(bound_box[0] + bound_box[2]),  \
                                int(bound_box[1] + bound_box[3])), \
                            (255, 0, 0), 1)

        self.show_text_info(state, img)
        self.show_scroll_bar(state, img, current_frame_no)
        
        cv2.imshow(window_name, img)

        return ret_img, bound_box

    def show_text_info(self, state, img):
        '''add misc text info to img, based on the state
        '''
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Speed:{}, MaxFPS:{}, CurFPS:{},  Frame:{}/{}"\
                .format(state.speed, int(state.max_fps), int(state.cur_fps),
                        int(state.cap.get(cv2.CAP_PROP_POS_FRAMES)-1),
                        int(state.cap.get(cv2.CAP_PROP_FRAME_COUNT)) )
        textsize = cv2.getTextSize(text, font, 0.5, 2)[0]
        textX = max( (img.shape[1] - textsize[0]) / 2, 0)
        #display the speed text to the bottom (offset 20) center
        cv2.putText(img, text, (textX, img.shape[0]-15), font, 0.5, (0, 255, 0), 1)

        text2 = "Save:{}, Mode:{}, MosaicSize:{}, Tracker:{}".format(
                     state.save, self.__mode, 
                     state.mosaic_size, state.tracker_type)
        if state.iou is not None:
            text2 += " , IOU: {:.2%}".format(state.iou)
        textsize2 = cv2.getTextSize(text2, font, 0.5, 2)[0]
        textX = max( (img.shape[1] - textsize2[0]) / 2, 0)
        #display the speed text to the bottom (offset 20) center
        cv2.putText(img, text2, (textX, img.shape[0]-30), font, 0.5, (0, 255, 0), 1)

        
    def show_scroll_bar(self, state, img, current_frame_no):
        ''' add scroll bar to the bottom of the image based on the state
        '''
        raw = img.copy()
        height = img.shape[0]
        width = img.shape[1]
        MAX_FRAME_NO = state.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        percentage = current_frame_no/MAX_FRAME_NO
        end = int(width*percentage)
        if end <= 10:
            end = 10
        cv2.line(img, (0, int(height-10)), (end, int(height-10)), (255, 0, 0), 5) 

        #make the line to be smi-transparent
        alpha = 0.5
        cv2.addWeighted(img, alpha, raw, 1 - alpha, 0, img)

        #show the saved mosaiced image pos to the scroll bar, using red circle
        pos = np.nonzero(state.mosaiced_frames)[0]
        if len(pos) == 0:
            return
        pos = pos/MAX_FRAME_NO*width
        for i in pos.astype(np.int32):
            cv2.circle(img, (i, int(height-10)), 3, (0,0,255), 1) 

    def mosaic_on_bound_box(self, image, bound_box, mosaic_size=10):
        height, width = image.shape[:2]
        scale_mosaic = 1 / float(mosaic_size)
        mosaic_image = cv2.resize(image, (0, 0), fx=scale_mosaic, fy=scale_mosaic)
        mosaic_image = cv2.resize(mosaic_image, (width, height), interpolation=cv2.INTER_NEAREST)

        #this is to make the x,y are in the scope of given img indices
        #because some times, the bound box given by the tracker, will be out the scope
        x_start = max(int(bound_box[0]), 0)
        y_start = max(int(bound_box[1]), 0)
        x_end = min(int(bound_box[0] + bound_box[2]+1), width)
        y_end = min(int(bound_box[1] + bound_box[3]+1), height)

        image[y_start:y_end, x_start:x_end, :] = mosaic_image[y_start:y_end, x_start:x_end, :]

    def update_on_key(self, k):
        if k == ord('c'):
            self.__mode = Mode.CROP
        elif k == ord('m'):
            self.__mode = Mode.MOSAIC
        elif k == ord('a'):
            self.__mode = Mode.ANN

def draw_rect(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN:
        state.clear_bbox()
        state.clear_tracker()
        state.drawing = True
        state.pause = True
        state.start_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        state.drawing = False
        state.end_point = (x, y)
        state.init_tracker()
        ## Do not change the pause status when the mouse is gone
        ## User need to type space to resume
        #state.pause = False

    elif event==cv2.EVENT_MOUSEMOVE:
        if state.drawing == True:
            state.end_point = (x, y)
 
    elif event == cv2.EVENT_RBUTTONDOWN:
        state.clear_bbox()
        state.clear_tracker()


def main():
    '''
        An cv2 app allow you to draw bounding box to an video
        usage: python script_name.py input_file output_file
    '''

    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    assert os.path.dirname(input_file) != output_dir, \
            "Output file is same with input file, this will overwrite the input"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    cap = VideoCapture(input_file)
    fps = cap.get(cv2.CAP_PROP_FPS)  
    fps = 25
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),   
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  

    basename = os.path.basename(os.path.splitext(input_file)[0])

    out = None
    if SAVE_VIDEO:
        output_file = os.path.join(output_dir, "{}.mp4".format(basename))

        out = cv2.VideoWriter(output_file, \
                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
        assert out.isOpened()
    
    state = State(size, 1, cap)
    render = Render()
    saver = SamplePairSaver(output_dir, basename, True)

    if sys.platform.startswith('linux'):
        cv2.namedWindow(WINDOW_NAME,  cv2.WINDOW_GUI_NORMAL+ \
                                     cv2.WINDOW_AUTOSIZE + \
                                     cv2.WINDOW_KEEPRATIO)
    else:
        cv2.namedWindow(WINDOW_NAME)

    cv2.setMouseCallback(WINDOW_NAME, draw_rect, state)
    
    img = np.zeros((size[0], size[1], 3), np.uint8)
    cv2.imshow('window', img)
    raw_img = None

    max_frame_no = state.cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_cnt = 0
    last_saved_frame = -1
    
    current_frame_no = 0
    
    while(not state.terminate):
        # pause on last frame, wait user to quit or jump
        if state.cap.get(cv2.CAP_PROP_POS_FRAMES) == max_frame_no:
            state.pause = True
        if not state.pause:
            ok, (img, boxes) = cap.read()
            current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)-1)
            frame_cnt += 1
            if not ok:
                break
            # Backup the orignal raw image, will save later
            raw_img = img.copy()
            # This is needed, because the mouse callback need this
            # To init the tracker
            state.img = img

        timer = cv2.getTickCount()
        processed_img, bound_box = render.render(state, WINDOW_NAME, 
                                                 current_frame_no)

        if not state.pause and state.save:
            if out is not None:
                out.write(processed_img)
            #only save the image with valid bounding box
            #and save at least every other INTERVALS images
            if bound_box is not None:
                save_this_frame = False
                if last_saved_frame == -1:
                    save_this_frame = True
                elif frame_cnt - last_saved_frame > state.intervals:
                    save_this_frame = True
                if save_this_frame:
                    last_saved_frame = frame_cnt
                    saver.save_images(raw_img, processed_img, 
                            (state.iou, bound_box), current_frame_no,
                            state.thread_save)
                    state.mark_frame_as_mosaic(current_frame_no)

        cur_fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        #speed control
        target_fps = fps * state.speed
        if not state.pause:
            if frame_cnt % 25 == 0:
                state.max_fps = cur_fps
            state.cur_fps = min(cur_fps, target_fps)
        idle_time = int(1000/target_fps - 1000/cur_fps)
        if idle_time <= 0:
            idle_time = 1
        #print("cur_fps:{}, target_fps:{}, idle:{} ms".format(cur_fps, target_fps, idle_time))

        k = cv2.waitKey(idle_time) & 0xFF
        #handle key event
        state.update_on_key(k)
        render.update_on_key(k)

    #This is needed, because we need to wait the saver's daemon thread
    #to be finished, otherwise, some images might be lost
    while not saver.finished():
        print "Waiting saver to be finished"
        time.sleep(3) #wait 3s

    if out is not None:
        out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
