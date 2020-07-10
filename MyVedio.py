# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:24:28 2020

@author: BEL
"""


import tkinter as tk
import cv2
from tkinter import ttk
import pandas as pd
from multiprocessing import Process
# Added by BSTC Team for Bundle executable
#Enable for Bundling executable
import sklearn.ensemble
import sklearn.tree
import pickle
import sklearn.neighbors.typedefs
import sklearn.neighbors.quad_tree
import sklearn.tree._utils
import cython
import sklearn
#import sklearn.utils._cython_blas
import numpy as np
#import joblib
from sklearn.preprocessing import StandardScaler
# import utility.h5pb as h5pb
#end

import PIL.Image, PIL.ImageTk
import datetime
import time
from random import shuffle
from tkinter import filedialog
# from utility.facerecognition import FaceRecognition
from keras.models import load_model
# from utility.facedetect import FaceDetector
from keras.layers import LeakyReLU
import sqlite3
import sys
import threading

# from utility.align_face import FaceAlign
import os, csv

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from tkinter import messagebox
import tensorflow as tf

# import utility.facenet as facenet

from pandas.io import sql
from six import string_types, iteritems

# K.set_image_dim_ordering('tf')

from keras import backend as K
from statistics import mode
# from track.centroidtracker import CentroidTracker
# Added by BSTC Team for configuration file reading
import numpy as np
import configparser
from distutils.dir_util import copy_tree
import shutil

config=configparser.ConfigParser()

config.read("FRS_config_params.txt")
config.read("FRS_IP_Config_List.txt")

model_name = config.get('Config_Parameters' , 'model_name')

WEB_CAM = config.get('Config_Parameters_Loc' , 'Web_Cam')

DB_FILE=config.get('Config_Parameters' , 'DB_FILE')

ENROLLED_FACES_DIR = config.get('Config_Parameters' , 'ENROLLED_FACES_DIR')
TRAINED_FACES_DIR = config.get('Config_Parameters' , 'TRAINED_FACES_DIR')
BASE_DIR = os.path.dirname(__file__) + '/'

CSV_FILE = config.get('Config_Parameters' , 'CSV_FILE')
DB_FILE_CSV = config.get('Config_Parameters' , 'DB_FILE_CSV')
table_name = 'my_table'
OS_NAME = config.get('Config_Parameters' , 'OS_NAME')

DB_FILE_PATH = config.get('Config_Parameters' , 'DB_FILE_PATH')

IMAGE_NOVIDEO = config.get('Config_Parameters' , 'IMAGE_NOVIDEO')
IMAGE_TRAINING = config.get('Config_Parameters' , 'IMAGE_TRAINING')
IMAGE_HEADER = config.get('Config_Parameters' , 'IMAGE_HEADER')

HLS_VERSION = config.get('Config_Parameters' , 'HLS_VERSION')

TEST_FEATURE = config.get('Config_Parameters' , 'TEST_FEATURE')
TEST_LABEL= config.get('Config_Parameters' , 'TEST_LABEL')
TRAIN_FEATURE= config.get('Config_Parameters' , 'TRAIN_FEATURE')
TRAIN_LABEL= config.get('Config_Parameters' , 'TRAIN_LABEL')
VIDEO_SCROLL_BAR  = config.get('Config_Parameters' , 'VIDEO_SCROLL_BAR')

detector=FaceDetector()
face_rec=FaceRecognition()

ct = CentroidTracker()

TRAIN_PB_MODEL = config.get('Config_Parameters' , 'MODEL_TRAIN_PB')

model_pred_dir = config.get('Config_Parameters' , 'MTCNN_PB')

npy = config.get('Config_Parameters' , 'NPY_FILES')

#changed by shwetha
details_dict = dict(config.items('Config_Parameters_Loc'))

PEOPLE_COUNT = config.get('Config_Parameters' , 'PEOPLE_COUNT')
TOTAL_EPOCHS = config.get('Config_Parameters' , 'TOTAL_EPOCHS')
EVAL_PERIOD = config.get('Config_Parameters' , 'EVAL_PERIOD')
TOTAL_IMGS = config.get('Config_Parameters' , 'TOTAL_IMGS')


MODEL_FILE_DIR = config.get('Config_Parameters' , 'MODEL_FILE_DIR')
MODEL_PB_FILE = config.get('Config_Parameters' , 'MODEL_PB_FILE')

global face_recognizer


global face_folder

model_path = config.get('Config_Parameters' , 'SHARE_PREDICTOR')
face_aligner = FaceAlign(model_path)




class App(tk.Canvas):
    def __init__(self, window, window_title, master=None, video_source=0, **kw):
        if OS_NAME == "windows":
            self.db_file = DB_FILE_PATH+'\\'+DB_FILE
        elif OS_NAME == "ubuntu":
            self.db_file = DB_FILE_PATH+'/'+DB_FILE

        print(DB_FILE)
        print(ENROLLED_FACES_DIR)
        print(TRAINED_FACES_DIR)
        print(CSV_FILE)
        print(model_name)

        print(self.db_file)

        self.people_count = int(PEOPLE_COUNT) #50
        self.total_no_epochs= int(TOTAL_EPOCHS) #800
        print(OS_NAME)
        CurrentDate = datetime.datetime.now()
        print("Current_Date =",CurrentDate )

        config.read("FRS_config_params.txt")
        last_date1=config.get('Config_Parameters' , 'last_date')
        print("Config_Date =",last_date1)
        config_date = datetime.datetime.strptime(str(last_date1), '%Y-%m-%d %H:%M:%S')

        evaluation_period= int(EVAL_PERIOD) #30
        date_limit = config_date+datetime.timedelta(evaluation_period)
        print("Config_Date Plus days to add =",date_limit)

        if CurrentDate<=date_limit:
            self.image = kw.pop('image', None)
            super(App, self).__init__(master=master, **kw)
            self['highlightthickness'] = 4
            self.propagate(0)
            self.window = window
            self.flag  = 0
            self.webcam_flag=1
            self.Temp_face_id=0
            # Define frame size and position in the screen :
            self.ScreenSizeX=window.winfo_screenwidth()
            self.ScreenSizeY=window.winfo_screenheight()

            screen_resolution = str(self.ScreenSizeX) + 'x' + str(self.ScreenSizeY)
            self.window.geometry(screen_resolution)

            print("width and height=", self.ScreenSizeX, 'x' ,self.ScreenSizeY)
            self.widthnew = self.ScreenSizeX
            self.heightnew = self.ScreenSizeY

            self.window.configure(background='white')
            self.window.title(window_title)

            self.left = tk.Frame(self.window, borderwidth=1, relief="solid")

            self.right = tk.Frame(self.window, borderwidth=1, relief="solid")


            self.leftframe =tk.Frame(self.window)
            self.leftframe.place(x=5, y=70)
#open video source (by default this will try to open the computer webcam)
            if video_source:
	            self.video_source = 0
            elif OS_NAME == "windows":
                self.video_source = IMAGE_NOVIDEO # ".\image_800.png"
            else:
                self.video_source =  os.path.expanduser(os.path.join(IMAGE_NOVIDEO))

            if OS_NAME == "windows":
                self.frame2 = cv2.imread(IMAGE_TRAINING)
                self.frame1 = cv2.imread(IMAGE_NOVIDEO)
            else:
                self.frame2 = cv2.imread(os.path.expanduser(os.path.join(IMAGE_TRAINING)))
                self.frame1 = cv2.imread(os.path.expanduser(os.path.join(IMAGE_NOVIDEO)))


            self.length = 0

            self.vid = cv2.VideoCapture(int(WEB_CAM))

            self.video = MyVideoCapture(self.video_source)

            if self.vid.isOpened():
                self.width = self.vid.set(cv2.CAP_PROP_FRAME_WIDTH,self.ScreenSizeX*0.80)
                self.height = self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT,self.ScreenSizeY*0.80)

                self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
                self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

                print("width : ",self.width)
                print("height : ",self.height)
            else :
                self.width = self.ScreenSizeX*0.80
                self.height = self.ScreenSizeY*0.80

            self.width = self.ScreenSizeX*0.80
            self.height = self.ScreenSizeY*0.80
            self.frame1 = cv2.resize(self.frame1, (int(self.width), int(self.height)))
            self.frame2 = cv2.resize(self.frame2, (int(self.width), int(self.height)))

            self.count = 0
            self.init_img_no = 0
            self.flag_enrolment=0
            self.total_imgs = int(TOTAL_IMGS) # 50
            self.recognition_flag = 0
            self.browse_flag = 0
            self.emptystr_flag=0
            self.error_flag = 0
            self.ret_image =  0
            self.frame_count = 0
            self.last_face_id = []
            self.array_face_id = []
            self.face_indexX=[]
            self.face_indexY=[]
            self.csv_header =[]
            # create all of the main containers

            self.topframe=tk.Frame(self.window)
            self.topframe.pack(side= tk.TOP)

             # Create a top heading
            if OS_NAME == "windows":
                 path = IMAGE_HEADER
            else:
                 path =  os.path.expanduser(os.path.join(IMAGE_HEADER))

            img = PIL.ImageTk.PhotoImage(PIL.Image.open(path))
            self.panel = tk.Label(self.topframe, image = img)

            self.panel.pack(anchor = "center")

            self.face_index=0
            self.tag=[[None for y in range(7)] for x in range(self.people_count)]
            self.conn_csv = sqlite3.connect(DB_FILE_CSV)
            self.cursor_csv = self.conn_csv.cursor()
            self.datalink=open(CSV_FILE ,"r")
            self.dataset=csv.reader(self.datalink,delimiter=",")
            self.face_avg_dict=dict()

            self.face_names = self.get_face_names(self.db_file)
            self.latest_frame=None
            self.last_ret = None

            button_width = ((self.ScreenSizeX*0.20) * 0.07)
            print(button_width)

            button_width2 = ((self.ScreenSizeX*0.20) * 0.3166)
            print(button_width2)

            button_width3 = ((self.ScreenSizeX*0.20) * 0.2125)
            print(button_width3)

            button_width4 = ((self.ScreenSizeX*0.20) * 0.2472)
            print(button_width4)


            button_height1 = (self.ScreenSizeY * 0.622)
            print(button_height1)

            button_height2 = (self.ScreenSizeY * 0.655)
            print(button_height2)
            button_height3 = (self.ScreenSizeY * 0.430)
            print(button_height3)
            button_height4 = (self.ScreenSizeY * 0.488)
            print(button_height4)
            button_height5 = (self.ScreenSizeY * 0.522)
            print(button_height5)
            button_height6 = (self.ScreenSizeY * 0.555)
            print(button_height6)
            button_height7 = (self.ScreenSizeY * 0.588)
            print(button_height7)
            button_height8 = (self.ScreenSizeY * 0.088)
            print(button_height8)
            button_height9 = (self.ScreenSizeY * 0.144)
            print(button_height9)
            button_height10 = (self.ScreenSizeY * 0.177)
            print(button_height10)
            button_height11 = (self.ScreenSizeY * 0.211)
            print(button_height11)
            button_height12 = (self.ScreenSizeY * 0.255)
            print(button_height12)
            button_height13 = (self.ScreenSizeY * 0.3)
            print(button_height13)
            button_height14 = (self.ScreenSizeY * 0.344)
            print(button_height14)
            button_height15 = (self.ScreenSizeY * 0.377)
            print(button_height15)
            button_height16 = (self.ScreenSizeY * 0.411)
            print(button_height16)
            button_height17 = (self.ScreenSizeY * 0.711)
            print(button_height17)
            button_height18 = (self.ScreenSizeY * 0.751)
            print(button_height18)
            button_height19 = (self.ScreenSizeY * 0.791)
            print(button_height19)


            self.progress_var1 = tk.DoubleVar() #here you have ints but when calc. %'s usually floats
            self.prog_MAX1=100
            self.progress_var2 = tk.DoubleVar() #here you have ints but when calc. %'s usually floats
            self.prog_MAX2=100
            self.progress_var3 = tk.DoubleVar()
            self.prog_MAX3=100

            self.lbl_p1 = tk.Label(self.window, text='Stage1',width=6,bg = "gray", fg = 'white',font = "Ms 10")
            self.lbl_p1.place(x=(self.width+button_width), y=button_height1)
            self.progress1=ttk.Progressbar(self.window,orient=tk.HORIZONTAL, variable=self.progress_var1, length=150,maximum=self.prog_MAX1,mode='determinate')
            self.progress1.place(x=(self.width+button_width4), y=button_height1)

            self.lbl_p2 = tk.Label(self.window, text='Stage2',width=6,bg = "gray", fg = 'white',font = "Ms 10")
            self.lbl_p2.place(x=(self.width+button_width), y=button_height2)
            self.progress2=ttk.Progressbar(self.window,orient=tk.HORIZONTAL, variable=self.progress_var2,length=150, maximum=self.prog_MAX2,mode='determinate')
            self.progress2.place(x=(self.width+button_width4), y=button_height2)

#            self.lbl_p3 = tk.Label(self.window, text='video file progress bar',width=25,bg = "gray", fg = 'white',font = "Ms 10")
#            self.lbl_p3.place(x=(self.width+button_width), y=button_height2+50)
#            self.progress3 = 0
            self.video_file_progress_bar = tk.Label(self.window, text="VIDEO FILE PROGRESS BAR", width=24,bg = "grey", fg = 'white',font = "Ms 10")
            self.video_file_progress_bar.place(x=(self.width+button_width), y=button_height18)
            self.progress3=ttk.Progressbar(self.window,orient=tk.HORIZONTAL, variable=self.progress_var3, length=200, maximum=self.prog_MAX3,mode='determinate')
            self.progress3.place(x=(self.width+button_width), y=button_height19)
            #Processing bar

            if VIDEO_SCROLL_BAR == '1':
                self.scrollbar = tk.Scrollbar(self.leftframe,orient='vertical', width=16)
                self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y, expand=False)

                self.scrollbar1 = tk.Scrollbar(self.leftframe, orient='horizontal', width=16)
                self.scrollbar1.pack(side=tk.BOTTOM, fill=tk.X, expand=False)

    #            # Create a canvas that can fit the a.bove video source size
            self.canvas =tk.Canvas(self.leftframe, width=self.width, height=self.height)
            self.canvas.pack(side=tk.LEFT, anchor=tk.NW)

            if VIDEO_SCROLL_BAR == '1':
                self.scrollbar.config(command=self.canvas.yview)
                self.scrollbar1.config(command=self.canvas.xview)
                self.config(scrollregion=self.bbox('all'))


            if not self.vid.isOpened():
                print("VIDEO NOT FOUND....................")
                self.frame = self.frame1
                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.frame))
                self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
                self.canvas.update()


            self.left_frame_title1 = tk.Label(self.window, text='  Enter the following details \n     for Enrolment  ', bg = "white", fg = 'black',font = "Ms 10 ")
            self.left_frame_title1.place(x=(self.width+button_width), y=button_height3)

            self.btn_enrolment = tk.Button(self.window, text="Enrolment", width=24, bg = "blue", fg = 'white',font = "Ms 10 " ,command=self.enrolment, state=tk.NORMAL)
            self.btn_enrolment.place(x=(self.width+button_width), y=button_height4)

            self.lbl1 = tk.Label(self.window, text='Name',width=8,bg = "gray", fg = 'white',font = "Ms 10")
            self.lbl1.place(x=(self.width+button_width), y=button_height5)

            self.t1 = tk.Entry(self.window, text="Name", width =18 , font ="Ms 10 ",  bd=3, state='disabled')

            self.t1.place(x=(self.width+button_width2), y=button_height5)

            self.btn_start = tk.Button(self.window, text="Capture Faces", width=24, bg = "blue", fg = 'white',font = "Ms 10 " ,command=self.capture_faces, state=tk.NORMAL)

            self.btn_start.place(x=(self.width+button_width), y=button_height6)

            self.btn_training = tk.Button(self.window, text="Training", width=24,bg = "blue", fg = 'white',font = "Ms 10 " , state=tk.DISABLED, command=self.training)
            self.btn_training.place(x=(self.width+button_width),y=button_height7)



            self.lbl2 = tk.Label(self.window, text='Camera Selection',width=24,bg = "gray", fg = 'white',font = "Ms 10")
            self.lbl2.place(x=(self.width+button_width), y=button_height9)



            self.t2 = ttk.Combobox(self.window,
                           values=list(details_dict.keys()),
                            state="readonly")

            self.t2.place(x=(self.width+button_width+10), y=button_height10)
            self.t2.current(0)


            self.btn_IP_cam = tk.Button(self.window, text="Camera Select", width=24, bg = "blue", fg = 'white',font = "Ms 10 " ,command=self.browsefunc_ipcam, state=tk.NORMAL)
            self.btn_IP_cam.place(x=(self.width+button_width), y=button_height11)


            self.btn_browse = tk.Button(self.window, text="Browse Image/video File", width=24,bg = "blue", fg = 'white',font = "Ms 10" , state=tk.NORMAL, command=self.browsefunc)
            self.btn_browse.place(x=(self.width+button_width), y=button_height12)

            self.btn_recognize = tk.Button(self.window, text="Start Recognition", width=24,bg = "blue", fg = 'white',font = "Ms 10 " , state=tk.NORMAL, command=self.start_recognition)
            self.btn_recognize.place(x=(self.width+button_width), y=button_height13)

            self.label_status = tk.Label(self.window, text="Status", width=24,bg = "green", fg = 'white',font = "Ms 10")
            self.label_status.place(x=(self.width+button_width), y=button_height14)

            self.camera_status = tk.Label(self.window, text=" Web Cam", width=24,bg = "green", fg = 'white',font = "Ms 10")
            self.camera_status.place(x=(self.width+button_width), y=button_height15)

            self.Training_Phase = tk.Label(self.window, text="Training Phase:", width=24,bg = "green", fg = 'white',font = "Ms 10")
            self.Training_Phase.place(x=(self.width+button_width), y=button_height17)

#

            self.Recognition_Status()

            self.delay = 1

            self.btn_start.config(state=tk.DISABLED)
            self.btn_training.configure(state=tk.NORMAL)
            self.detection_graph = tf.Graph()
            self.sess=tf.Session(config=tf.ConfigProto(log_device_placement=True), graph=self.detection_graph )

            if not self.vid.isOpened():
                self.camera_status.config(text="No camera")
                self.camera_status.config(bg="red")
                self.btn_recognize.config(state=tk.DISABLED)
                self.btn_enrolment.config(state=tk.DISABLED)

            with self.sess.as_default():
                with self.detection_graph.as_default():
                    self.pnet, self.rnet, self.onet = self.create_mtcnn(self.sess, npy)
                    self.minsize = 20  # minimum size of face
                    self.threshold = [0.85, 0.85, 0.85]  # three steps's threshold
                    self.factor = 0.709  # scale factor
                    self.margin = 44

                    self.image_size = 182
                    self.input_image_size = 160

                    facenet.load_model(TRAIN_PB_MODEL)
                    self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                    self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                    self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                    self.embedding_size = self.embeddings.get_shape()[1]
                    facenet.load_model(model_pred_dir)
                    self.model_pred_input = tf.get_default_graph().get_tensor_by_name("dense_1_input:0")
                    self.model_pred_output = tf.get_default_graph().get_tensor_by_name("dense_4/Softmax:0")

            self.emb_array = np.zeros((1, self.embedding_size))
            self.update()

            if HLS_VERSION == '1':
                self.btn_enrolment.config(state=tk.DISABLED)
                self.btn_browse.config(state=tk.DISABLED)
                self.btn_enrolment.config(state=tk.DISABLED)
                self.btn_training.config(state=tk.DISABLED)

            self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.window.mainloop()



    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.vid.release()
            self.window.destroy()

    def enrolment(self):
        self.btn_recognize.config(state=tk.NORMAL)
        self.btn_training.configure(state=tk.DISABLED)
        self.btn_enrolment.config(state=tk.DISABLED)
        self.btn_start.config(state=tk.NORMAL)
        self.btn_browse.config(state=tk.DISABLED)

        ##enrolment code
        self.t1.config(state=tk.NORMAL)
        self.init_img_no=0
        self.flag_enrolment=1
        self.recognition_flag = 0
        self.browse_flag = 0
        self.Training_Phase.config(text="Enrolment Started")


    def update_capture_face(self):
        self.btn_start.config(state=tk.DISABLED)

        frame = self.frame.copy()

        if frame.ndim == 2:
            frame = facenet.to_rgb(frame)
        try:
            frame = frame[:, :, 0:3]
            bounding_boxes, _ = self.detect_face(frame)
            nrof_faces = bounding_boxes.shape[0]
            if nrof_faces > 1:
                self.Training_Phase.config(text="Multiple Faces detected ")
                self.Training_Phase.config(bg="red")
                self.btn_enrolment.config(state=tk.NORMAL)
                self.btn_start.config(state=tk.DISABLED)
                self.flag = 0
                self.flag_enrolment=0

                self.btn_recognize.config(state=tk.NORMAL)
                self.btn_training.configure(state=tk.NORMAL)
                self.btn_browse.config(state=tk.NORMAL)
                nrof_faces = 0

                return 0

            if(self.init_img_no < self.total_imgs):
                print("capturing count :  ", self.init_img_no, "/",self.total_imgs)

                self.init_img_no += 1
                self.frame = frame
                return 1

            else:
                self.btn_enrolment.config(state=tk.NORMAL)
                self.btn_start.config(state=tk.DISABLED)
                self.flag = 0
                self.flag_enrolment=0

                self.btn_recognize.config(state=tk.NORMAL)
                self.btn_training.configure(state=tk.NORMAL)
                self.btn_browse.config(state=tk.NORMAL)

                self.Training_Phase.config(text="Enrolment Completed")
                return 0
        except:
            self.Training_Phase.config(text="Enrolment Exception")
            pass

    def capture_faces(self):
        self.Training_Phase.config(text="Enrolment : Capture Faces")
        self.btn_recognize.config(state=tk.DISABLED)
        self.create_folder(TRAINED_FACES_DIR)     # NEW_FACE_DIR  Changed by BSTC
        nameofperson =self.t1.get()
        if(len(nameofperson) == 0):
            messagebox.showerror('Error', 'Please enter the name')
            self.btn_start.config(state=tk.NORMAL)

        else:
            self.flag = 1
            conn = self.create_connection(self.db_file)

            cur = conn.cursor()
            last_row = cur.execute('select * from faces').fetchall()[-1]
            face_id = last_row[0]+1
            face_id = int(face_id)
            last_row = self.create_record(conn,face_id,nameofperson)
            # NEW_FACE_DIR  Changed by BSTC
            self.face_folder = TRAINED_FACES_DIR +"/"+ str(face_id) + "/"   # NEW_FACE_DIR  Changed by BSTC
            self.create_folder(self.face_folder)
            self.t1.delete(0, tk.END)
            self.t1.config(state=tk.DISABLED)
            self.Temp_face_id = face_id
            return

    def get_enrolled_count(self,db_file):

        import sqlite3
        conn = sqlite3.connect(self.db_file)
        print("db : ",self.db_file)
        cur =  conn.cursor()
        cur_db_count = cur.execute("SELECT COUNT(*) FROM faces")
        cur_db_count = cur.fetchone()
        cur_db_count = cur_db_count[0]

        return cur_db_count

    def get_Folder_count(self):
       folder=-1

       self.path =TRAINED_FACES_DIR # ".\FRS_RUNTIME_FILES\FRS_DB\enrolled_faces_26"

       for dirnames in os.walk(self.path):
           folder = folder+1

       return folder

    def training(self):
        self.total_db_count = self.get_enrolled_count(self.db_file)
        print("Face Count:",self.total_db_count)
        self.total_folders = self.get_Folder_count()
        print("Total folders:",self.total_folders)

        if self.total_db_count == self.total_folders:

            self.Training_Phase.config(text="Training Initiated")
            self.recognition_flag = 2
            self.btn_IP_cam.config(state=tk.DISABLED)
            self.btn_enrolment.config(state=tk.DISABLED)
            self.btn_training.config(state=tk.DISABLED)
            self.btn_browse.config(state=tk.DISABLED)
            self.btn_start.config(state=tk.DISABLED)
            self.btn_recognize.config(state=tk.DISABLED)
            #  write code for  model enrolment here
            self.vid.release()

            frame = self.frame2
            self.ret_image=0
            self.folder_count=0
            self.eph=0

            f = open(CSV_FILE,"w")
            f.truncate()
            f.close()
            f = open(DB_FILE_CSV,"w")
            f.truncate()
            f.close()
            f = open(model_name,"w")
            f.truncate()
            f.close()

            f = open(model_pred_dir,"w")
            f.truncate()
            f.close()

            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
            self.canvas.update()
            self.training_process1()
        else:
            self.Training_Phase.config(text="Missmatch in the enroled faces")
            return


    def training_process1(self):
        self.Training_Phase.config(text="Training Process-1 Starts")
        emb_array = np.zeros((1, self.embedding_size))
        folder_names = os.listdir(TRAINED_FACES_DIR)

        if(self.folder_count<len(folder_names)):

            full_folder_path  = TRAINED_FACES_DIR +"/"+folder_names[self.folder_count]+"/"

            images = os.listdir(full_folder_path)
            for image in images:

                full_image_path = full_folder_path+image

                img = cv2.imread(full_image_path)

                scaled = cv2.resize(img, (self.input_image_size,self.input_image_size),interpolation=cv2.INTER_CUBIC)
                scaled = self.prewhiten(scaled)
                scaled_reshape=scaled.reshape(-1,self.input_image_size,self.input_image_size,3)
                feed_dict = {self.images_placeholder: scaled_reshape, self.phase_train_placeholder: False}
                emb_array[0, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)
                face_descriptor = list(emb_array[0, :])
                face_descriptor.insert(0, int(folder_names[self.folder_count]))

                with open(CSV_FILE, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)

                    writer.writerow(face_descriptor)

            self.folder_count=self.folder_count+1
            self.progress_var1.set(self.folder_count*100/len(folder_names))
            self.window.update_idletasks()
        else:
            self.ret_image=1
            print("training_process1 ends")
            self.Training_Phase.config(text="Training Process-1 Completed")
            self.training_process2()
            self.conn_csv = sqlite3.connect(DB_FILE_CSV)
            return

        self.window.after(self.delay, self.training_process1)

    def training_process2(self):

        if( self.ret_image==1):
            self.Training_Phase.config(text="Training Process-2 Starts")
            face_descriptor = []
            features = []
            labels = []
            row = ""
            with open(CSV_FILE) as f:
                li = f.readlines()
            shuffle(li)
            shuffle(li)
            shuffle(li)
            with open(CSV_FILE, 'w') as f:
                f.writelines(li)

    #        #####step2:
            self.conn_csv.close()
            os.remove(DB_FILE_CSV)
            cnx = sqlite3.connect(DB_FILE_CSV)
            df = pd.read_csv(CSV_FILE,  names = self.csv_header)
            sql.to_sql(df,  name=table_name, con=cnx )
            cnx.close()

            with open(CSV_FILE, newline="") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    labels.append(int(row[0]))
                    features.append(np.array(row[1:], dtype=np.float32))

            no_of_faces = len(features)
            train_features_def = features[:int(0.9*no_of_faces)]
            train_labels_def = labels[:int(0.9*no_of_faces)]
            test_features_def = features[int(0.9*no_of_faces):]
            test_labels_def = labels[int(0.9*no_of_faces):]



            with open(TRAIN_FEATURE, 'wb') as f:
            	pickle.dump(train_features_def, f)
            del train_features_def


            with open(TRAIN_LABEL, 'wb') as f:
            	pickle.dump(train_labels_def, f)
            del train_labels_def


            with open(TEST_FEATURE, 'wb') as f:
            	pickle.dump(test_features_def, f)
            del test_features_def


            with open(TEST_LABEL, 'wb') as f:
            	pickle.dump(test_labels_def, f)
            del test_labels_def


            with open(TRAIN_FEATURE, "rb") as f:
                self.train_images = np.array(pickle.load(f))

            with open(TRAIN_LABEL, "rb") as f:
                self.train_labels = np.array(pickle.load(f), dtype=np.int32)


            with open(TEST_FEATURE, "rb") as f:
                self.test_images = np.array(pickle.load(f))

            with open(TEST_LABEL, "rb") as f:
                self.test_labels = np.array(pickle.load(f), dtype=np.int32)

            self.train_labels = np_utils.to_categorical(self.train_labels)
            self.test_labels = np_utils.to_categorical(self.test_labels)
            self.model, self.callbacks_list = self.mlp_model()
            self.Training_Phase.config(text="Training Process-2 Completed")
            self.ret_image=2
            self.training_process3()
        else:
            return

    def training_process3(self):

        if(self.ret_image==2):
            if(self.eph<self.total_no_epochs):
                self.Training_Phase.config(text="Training Process-3 Starts")

                validation_data=(self.test_images, self.test_labels)
                self.model.fit(self.train_images, self.train_labels, validation_data=(self.test_images, self.test_labels), epochs=1, batch_size=32, callbacks=self.callbacks_list)
                self.eph+=1
                self.progress_var2.set(self.eph*100/self.total_no_epochs)
                self.window.update_idletasks()
            else:
                self.ret_image=3
                self.Training_Phase.config(text="Training Process-3 Compeleted")
                self.training_process4()
                return
        self.window.after(self.delay, self.training_process3)

    def training_process4(self):
        if(self.ret_image==3 ):
            self.Training_Phase.config(text="Training Process-4 Starts")
            scores = self.model.evaluate(self.test_images, self.test_labels, verbose=3)
            self.btn_enrolment.configure(state=tk.NORMAL)
            self.btn_browse.config(state=tk.NORMAL)
            self.btn_recognize.config(state=tk.NORMAL)

            model_name1 =load_model(model_name)


            frozen_graph= h5pb.freeze_session(K.get_session(),model_name1,output_names=[out.op.name for out in self.model.outputs])

            tf.train.write_graph(frozen_graph, MODEL_FILE_DIR, MODEL_PB_FILE, as_text=False)

            self.ret_image=4
            self.Training_Phase.config(text="Training Process-4 Completed")
            messagebox.showerror('Message', 'Training Completed')
            self.btn_IP_cam.config(state=tk.NORMAL)
            self.recognition_flag = 0
            self.browse_flag = 0
            self.Training_Phase.config(text="Training Phase")
        return


    def get_num_of_faces(self):
        return len(os.listdir(TRAINED_FACES_DIR))

    def mlp_model(self):
        num_of_faces = self.get_num_of_faces()
        model = Sequential()
        model.add(Dense(128, input_shape=(128, )))
        model.add(LeakyReLU(alpha=0.5))
        model.add(Dropout(0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.3))

        model.add(Dropout(0.3))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.3))

        model.add(Dropout(0.2))
        model.add(Dense(num_of_faces, activation='softmax'))


        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        checkpoint1 = ModelCheckpoint(model_name, monitor='acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint1]

        return model, callbacks_list

    def Recognition_Status(self):
        if  self.recognition_flag == 1:
            self.label_status.config(text="Recognition started")
            self.btn_recognize.config(text="Stop_Recognition")

        else:
            self.label_status.config(text="Recognition stopped")
            self.btn_recognize.config(text="Start_Recognition")

    def update(self):
        try:
            t0 = time.clock()
            self.count=self.count+1
            if (self.browse_flag ==1):     #Condition used when Offline Image/Video browsed
                ret, frame = self.video.get_frame()
#

                if self.video.Isvideocheck():
                    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (int(self.width), int(self.height)))
                else:
                    if self.recognition_flag == 2:
                        self.frame= self.frame2
                        self.webcam_flag =0
                    else:
                       # self.frame= self.frame1
                        self.webcam_flag = 0

            else:                           #Condition used when Live video streaming (Webcam/IP Cam)
                self.VideoCapture_IP()
                frame = self.frame.copy()

                frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ret = self.ret
                frame = cv2.resize(frame, (int(self.width), int(self.height)))
            if(self.recognition_flag == 0 ):      #Condition useed while Recognition if OFF

                if(self.flag_enrolment==1):
                    if(self.init_img_no == self.total_imgs):
                        self.flag_enrolment=0

                    if ( self.flag==1) and (self.webcam_flag==1):
                        result = self.update_capture_face()
                        if result:
                            bounding_boxes, _ = self.detect_face(frame)
                            nrof_faces = bounding_boxes.shape[0]
                            det = bounding_boxes[:, 0:4]
                            bb = np.zeros((nrof_faces,4), dtype=np.int32)
                            bb[0][0] = det[0][0]
                            bb[0][1] = det[0][1]
                            bb[0][2] = det[0][2]
                            bb[0][3] = det[0][3]

                            face_img=frame[bb[0][1]: bb[0][3],bb[0][0]: bb[0][2]]
                            img_path = self.face_folder+str(self.init_img_no)+".jpg"
                            cv2.imwrite(img_path, face_img)
                            cv2.rectangle(frame, (bb[0][0], bb[0][1]), (bb[0][2], bb[0][3]), (255, 255, 0), 2)

                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
            elif(self.recognition_flag == 1 ):      #Condition useed while Recognition if ON

                if ret :# and (frame):
                    self.extract_face_info(frame)

                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)


            t1 = time.clock()

            t3 = t1-t0
#            print("Time Taken Update function : ", t3)

        except:
            pass
        self.window.after(self.delay, self.update)

    def browsefunc(self):
        if(self.recognition_flag != 2):

            self.browse_flag = 1
            self.tag=[[None for y in range(7)] for x in range(self.people_count)]
            filename = ""
            filename = filedialog.askopenfilename()
            print('filename ', filename)

            if filename != "":
                self.vid.release()

                self.recognition_flag = 0
                self.webcam_flag=0
                self.btn_enrolment.config(state=tk.DISABLED)
                self.btn_start.config(state=tk.DISABLED)
                self.btn_training.configure(state=tk.DISABLED)
                self.video= MyVideoCapture(filename)
#                MyVideoCapture.check(self,filename)

                if filename.find(".mp4")>0:
                    print('file')
#                    self.video= MyVideoCapture(filename)
                    self.vid= cv2.VideoCapture(filename)
                    print('hello')
                    fps =self.vid.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
                    print('fps = ' + str(fps))
                    self.frame_count = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
                    print('number of frames = ' + str(self.frame_count))
                    duration = self.frame_count/fps
                    print('duration (S) = ' + str(duration))
                    minutes = int(duration/60)
                    seconds = duration%60
                    print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))
                    while(self.frame_count):
                        self.vid.read()
                        self.frame_index=self.vid.get(cv2.CAP_PROP_POS_FRAMES)
                        print('frame_index=' + str(self.frame_index))

                        self.progress3 = (self.frame_index/self.frame_count)*100;
                        # Vedio Play
                        self.progress_var3.set(self.progress3)
#                        self.vid.set(self.progress3)

                        self.window.update()
                        time.sleep(0.01)

                        if self.progress3 > 99:
                            return

                self.count = 0
                self.camera_status.config(text="Offline Video/Image")
#

    def browsefunc_ipcam(self):

        self.video.delete()
        self.browse_flag= 0
        self.tag=[[None for y in range(7)] for x in range(self.people_count)]
        self.face_index = self.people_count

        filename=config.get('Config_Parameters_Loc', self.t2.get())
        self.camera_status.config(text='IP:'+self.t2.get())

        self.recognition_flag = 0
        self.btn_start.config(state=tk.DISABLED)
        self.btn_training.configure(state=tk.DISABLED)
        if(filename == "0"):

            self.vid = cv2.VideoCapture(int(WEB_CAM))

            self.width = self.vid.set(cv2.CAP_PROP_FRAME_WIDTH,self.ScreenSizeX*0.80)
            self.height = self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT,self.ScreenSizeY*0.80)

            self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.width = self.ScreenSizeX*0.80
            self.height = self.ScreenSizeY*0.80
            print("width",self.width)
            print("height",self.height)
            self.webcam_flag=1
            self.btn_enrolment.config(state=tk.NORMAL)

        else:
            self.vid = cv2.VideoCapture(filename)
            self.webcam_flag=0
            self.btn_enrolment.config(state=tk.DISABLED)

        self.Recognition_Status()

        self.btn_IP_cam.configure(state=tk.NORMAL)
        if self.vid.isOpened():
            self.btn_recognize.config(state=tk.NORMAL)
            self.camera_status.config(bg="green")
        else:
            self.camera_status.config(text="No camera")
            self.camera_status.config(bg="red")

        if HLS_VERSION == '1' :
            self.btn_browse.config(state=tk.DISABLED)
            self.btn_training.configure(state=tk.DISABLED)
        else:
            self.btn_browse.config(state=tk.NORMAL)
            self.btn_training.configure(state=tk.DISABLED)

    def VideoCapture_IP(self):

            if self.vid.isOpened():
                self.ret, self.frame = self.vid.read()

            else:
                self.ret = True
                if self.recognition_flag == 2:
                    self.frame= self.frame2
                else:
                    self.frame= self.frame1


    def recognize_face(self,face_descriptor):
        pred = self.face_recognizer.predict(face_descriptor)
        pred_probab = pred[0]
        pred_class = list(pred_probab).index(max(pred_probab))
        return max(pred_probab), pred_class


    def get_face_names(self,db_file):
        face_ids = dict()
        import sqlite3
        conn = sqlite3.connect(db_file)
        sql_cmd = "SELECT * FROM faces"
        cursor = conn.execute(sql_cmd)
        for row in cursor:
            face_ids[row[0]] = row[1]
        return face_ids


    def extract_face_info(self,frame):


        if frame.ndim == 2:
            frame = facenet.to_rgb(frame)

        try:
            if self.face_index >= self.people_count-1 :
                self.face_index=0
                nrof_faces =0
                det=[]

            else:

                bounding_boxes, _ = self.detect_face(frame)

                nrof_faces = bounding_boxes.shape[0]
                det = bounding_boxes[:, 0:4]

            if len(det) == 0:
                self.tag=[[None for y in range(7)] for x in range(self.people_count)]

            objects=ct.update(det)
            if nrof_faces > 0:

                cropped = []
                scaled = []
                scaled_reshape = []
                bb = np.zeros((nrof_faces,4), dtype=np.int32)

                images = np.zeros((nrof_faces, self.input_image_size, self.input_image_size, 3))
                emb_array = np.zeros((nrof_faces, self.embedding_size))
                for i in range(nrof_faces):
                    bb[i][0] = det[i][0]
                    bb[i][1] = det[i][1]
                    bb[i][2] = det[i][2]
                    bb[i][3] = det[i][3]

                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)



                    if bb[i][0] <= 10 or bb[i][1] <= 10 or bb[i][2] >=( len(frame[0])-10 )or bb[i][3] >=( len(frame)-10):

                        continue
                    else:

                        cropped=frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]

                        scaled = cv2.resize(cropped, (self.input_image_size,self.input_image_size),interpolation=cv2.INTER_CUBIC)
                        scaled = self.prewhiten(scaled)

                        scaled_reshape=scaled.reshape(-1,self.input_image_size,self.input_image_size,3)
                        images[i,:,:,:] = scaled_reshape


                feed_dict = {self.images_placeholder:images, self.phase_train_placeholder: False}
                emb_array[0:nrof_faces, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)


                for i in range(nrof_faces):

                    if bb[i][0] <= 10 or bb[i][1] <= 10 or bb[i][2] >=( len(frame[0])-10 )or bb[i][3] >=( len(frame)-10):

                        continue
                    else:
                        cX = int((bb[i][0] + bb[i][2]) / 2.0)
                        cY = int((bb[i][1] + bb[i][3]) / 2.0)

                        self.emb_array[0,:] = emb_array[i,:].copy()
                        features=self.emb_array

                        feed_dict_pred = {self.model_pred_input: features}

                        output = self.sess.run(self.model_pred_output, feed_dict=feed_dict_pred)


                        pred_probab = output[0,:]
                        face_id = list(pred_probab).index(max(pred_probab))

                        probability = max(pred_probab)


                        ft=self.me(features,face_id)
#                        print("face_id",face_id,probability,ft)

                        if probability > 0.3:
                            if(self.browse_flag == 1 and self.length==1):
                                if (ft<1):
                                    cv2.putText(frame, self.face_names[face_id],  (bb[i][0], bb[i][1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

                            elif ft < 0.7:

                                for (objectID, centroid) in objects.items():

                                    if objectID > self.people_count-1:
                                        self.face_index=objectID
                                        break

                                    if(abs(centroid[0]-cX)<2 and abs(centroid[1]-cY)<2):

                                        self.tag[objectID].pop(0)
                                        self.tag[objectID].append(face_id)
                                        break

                                mod_faceid=mode(self.tag[objectID])
                                if(self.browse_flag == 1):
                                    if(self.length>1 and mod_faceid !=None):
                                        cv2.putText(frame, self.face_names[mod_faceid],  (bb[i][0], bb[i][1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

                                elif(mod_faceid !=None) :
                                    cv2.putText(frame, self.face_names[mod_faceid],  (bb[i][0], bb[i][1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


                            else:

                                for (objectID, centroid) in objects.items():
                                    if(abs(centroid[0]-cX)<2 and abs(centroid[1]-cY)<2):

                                        break

                                mod_faceid=mode(self.tag[objectID])
                                if (mod_faceid !=None) :
                                    cv2.putText(frame, self.face_names[mod_faceid],  (bb[i][0], bb[i][1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

                        else:

                            for (objectID, centroid) in objects.items():
                                if(abs(centroid[0]-cX)<2 and abs(centroid[1]-cY)<2):

                                    break

                            mod_faceid=mode(self.tag[objectID])
                            if (mod_faceid !=None) :
                                cv2.putText(frame, self.face_names[mod_faceid],  (bb[i][0], bb[i][1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

            else:
                print("Alignment Failure!!!")


        except:
            pass



    def start_recognition(self):
        if(self.recognition_flag == 0):
           self.recognition_flag = 1
           self.Recognition_Status()

           self.flag = 0
       	   self.tag=[[None for y in range(7)] for x in range(self.people_count)]
           self.face_index = self.people_count

           self.btn_enrolment.config(state=tk.DISABLED)
           self.btn_start.config(state=tk.DISABLED)
           self.btn_browse.config(state=tk.DISABLED)
           self.btn_training.configure(state=tk.DISABLED)
           self.btn_IP_cam.configure(state=tk.DISABLED)


           self.t1.delete(0, tk.END)
           self.t1.config(state=tk.DISABLED)

#           if(self.browse_flag == 1):
#               self.browse_flag = 0


           self.canvas.config(width=self.width)
           self.canvas.config(height=self.height)
        else:
            self.recognition_flag = 0
            self.Recognition_Status()

            self.btn_start.config(state=tk.DISABLED)
            self.btn_IP_cam.configure(state=tk.NORMAL)

            if HLS_VERSION == '1' :
               self.btn_enrolment.config(state=tk.DISABLED)
               self.btn_browse.config(state=tk.DISABLED)
               self.btn_training.configure(state=tk.DISABLED)
            else:
               self.btn_enrolment.config(state=tk.NORMAL)
               self.btn_browse.config(state=tk.NORMAL)
               self.btn_training.configure(state=tk.DISABLED)


    def stop_recognition(self):
        ###use self.vid for vs in recognition
       self.btn_start.config(state=tk.DISABLED)
       self.recognition_flag = 0
       if HLS_VERSION == '1' :
           self.btn_enrolment.config(state=tk.DISABLED)
           self.btn_browse.config(state=tk.DISABLED)
           self.btn_training.configure(state=tk.DISABLED)
       else:
           self.btn_enrolment.config(state=tk.NORMAL)
           self.btn_browse.config(state=tk.NORMAL)
           self.btn_training.configure(state=tk.DISABLED)


    def prewhiten(self,x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y


    def create_connection(self,db_file):
        try:
            conn = sqlite3.connect(db_file)

            return conn
        except Exception as e:
            print(e)
        return None


    def create_record(self,conn, face_id,name):
        my_list=(face_id,name)
        sql = ''' INSERT INTO faces(face_id, name)
                  VALUES(?,?) '''
        cur = conn.cursor()
        cur.execute(sql,my_list)
        conn.commit()
        return cur.lastrowid

    def create_folder(self,folder_name):
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

            ####for image in .exe
    def resource_path(self, relative_path):
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        return(os.path.join(base_path, relative_path))


    def readfromdatabase(self):
        self.cur.execute("SELECT * FROM table_record")
        return self.cur.fetchall()

    def create_mtcnn(self,sess, model_path):
        if not model_path:
            model_path,_ = os.path.split(os.path.realpath(__file__))

        with tf.variable_scope('pnet'):
            data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
            self.pnet = PNet({'data':data})
            self.pnet.load(os.path.join(model_path, 'det1.npy'), sess)
        with tf.variable_scope('rnet'):
            data = tf.placeholder(tf.float32, (None,24,24,3), 'input')
            self.rnet = RNet({'data':data})
            self.rnet.load(os.path.join(model_path, 'det2.npy'), sess)
        with tf.variable_scope('onet'):
            data = tf.placeholder(tf.float32, (None,48,48,3), 'input')
            self.onet = ONet({'data':data})
            self.onet.load(os.path.join(model_path, 'det3.npy'), sess)

        pnet_fun = lambda img : sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/input:0':img})
        rnet_fun = lambda img : sess.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'), feed_dict={'rnet/input:0':img})
        onet_fun = lambda img : sess.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'), feed_dict={'onet/input:0':img})
        return pnet_fun, rnet_fun, onet_fun

    def detect_face(self,img):

    # im: input image
    # minsize: minimum of faces' size
    # pnet, rnet, onet: caffemodel
    # threshold: threshold=[th1 th2 th3], th1-3 are three steps's threshold
    # fastresize: resize img from last scale (using in high-resolution images) if fastresize==true
        factor_count=0
        total_boxes=np.empty((0,9))

        points=np.empty(0)
        h=img.shape[0]
        w=img.shape[1]
        minl=np.amin([h, w])
        m=12.0/self.minsize
        minl=minl*m

        scales=[]
        while minl>=12:
            scales += [m*np.power(self.factor, factor_count)]
            minl = minl*self.factor
            factor_count += 1


        # first stage
        for j in range(len(scales)):
            scale=scales[j]
            hs=int(np.ceil(h*scale))
            ws=int(np.ceil(w*scale))
            im_data = self.imresample(img, (hs, ws))
            im_data = (im_data-127.5)*0.0078125
            img_x = np.expand_dims(im_data, 0)
            img_y = np.transpose(img_x, (0,2,1,3))
            out = self.pnet(img_y)
            out0 = np.transpose(out[0], (0,2,1,3))

            out1 = np.transpose(out[1], (0,2,1,3))

            boxes, _ = self.generateBoundingBox(out1[0,:,:,1].copy(), out0[0,:,:,:].copy(), scale)

            # inter-scale nms
            pick = self.nms(boxes.copy(), 0.5, 'Union')
            if boxes.size>0 and pick.size>0:
                boxes = boxes[pick,:]
                total_boxes = np.append(total_boxes, boxes, axis=0)


        numbox = total_boxes.shape[0]

        if numbox>0:
            pick = self.nms(total_boxes.copy(), 0.7, 'Union')
            total_boxes = total_boxes[pick,:]

            regw = total_boxes[:,2]-total_boxes[:,0]

            regh = total_boxes[:,3]-total_boxes[:,1]
            qq1 = total_boxes[:,0]+total_boxes[:,5]*regw
            qq2 = total_boxes[:,1]+total_boxes[:,6]*regh
            qq3 = total_boxes[:,2]+total_boxes[:,7]*regw
            qq4 = total_boxes[:,3]+total_boxes[:,8]*regh
            total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:,4]]))
            total_boxes = self.rerec(total_boxes.copy())
            total_boxes[:,0:4] = np.fix(total_boxes[:,0:4]).astype(np.int32)
            dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = self.pad(total_boxes.copy(), w, h)

        numbox = total_boxes.shape[0]
        if numbox>0:
            # second stage
            tempimg = np.zeros((24,24,3,numbox))

            for k in range(0,numbox):
                tmp = np.zeros((int(tmph[k]),int(tmpw[k]),3))
                tmp[dy[k]-1:edy[k],dx[k]-1:edx[k],:] = img[y[k]-1:ey[k],x[k]-1:ex[k],:]
                if tmp.shape[0]>0 and tmp.shape[1]>0 or tmp.shape[0]==0 and tmp.shape[1]==0:
                    tempimg[:,:,:,k] = self.imresample(tmp, (24, 24))
                else:
                    return np.empty()
            tempimg = (tempimg-127.5)*0.0078125
            tempimg1 = np.transpose(tempimg, (3,1,0,2))
            out = self.rnet(tempimg1)
            out0 = np.transpose(out[0])
            out1 = np.transpose(out[1])
            score = out1[1,:]
            ipass = np.where(score>self.threshold[1])
            total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(), np.expand_dims(score[ipass].copy(),1)])
            mv = out0[:,ipass[0]]
            if total_boxes.shape[0]>0:
                pick = self.nms(total_boxes, 0.7, 'Union')
                total_boxes = total_boxes[pick,:]
                total_boxes = self.bbreg(total_boxes.copy(), np.transpose(mv[:,pick]))
                total_boxes = self.rerec(total_boxes.copy())

        numbox = total_boxes.shape[0]

        if numbox>0:
            # third stage
            total_boxes = np.fix(total_boxes).astype(np.int32)
            dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = self.pad(total_boxes.copy(), w, h)
            tempimg = np.zeros((48,48,3,numbox))
            for k in range(0,numbox):
                tmp = np.zeros((int(tmph[k]),int(tmpw[k]),3))
                tmp[dy[k]-1:edy[k],dx[k]-1:edx[k],:] = img[y[k]-1:ey[k],x[k]-1:ex[k],:]
                if tmp.shape[0]>0 and tmp.shape[1]>0 or tmp.shape[0]==0 and tmp.shape[1]==0:
                    tempimg[:,:,:,k] = self.imresample(tmp, (48, 48))
                else:
                    return np.empty()
            tempimg = (tempimg-127.5)*0.0078125
            tempimg1 = np.transpose(tempimg, (3,1,0,2))
            out = self.onet(tempimg1)
            out0 = np.transpose(out[0])
            out1 = np.transpose(out[1])
            out2 = np.transpose(out[2])
            score = out2[1,:]
            points = out1
            ipass = np.where(score>self.threshold[2])
            points = points[:,ipass[0]]
            total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(), np.expand_dims(score[ipass].copy(),1)])
            mv = out0[:,ipass[0]]

            w = total_boxes[:,2]-total_boxes[:,0]+1
            h = total_boxes[:,3]-total_boxes[:,1]+1
            points[0:5,:] = np.tile(w,(5, 1))*points[0:5,:] + np.tile(total_boxes[:,0],(5, 1))-1
            points[5:10,:] = np.tile(h,(5, 1))*points[5:10,:] + np.tile(total_boxes[:,1],(5, 1))-1
            if total_boxes.shape[0]>0:
                total_boxes = self.bbreg(total_boxes.copy(), np.transpose(mv))
                pick = self.nms(total_boxes.copy(), 0.7, 'Min')
                total_boxes = total_boxes[pick,:]
                points = points[:,pick]

        return total_boxes, points

    def bbreg(self,boundingbox,reg):
    # calibrate bounding boxes
        if reg.shape[1]==1:
            reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

        w = boundingbox[:,2]-boundingbox[:,0]+1
        h = boundingbox[:,3]-boundingbox[:,1]+1
        b1 = boundingbox[:,0]+reg[:,0]*w
        b2 = boundingbox[:,1]+reg[:,1]*h
        b3 = boundingbox[:,2]+reg[:,2]*w
        b4 = boundingbox[:,3]+reg[:,3]*h
        boundingbox[:,0:4] = np.transpose(np.vstack([b1, b2, b3, b4 ]))
        return boundingbox

    def generateBoundingBox(self,imap, reg, scale):
        # use heatmap to generate bounding boxes
        stride=2
        cellsize=12
        t=self.threshold[0]
        imap = np.transpose(imap)
        dx1 = np.transpose(reg[:,:,0])
        dy1 = np.transpose(reg[:,:,1])
        dx2 = np.transpose(reg[:,:,2])
        dy2 = np.transpose(reg[:,:,3])
        y, x = np.where(imap >= t)
        if y.shape[0]==1:
            dx1 = np.flipud(dx1)
            dy1 = np.flipud(dy1)
            dx2 = np.flipud(dx2)
            dy2 = np.flipud(dy2)
        score = imap[(y,x)]
        reg = np.transpose(np.vstack([ dx1[(y,x)], dy1[(y,x)], dx2[(y,x)], dy2[(y,x)] ]))
        if reg.size==0:
            reg = np.empty((0,3))
        bb = np.transpose(np.vstack([y,x]))
        q1 = np.fix((stride*bb+1)/scale)
        q2 = np.fix((stride*bb+cellsize-1+1)/scale)
        boundingbox = np.hstack([q1, q2, np.expand_dims(score,1), reg])
        return boundingbox, reg

    # function pick = nms(boxes,threshold,type)
    def nms(self,boxes, threshold, method):
        if boxes.size==0:
            return np.empty((0,3))
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        s = boxes[:,4]
        area = (x2-x1+1) * (y2-y1+1)
        I = np.argsort(s)
        pick = np.zeros_like(s, dtype=np.int16)
        counter = 0
        while I.size>0:
            i = I[-1]
            pick[counter] = i
            counter += 1
            idx = I[0:-1]
            xx1 = np.maximum(x1[i], x1[idx])
            yy1 = np.maximum(y1[i], y1[idx])
            xx2 = np.minimum(x2[i], x2[idx])
            yy2 = np.minimum(y2[i], y2[idx])
            w = np.maximum(0.0, xx2-xx1+1)
            h = np.maximum(0.0, yy2-yy1+1)
            inter = w * h
            if method is 'Min':
                o = inter / np.minimum(area[i], area[idx])
            else:
                o = inter / (area[i] + area[idx] - inter)
            I = I[np.where(o<=threshold)]
        pick = pick[0:counter]
        return pick

    # function [dy edy dx edx y ey x ex tmpw tmph] = pad(total_boxes,w,h)
    def pad(self,total_boxes, w, h):
        # compute the padding coordinates (pad the bounding boxes to square)
        tmpw = (total_boxes[:,2]-total_boxes[:,0]+1).astype(np.int32)

        tmph = (total_boxes[:,3]-total_boxes[:,1]+1).astype(np.int32)
        numbox = total_boxes.shape[0]

        dx = np.ones((numbox), dtype=np.int32)
        dy = np.ones((numbox), dtype=np.int32)
        edx = tmpw.copy().astype(np.int32)
        edy = tmph.copy().astype(np.int32)

        x = total_boxes[:,0].copy().astype(np.int32)
        y = total_boxes[:,1].copy().astype(np.int32)
        ex = total_boxes[:,2].copy().astype(np.int32)
        ey = total_boxes[:,3].copy().astype(np.int32)

        tmp = np.where(ex>w)
        edx.flat[tmp] = np.expand_dims(-ex[tmp]+w+tmpw[tmp],1)
        ex[tmp] = w

        tmp = np.where(ey>h)
        edy.flat[tmp] = np.expand_dims(-ey[tmp]+h+tmph[tmp],1)
        ey[tmp] = h

        tmp = np.where(x<1)
        dx.flat[tmp] = np.expand_dims(2-x[tmp],1)
        x[tmp] = 1

        tmp = np.where(y<1)
        dy.flat[tmp] = np.expand_dims(2-y[tmp],1)
        y[tmp] = 1

        return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph

    # function [bboxA] = rerec(bboxA)
    def rerec(self,bboxA):
        # convert bboxA to square
        h = bboxA[:,3]-bboxA[:,1]
        w = bboxA[:,2]-bboxA[:,0]
        l = np.maximum(w, h)
        bboxA[:,0] = bboxA[:,0]+w*0.5-l*0.5
        bboxA[:,1] = bboxA[:,1]+h*0.5-l*0.5
        bboxA[:,2:4] = bboxA[:,0:2] + np.transpose(np.tile(l,(2,1)))
        return bboxA

    def imresample(self,img, sz):
        im_data = cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_AREA) #@UndefinedVariable
        return im_data


    def me(self,comp,id):

        if self.face_avg_dict.__contains__(str(id)):

            return np.linalg.norm(comp-self.face_avg_dict[str(id)])

        ar=[]

        self.cursor_csv.execute("select * from my_table WHERE level_0=?",(str(id),))

        rows = self.cursor_csv.fetchall()

        for row in rows:
           ar.append(np.array(row[1:],dtype=np.float))


        ot=np.mean(ar,axis=0)
        self.face_avg_dict[str(id)]=ot
        ot=np.linalg.norm(comp-ot)

        return ot

class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source

        self.video = cv2.VideoCapture(video_source)


#        if OS_NAME == "windows":
#            self.frame1 = cv2.imread(IMAGE_NOVIDEO)
#        else:
#            self.frame1 = cv2.imread(os.path.expanduser(os.path.join(IMAGE_NOVIDEO)))
#        print("width : ",self.width)
#        print("height : ",self.height)
        if not self.video.isOpened():
            self.width = self.frame1.shape[1]
            self.height = self.frame1.shape[0]

            self.frame1=cv2.cvtColor(self.frame1, cv2.COLOR_RGB2BGR)
            messagebox.showerror('Error', 'Unable to open video source')
#        print("width : ",self.width)
#        print("height : ",self.height)


#    def check(self,filename):
#
#        if filename.find(".mp4")>0:
#            print('file')
#            self.video = cv2.VideoCapture(filename)
#
#            #self.video= MyVideoCapture(filename)
##            self.cap= cv2.VideoCapture(filename)
#            print('hello')
#            fps =self.video.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
#            print('fps = ' + str(fps))
#            self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
#            print('number of frames = ' + str(self.frame_count))
#            duration = self.frame_count/fps
#            print('duration (S) = ' + str(duration))
#            minutes = int(duration/60)
#            seconds = duration%60
#            print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))
#            while(self.frame_count):
#                self.video.read()
#                self.frame_index=self.video.get(cv2.CAP_PROP_POS_FRAMES)
#                print('frame_index=' + str(self.frame_index))
#                #self.video= MyVideoCapture(filename)
#                self.progress3 = (self.frame_index/self.frame_count)*100;
#               # self.progress_var3.set(self.folder_count*100/frame_index)
#                self.progress_var3.set(self.progress3)
##                self.video.update()
#                self.window.update()
#                time.sleep(0.01)
#
#                if self.progress3 > 99:
#                    return


    def get_frame(self):
        if self.video.isOpened():
            ret, frame = self.video.read()


            if ret:
                # Return a boolean success flag and the current frame converted to BGR
#                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                return(ret,frame)
            else:
                return (ret, 0)
        else:
            ret = True
            return (ret, self.frame1)

    def delete(self):
        if self.video.isOpened():
            self.video.release()

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def Isvideocheck(self):
        if self.video.isOpened():
            return 1
        else:
            return 0

def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated

class Network(object):

    def __init__(self, inputs, trainable=True):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable

        self.setup()

    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path, encoding='latin1').item() #pylint: disable=no-member

        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in iteritems(data_dict[op_name]):
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, string_types):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape, trainable=self.trainable)

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self,
             inp,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding='SAME',
             group=1,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = int(inp.get_shape()[-1])
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i // group, c_o])
            # This is the common-case. Convolve the input without any further complications.
            output = convolve(inp, kernel)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def prelu(self, inp, name):
        with tf.variable_scope(name):
            i = int(inp.get_shape()[-1])
            alpha = self.make_var('alpha', shape=(i,))
            output = tf.nn.relu(inp) + tf.multiply(alpha, -tf.nn.relu(-inp))
        return output

    @layer
    def max_pool(self, inp, k_h, k_w, s_h, s_w, name, padding='SAME'):
        self.validate_padding(padding)
        return tf.nn.max_pool(inp,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def fc(self, inp, num_out, name, relu=True):
        with tf.variable_scope(name):
            input_shape = inp.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= int(d)
                feed_in = tf.reshape(inp, [-1, dim])
            else:
                feed_in, dim = (inp, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=name)
            return fc


    """
    Multi dimensional softmax,
    refer to https://github.com/tensorflow/tensorflow/issues/210
    compute softmax along the dimension of target
    the native softmax only supports batch_size x dimension
    """
    @layer
    def softmax(self, target, axis, name=None):
        max_axis = tf.reduce_max(target, axis, keepdims=True)
        target_exp = tf.exp(target-max_axis)
        normalize = tf.reduce_sum(target_exp, axis, keepdims=True)
        softmax = tf.div(target_exp, normalize, name)
        return softmax

class PNet(Network):
    def setup(self):
        (self.feed('data') #pylint: disable=no-value-for-parameter, no-member
             .conv(3, 3, 10, 1, 1, padding='VALID', relu=False, name='conv1')
             .prelu(name='PReLU1')
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(3, 3, 16, 1, 1, padding='VALID', relu=False, name='conv2')
             .prelu(name='PReLU2')
             .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv3')
             .prelu(name='PReLU3')
             .conv(1, 1, 2, 1, 1, relu=False, name='conv4-1')
             .softmax(3,name='prob1'))

        (self.feed('PReLU3') #pylint: disable=no-value-for-parameter
             .conv(1, 1, 4, 1, 1, relu=False, name='conv4-2'))

class RNet(Network):
    def setup(self):
        (self.feed('data') #pylint: disable=no-value-for-parameter, no-member
             .conv(3, 3, 28, 1, 1, padding='VALID', relu=False, name='conv1')
             .prelu(name='prelu1')
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(3, 3, 48, 1, 1, padding='VALID', relu=False, name='conv2')
             .prelu(name='prelu2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .conv(2, 2, 64, 1, 1, padding='VALID', relu=False, name='conv3')
             .prelu(name='prelu3')
             .fc(128, relu=False, name='conv4')
             .prelu(name='prelu4')
             .fc(2, relu=False, name='conv5-1')
             .softmax(1,name='prob1'))

        (self.feed('prelu4') #pylint: disable=no-value-for-parameter
             .fc(4, relu=False, name='conv5-2'))

class ONet(Network):
    def setup(self):
        (self.feed('data') #pylint: disable=no-value-for-parameter, no-member
             .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv1')
             .prelu(name='prelu1')
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv2')
             .prelu(name='prelu2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv3')
             .prelu(name='prelu3')
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(2, 2, 128, 1, 1, padding='VALID', relu=False, name='conv4')
             .prelu(name='prelu4')
             .fc(256, relu=False, name='conv5')
             .prelu(name='prelu5')
             .fc(2, relu=False, name='conv6-1')
             .softmax(1, name='prob1'))

        (self.feed('prelu5') #pylint: disable=no-value-for-parameter
             .fc(4, relu=False, name='conv6-2'))

        (self.feed('prelu5') #pylint: disable=no-value-for-parameter
             .fc(10, relu=False, name='conv6-3'))
App(tk.Tk(),"BEL Face Identity Registration and Recognition System ")

