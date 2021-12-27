# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 18:22:48 2021

@author: user1
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 13:37:27 2021

@author: user1
"""

from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout,QDesktopWidget,QWidget,QTableWidget,QTableView,QTableWidgetItem,QHeaderView,QGraphicsScene,QGraphicsPixmapItem,QFileDialog

import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgb2hsv,rgb2lab,rgb2gray
from skimage.feature import daisy


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import math
import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier
from sklearn import decomposition
from sklearn import datasets
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
import sys
from PyQt5.QtWidgets import QApplication, QTableView,QFileDialog
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QAbstractTableModel, Qt
from sklearn.model_selection import KFold
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.decomposition import PCA
import numpy as np
import cv2
import numpy as np
import os 
import glob
import scipy.cluster

from PyQt5.QtWidgets import QApplication, QTableView,QFileDialog
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QAbstractTableModel, Qt
from PyQt5.QtGui import *
from goruntu import Ui_Form

import sys
import os
import pandas as pd
import cv2
import numpy as np
import shutil
import random
from PyQt5 import QtWidgets, QtCore
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from PyQt5.QtWidgets import QMessageBox   
from pathlib import Path
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Input, merge, UpSampling2D,BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf
from skimage.feature import daisy
from skimage.color import rgb2hsv,rgb2lab,rgb2gray,rgb2gray


rgbPath="C:/Users/user1/Desktop/Deneme/cicek/" 
hsvPath="C:/Users/user1/Desktop/Deneme/cicekhsv/"
ciePath="C:/Users/user1/Desktop/Deneme/cicekcie/"
futurePath="C:/Users/yalci/Desktop/dersler/futuress/"

class FileDialogs(QtWidgets.QFileDialog):
    def __init__(self, *args, **kwargs):
        super(FileDialogs, self).__init__(*args, **kwargs)
        self.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        self.setFileMode(QtWidgets.QFileDialog.ExistingFiles)

    def accept(self):
        super(FileDialogs, self).accept()
class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
       
      #bu(3,3,136 ) feturus'un tututlduğu koordinat ne dxemeek 
      #379 ne demek
      #mutlak değerler içerisindeeki giridler ne 
      #daisyi nasıl büyütebilirim
      #bir de sen sift için yaptın mı
      #klasörün içeriisndeki bütün resimler için yapabiliyor mu 

        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.verisetiyukle)
        self.ui.pushButton_2.clicked.connect(self.kfoldfonk)
        self.ui.pushButton_3.clicked.connect(self.hold)
  
        
        self.ui.pushButton_8.clicked.connect(self.knn)
        self.ui.pushButton_9.clicked.connect(self.svc)
        self.ui.pushButton_11.clicked.connect(self.unet_model)
        self.ui.pushButton_10.clicked.connect(self.randomforest)
        self.ui.pushButton_12.clicked.connect(self.orb)
        self.ui.pushButton_13.clicked.connect(self.upload_rgb)
        self.ui.pushButton_15.clicked.connect(self.sift)
        
        #self.imag_tipi=""
        self.images=[]
        self.show()
    def upload_rgb(self):
        self.dosyaadi, isim = QtWidgets.QFileDialog.getOpenFileName(None,"Resim Dosyası Seç","","Resim Türü(*.jpg)")   
    def siftcevirme(self,image,i):
         print("serrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
         print(image)
         img = cv2.imread(image)
         sift = cv2.xfeatures2d.SIFT_create()
         kp, des = sift.detectAndCompute(img,None)
         x,y=[],[]
         for keyPoint in kp:
             x.append(int(keyPoint.pt[0]))
             y.append(int(keyPoint.pt[1]))
             for i in range(5):
                 try:
                     
                     
                     region=img[abs((y[i]-25)):abs((y[i]+25)),abs((x[i]-25)):abs((x[i]+25))]
                     region=rgb2gray(region)
                     desc, descs_img = daisy(region, step=10, radius=10, rings=2, histograms=8,orientations=8,visualize=True)
                     print("Features:",desc.shape)
                     print(desc)
                     print(i)
                     cv2.imwrite("./sift_image/sift"+str(i)+".jpg",np.array(descs_img))
                      
                 except:
                        
                        continue
                     
                   
                        
    def sift(self):
        yol = 'C:/Users/user1/Desktop/Deneme'
        for folder in os.listdir(yol):
            for filename in os.listdir(yol+'/'+folder):
                img = cv2.imread(yol+'/'+folder+'/'+filename, 1);
                sift = cv2.xfeatures2d.SIFT_create()
                kp, des = sift.detectAndCompute(img,None)
                
                x,y=[],[]
                for keyPoint in kp:
                    x.append(int(keyPoint.pt[0]))
                    y.append(int(keyPoint.pt[1]))
                        
                for i in range(5):
                    try:
                        region=img[abs((y[i]-25)):abs((y[i]+25)),abs((x[i]-25)):abs((x[i]+25))]
                        region=rgb2gray(region)
                        desc, descs_img = daisy(region, step=10, radius=10, rings=2, histograms=8,orientations=8,visualize=True)
                        print("Features:",desc.shape)
                        print(desc)
                        print(i)
                    except:
                        continue
                
          
        print("girdi")              
    def orb(self):
       
        yol = './flowers'
        for folder in os.listdir(yol):
            for filename in os.listdir(yol+'/'+folder):
                img = cv2.imread(yol+'/'+folder+'/'+filename, 1);
                orb = cv2.ORB_create()
                kp = orb.detect(img,None)
                kp, des = orb.compute(img, kp)
                x,y=[],[]
                for keyPoint in kp:
                    x.append(int(keyPoint.pt[0]))
                    y.append(int(keyPoint.pt[1]))
                        
                for i in range(5):
                    try:
                        region=img[abs((y[i]-25)):abs((y[i]+25)),abs((x[i]-25)):abs((x[i]+25))]
                        region=rgb2gray(region)
                        desc, descs_img = daisy(region, step=10, radius=10, rings=2, histograms=8,orientations=8,visualize=True)
                        print("Features:",desc.shape)
                        print(desc)
                        print(i)
                    except:
                        continue
                
          
        print("girdi")
        
    def rgb2sift(self):
         
        rgb_path = "./cicekrgb/"
        rgb_image = glob.glob(rgb_path + "*.jpg")
         
        rgb_image = [x.replace('\\', '/') for x in rgb_image]
         
        for i,image in enumerate(rgb_image):
             
            self.sift(image,i)
            
    def hsvsift(self):
         
        hsv_path = "./cicekhsv/"
        hsv_image = glob.glob(hsv_path + "*.jpg")
         
        hsv_image = [x.replace('\\', '/') for x in hsv_image]
         
        for i,image in enumerate(hsv_image):
             
            self.sift(image,i)
    
    def ciesift(self):
         
        cie_path = "./cicekhsv/"
        cie_image = glob.glob(cie_path + "*.jpg")
         
        cie_image = [x.replace('\\', '/') for x in cie_image]
       #  
        for i,image in enumerate(cie_image):
             
            self.sift(image,i)        
             
        
    def ac(self): 
        self.dialog = FileDialogs()
        if self.dialog.exec() == QtWidgets.QDialog.Accepted:
            print("sasasa")
        self.ui.textEdit.setText(str(self.dialog.selectedFiles()))
        #self.ui.tableView_2.setModel(self.dialog.selectedFiles())
        #self.ui.tableView_6.setText(str(self.dialog.selectedFiles()))
   
    def cie (self):
        
        cie_image = []
        
        for i,image in enumerate(self.X):
            img = cv2.imread(image)     
            img = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
            cv2.imwrite("./cicekcie/"+str(i)+".jpg",img)
            
            cie_image.append(np.array(img))
        
        print(cie_image)
    def hsv(self):
        hsv_image = []
        for i,image in enumerate(self.X):
            img = cv2.imread(image)     
            img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            cv2.imwrite("./cicekhsv/"+str(i)+".jpg",img) 
            
            hsv_image.append(np.array(img))
        
        
    def rgb(self):
            
        rgb_image = []
        for i,image in enumerate(self.X):
            img = cv2.imread(image)     
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            cv2.imwrite("./cicekrgb/"+str(i)+".jpg",img) 
            
            rgb_image.append(np.array(img))
        
            
    def verisetiyukle(self):
        self.folder = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Resim klasörünüzü seçin"))
        self.dosyaislem=self.folder
        print(self.dosyaislem)
        
        genelpath = self.dosyaislem
        print(genelpath)
        pathDaisy = genelpath + "/daisy/"
        pathDandelion = genelpath +"/dandelion/"
        pathRose = genelpath +"/rose/"
        pathSunflower =genelpath + "/sunflower/"
        pathTulip = genelpath +"/tulip/"
        
        
        dataDaisy = glob.glob(pathDaisy + "*.jpg")
        dataDandelion = glob.glob(pathDandelion + "*.jpg")
        dataRose = glob.glob(pathRose + "*.jpg")
        dataSunflower = glob.glob(pathSunflower + "*.jpg")
        dataTulip = glob.glob(pathTulip + "*.jpg")
        
        
        dataDaisy = [x.replace('\\', '/') for x in dataDaisy]
        dataDandelion = [x.replace('\\', '/') for x in dataDandelion]
        dataRose = [x.replace('\\', '/') for x in dataRose]
        dataSunflower = [x.replace('\\', '/') for x in dataSunflower]
        dataTulip = [x.replace('\\', '/') for x in dataTulip]
        
        
        data_expand = []
        
        for x in dataDaisy:
            data_expand.append([x,0])
        for x in dataDandelion:
            data_expand.append([x,1])
        for x in dataRose:
            data_expand.append([x,2])
        for x in dataSunflower:
            data_expand.append([x,3])
        for x in dataTulip:
            data_expand.append([x,4])
        
        self.df = pd.DataFrame(data_expand, columns = ["image", "label"])
        
        self.ui.tableView.setModel(pandasModel(self.df))
        self.ui.tableView_2.setModel(pandasModel(self.df))
      
        self.X = self.df["image"]
        
        
            
        self.y = self.df["label"]
        
        
        print(self.X)
        print(self.y)
  
    def kfoldfonk(self):## X=self.veriseti.iloc[:, :-1].values   
        kf = KFold(n_splits=int(self.ui.comboBox.currentText()))
        KFold(n_splits=2, random_state=None, shuffle=False)
        for train_index, test_index in kf.split(self.X):
            print("TRAIN:", train_index, "TEST:", test_index)
            self.X_train, self.X_test = self.X[train_index], self.X[test_index]
            self.Y_train, self.Y_test = self.y[train_index], self.y[test_index]
            print(self.X_train)
    def hold(self):
        print("x")
        print(self.X)
        
        image_list = []
        
        for image in self.X:
            image_list.append(self.string2img(image))
        image_list = np.array(image_list)
        
        print("image converted")
        print(image_list)
        self.X_train, self.X_test, self.Y_train, self.Y_test = model_selection.train_test_split(image_list, self.y, test_size=float(self.ui.comboBox_2.currentText()), random_state=0)
        print("Holdout")
        print("Train veri sayısı: {}".format(self.X_train.shape[0]))
    def string2img(self, img_path):
        
        image = cv2.imread(img_path)
        image = cv2.resize(image, (100,100))
        
        return image.flatten()
    def  randomforest(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import confusion_matrix
        rfc = RandomForestClassifier()
        rfc.fit(self.X_train, self.Y_train)
        self.y_pred=rfc.predict(self.X_test)
        cm=confusion_matrix(self.Y_test,self.y_pred)
        print(str("karmaşıklık matrisi:\n{}".format(str(confusion_matrix(self.Y_test,self.y_pred)))))
        TP =cm[1,1]
        TN=cm[0,0]
        FP= cm[0,1]
        FN= cm[1,0]
        sensi= TP/(TP+FN)
        spec= TN/(TN+FP)
        print("TP:{}".format(TP))
        print("FN: {}".format(FN))
        print("SENSİVİTİC = {}".format(sensi*100.0))
        print("spec = {}".format(spec*100.0))
    def knn(self):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import confusion_matrix
        knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski")
        knn.fit(self.X_train, self.Y_train)
        self.y_pred=knn.predict(self.X_test)
        cm=confusion_matrix(self.Y_test,self.y_pred)
        print(str("karmaşıklık matrisi:\n{}".format(str(confusion_matrix(self.Y_test,self.y_pred)))))
        #self.ui.textEdit_12.setText(str("karmaşıklık matrisi:\n{}".format(str(confusion_matrix(self.Y_test,self.y_pred)))))
        TP =cm[1,1]
        TN=cm[0,0]
        FP= cm[0,1]
        FN= cm[1,0]
        sensi= TP/(TP+FN)
        spec= TN/(TN+FP)
        print("TP:{}".format(TP))
        print("FN: {}".format(FN))
        print("SENSİVİTİC = {}".format(sensi*100.0))
        print("spec = {}".format(spec*100.0))
        
    def svc(self):
       
        svc=SVC(kernel="poly")
        svc.fit(self.X_train,self.Y_train)
        y_pred=svc.predict(self.X_test)
        cm=confusion_matrix(self.Y_test,y_pred)
        print(str("karmaşıklık matrisi:\n{}".format(str(confusion_matrix(self.Y_test,y_pred)))))
        TP =cm[1,1]
        TN=cm[0,0]
        FP= cm[0,1]
        FN= cm[1,0]
        sensi= TP/(TP+FN)
        spec= TN/(TN+FP)
        print("SENSİPİCİK:\n" " %.2f%%" %  (sensi*100.0))
        print("SPECTİF:\n" " %.2f%%" % (spec*100.0))
        print(str(sensi))
        print(str(spec))
        
#        pyplot.plot(self.X_train,self.Y_train,color='red')
#        pyplot.plot(self.X_test,y_pred,color='black')
#        pyplot.savefig("./SVC.png")
        #self.pixmap = QPixmap("./SVC.png")
       # self.ui.label.setPixmap(self.pixmap)
       
    def unet_model(self):
    
        inputs = Input((100,100,3))
        
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same') (inputs)
        batch1 = BatchNormalization(axis=1)(conv1)
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same') (batch1)
        batch1 = BatchNormalization(axis=1)(conv1)
        pool1 = MaxPooling2D((2, 2)) (batch1)
        
        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same') (pool1)
        batch2 = BatchNormalization(axis=1)(conv2)
        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same') (batch2)
        batch2 = BatchNormalization(axis=1)(conv2)
        pool2 = MaxPooling2D((2, 2)) (batch2)
        
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same') (pool2)
        batch3 = BatchNormalization(axis=1)(conv3)
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same') (batch3)
        batch3 = BatchNormalization(axis=1)(conv3)
        pool3 = MaxPooling2D((2, 2)) (batch3)
        
        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same') (pool3)
        batch4 = BatchNormalization(axis=1)(conv4)
        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same') (batch4)
        batch4 = BatchNormalization(axis=1)(conv4)
      
        conv5 = Conv2D(1, (1, 1), activation='sigmoid')(batch4)
    
        model = Model(inputs=[inputs], outputs=[conv5])
    
        model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        
        self.history = model.fit(self.X_train,validation_split=0.05,batch_size=1,epochs=3,shuffle=True,verbose=1)
        
        self.y_pred = model.predict(self.X_test)
        conf_matrix = confusion_matrix(self.Y_test, self.Y_pred)
        
        print(str(conf_matrix))
      
class pandasModel(QAbstractTableModel):
    def __init__(self, data):
            QAbstractTableModel.__init__(self)
            self._data = data
    def rowCount(self, parent=None):
            return self._data.shape[0]
    def columnCount(self, parnet=None):
            return self._data.shape[1]
            
    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None
     

    def headerData(self, col, orientation, role):
            if orientation == Qt.Horizontal and role == Qt.DisplayRole:
                return self._data.columns[col]
            return None


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = MyWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

               
            
            
                
        
        
        
        
        
        
        
        
        
        