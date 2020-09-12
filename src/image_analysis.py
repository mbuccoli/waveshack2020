# %%
import os 
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.cluster import DBSCAN
# %%

class Image_Analysis:
    def __init__(self, name="", fn=None):
        if fn is None:
            root = tk.Tk()
            root.withdraw()
            fn = filedialog.askopenfilename()
            name = os.path.split(fn)[-1].split(".")[0]
        self.fn=fn
        self.name=name
        self.im =cv2.imread(self.fn)
        if self.im.shape[0]>720 or  self.im.shape[1]>1280:
            scale = min(720/self.im.shape[0], 1280/self.im.shape[1])
            self.im = cv2.resize(self.im, (int(scale*self.im.shape[1]), int(scale*self.im.shape[0])))
            
        self.imgray = cv2.cvtColor(self.im,cv2.COLOR_BGR2GRAY)
        self.ret,self.thresh = cv2.threshold(self.imgray,127,255,0)
        

    def show(self,im=None):
        if im is None:
            cv2.imshow(self.name, self.im)
        else:
            cv2.imshow(self.name, im)
        cv2.waitKey()
    def get_stats(self):
        means, stddevs  = cv2.meanStdDev(self.im)
        return means, stddevs
    def get_contours(self):
        contours, hierarchy = cv2.findContours(self.thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        len_contours=[len(cont) for cont in contours]        
        return contours, len_contours

    def get_lines(self):   
        edges = cv2.Canny(self.imgray,50,150,apertureSize = 3)
        lines = cv2.HoughLines(edges,1,np.pi/180,200)
        return lines
    
    def get_main_colors(self):
        pixels = self.im.reshape((-1,3))
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 3
        _, labels, (centers) = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        return centers

# %%
if __name__=="__main__":    
    city_fns={"milan":"../images/milan.jpg"}
    city="milan"

    #Ia= Image_Analysis(name=city.title(), fn=city_fns[city])
    Ia= Image_Analysis(name=city.title(), fn=None)
    means, stddevs=Ia.get_stats()
    print(means, stddevs)
    contours,_=Ia.get_contours()
    lines=Ia.get_lines()
    imcontours = cv2.drawContours(Ia.im, contours, -1, (0,255,0), 3)
    Ia.show(imcontours)
    centres = Ia.get_main_colors()
    print(centres)

# %%
