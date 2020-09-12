# %%
import os 
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt
from scipy import stats

# %%
city="milan"

#root = tk.Tk()
#root.withdraw()

#fn = filedialog.askopenfilename()
#title = os.path.split(fn)[-1].split(".")[0]

city_fns={"milan":"../images/milan.jpg"}
im=cv2.imread(city_fns[city])
#cv2.imshow(city.title(), im)
#print("hello world")
#cv2.waitKey()
# %%
means, stddevs  = cv2.meanStdDev(im)
print(means)
print(stddevs)
#ignore histogram mean and standard deviation
color = ('b','g','r')
mean_list = []
std_list = []
for i,col in enumerate(color):
    histr = cv2.calcHist([im],[i],None,[256],[0,256])
    mean = np.mean(histr).item()
    std = np.std(histr).item()
    mean_list.append(mean)
    std_list.append(std)
    #plt.plot(histr,color = col)
    #plt.xlim([0,256])

#plt.show()
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
im = cv2.drawContours(im, contours, -1, (0,255,0), 3)
print(len(contours))
#cv2.imshow(city.title(),im)
#cv2.waitKey()
edges = cv2.Canny(imgray,50,150,apertureSize = 3)
lines = cv2.HoughLines(edges,1,np.pi/180,200)
print(len(lines))
