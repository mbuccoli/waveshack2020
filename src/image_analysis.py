# %%
import os 
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# %%
city="milan"

#root = tk.Tk()
#root.withdraw()

#fn = filedialog.askopenfilename()
#title = os.path.split(fn)[-1].split(".")[0]

city_fns={"milan":"../images/milan.jpg"}
im=cv2.imread(city_fns[city])
cv2.imshow(city.title(), im)
print("hello world")
cv2.waitKey()
# %%
