from PIL import Image
import numpy as np
import os

img_path = r"/home/guatambu/bauce_ds/projeto/dataset/data_am/treino_am/label/512/"

for pic in os.listdir(img_path):
   img = np.array(Image.open(img_path+pic))
   print(np.unique(img))