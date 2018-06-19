import os
import cv2
import numpy as np
from PIL import Image


def split():
    kernel_3x3 = np.ones((3, 3), np.float32) / 9
    list_of_classes = os.listdir('Training')
    for i in list_of_classes:
        path = 'Training/'+i
        list_of_img = os.listdir(path)
        length = len(list_of_img)
        for j in range(0,length):
            path1 = 'Training/'+i+'/'+list_of_img[j]
            im = Image.open(path1)
            #blur = cv2.filter2D(im, -1,kernel_3x3) 
            if(j%5==0): 
                path2 = 'Validation/'+i+'/'+list_of_img[j] 
                im.save(path2,"ppm")
                #path2_ = 'Validation/'+i+'/b_'+list_of_img[j] 
                #cv2.imwrite(path2_,blur)
                os.remove(path1)
            #else:
                #path1_ = 'Training/'+i+'/b_'+list_of_img[j]	
                #cv2.imwrite(path1_,blur)
         
split()
