# -*- coding: utf-8 -*-
"""Nature_ComputerVisionAprilTag.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1e76ralGnvLUeyKdBmejCb2osQ5prVabZ
"""

pip install opencv-python

pip install apriltag

import cv2
import time
import apriltag
from google.colab.patches import cv2_imshow

from matplotlib.lines import Line2D
import cv2
import time
import apriltag
import numpy
import pandas as pd
import matplotlib as plt
Ptom = 2/45 
A0x = []
A1x = []
A3x = []
A4x = []
A7x = []
A8x = []
A5x = []
A9x = []
A10x = []
A11x = []
A0y = []
A1y = []
A3y = []
A4y = []
A5y = []
A9y = []
A7y = []
A8y = []
A10y = []
A11y = []
L4 = []
L3 = []
L2 = []
L1 = []
count = 0
cap = cv2.VideoCapture('DSC_0893.MOV')

if cap.isOpened()==False:
    print('Error File Not Found')
    
while cap.isOpened():
    
    ret,frame = cap.read()
    
    if ret == True:
        count = count+1
        grayimage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ATFind = apriltag.Detector()
        AL = ATFind.detect(grayimage)
        if type(AL) == list: 
          arr = numpy.array(AL)
          if len(arr.shape) == 2: 
            det,_ = arr.shape
            if det == 4 or 3 or 2 or 1:
              pass
            else:
              print(det)
            if det == 1:
              if arr[0][1] == 9:
                A9x.append(arr[0][6][0])
                A9y.append(arr[0][6][1])
              if arr[0][1] == 7:
                A7x.append(arr[0][6][0])
                A7y.append(arr[0][6][1])
              if arr[0][1] == 5:
                A5x.append(arr[0][6][0])
                A5y.append(arr[0][6][1])
              if arr[0][1] == 0:
                A0x.append(arr[0][6][0])
                A0y.append(arr[0][6][1])
              if arr[0][1] == 1:
                A1x.append(arr[0][6][0])
                A1y.append(arr[0][6][1])
              if arr[0][1] == 3:
                A3x.append(arr[0][6][0])
                A3y.append(arr[0][6][1])
              if arr[0][1] == 8:
                A8x.append(arr[0][6][0])
                A8y.append(arr[0][6][1])
              if arr[0][1] == 4:
                A4x.append(arr[0][6][0])
                A4y.append(arr[0][6][1])
              if arr[0][1] == 10:
                A10x.append(arr[0][6][0])
                A10y.append(arr[0][6][1])
              if arr[0][1] == 11:
                A11x.append(arr[0][6][0])
                A11y.append(arr[0][6][1])

            elif det == 2:
              if arr[0][1] == 9:
                A9x.append(arr[0][6][0])
                A9y.append(arr[0][6][1])
              if arr[0][1] == 5:
                A5x.append(arr[0][6][0])
                A5y.append(arr[0][6][1])
              if arr[0][1] == 7:
                A7x.append(arr[0][6][0])
                A7y.append(arr[0][6][1])
              if arr[0][1] == 0:
                A0x.append(arr[0][6][0])
                A0y.append(arr[0][6][1])
              if arr[0][1] == 1:
                A1x.append(arr[0][6][0])
                A1y.append(arr[0][6][1])
              if arr[0][1] == 3:
                A3x.append(arr[0][6][0])
                A3y.append(arr[0][6][1])
              if arr[0][1] == 8:
                A8x.append(arr[0][6][0])
                A8y.append(arr[0][6][1])
              if arr[0][1] == 4:
                A4x.append(arr[0][6][0])
                A4y.append(arr[0][6][1])
              if arr[0][1] == 10:
                A10x.append(arr[0][6][0])
                A10y.append(arr[0][6][1])
              if arr[0][1] == 11:
                A11x.append(arr[0][6][0])
                A11y.append(arr[0][6][1])

              if arr[1][1] == 7:
                A7x.append(arr[1][6][0])
                A7y.append(arr[1][6][1])
              if arr[1][1] == 9:
                A9x.append(arr[1][6][0])
                A9y.append(arr[1][6][1])
              if arr[1][1] == 5:
                A5x.append(arr[1][6][0])
                A5y.append(arr[1][6][1])
              if arr[1][1] == 0:
                A0x.append(arr[1][6][0])
                A0y.append(arr[1][6][1])
              if arr[1][1] == 1:
                A1x.append(arr[1][6][0])
                A1y.append(arr[1][6][1])
              if arr[1][1] == 3:
                A3x.append(arr[1][6][0])
                A3y.append(arr[1][6][1])
              if arr[1][1] == 8:
                A8x.append(arr[1][6][0])
                A8y.append(arr[1][6][1])
              if arr[1][1] == 4:
                A4x.append(arr[1][6][0])
                A4y.append(arr[1][6][1])
              if arr[1][1] == 10:
                A10x.append(arr[1][6][0])
                A10y.append(arr[1][6][1])
              if arr[1][1] == 11:
                A11x.append(arr[1][6][0])
                A11y.append(arr[1][6][1])

            elif det == 3:
              if arr[0][1] == 7:
                A7x.append(arr[0][6][0])
                A7y.append(arr[0][6][1])
              if arr[0][1] == 9:
                A9x.append(arr[0][6][0])
                A9y.append(arr[0][6][1])
              if arr[0][1] == 5:
                A5x.append(arr[0][6][0])
                A5y.append(arr[0][6][1])
              if arr[0][1] == 0:
                A0x.append(arr[0][6][0])
                A0y.append(arr[0][6][1])
              if arr[0][1] == 1:
                A1x.append(arr[0][6][0])
                A1y.append(arr[0][6][1])
              if arr[0][1] == 3:
                A3x.append(arr[0][6][0])
                A3y.append(arr[0][6][1])
              if arr[0][1] == 8:
                A8x.append(arr[0][6][0])
                A8y.append(arr[0][6][1])
              if arr[0][1] == 4:
                A4x.append(arr[0][6][0])
                A4y.append(arr[0][6][1])
              if arr[0][1] == 10:
                A10x.append(arr[0][6][0])
                A10y.append(arr[0][6][1])
              if arr[0][1] == 11:
                A11x.append(arr[0][6][0])
                A11y.append(arr[0][6][1])

              if arr[1][1] == 9:
                A9x.append(arr[1][6][0])
                A9y.append(arr[1][6][1])
              if arr[1][1] == 7:
                A7x.append(arr[1][6][0])
                A7y.append(arr[1][6][1])
              if arr[1][1] == 5:
                A5x.append(arr[1][6][0])
                A5y.append(arr[1][6][1])
              if arr[1][1] == 0:
                A0x.append(arr[1][6][0])
                A0y.append(arr[1][6][1])
              if arr[1][1] == 1:
                A1x.append(arr[1][6][0])
                A1y.append(arr[1][6][1])
              if arr[1][1] == 3:
                A3x.append(arr[1][6][0])
                A3y.append(arr[1][6][1])
              if arr[1][1] == 8:
                A8x.append(arr[1][6][0])
                A8y.append(arr[1][6][1])
              if arr[1][1] == 4:
                A4x.append(arr[1][6][0])
                A4y.append(arr[1][6][1])
              if arr[1][1] == 10:
                A10x.append(arr[1][6][0])
                A10y.append(arr[1][6][1])
              if arr[1][1] == 11:
                A11x.append(arr[1][6][0])
                A11y.append(arr[1][6][1])

              if arr[2][1] == 9:
                A9x.append(arr[2][6][0])
                A9y.append(arr[2][6][1])
              if arr[2][1] == 7:
                A7x.append(arr[2][6][0])
                A7y.append(arr[2][6][1])
              if arr[2][1] == 5:
                A5x.append(arr[2][6][0])
                A5y.append(arr[2][6][1])
              if arr[2][1] == 0:
                A0x.append(arr[2][6][0])
                A0y.append(arr[2][6][1])
              if arr[2][1] == 1:
                A1x.append(arr[2][6][0])
                A1y.append(arr[2][6][1])
              if arr[2][1] == 3:
                A3x.append(arr[2][6][0])
                A3y.append(arr[2][6][1])
              if arr[2][1] == 8:
                A8x.append(arr[2][6][0])
                A8y.append(arr[2][6][1])
              if arr[2][1] == 4:
                A4x.append(arr[2][6][0])
                A4y.append(arr[2][6][1])
              if arr[2][1] == 10:
                A10x.append(arr[2][6][0])
                A10y.append(arr[2][6][1])
              if arr[2][1] == 11:
                A11x.append(arr[2][6][0])
                A11y.append(arr[2][6][1])
            
            elif det == 4:
              if arr[0][1] == 9:
                A9x.append(arr[0][6][0])
                A9y.append(arr[0][6][1])
              if arr[0][1] == 7:
                A7x.append(arr[0][6][0])
                A7y.append(arr[0][6][1])
              if arr[0][1] == 5:
                A5x.append(arr[0][6][0])
                A5y.append(arr[0][6][1])
              if arr[0][1] == 0:
                A0x.append(arr[0][6][0])
                A0y.append(arr[0][6][1])
              if arr[0][1] == 4:
                A1x.append(arr[0][6][0])
                A1y.append(arr[0][6][1])
              if arr[0][1] == 3:
                A3x.append(arr[0][6][0])
                A3y.append(arr[0][6][1])
              if arr[0][1] == 8:
                A8x.append(arr[0][6][0])
                A8y.append(arr[0][6][1])
              if arr[0][1] == 4:
                A4x.append(arr[0][6][0])
                A4y.append(arr[0][6][1])
              if arr[0][1] == 10:
                A10x.append(arr[0][6][0])
                A10y.append(arr[0][6][1])
              if arr[0][1] == 11:
                A11x.append(arr[0][6][0])
                A11y.append(arr[0][6][1])

              if arr[1][1] == 9:
                A9x.append(arr[1][6][0])
                A9y.append(arr[1][6][1])
              if arr[1][1] == 7:
                A7x.append(arr[1][6][0])
                A7y.append(arr[1][6][1])
              if arr[1][1] == 5:
                A5x.append(arr[1][6][0])
                A5y.append(arr[1][6][1])
              if arr[1][1] == 0:
                A0x.append(arr[1][6][0])
                A0y.append(arr[1][6][1])
              if arr[1][1] == 1:
                A1x.append(arr[1][6][0])
                A1y.append(arr[1][6][1])
              if arr[1][1] == 3:
                A3x.append(arr[1][6][0])
                A3y.append(arr[1][6][1])
              if arr[1][1] == 8:
                A8x.append(arr[1][6][0])
                A8y.append(arr[1][6][1])
              if arr[1][1] == 4:
                A4x.append(arr[1][6][0])
                A4y.append(arr[1][6][1])
              if arr[1][1] == 10:
                A10x.append(arr[1][6][0])
                A10y.append(arr[1][6][1])
              if arr[1][1] == 11:
                A11x.append(arr[1][6][0])
                A11y.append(arr[1][6][1])
              
              if arr[2][1] == 9:
                A9x.append(arr[2][6][0])
                A9y.append(arr[2][6][1])
              if arr[2][1] == 7:
                A7x.append(arr[2][6][0])
                A7y.append(arr[2][6][1])
              if arr[2][1] == 5:
                A5x.append(arr[2][6][0])
                A5y.append(arr[2][6][1])

              if arr[2][1] == 0:
                A0x.append(arr[2][6][0])
                A0y.append(arr[2][6][1])

              if arr[2][1] == 1:
                A1x.append(arr[2][6][0])
                A1y.append(arr[2][6][1])

              if arr[2][1] == 3:
                A3x.append(arr[2][6][0])
                A3y.append(arr[2][6][1])

              if arr[2][1] == 8:
                A8x.append(arr[2][6][0])
                A8y.append(arr[2][6][1])
              if arr[2][1] == 4:
                A4x.append(arr[2][6][0])
                A4y.append(arr[2][6][1])
              if arr[2][1] == 10:
                A10x.append(arr[2][6][0])
                A10y.append(arr[2][6][1])
              if arr[2][1] == 11:
                A11x.append(arr[2][6][0])
                A11y.append(arr[2][6][1])
              
              if arr[3][1] == 9:
                A9x.append(arr[3][6][0])
                A9y.append(arr[3][6][1])
              if arr[3][1] == 7:
                A7x.append(arr[3][6][0])
                A7y.append(arr[3][6][1])
              if arr[3][1] == 5:
                A5x.append(arr[3][6][0])
                A5y.append(arr[3][6][1])

              if arr[3][1] == 0:
                A0x.append(arr[3][6][0])
                A0y.append(arr[3][6][1])
 
              if arr[3][1] == 1:
                A1x.append(arr[3][6][0])
                A1y.append(arr[3][6][1])
  
              if arr[3][1] == 3:
                A3x.append(arr[3][6][0])
                A3y.append(arr[3][6][1])

              if arr[3][1] == 8:
                A8x.append(arr[3][6][0])
                A8y.append(arr[3][6][1])
              if arr[3][1] == 4:
                A4x.append(arr[3][6][0])
                A4y.append(arr[3][6][1])
              if arr[2][1] == 10:
                A10x.append(arr[2][6][0])
                A10y.append(arr[2][6][1])
              if arr[2][1] == 11:
                A11x.append(arr[2][6][0])
                A11y.append(arr[2][6][1])

            elif det == 5:
              
              if arr[0][1] == 9:
                A9x.append(arr[0][6][0])
                A9y.append(arr[0][6][1])
              if arr[0][1] == 7:
                A7x.append(arr[0][6][0])
                A7y.append(arr[0][6][1])
              if arr[0][1] == 5:
                A5x.append(arr[0][6][0])
                A5y.append(arr[0][6][1])
   
              if arr[0][1] == 0:
                A0x.append(arr[0][6][0])
                A0y.append(arr[0][6][1])

              if arr[0][1] == 4:
                A1x.append(arr[0][6][0])
                A1y.append(arr[0][6][1])
 
              if arr[0][1] == 3:
                A3x.append(arr[0][6][0])
                A3y.append(arr[0][6][1])

              if arr[0][1] == 8:
                A8x.append(arr[0][6][0])
                A8y.append(arr[0][6][1])
              if arr[0][1] == 4:
                A4x.append(arr[0][6][0])
                A4y.append(arr[0][6][1])
              if arr[0][1] == 10:
                A10x.append(arr[0][6][0])
                A10y.append(arr[0][6][1])
              if arr[0][1] == 11:
                A11x.append(arr[0][6][0])
                A11y.append(arr[0][6][1])

              if arr[1][1] == 9:
                A9x.append(arr[1][6][0])
                A9y.append(arr[1][6][1])
              if arr[1][1] == 7:
                A7x.append(arr[1][6][0])
                A7y.append(arr[1][6][1])
              if arr[1][1] == 5:
                A5x.append(arr[1][6][0])
                A5y.append(arr[1][6][1])

              if arr[1][1] == 0:
                A0x.append(arr[1][6][0])
                A0y.append(arr[1][6][1])

              if arr[1][1] == 1:
                A1x.append(arr[1][6][0])
                A1y.append(arr[1][6][1])

              if arr[1][1] == 3:
                A3x.append(arr[1][6][0])
                A3y.append(arr[1][6][1])

              if arr[1][1] == 8:
                A8x.append(arr[1][6][0])
                A8y.append(arr[1][6][1])
              if arr[1][1] == 4:
                A4x.append(arr[1][6][0])
                A4y.append(arr[1][6][1])
              if arr[1][1] == 10:
                A10x.append(arr[1][6][0])
                A10y.append(arr[1][6][1])
              if arr[1][1] == 11:
                A11x.append(arr[1][6][0])
                A11y.append(arr[1][6][1])
              
              if arr[2][1] == 9:
                A9x.append(arr[2][6][0])
                A9y.append(arr[2][6][1])
              if arr[2][1] == 7:
                A7x.append(arr[2][6][0])
                A7y.append(arr[2][6][1])
              if arr[2][1] == 5:
                A5x.append(arr[2][6][0])
                A5y.append(arr[2][6][1])
    
              if arr[2][1] == 0:
                A0x.append(arr[2][6][0])
                A0y.append(arr[2][6][1])
      
              if arr[2][1] == 1:
                A1x.append(arr[2][6][0])
                A1y.append(arr[2][6][1])
      
              if arr[2][1] == 3:
                A3x.append(arr[2][6][0])
                A3y.append(arr[2][6][1])
  
              if arr[2][1] == 8:
                A8x.append(arr[2][6][0])
                A8y.append(arr[2][6][1])
              if arr[2][1] == 4:
                A4x.append(arr[2][6][0])
                A4y.append(arr[2][6][1])
              if arr[2][1] == 10:
                A10x.append(arr[2][6][0])
                A10y.append(arr[2][6][1])
              if arr[2][1] == 11:
                A11x.append(arr[2][6][0])
                A11y.append(arr[2][6][1])

              if arr[3][1] == 9:
                A9x.append(arr[3][6][0])
                A9y.append(arr[3][6][1])
              if arr[3][1] == 7:
                A7x.append(arr[3][6][0])
                A7y.append(arr[3][6][1])
              if arr[3][1] == 5:
                A5x.append(arr[3][6][0])
                A5y.append(arr[3][6][1])
  
              if arr[3][1] == 0:
                A0x.append(arr[3][6][0])
                A0y.append(arr[3][6][1])
 
              if arr[3][1] == 1:
                A1x.append(arr[3][6][0])
                A1y.append(arr[3][6][1])
  
              if arr[3][1] == 3:
                A3x.append(arr[3][6][0])
                A3y.append(arr[3][6][1])
    
              if arr[3][1] == 8:
                A8x.append(arr[3][6][0])
                A8y.append(arr[3][6][1])
              if arr[3][1] == 4:
                A4x.append(arr[3][6][0])
                A4y.append(arr[3][6][1])
              if arr[3][1] == 10:
                A10x.append(arr[3][6][0])
                A10y.append(arr[3][6][1])
              if arr[3][1] == 11:
                A11x.append(arr[3][6][0])
                A11y.append(arr[3][6][1])

              if arr[4][1] == 9:
                A9x.append(arr[4][6][0])
                A9y.append(arr[4][6][1])
              if arr[4][1] == 7:
                A7x.append(arr[4][6][0])
                A7y.append(arr[4][6][1])
              if arr[4][1] == 5:
                A5x.append(arr[4][6][0])
                A5y.append(arr[4][6][1])
                #print("8 in 4")
              if arr[4][1] == 0:
                A0x.append(arr[4][6][0])
                A0y.append(arr[4][6][1])
                #print("0 in 4")
              if arr[4][1] == 1:
                A1x.append(arr[4][6][0])
                A1y.append(arr[4][6][1])
                #print("4 in 4")
              if arr[4][1] == 3:
                A3x.append(arr[4][6][0])
                A3y.append(arr[4][6][1])
                #print("3 in 4")
              if arr[4][1] == 8:
                A8x.append(arr[4][6][0])
                A8y.append(arr[4][6][1])
              if arr[4][1] == 4:
                A4x.append(arr[4][6][0])
                A4y.append(arr[4][6][1])
              if arr[4][1] == 10:
                A10x.append(arr[4][6][0])
                A10y.append(arr[4][6][1])
              if arr[4][1] == 11:
                A11x.append(arr[4][6][0])
                A11y.append(arr[4][6][1])
            
            else:
              print(det)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    else:
        break
       
cap.release()
cv2.destroyAllWindows()

print("Path of 11 Robot")
print(len(A11x))
print(len(A11y))
print("Path of 10 Robot")
print(len(A10x))
print(len(A10y))
print("Path of 9 Robot")
print(len(A9x))
print(len(A9y))
print("Path of 5 Robot")
print(len(A5x))
print(len(A5y))
print("Path of 7 Robot")
print(len(A7x))
print(len(A7y))
print("Path of 8 Robot")
print(len(A8x))
print(len(A8y))
print("Corner Lower Left")
print(len(A1x))
print(len(A1y))
print("Corner Lower Right")
print(len(A0x))
print(len(A0y))
print("Corner Upper Right")
print(len(A3x))
print(len(A3y))
print("Corner Upper Left")
print(len(A4x))
print(len(A4y))

import numpy as np
AX5mm = (np.array(A5x)/20)
AY5mm = (np.array(A5y)/20)
AX9mm = (np.array(A9x)/20)
AY9mm = (np.array(A9y)/20)
AX7mm = (np.array(A7x)/20)
AY7mm = (np.array(A7y)/20)
AX8mm = (np.array(A8x)/20)
AY8mm = (np.array(A8y)/20)
AX1mm = (np.array(A1x)/20)
AY1mm = (np.array(A1y)/20)
AX0mm = (np.array(A0x)/20)
AY0mm = (np.array(A0y)/20)
AX3mm = (np.array(A3x)/20)
AY3mm = (np.array(A3y)/20)
AX4mm = (np.array(A4x)/20)
AY4mm = (np.array(A4y)/20)
AX10mm = (np.array(A10x)/20)
AY10mm = (np.array(A10y)/20)
AX11mm = (np.array(A11x)/20)
AY11mm = (np.array(A11y)/20)

print("Start")
X1 = AX8mm[0]
print(X1)
Y1 = AY8mm[0]

print("End")
X2 = AX8mm[(len(AX8mm)-1)]
print(X2)
Y2 = AY8mm[(len(AX8mm)-1)]
print(Y2)
print("Distance in cm")
dist = (((X1-X2)**2)+((Y1-Y2)**2))**0.5
print(dist)
Speed = dist/2
print('Speed in cm/min')
print(Speed)

import matplotlib.pyplot as plt
Xdist = [X1,X2]
Ydist = [Y1,Y2]
plt.plot(AX8mm,AY8mm,label="Robot", color="blue")
plt.plot(Xdist,Ydist, label="Line Dist", color="red")
plt.scatter(AX1mm,AY1mm,label="LL", color="green")
plt.scatter(AX0mm,AY0mm,label="LR", color="green")
plt.scatter(AX3mm,AY3mm,label="UR", color="green")
plt.scatter(AX4mm,AY4mm,label="UL", color="green")
plt.xlabel('x axis in cm')
plt.ylabel("y axis in cm")
plt.title('Pleuro Robot Untethered Path')
plt.legend()
plt.grid()
plt.show()