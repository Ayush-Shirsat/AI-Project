import numpy as np
import cv2
import csv

data = np.genfromtxt('./annot_train.csv', delimiter=',', dtype = 'str')
length = len(data)
path = './train/'

X = data[1:length,0]
Y = data[1:length,1]

with open("X.csv", 'w', newline = '') as f:
    writer = csv.writer(f, delimiter =',')
    for i in range(length-1):
        filename = X[i]
        img = cv2.imread(path+filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray,(96,96))
        flat_gray = gray.flatten()
        writer.writerow(flat_gray)
        print(i)
        
with open("Y.csv", 'w', newline = '') as f:
    writer = csv.writer(f, delimiter = ',')
    for i in range(length-1):
        label = Y[i]
        writer.writerow(label)
        print(i)
        