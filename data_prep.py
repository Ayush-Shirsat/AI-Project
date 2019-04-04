import numpy as np
import cv2
import csv

data = np.genfromtxt('./annot_train.csv', delimiter=',', dtype = 'str')
length = len(data)
path = './train/'

X = data[1:length,0]
Y = data[1:length,2:12]

with open("X.csv", 'w', newline = '') as f:
    writer = csv.writer(f, delimiter =',')
    for i in range(length-1):
        filename = X[i]
        img = cv2.imread(path+filename,1)
        img_re = cv2.resize(img,(224,224))
        flat_img = img_re.flatten()
        writer.writerow(flat_img)
        print(i)
        
with open("Y.csv", 'w', newline = '') as f:
    writer = csv.writer(f, delimiter = ',')
    for i in range(length-1):
        label = Y[i]
        writer.writerow(label)
        print(i)
        
