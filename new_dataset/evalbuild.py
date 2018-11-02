import os
import csv
import cv2
import math
import numpy as np

title = ['disease_class','image_id']
out = open('./eval.csv','a', newline='')
csv_write = csv.writer(out,dialect='excel')
csv_write.writerow(title)
for q in os.listdir(r'/media/hai/SANDISK/eval/'):
    print(q)
    name=[0,q]
    out = open('./eval.csv','a', newline='')
    csv_write = csv.writer(out,dialect='excel')
    csv_write.writerow(name)
