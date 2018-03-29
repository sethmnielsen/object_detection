from __future__ import print_function
import random
import os
import shutil
import argparse

loc = './images/train/'
testloc = './images/test/'

test_count = len(os.listdir(testloc))

if test_count == 0:
    file_count = len(os.listdir(loc))
    count = 0
    for filename in os.listdir(loc):
        if filename.endswith('xml') and count <= file_count // 10:
            filename_jpg = filename[:-3]+'jpg'
            filename_xml = filename
            print(filename_jpg,"-->",testloc+filename_jpg)
            print(filename_xml,"-->",testloc+filename_xml)
            os.rename(loc+filename_jpg, testloc+filename_jpg)
            os.rename(loc+filename_xml, testloc+filename_xml)
            count += 1
else:
    print("Images already sorted into train and test folders.")
