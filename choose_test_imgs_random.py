import random
import os
import shutil
import argparse

loc = './images/train/'
testloc = './images/test/'

test_count = 0
for filename in os.listdir(testloc):
    test_count += 1

if test_count == 0:
    file_count = 0;
    for filename in  os.listdir(loc):
        file_count += 1

    count = 0
    for filename in os.listdir(loc):
        if filename.endswith('xml') and count <= file_count // 20:
            filename_jpg = filename[:-3]+'jpg'
            filename_xml = filename
            print(filename_jpg,"-->",testloc+filename_jpg)
            print(filename_xml,"-->",testloc+filename_xml)
            os.rename(loc+filename_jpg, testloc+filename_jpg)
            os.rename(loc+filename_xml, testloc+filename_xml)
            count += 1
else:
    print("Images already sorted into train and test folders.")
