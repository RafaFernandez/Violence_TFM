# LAEO annotator

import numpy as np
import cv2
import glob
import sys
import os
from shutil import copy


def anotator(framesdir, annotation_filename):

    image_types = ("*.jpg", "*.png", "*.jpeg") # supported extension files
    allFiles = []
    for image_type in image_types:
        allFiles.extend( glob.glob(framesdir+image_type) )
    allFiles = sorted(allFiles,key=os.path.getmtime)  # Make sure files are sorted

    nfiles = len(allFiles)
    print('* Found:', nfiles)

    for fnames in allFiles:
        print(fnames)

    # The vector of labels
    #Load annotations from existing file
    if os.path.exists(annotation_filename):
        labels=np.loadtxt(annotation_filename,delimiter=" ",usecols=1)
    # The vector of labels
    else:
        labels = np.zeros([nfiles], dtype=np.int8 ) - 1

    font = cv2.FONT_HERSHEY_SIMPLEX     # For drawing on image

    terminate = 0
    lastLabel = 0
    i = 0

    while terminate == 0:
        # Load a color image
        print(allFiles[i])
        img_ = cv2.imread(allFiles[i])
        img = img_.copy()

        #cv2.imshow('image',img)

        moveImg = 0

        while moveImg == 0 and terminate == 0:

            # Initialize label
            if labels[i] == -1:
                labels[i] = lastLabel

            img = img_.copy()
            cv2.putText(img, str(int(labels[i])), (10, 100), font, 2, (255, 255, 255), 2) # cv2.LINE_AA)
            cv2.imshow('image', img)

            key = cv2.waitKey(0)

            # Moving frames
            if key == ord('n') or key == ord('N'):
                #img = img_.copy()
                #cv2.putText(img, 'Next image', (10, 500), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
                print('Next image')
                #cv2.imshow('image', img)
                i = i+1
                moveImg = 1
                if i == nfiles:
                    print("End of video")
            if key == ord('z') or key == ord('Z'):
                #img = img_.copy()
                #cv2.putText(img, 'Previous image', (10, 500), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
                print('Previous image')
                #cv2.imshow('image', img)
                i = i - 1
                moveImg = 1
            if key == 27:
                terminate = 1

            # Annotation keys
            if key >= ord('0') and key <= ord('9'):
                theLabel = key-ord('0')
                labels[i] = theLabel
                lastLabel = theLabel
                print(theLabel)

            # Check boundaries
            i = max(0,i)
            i = min(nfiles-1,i)

    cv2.destroyAllWindows()

    # Save to file
    file = open(annotation_filename, "w")
    for ix in range(0,nfiles):
        file.write(str(ix+1)+' '+str(int(labels[ix]))+'\n')
    file.close()


def extract_key_frames(source_frames_dir, dest_frames_dir):
    files = glob.glob(source_frames_dir+"/*")

    for movie in files:
        name = movie[37:][:-4]
        os.system("ffmpeg -threads 1 -skip_frame nokey -i " + movie + " -vsync 0 -r 30 -f image2 "+dest_frames_dir + name + "-%02d.jpeg")


def separateF(framesdir,file,outdir):
    frames=[]
    labels=[]

    f = open(file,"r")

    lines = [line.rstrip('\n') for line in open(file)]
    for x in lines:
        frames=np.append(frames,x.split(' ')[0])
        labels=np.append(labels,x.split(' ')[1])
    f.close()

    indices0=(labels=='0').nonzero()[0]
    indices1=(labels=='1').nonzero()[0]
    indices2=(labels=='2').nonzero()[0]
    indices3=(labels=='3').nonzero()[0]
    indices4=(labels=='4').nonzero()[0]
    indices9=(labels=='9').nonzero()[0]

    images = glob(framesdir+'*.jpeg')
    images = sorted(images, key=os.path.getmtime) #Ordenamos las imagenes
    for i in range(len(indices0)):
        copy(images[indices0[i]], outdir + "/clase0/")
    for i in range(len(indices1)):
        copy(images[indices1[i]], outdir + "/clase1/")
    for i in range(len(indices2)):
        copy(images[indices2[i]], outdir + "/clase2/")
    for i in range(len(indices3)):
        copy(images[indices3[i]], outdir + "/clase3/")
    for i in range(len(indices4)):
        copy(images[indices4[i]], outdir + "/clase4/")
    for i in range(len(indices9)):
        copy(images[indices9[i]], outdir + "/clase9/")

