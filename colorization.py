import numpy as np
import argparse
import cv2

w = 224
h = 224
imshowSize = (640,480)

# Select desired model
net = cv2.dnn.readNetFromCaffe('colorization_deploy_v2.prototxt', 'colorization_release_v2.caffemodel')
pts = np.load('pts_in_hull.npy') # load cluster centers

# populate cluster centers as 1x1 convolution kernel
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts.astype(np.float32)]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

cap = cv2.VideoCapture(r'chaplin.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    rgb = (frame[:,:,[2, 1, 0]] * 1.0 / 255).astype(np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    l = lab[:,:,0] # pull out L channel
    (H_orig,W_orig) = rgb.shape[:2] # original image size

    # resize image to network input size
    rs = cv2.resize(rgb, (w, h)) # resize image to network input size
    lab_rs = cv2.cvtColor(rs, cv2.COLOR_RGB2Lab)
    l_rs = lab_rs[:,:,0]
    l_rs -= 50 # subtract 50 for mean-centering

    net.setInput(cv2.dnn.blobFromImage(l_rs))
    ab= net.forward()[0,:,:,:].transpose((1,2,0)) # this is our result

    (H_out,W_out) = ab.shape[:2]
    ab_us = cv2.resize(ab, (W_orig, H_orig))
    lab_out = np.concatenate((l[:,:,np.newaxis],ab_us),axis=2) # concatenate with original image L
    bgr_out = np.clip(cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR), 0, 1)

    frame = cv2.resize(frame, imshowSize)
    cv2.imshow('origin', frame)
    # cv2.imshow('gray', cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
    cv2.imshow('colorized', cv2.resize(bgr_out, imshowSize))
    if cv2.waitKey(1) == 27 & 0xff:
        break

cap.release()
cv2.destroyAllWindows()
