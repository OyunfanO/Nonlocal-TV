import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import argparse
import math
import pylab
from sklearn.preprocessing import normalize
import PIL.Image as Image

import pydensecrf.densecrf as dcrf
import cv2
import time

caffe_root = '/home/yunfan/Documents/SegNet/caffe-segnet/' 	# Change this to the absolute directoy to SegNet Caffe
import sys
sys.path.insert(0, caffe_root + 'python')
os.environ['GLOG_minloglevel'] = '2'
import caffe

def prediction(Data,weights,model,IMAGE_FILE,CRF):
	caffe.set_mode_gpu()
	caffe.set_device(0)
	net = caffe.Net(model,weights,caffe.TEST)

	'''
	folder = "/home/yunfan/Documents/SegNet/WhiteCell/val_data/Dataset2_val/"
	imgfiles = os.listdir(folder)
	imgfiles = [folder + x for x in imgfiles if x.find("bmp")!=-1]
	imgfiles.sort()
	'''
	folder = "/home/yunfan/Documents/SegNet/CamVid/Dataset/test.txt"
	indexlist = [line.rstrip('\n') for line in open(folder)]

	timing = 0

	for i in range(0, iteration):

		line = indexlist[i]
		indexes = line.split(" ")
		imgfile = indexes[0]

		nameidx = imgfile.rfind('/')
		imgname = imgfile[nameidx+1:-4]

		start = time.time()
		net.forward()
		end = time.time()
		timing = timing + (end-start)

		image = net.blobs['data'].data
		label = net.blobs['label'].data
		predicted = net.blobs['prob'].data
		#predicted = net.blobs['A1'].data
		image = np.squeeze(image[0,:,:,:])
		output = np.squeeze(predicted[0,:,:,:])
		O = net.blobs['O'].data
		O = np.squeeze(O[0,:,:,:])

		Is = output.shape

		if CRF:
			#start = time.time()
			d = dcrf.DenseCRF2D(Is[2], Is[1], Is[0])
			output = np.where(output==0,1e-10,output)
			U = -np.log(output)
			U = U.reshape((Is[0],-1))
			d.setUnaryEnergy(U)
			#U = U.transpose(2, 0, 1).reshape((Is[0], -1))
			im = cv2.imread(imgfile)
			d.addPairwiseGaussian(sxy=(3,3), compat=4, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
			d.addPairwiseBilateral(sxy=(49,49), srgb=(13,13,13), rgbim=im, compat=5, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
			Q = d.inference(10)
			#end = time.time()
			#print(end-start)
			ind = np.argmax(Q, axis=0).reshape((Is[1],Is[2]))
		else:
			ind = np.argmax(output, axis=0)

		r = ind.copy()
		g = ind.copy()
		b = ind.copy()
		r_gt = label.copy()
		g_gt = label.copy()
		b_gt = label.copy()

		Sky = [128,128,128]
		Building = [128,0,0]
		Pole = [192,192,128]
		Road = [128,64,128]
		Pavement = [60,40,222]
		Tree = [128,128,0]
		SignSymbol = [192,128,128]
		Fence = [64,64,128]
		Car = [64,0,128]
		Pedestrian = [64,64,0]
		Bicyclist = [0,128,192]
		Unlabelled = [0,0,0]

		label_colours = np.array([Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled],dtype=np.uint8)
		for l in range(0,12):
			r[ind==l] = label_colours[l,0]
			g[ind==l] = label_colours[l,1]
			b[ind==l] = label_colours[l,2]
			r_gt[label==l] = label_colours[l,0]
			g_gt[label==l] = label_colours[l,1]
			b_gt[label==l] = label_colours[l,2]

		rgb = np.zeros((ind.shape[0], ind.shape[1], 3))
		rgb[:,:,0] = r/255.0
		rgb[:,:,1] = g/255.0
		rgb[:,:,2] = b/255.0
		rgb_gt = np.zeros((ind.shape[0], ind.shape[1], 3))
		rgb_gt[:,:,0] = r_gt/255.0
		rgb_gt[:,:,1] = g_gt/255.0
		rgb_gt[:,:,2] = b_gt/255.0

		image = image/255.0

		image = np.transpose(image, (1,2,0))
		output = np.transpose(output, (1,2,0))
		image = image[:,:,(2,1,0)]

		rgb = rgb*255
		rgb_im = Image.fromarray(rgb.astype('uint8'))
		if CRF:
			rgb_im.save(IMAGE_FILE+imgname+'_prediction_'+"Segnet_CRF"+'.png')
		else:
			rgb_im.save(IMAGE_FILE + imgname + '_prediction_' + "NLSegnet" + '.png')
		rgb_gt = rgb_gt*255
		rgbgt_im = Image.fromarray(rgb_gt.astype('uint8'))
		rgbgt_im.save(IMAGE_FILE+imgname+'_gt.png')
		image = image*255
		image_im = Image.fromarray(image.astype('uint8'))
		image_im.save(IMAGE_FILE+imgname+'.png')
		np.save(IMAGE_FILE+imgname+'.npy',O)
	print(timing)
	print('Success!')

def path_process(Data):
	model = FILEROOT+"prototxt/NLTV/b4_80k_eps/NL"+Net+"_inference.prototxt"
	#model = FILEROOT + "prototxt/" + Net + "/b4_80k_rf/" + "Segnet" + "_inference.prototxt"
	#weights = FILEROOT+"model/"+Net+"/Dataset/b4_80k3/"+MODEL
	weights = FILEROOT + "model/NLTV/b4_80k_eps/" + MODEL
	#weights = FILEROOT+"model/Segnet/Dataset/b4_80k_rf/"+MODEL
	IMAGE_FILE = FILEROOT+"inference/"+"NLTV"+"/b4_80k_eps/"#+lambdal
	#IMAGE_FILE = FILEROOT+"inference/Segnet/b4_80k_rf/"#+lambdal

	if not os.path.isdir(IMAGE_FILE):
		print("creating folder "+IMAGE_FILE)
		os.mkdir(IMAGE_FILE)
	return Data,weights,model,IMAGE_FILE


FILEROOT = '/home/yunfan/Documents/SegNet/CamVid/'
Data = "Dataset"
Net = "Segnet"
MODEL = "_iter_35000.caffemodel"
CRF = 0

iteration = 233

Datapath,weights,model,IMAGE_FILE = path_process(Data)
prediction(Datapath,weights,model,IMAGE_FILE,CRF)
