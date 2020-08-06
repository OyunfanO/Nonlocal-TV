import numpy as np
import os
from multiprocessing import Pool
processNum = 8

topk = 40


def newTopK(folder,folderpath,npyfiles,i):
	npy = folder+npyfiles[i]
	W = np.load(npy)
	Widx = np.load(npy.replace("W.npy","Widx.npy"))
	s = W.shape
	newW = np.zeros((s[0],topk),dtype=np.float16)
	newWidx = np.zeros((s[0],topk),dtype=np.int32)
	#kidx = np.zeros(topk,dtype=np.int32)

	for jj in range(0,s[0]):
		kidx = W[jj,:].argsort()[-topk:][::-1]
		newW[jj,:] = W[jj,kidx]
		newWidx[jj,:] = Widx[jj,kidx]

	newpath = folderpath+"/"+npyfiles[i]
	#newpath = folder + "/" + npyfiles[i]
	np.save(newpath,newW)
	np.save(newpath.replace("W","Widx"),newWidx)
	print('PID: %d processing the %d th file %s' % (os.getpid(), i, newpath))

#folderpath = input("input image folder:").strip("' ")
folderpath = "/home/yunfan/Documents/SegNet/CamVid/Dataset/test"

folder = folderpath +"/topk100/"

files = os.listdir(folder)
npyfiles = [x for x in files if x.find("W.npy")!=-1]

num = len(npyfiles)

p = Pool(processNum)
for i in range(0,num):
	p.apply_async(newTopK, args=(folder,folderpath,npyfiles, i,))
p.close()
p.join()
print("all subprocesses done")
