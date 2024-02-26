from keras.models import load_model
import numpy as np 
from keras.preprocessing.image import img_to_array,load_img
import numpy as np
import glob
import os 
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import argparse
from PIL import Image
import matplotlib.pyplot as plt



imagestore=[]

video_source_path='/content/'
fps=5
#fps refers to the number of seconds after which one frame will be taken . fps=5 means 1 frame after every 5 seconds. More like seconds per frame.

def create_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def remove_old_images(path):
	filelist = glob.glob(os.path.join(path, "*.png"))
	for f in filelist:
		os.remove(f)

def store(image_path):
	img=load_img(image_path)
	img=img_to_array(img)
	#Resize the Image to (227,227,3) for the network to be able to process it. 

	img=resize(img,(224,224,3))
	imagestore.append(img)



#List of all Videos in the Source Directory.
videos=os.listdir(video_source_path)
print("Found ",len(videos)," files")


#Make a temp dir to store all the frames
create_dir(video_source_path+'/frames')

#Remove old images
remove_old_images(video_source_path+'/frames')

framepath=video_source_path+'/frames'
flag=0
for video in videos:
		if (video=="test.avi" or video=="test.mp4"):
			print("Test video found")
			flag=1
			os.system( 'ffmpeg -i {}/{} -r 1/{}  {}/frames/%03d.jpg'.format(video_source_path,video,fps,video_source_path))
			images=os.listdir(framepath)
			for image in images:
				image_path=framepath+ '/'+ image
				store(image_path)

if flag==0:
	print("Couldn't find test.mp4 or test.avi. Make sure you reupload and try this")
	exit()
imagestore=np.array(imagestore)
print(imagestore.shape)
imagestore=imagestore/255
#Clip negative Values
imagestore=np.clip(imagestore,0,1)
np.save('sample.npy',imagestore)
#Remove Buffer Directory
os.system('rm -r {}'.format(framepath))
print("Please wait while video is processed. \nRefresh when needed")


def mean_squared_loss(x1,x2):


	''' Compute Euclidean Distance Loss  between 
	input frame and the reconstructed frame'''

	diff=x1-x2
	a,b,c,d=diff.shape
	n_samples=a*b*c*d
	sq_diff=diff**2
	Sum=sq_diff.sum()
	dist=np.sqrt(Sum)
	mean_dist=dist/n_samples

	return mean_dist


'''Define threshold for Sensitivity
Lower the Threshhold,higher the chances that a bunch of frames will be flagged as Anomalous.

'''

#threshold=0.0004 #(Accuracy level 1)
#threshold=0.00042 #(Accuracy level 2)
threshold=0.0008#(Accuracy level 3)
d={}
model=load_model('AnomalyDetector.h5')

X_test=np.load('sample.npy')
i=0
for frame in X_test:
  # frame=np.random.randint(0,255,size=224*224*3)
  frame.resize(224,224,3)
  frame_normalized = frame/ 255.0  # Normalize to [0, 1] assuming the model expects input in this range
  frame_input = np.expand_dims(frame_normalized, axis=0)  # Add batch dimension
  reconstructed_frame=model.predict(frame_input)
  loss=mean_squared_loss(frame_input,reconstructed_frame)
  print(loss)
