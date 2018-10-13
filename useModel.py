from tensorflow.keras.models import load_model
from PIL import Image
from skimage import io,transform
import numpy as np
import scipy
import os

img_rows = 128
img_cols = 128
channels = 3
img_shape = (img_rows,img_cols)

modelA2B = load_model('./GAN/models/generator_A2B.h5')
modelB2A = load_model('./GAN/models/generator_B2A.h5')

def dealImages(imgDir):
    for root , dirs , files in os.walk(imgDir):
        for f in files:
            imgFile = os.path.join(root , f)
            imageOrigin = io.imread(imgFile)
            image = scipy.misc.imresize(imageOrigin,(img_cols,img_rows))
            image2Save = image
            image = np.asarray(image) *1.0/ 127.5 -1
            image = np.expand_dims(image, axis=0)
            resultImgA2A = modelB2A.predict(image)
            resultImgA2B = modelA2B.predict(image)
            resultImgA2A = (resultImgA2A[0] +1.0)*127.5
            resultImgA2B = (resultImgA2B[0] +1.0)*127.5
            resultImgA2A = Image.fromarray(np.uint8(resultImgA2A))
            resultImgA2B = Image.fromarray(np.uint8(resultImgA2B))
            image2Save = Image.fromarray(np.uint8(image2Save))
            savedDir_resultImgA2B = os.path.join('./GAN/use/target' ,'%s【转换】.jpg'%f.split('.')[0])
            savedDir_resultImgA2A = os.path.join('./GAN/use/target' ,'%s【Back】.jpg'%f.split('.')[0])
            savedDir_image2Save = os.path.join('./GAN/use/target' ,'%s【原图】.jpg'%f.split('.')[0])
            resultImgA2B.save(savedDir_resultImgA2B)
            resultImgA2A.save(savedDir_resultImgA2A)
            image2Save.save(savedDir_image2Save)



dealImages('./GAN/use/origin')


'''
imageA = io.imread('./GAN/data/test/A/2012-05-17 14:33:00.jpg')
#io.imshow(image)
#image = Image.open('./GAN/data/test/A/2011-06-03 21:27:20.jpg')
orgin_img = Image.fromarray(imageA)
orgin_img.show()
#image.resize(img_shape)
image = scipy.misc.imresize(imageA,(img_cols,img_rows)) 
tmpimg = Image.fromarray(np.uint8(image ))
tmpimg.show()
image = np.asarray(image) *1.0/ 127.5 -1
tmpimg = Image.fromarray(np.uint8((image +1.0 ) *127.5))
tmpimg.show()
image = np.expand_dims(image, axis=0)

print('---------->')
print(image)
resultImgA2A = modelB2A.predict(image)
print('---------->')
print(resultImgA2A)
resultImg = modelA2B.predict(image)
resultImg = resultImg[0]
resultImg = (resultImg +1.0)*127.5
print(np.shape(resultImg))
print(resultImg)
resultImg = Image.fromarray(np.uint8(resultImg))
resultImg.show()
'''
