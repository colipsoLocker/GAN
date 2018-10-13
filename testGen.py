from tensorflow.keras.preprocessing.image import ImageDataGenerator
img_rows = 128
img_cols = 128
channels = 3
img_shape = (img_rows,img_cols)
dropout_rate = 0.2
from skimage import io

valImgA_genF = ImageDataGenerator(
    #rescale=1./127.5-1.
)


valImgA_gen = valImgA_genF.flow_from_directory(
    directory = './GAN/data/test',
    target_size=img_shape,
    batch_size=2,
    class_mode = 'categorical'
)
import numpy as np
classNum2Name_test = valImgA_gen.class_indices
print(classNum2Name_test)
data = valImgA_gen.next()
print(data)
img = data[0][0]
kind = data[1][0]
from PIL import Image
print(img)
print(np.shape(img))
print(kind)
img2show = Image.fromarray(img.astype('uint8'))
img2show.show()
'''
import time
import numpy as np
from skimage import io,transform
imgs = io.ImageCollection('./GAN/data/test/A/*.jpg')

def getImages(imageDir):
    pattern = imageDir + '/*.jpg'
    imgs = io.ImageCollection(pattern)
    data = []
    for img in imgs:
        img = transform.resize(img,(img_cols,img_rows)) 
        img = img *1.0/127.5 -1
        data.append(img)
    data = np.asarray(data,dtype='float')
    return data

data = getImages('./GAN/data/test/A')
print(np.shape(data))
'''