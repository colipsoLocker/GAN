from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D,BatchNormalization,Dropout,LeakyReLU,ZeroPadding2D,UpSampling2D,Input,Concatenate
import scipy
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from skimage import io,transform
import numpy as np
from utils import sample_images

img_rows = 128
img_cols = 128
channels = 3
img_shape = (img_rows,img_cols,channels)
dropout_rate = 0.2

#定义生成器
def generator():
    d0 = Input(shape=img_shape)

    d1 = Conv2D(
        filters = 32 , 
        kernel_size = (4,4),
        strides=(2,2) , 
        padding='same',
        #activation='LeakyReLU',
        )(d0)
    d1 = LeakyReLU(alpha=0.2)(d1)
    d1 = BatchNormalization()(d1)

    d2 = Conv2D(
        filters = 32*2 , 
        kernel_size = (4,4),
        strides=(2,2) , 
        padding='same',
        #activation='LeakyReLU',
        )(d1)
    d2 = LeakyReLU(alpha=0.2)(d2)
    d2 = BatchNormalization()(d2)

    d3 = Conv2D(
        filters = 32*4 , 
        kernel_size = (4,4),
        strides=(2,2) , 
        padding='same',
        #activation='LeakyReLU',
        )(d2)
    d3 = LeakyReLU(alpha=0.2)(d3)
    d3 = BatchNormalization()(d3)

    d4 = Conv2D(
        filters = 32*8 , 
        kernel_size = (4,4),
        strides=(2,2) , 
        padding='same',
        #activation='LeakyReLU',
        )(d3)
    d4 = LeakyReLU(alpha=0.2)(d4)
    d4 = BatchNormalization()(d4)

    u1 = UpSampling2D(size=2)(d4)
    u1 = Conv2D(
        filters = 64*4 , 
        kernel_size = (4,4),
        strides=(1,1),
        padding='same',
        activation='relu'
        )(u1)
    u1 = Dropout(dropout_rate)(u1)
    u1 = BatchNormalization()(u1)
    u1 = Concatenate()([u1,d3])

    u2 = UpSampling2D(size=2)(u1)
    u2 = Conv2D(
        filters = 64*2 , 
        kernel_size = (4,4),
        strides=(1,1),
        padding='same',
        activation='relu'
        )(u2)
    u2 = Dropout(dropout_rate)(u2)
    u2 = BatchNormalization()(u2)
    u2 = Concatenate()([u2,d2])

    u3 = UpSampling2D(size=2)(u2)
    u3 = Conv2D(
        filters = 64 , 
        kernel_size = (4,4),
        strides=(1,1),
        padding='same',
        activation='relu'
        )(u3)
    u3 = Dropout(dropout_rate)(u3)
    u3 = BatchNormalization()(u3)
    u3 = Concatenate()([u3,d1])

    u4 = UpSampling2D(size=(2,2))(u3)
    outPutImg = Conv2D(
        filters = 3,
        kernel_size=(4,4),
        strides=(1,1),
        padding='same',
        activation='tanh'
        )(u4)
    return Model(d0 , outPutImg)

#定义鉴别器
def discriminator():
    d0 = Input(shape=img_shape)

    d1 = Conv2D(
        filters = 64 , 
        kernel_size = (4,4),
        strides=(2,2),
        padding='same'
    )(d0)
    d1 = LeakyReLU(alpha=0.2)(d1)

    d2 =  Conv2D(
        filters = 64*2 , 
        kernel_size = (4,4),
        strides=(2,2),
        padding='same'
    )(d1)
    d2 = LeakyReLU(alpha=0.2)(d2)
    d2 = BatchNormalization()(d2)

    d3 =  Conv2D(
        filters = 64*4 , 
        kernel_size = (4,4),
        strides=(2,2),
        padding='same'
    )(d2)
    d3 = LeakyReLU(alpha=0.2)(d3)
    d3 = BatchNormalization()(d3)

    d4 =  Conv2D(
        filters = 64*4 , 
        kernel_size = (4,4),
        strides=(2,2),
        padding='same'
    )(d3)
    d4 = LeakyReLU(alpha=0.2)(d4)
    d4 = BatchNormalization()(d4)

    validity = Conv2D(
        filters = 1,
        kernel_size=(4,4),
        strides=(1,1),
        padding='same',
    )(d4)

    return Model(d0 , validity)


#定义两个类别的鉴定器
optimizer = Adam(lr = 0.0002 , beta_1=0.5)
discriminator_A = discriminator()
discriminator_B = discriminator()
discriminator_A.compile(
    optimizer = optimizer,
    loss = 'mse',
    metrics = ['accuracy']
)
discriminator_B.compile(
    optimizer = optimizer,
    loss = 'mse',
    metrics = ['accuracy']
)


#定义两个类别相互的生成器

generator_A2B = generator()
generator_B2A = generator()

imgA = Input(shape=img_shape)
imgB = Input(shape=img_shape)

fakeB = generator_A2B(imgA)
fakeA = generator_B2A(imgB)

#用A生成伪装的B，再重构回A
reproductA = generator_B2A(fakeB)
reproductB = generator_A2B(fakeA)

#以冬夏互换举例，如果给夏天->冬天生成器输入冬天的图像，那么
#输出的也应该是冬天，所谓的一致性
imgA_identity = generator_B2A(imgA)
imgB_identity = generator_A2B(imgA)

#组合后的模型只训练生成器，不训练鉴别器

discriminator_A.trainable = False
discriminator_B.trainable = False

validA = discriminator_A(fakeA)
validB = discriminator_B(fakeB)

combinedModel = Model(
    inputs = [imgA , imgB],
    outputs = [
        validA , validB,
        reproductA,reproductB,
        imgA_identity,imgB_identity]
)
combinedModel.compile(
    loss = [
        'mse','mse',
        'mae','mae',
        'mae','mae'
    ],
    loss_weights = [
        1,1,
        10,10,
        1,1
    ],
    optimizer = optimizer
)


#组合数据迭代器

batch_size = 1
disc_patch = (8,8,1)
'''
trainImg_genF = ImageDataGenerator(
    rescale=1./127.5-1.,
    horizontal_flip=True
)


valImg_genF = ImageDataGenerator(
    rescale=1./127.5-1.
)


trainImg_gen = trainImg_genF.flow_from_directory(
    directory = './GAN/data/train',
    target_size=img_shape,
    batch_size=batch_size,
    class_mode = 'categorical'
)
valImg_gen = valImg_genF.flow_from_directory(
    directory = './GAN/data/test',
    target_size=img_shape,
    batch_size=batch_size,
    class_mode = 'categorical'
)


def MultipleGen(gen):
    while True:
        x = []
        y = []
        xResult = gen.next()[0]
        yResult =gen.next()[1][0]
        if result[0] == 0.:
            A = 

'''
import scipy

def getImages(imageDir):
    pattern = imageDir + '/*.jpg'
    imgs = io.ImageCollection(pattern)
    data = []
    for img in imgs:
        img = scipy.misc.imresize(img,(img_cols,img_rows)) 
        img = img *1.0/127.5 -1
        data.append(img)
    data = np.asarray(data,dtype='float')
    return data


trainImgsA = getImages('./GAN/data/train/A')
trainImgsB = getImages('./GAN/data/train/B')
valImgsA = getImages('./GAN/data/test/A')
valImgsB = getImages('./GAN/data/test/B')

#test
print('================')
print(np.shape(trainImgsA))
print(len(trainImgsA))
print(np.shape(trainImgsB))

batch_size = 32
trainMinL = min(len(trainImgsA),len(trainImgsB),batch_size)
valMinL = min(len(valImgsA) , len(valImgsB),batch_size)

oneValid = np.ones(disc_patch)
oneFake = np.zeros(disc_patch)
valid = np.ones((trainMinL,) + disc_patch)
fake = np.zeros((trainMinL,) + disc_patch)

epochs = 200
sample_interval = 1
import datetime
import sklearn
start_time = datetime.datetime.now()
print("=================Begin Training===========================")
for epoch in range(epochs):
    trainImgsA = sklearn.utils.shuffle(trainImgsA)
    trainImgsB = sklearn.utils.shuffle(trainImgsB)
    valImgsA = sklearn.utils.shuffle(valImgsA)
    valImgsB = sklearn.utils.shuffle(valImgsB)
    x = []
    x1 = []
    x2 = []
    validList = []
    y = []
    valid_y = []
    fake_y = []
    for i in range(trainMinL):
        #x.append([trainImgsA[i] , trainImgsB[i]])
        x1.append(trainImgsA[i])
        x2.append(trainImgsB[i])
        valid_y.append(oneValid)
        fake_y.append(oneFake)
        validList.append(oneValid)
        '''
        y.append([
            oneValid,oneValid,
            trainImgsA[i] , trainImgsB[i],
            trainImgsA[i] , trainImgsB[i]
        ])
        '''
    x = [x1,x2]
    y = [
            validList,validList,
            x1 , x2,
            x1 , x2
        ]
    '''
    Tx = []
    Ty = []
    Tvalid_y = []
    Tfake_y = []
    for i in range(valMinL):
        Tx.append([valImgsA[i] , valImgsB[i]])
        Tvalid_y.append(valid)
        Tfake_y.append(fake)
        Ty.append([
            valid,valid,
            valImgsA[i] , valImgsB[i],
            valImgsA[i] , valImgsB[i]
        ])
    '''
    fake_B = generator_A2B.predict(trainImgsA[:trainMinL])
    fake_A = generator_B2A.predict(trainImgsB[:trainMinL])
    discriminator_A.trainable=True
    dA_loss_real = discriminator_A.train_on_batch(trainImgsA[:trainMinL] , valid)
    dA_loss_fake = discriminator_A.train_on_batch(fake_A,fake)
    dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

    discriminator_B.trainable = True
    dB_loss_real = discriminator_B.train_on_batch(trainImgsB[:trainMinL] , valid)
    dB_loss_fake = discriminator_B.train_on_batch(fake_B,fake)
    dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

    d_loss = 0.5 * np.add(dA_loss, dB_loss)

    discriminator_A.trainable = False
    discriminator_B.trainable = False
    g_loss = combinedModel.train_on_batch(x,y)

    elapsed_time = datetime.datetime.now() - start_time

    print ("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6]),
                                                                            elapsed_time))

    if epoch % sample_interval == 0:
        numA = np.random.randint(0,len(valImgsA))
        numB = np.random.randint(0,len(valImgsB))
        sample_images(
            img_A =(valImgsA[numA]+1.0) *127.5,
            img_B = (valImgsB[numB]+1.0) *127.5 ,
            g_AB = generator_A2B,
            g_BA = generator_B2A,
            epoch=epoch)


generator_A2B.save('./GAN/models/generator_A2B.h5')
generator_B2A.save('./GAN/models/generator_B2A.h5')
discriminator_A.save('./GAN/models/discriminator_A.h5')
discriminator_B.save('./GAN/models/discriminator_B.h5')
combinedModel.save('./GAN/models/combinedModel.h5')