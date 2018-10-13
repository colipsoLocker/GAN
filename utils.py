import os
import numpy as np
import matplotlib.pyplot as plt

def sample_images(img_A,img_B, g_AB,g_BA,epoch):
        os.makedirs('./GAN/images_gen', exist_ok=True)
        r, c = 2, 3

        # Translate images to the other domain
        img_A = np.expand_dims(img_A, axis=0)
        img_B = np.expand_dims(img_B,axis = 0)
        fake_B = g_AB.predict(img_A)
        fake_A = g_BA.predict(img_B)
        # Translate back to original domain
        reconstr_A = g_BA.predict(fake_B)
        reconstr_B = g_AB.predict(fake_A)

        gen_imgs = np.concatenate([img_A, fake_B, reconstr_A, img_B, fake_A, reconstr_B])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("./GAN/images_gen/第%d次迭代抽样结果.png" % (epoch))
        plt.close()
