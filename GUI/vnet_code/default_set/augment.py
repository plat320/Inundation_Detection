import numpy as np

def np_select_rand_pos(img):
    max_x = img.shape[1]-128
    max_y = img.shape[0]-128
    if max_x<0 or max_y<0:
        print("This image size is too small.")
        return 0, 0
    return np.random.randint(max_x), np.random.randint(max_y)

def np_crop_img(img,x,y):
    return img[y:y+128,x:x+128,:]

def np_rand_flip(img,flip_flag):
    if flip_flag:
        return np.flip(img,1)
    else:
        return img

def np_rand_noise(img):
    noise_flag = np.random.randint(2)
    if noise_flag:
        s = np.random.normal(0, 25, (128, 128, 3))
        tmp = img + s
        tmp[tmp>255] = 255
        tmp[tmp<0] = 0
        tmp = tmp.astype(np.uint8)
        return tmp
    else :
        return img