"""
참고 URL
        https://www.imageeprocessing.com/2011/10/add-salt-and-pepper-noise-to-image.html

    영상에 잡음을 첨가하는 MATLAB 함수, imnoise와 유사한 동작을 하는 함수를 구현하는 방법에 대해 고찰한다.
        방법 1) sckit-image를 이용해 함수를 통해 쉽게 구현 -> 추천
        방법 2) 자체 아이디어로 제작. 아직 제작 중....
        방법 3) 외부 자료를 활용. -> 미 검토
    Name으로 주어지는 파일이름에 .PNG 파일로 잡음를 첨가하여 파일로 저장하여 출력한다.


"""
#import cv2 as cv
#import matplotlib.pyplot as plt
import numpy as np, cv2 as cv, matplotlib.pyplot as plt

# 영상의 기본적인 정보를 출력하는 함수
def printImgBriefs(img_str):
    im = eval(img_str)
    print(img_str+'.shape=', im.shape)
    print(img_str+'.dtype=', im.dtype)
    print('np.amax('+img_str+')=', np.amax(im))     # 값의 최대 범위를 점검한다.
    #print('np.amin('+img_str+')=', np.amin(im))     # 값의 최소 범위를 점검한다.
    print('np.amin(' + img_str + ')=', im.min())  # 값의 최소 범위를 점검한다.


#------------------------------------------------------------------------------------
# 단계 0 : 영상이 존재하는 폴더와 파일 이름을 지정하여 영상 파일 읽어 내기
# 다음 사례를 활용하여 영상 파일이 있는 폴더를 적절히 지정하시오.
#------------------------------------------------------------------------------------
#Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
Path = '../data/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 Images 폴더.
Name = 'bone(m).bmp'
Name = 'lenna.tif'
#Name = 'monarch.png'
#Name = 'fruits.jpg'
Name = 'BAR.bmp'




FullName = Path + Name
imgOrg = cv.imread(FullName, 0)

assert imgOrg is not None, "Failed to load image file.."

cv.imshow('original image', imgOrg)


#"""
#------------------------------------------------------------------------------------
# 방법 1 : scikitimage 에서 제공하는 함수, random_noise를 이용한다.
#   pip install scikit-image
# https://scikit-image.org/docs/dev/api/skimage.util.html?highlight=random_noise#skimage.util.random_noise
# 설치 상황에 따라 scipy도 추가 설치가 필요할 수도 있다. pip install scikit-image
# 사전에 미리 pip를 최신 버전으로 업그레이드하는 것이 좋다. pip install --upgrade pip
#------------------------------------------------------------------------------------


from skimage.util import random_noise       # 패키지 이름은 scikit-image, 모듈 이름은 skimage

# 입력 영상에 가우시안 잡음 첨가
# default: mean = 0, variance = 0.01 => sigma(표준편차) = 0.1
gauss = random_noise(imgOrg, mode='gaussian', seed=None, clip=True)

# 입력 영상에 salt & pepper 잡음 첨가
# default variance = 0.01 => sigma(표준편차) = 0.1
sp = random_noise(imgOrg, mode='s&p', seed=None, clip=True)

printImgBriefs('gauss')
printImgBriefs('sp')

cv.imshow('gauss', gauss)
cv.imshow('salt & pepper', sp)

# 확장자를 붙여 저장한다.
cv.imwrite(Name+'_gauss.png', (255*gauss).astype(np.uint8))
cv.imwrite(Name+'_sp.png', (255*sp).astype(np.uint8))
cv.waitKey()

exit(0)

#"""








#------------------------------------------------------------------------------------
# 방법 2 : 잡음 첨가 영상을 함수에 의존하지 않고 수작업으로 만들어 본다.
#   https://realpython.com/python-random/
#   https://www.imageeprocessing.com/2011/10/add-salt-and-pepper-noise-to-image.html
#------------------------------------------------------------------------------------
img = (imgOrg/255)
print("img.dtype=", img.dtype)

#im = np.zeros(img.shape, img.dtype)
# random.randrange(5)       # 0~4까지의 임의의 정수를 반환.
# random.randrange(1, 7)    # 1~6까지의 임의의 정수를 반환.
# random()                  # 0부터 1 사이의 임의의 float

# Samples are uniformly distributed over the half-open interval [low, high)
# (includes low, but excludes high).
# In other words, any value within the given interval is equally likely to be drawn by uniform.
#im=(np.random.uniform(-0.5, 0.5, size=img.shape)).astype(dtype=np.float32)
im=np.random.uniform(-0.2, 0.2, size=img.shape)
print('im.shape=', im.shape)
print("im.dtype=", im.dtype)
print("np.amax(im)=", np.amax(im))
print("np.amin(im)=", np.amin(im))

#img = cv.add(img, im) # add the noise to the original image
img = img + im

#thresh=1.0; maxval=0; retval, img	= cv.threshold(img, thresh, maxval, type=cv.THRESH_TRUNC)

print("img.dtype=", img.dtype)
print("np.amax(img)=", np.amax(img))
print("np.amin(img)=", np.amin(img))


cv.imshow('noisy image', img)

# medianBlur 함수의 입력 영상의 데이터형이 kisze=5일 때는 32비트 부동소수까지 가능
# uint8 형으로 바꾼다.
black = 0.0
BlackArray = ~(img <= black)
print('BlackArray.shape=', BlackArray.shape)
print('BlackArray.dtype=', BlackArray.dtype)
cv.imshow('BlackArray', (255*BlackArray).astype(np.uint8))

NoiseImg = img.copy()
NoiseImg *= BlackArray
cv.imshow('NoiseImg', NoiseImg)

thresh = 1; maxval = 0
retval, img	= cv.threshold(NoiseImg, thresh, maxval, type=cv.THRESH_TRUNC)
img = (255*img).astype(np.uint8)

#WhiteArray = img >=white
#print('WhiteArray.shape=', WhiteArray.shape)
#cv.imshow('WhiteArray', (255*WhiteArray.astype(np.uint8)))
#NoiseImg |= WhiteArray
#img = NoiseImg.copy()

print("img.dtype=", img.dtype)
print("np.amax(img)=", np.amax(img))
print("np.amin(img)=", np.amin(img))


#img = (255 * img).astype(dtype=np.uint8)
#cv.imwrite('tmp.jpg', img); img = cv.imread('tmp.jpg')

cv.imshow('noisy image in uint8', img)

exit(0)


#=================================================================
# 이하 기록용: 루틴을 설계한 사례. - 아직 검토 안함.
# gaussian , salt-pepper , poisson and speckle noise in an image
# https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
#=================================================================
"""
Parameters
----------
image : ndarray
    Input image data. Will be converted to float.
mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
"""

import numpy as np
import os
import cv2
def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
   elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
   elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)
      noisy = image + image * gauss
      return noisy


