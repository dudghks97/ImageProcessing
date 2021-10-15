"""
개요:
    영상의 히스토그램 명세화를 실현한다.
    1) skimage.exposure 모듈의 cumulative_distribution() 함수를 통해 ncdf 값을 연산하는 연습 프로그램을 학습한다.
    2) 소스 영상과 템플레이트 영상을 제공하여 주어진 템플레이트의 RGB 히스토그램이 나올 수 있도록 소스 영상의 히스토그램을 변환한다.
    이를 위해서는 히스토그램 명세화를 실행하는 맵핑함수의 개발이 필요하다.

One-dimensional linear interpolation.
numpy.interp(x, xp, fp, left=None, right=None, period=None)
    Returns the one-dimensional piecewise linear interpolant to a function
    with given discrete data points (xp, fp), evaluated at x.
    x: array_like
        The x-coordinates at which to evaluate the interpolated values.
    xp: 1-D sequence of floats
        The x-coordinates of the data points, must be increasing if argument period is not specified.
        Otherwise, xp is internally sorted after normalizing the periodic boundaries with xp = xp % period.
    fp: 1-D sequence of float or complex
        The y-coordinates of the data points, same length as xp.


"""

path = '../data/hist_spec/'

# s_file: source file. 변형된 히스토그램이 적용될 영상.
# t_file: template 영상. 닮고자하는 히스토그램을 갖고 있는 영상
#s_file = 'forest-park-trail.jpg'; t_file = 'snowy.jpg'
s_file = 'fseprd746674.jpg'; t_file = 'flames.jpg'
#s_file = 'CSLN-Member-gathering.jpg'; t_file = 'Autumn_forest-800x450.jpg'
#s_file = 'sirens-descansoenchanted-8.jpg'; t_file = 'Autumn_forest-800x450.jpg'
#s_file = 'sirens-descansoenchanted-8.jpg'; t_file = 'snowy.jpg'



from skimage.exposure import cumulative_distribution
from skimage.io import imread
from skimage.color import rgb2gray
# pylab combines pyplot with numpy into a single namespace.
# This is convenient for interactive work,
# but for programming it is recommended that the namespaces be kept separate
#import matplotlib.pylab as plt

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def maximizeWindow():     # Win10에서 창크기 최대화하는 방법
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

# 원본 영상의 cdf, c와 템플레이트 영상의 cdf, c_t 및 원본 영상, im을 받아
# 이에 대한 히스토그램 명세화를 행한 영상과 그 mapping 함수를 반환한다.
def hist_matching(c, c_t, im):

    # c: 원본 영상의 cdf. 변형된 히스토그램이 적용될 영상의 cdf
    # c_t: template 영상의 cdf. 닮고자하는 히스토그램을 갖고 있는 영상의 cdf
    # im: 원본 영상. 이것이 c_t의 히스토그램으로 변환된 영상이 반환된다.

    b = np.interp(c, c_t, np.arange(256))   # find closest matches to b_t
        # c: The x-coordinates at which to evaluate the interpolated values.
        # c_t: The x-coordinates of the data points
        # np.arange(256): The y-coordinates of the data points, same length as xp.

    pix_repl = {i:b[i] for i in range(256)} # dictionary to replace the pixels
    mp = np.arange(0, 256)
    for (k, v) in pix_repl.items():     # key:value
        mp[k] = v

    #s = im.shape; im = np.reshape(mp[im.ravel()], im.shape); im = np.reshape(im, s) # 아래 한 줄이면 간단히 처리.
    im = mp[im]
    return im, mp

def cdf(im):
    # 영상 im의 ncdf(normalized cumulative distribution function)를 반환한다.
    c, b = cumulative_distribution(im)  # 이 함수는 cdf와 bin을 반환하는데, 그대로 사용하기가 불편하다. 길이가 256개가 아니다.
    # 0~255까지의 cdf 값이 있는 것이 아니라, b(0)...b(n-1)까지의 n개의 cdf 값이 있다.
    # 따라서 아래와 같이 b(0) 앞부분에는 0을 채워 넣어야 하고, b(n-1) 이후 뒷부분에는 1을 추가해야 한다.
    for i in range(b[0]):
        c = np.insert(c, 0, 0)      # b[0]의 앞부분의 자리의 c 값(cdf)은 0을 그만큼 채워주어야 한다.
    for i in range(b[-1]+1, 256):   # b(n-1) 뒷부분에는 1의 값을 채워주어야 한다.
        c = np.append(c, 1)
    np.set_printoptions(precision=1)
    #print(f"ncdf: shape={c.shape},\n{c}")
    print(f"ncdf: shape={c.shape}")
    #print(f"bin_centers: shape={b.shape}, \n{b}")
    print(f"bin_centers: shape={b.shape}")
    return c

import sys
print(f"python version={sys.version}\nopenCV version={cv.__version__}")

"""
# -------------------------------------------------------------------------------------------------------------
# 사전연습
# from skimage.exposure import cumulative_distribution
#   cumulative_distribution() 함수를 연습해 본다. => 이 함수로 일단 histogram equalization을 행한다.
# -------------------------------------------------------------------------------------------------------------
im = imread(path + s_file)
# img_cdf, array: Values of cumulative distribution function.
# bin_centers, array: Centers of bins.
gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
gray_cdf = cdf(gray)

plt.figure()
#maximizeWindow()
plt.figure(figsize=(15, 8), num='Histogram Equalization') 
plt.suptitle("Histogram Equalization by skimage.exposure.cumulative_distribution", fontsize=15, fontweight='bold')
plt.subplot(221), plt.imshow(im)
plt.axis('off')
plt.title('Input Image', size=20)

plt.subplot(222), plt.plot(gray_cdf)
plt.grid('on')
plt.title('normalized cumulative distribution function', size=20)

LUT = (255 * gray_cdf).astype(np.uint8)
print(f"LUT: shape={LUT.shape}")
plt.subplot(223), plt.imshow(LUT[im])
plt.axis('off')
plt.title('Histogram Equalized Image', size=20)

plt.show()
exit(0)
"""


#"""

# -------------------------------------------------------------------------------------------------------------
# 시작 부분
# -------------------------------------------------------------------------------------------------------------

# skimage.io 함수 imread()는 RGB로 구성되어 있다.

im = imread(path + s_file)      # 원본 영상. 변형된 히스토그램이 적용될 영상.
im_t = imread(path + t_file)    # template 영상. 닮고자하는 히스토그램을 갖고 있는 영상
print(f'im: shape={im.shape}, size={im.size}, dtype={im.dtype}, max={np.max(im)}, min={np.min(im)}')

im1 = np.zeros(im.shape, dtype=np.uint8)

plt.figure()
maximizeWindow()
plt.suptitle("NCDFs of src(Cyan)/template(tab:pink) & mapping function(Black) for Each Color Plane", fontsize=15, fontweight='bold')
clr_title = ['Red', 'Green', 'Blue']
for i in range(3):          # RGB 채널에 대해 loop 작업
    c = cdf(im[..., i])     # 원본의 cdf
    c_t = cdf(im_t[..., i])     # template 영상의 cdf
    im1[..., i], mp = hist_matching(c, c_t, im[..., i]) # 히스토그램 명세화된 영상과 그 매핑함수를 반환
    plt.subplot(221 + i)
    plt.plot(c * 255, 'c', label='org(ncdf)')
    plt.plot(c_t * 255, 'tab:pink', label='template(ncdf)')
    plt.plot(mp, 'k', label='mapping')
    plt.title(clr_title[i], size=20)
    plt.grid('on'), plt.legend(fontsize=15)

#plt.subplot(224)        # src/template/변환 후 영상의 히스토그램 분포(rgb)를 보인다.


np.set_printoptions(precision=2)
#print(f'len(c)={len(c)}: c={c}')
#print(f'len(c_t)={len(c_t)}: c_t={c_t}')

#fig, ax = plt.subplots(2, 2)
#plt.subplots_adjust(left=0, top=0.95, right=1, bottom=0, wspace=0.05, hspace=0.05)

plt.figure()
maximizeWindow()
plt.suptitle("Source/Template/Output Images", fontsize=15, fontweight='bold')
plt.subplot(221), plt.imshow(im)
plt.axis('off')
plt.title('Input Image', size=20)

plt.subplot(222), plt.imshow(im_t), plt.axis('off'), plt.title('Template Image', size=20)
#plt.subplot(223), plt.imshow(im1[..., :3]), plt.axis('off'), plt.title('Output Image', size=20)
plt.subplot(223), plt.imshow(im1), plt.axis('off'), plt.title('Output Image', size=20)

#"""

# 미션: 새 창을 열고 src/template/output 영상의 RGB별 히스토그램을 그려 보자.


# img가 3채널 칼라 영상을 가정하고 r, g, b 히스토그램을 반환한다.
def myHist(img):            # img 영상은 RGB로 구성된 것으로 가정한다.
    hist_lst = []
    #for i in np.arange(2, -1, -1):      # b, g, r을 가정
    for i in np.arange(0, 3, 1):    # r, g, b를 가정
        hist, bin_edges = np.histogram(img[..., i],
                                   256,  # bins : int or sequence of scalars or str, optional
                                   [0, 255])  # range : (float, float), optional
        #print(len(bin_edges), bin_edges.astype("int"))
        print(len(bin_edges))
        hist_lst.append(hist/np.max(hist))  # 정규화된 히스토그램.
        #hist_lst.append(hist)           # 정규화되지 않은 히스토그램
    return hist_lst[0], hist_lst[1], hist_lst[2]        #r, g, b

plt.figure()
maximizeWindow()
plt.suptitle("Source/Template/Output Histogram", fontsize=15, fontweight='bold')
src = imread(path+s_file)           # skimage.io 함수 imread()는 RGB로 구성되어 있다.
template = imread(path+t_file)      # skimage.io 함수 imread()는 RGB로 구성되어 있다.
ttl_lst = ['source', 'template', 'output']
for i, img in enumerate([src, template, im1]):      # src/template/output
    r, g, b = myHist(img)
    plt.subplot(221 + i)
    plt.plot(r, color='r')
    plt.plot(g, color='g')
    plt.plot(b, color='b')
    #plt.axis('off')
    plt.grid('on')
    plt.title(ttl_lst[i])
plt.show()





