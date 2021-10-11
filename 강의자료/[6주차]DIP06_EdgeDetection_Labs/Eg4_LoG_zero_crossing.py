"""
Multidimensional Laplace filter using gaussian second derivatives.
scipy의 LoG 함수를 활용한다.

https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_laplace.html

Zero Crossing을 찾는 조건으로 무조건 +, - 교차점만을 찾는 것은 문제가 있다.
1) 작은 크기의 LoG 값의 부호 변화는 무시해야 한다.
2) +, -의 중간에 화소가 존재하지 않을 수도 있으므로 부호 변화가 일어나면 인정하는 전략을 취한다.
3) line의 방향 성분에 따라 처리할 필요가 있다.


"""

Path = '../data/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 Images 폴더.
#Path = 'd:/work/@@DIP/LectureMaterials/Images/'
#Name = 'monarch.jpg'
#Name = 'graphicShapes.png'
Name = 'circles_rects_2.png'
Name = 'lenna.tif'
#Name = 'circles_rects.png'
#Name = 'mdl.tiff'


from scipy import ndimage, misc
import matplotlib.pyplot as plt
import cv2 as cv
import scipy as sp
import numpy as np

FullName = Path + Name
img = cv.imread(FullName, 0)
# assert condition, message  : condition이 false이면  message 출력하면서 AssertError 발생.
assert img is not None, f'<{FullName}> is not found!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.
img = img/255.0


def maximizeWindow():     # Win10에서 창크기 최대화하는 방법
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")


#img = misc.lena()
#img = misc.ascent()
#img = misc.face()  # 주의! 3채널 영상

plt.figure(num='scipy LoG', figsize=(10, 8))
#maximizeWindow()
plt.suptitle("Laplacian of Gaussian by Numpy ", fontsize=12, fontweight='bold')

plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('Original'); plt.xticks([]), plt.yticks([])


log_lst = []        # list for log result
s_lst = [1.5, 3, 4]   # sigma values
for i, s in enumerate(s_lst):
    result = ndimage.gaussian_laplace(img, sigma=s)
    log_lst.append(result)
    plt.subplot(222 + i)
    plt.imshow(result, cmap='gray')
    plt.title(f'LOG: sigma={s}')
    plt.xticks([]), plt.yticks([])

plt.tight_layout() # 맨 나중에 써야 영향을 미침.
plt.waitforbuttonpress()

plt.figure(num='simple Zero Crossing', figsize=(10, 8))
#maximizeWindow()
plt.suptitle("simple Zero Crossing", fontsize=12, fontweight='bold')

plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('Original'); plt.xticks([]), plt.yticks([])


# 2차원데이터에 맨 아래 줄과 오른쪽 끝에 각각 0을 1행, 1열을 패딩하여 확장한다.
#a = 5*np.ones((3,3)); lap = np.pad(a, ((0, 1), (0, 1))); print(lap); exit()

# 2차원데이터에 맨 아래 줄과 오른쪽 끝에 각각 0을 2행, 1열을 패딩하여 확장한다.
#a = 5*np.ones((3,3)); lap = np.pad(a, ((0, 2), (0, 1))); print(lap); exit()

th_lst = [0.00001, 0.0000001, 0.00000001]
#th_lst = [0, 0, 0]
for i, (log, s) in enumerate(zip(log_lst, s_lst)):

    # log 어레이의 맨 아래 줄과 오른쪽 끝에 각각 0을 1행, 1열을 패딩하여 확장한다.
    lap = np.pad(log, ((0, 1), (0, 1)))
    # 2개의 포인트만 관찰

    # 방법 1: 관심 포인트의 우측, 하단과의 부호가 다르면 에지이다.
    # lap = np.pad(log, ((0, 1), (0, 1)))
    #diff_x = lap[:-1, :-1] * lap[:-1, 1:] < -th_lst[i]
    #diff_y = lap[:-1, :-1] * lap[1:, :-1] < -th_lst[i]
    # zeroCross = np.logical_or(diff_x, diff_y).astype(float)

    # 방법 2: 현 지점의 음수이고, 관심 포인트의 우측부호, 하단 부호가 +이면 에지이다. ==> 틀림
    # lap = np.pad(log, ((0, 1), (0, 1)))
    #diff_x = (lap[:-1, :-1] < 0) & (lap[:-1, 1:] > th_lst[i])
    #diff_y = (lap[:-1, :-1] < 0) * (lap[1:, :-1] > th_lst[i])
    #zeroCross = np.logical_or(diff_x, diff_y).astype(float)

    # 방법 3: x 에지가 판별될 때는 상하 에지가 없어야 한다. 즉, 곱해서 0보다 커야한다.
    lap = np.pad(log, ((0, 2), (0, 2)))
    diff_y_u = lap[1:-1, 1:-1] * lap[0:-2, 1:-1] < -th_lst[i]     # up. 상하 연속된 에지
    diff_y_d = lap[1:-1, 1:-1] * lap[2:, 1:-1] < -th_lst[i]       # down

    diff_x_l = lap[1:-1, 1:-1] * lap[1:-1, :-2] < -th_lst[i]     # left.좌우 연결된 에지
    diff_x_r = lap[1:-1, 1:-1] * lap[1:-1, 2:] < -th_lst[i]      # right

    diff_x = diff_x_l | diff_x_r
    diff_y = diff_y_u | diff_y_d

    #diff_x = diff_x_l           # 둘중의 하나만 취하면 부족함
    #diff_y = diff_y_u

    zeroCross = np.logical_or(diff_x, diff_y).astype(float)

    plt.subplot(222 + i)
    plt.imshow(zeroCross, cmap='gray')
    plt.title(f'zeroCross: sigma={s}')
    plt.xticks([]), plt.yticks([])

plt.tight_layout() # 맨 나중에 써야 영향을 미침.
plt.waitforbuttonpress()
#plt.show()












"""
# 모폴러지 연산에 의한 0점 교차점 찾기
th = 0.001
for i, (log, s) in enumerate(zip(log_lst, s_lst)):
    # erode 연산: 주어진 커널에서 가장 작은 값으로 중심 값(anchor point)을 교체
    minLoG = cv.morphologyEx(log, cv.MORPH_ERODE, np.ones((3, 3)))

    # dilate 연산: 주어진 커널에서 가장 큰 값으로 중심 값(anchor point)을 교체
    maxLoG = cv.morphologyEx(log, cv.MORPH_DILATE, np.ones((3, 3)))

    # anchor point 주변에 큰 값의 부호 변화가 있으면 영점 교차점으로 인정. 아직 방향성분은 고려하지 않았다.
    zeroCross = np.logical_and(minLoG < -th, log > th)

    # zero crossing은 log anchor point 주변에서 부호가 다르면 영점 교차점으로 인정한다. 2개의 조건이 OR 조건임.
    #zeroCross = np.logical_or(np.logical_and(minLoG < -10* th, log > +th), np.logical_and(maxLoG > 10*th, log < -th))
    #zeroCross = np.logical_or(np.logical_and(minLoG < -th, log > th), np.logical_and(maxLoG > th, log < -th))
    #zeroCross = np.logical_and(minLoG < -th, log > th)

    plt.subplot(222 + i)
    plt.imshow(zeroCross, cmap='gray')
    plt.title(f'zeroCross: sigma={s}')
    
    
    plt.xticks([]), plt.yticks([])

plt.show()
"""










