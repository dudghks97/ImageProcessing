"""
개요
    Sharpening Mask를 이용하여 영상의 첨예도를 강화한다.

목표
    Sharpening Mask가 [원본 영상 -(2차 미분)]으로 구성되어 있다는 사실을 밝힌다.
    이를 위해 실제로 마스크 상에서 위의 동작을 수행해서
    얻어진 필터를 filter2D 함수로 코릴레이션해서 첨예도가 향상된 영상을 얻는 과정을 보여준다.
    - 여기서 미분은 x좌표의 경우 중심 화소의 오른쪽 값에서 왼쪽화소 값을 빼는 동작을 1차 미분으로 정의하였다.

"""


import numpy as np
import cv2 as cv
# 만약 matplotlib 모듈이 없다면 윈도 컴맨드 창에서 "pip install matplotlib" 명령을 수행한다.
from matplotlib import pyplot as plt



# ----------------------------------------------------------------------------------------------------------------------
# 영상이 존재하는 폴더와 파일 이름을 지정하기.
# 다음 사례를 활용하여 영상 파일이 있는 폴더를 적절히 지정하시오.
# ----------------------------------------------------------------------------------------------------------------------
#Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
Path = '../data/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 Images 폴더.
#Path = 'd:/work/@@DIP/LectureMaterials/Images/'
Name = 'lenna.tif'
Name = 'monarch.png'
#Name = 'gray2.jpg'
#Name = 'GraphicShapes.png'
#Name = 'GraphicShapes.jpg'
#Name = 'graphicShapes.png'
#Name = 'circles_rects.png'
#Name = 'circles_rects_2.png'

def maximizeWindow():     # Win10에서 창크기 최대화하는 방법
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()


FullName = Path + Name
img = cv.imread(FullName)
# assert condition, message  : condition이 false이면  message 출력하면서 AssertError 발생.
assert img is not None, 'No image file....!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# ----------------------------------------------------------------------------------------------------------------------
# 단계 1
# 입력 영상을 1채널의 흑백 gray 영상으로 읽어들인다.
# ----------------------------------------------------------------------------------------------------------------------
#plt.figure(num='1.: Input images')


#plt.gray()
#maximizeWindow()
#plt.suptitle("1.: Input images", fontsize=12, fontweight='bold')
#plt.subplot(121); plt.imshow(img)
#plt.title('Original(color)'); plt.xticks([]), plt.yticks([])
#plt.waitforbuttonpress()


#"""
# ======================================================================================================================
# 실험 1: sharpening mask 커널을 정의한다.
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
# 단계 2
# 커널의 종류와 그 내부 계수 값을 정의한다.
# 라플라시안 연산자 - 2차 미분
# ----------------------------------------------------------------------------------------------------------------------

Laplacian4 = [[0, 1, 0], [1, -4, 1], [0, 1, 0]] # 4 neighbor Laplacian Kernel
Laplacian8 = [[1, 1, 1], [1, -8, 1], [1, 1, 1]] # 8 neighbor Laplacian Kernel
org_kernel = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]  # original image kernel

Lap4 = np.array(Laplacian4, dtype=float )     # print(type(Lap4))
Lap8 = np.array(Laplacian8, dtype=float )     # print(type(Lap8))
Org = np.array(org_kernel, dtype=float )

smask4 = Org - Laplacian4; smask8 = Org - Laplacian8
print("smask4=\n", smask4); print("smask8=\n", smask8)


# ----------------------------------------------------------------------------------------------------------------------
# 단계 3
# 2차원 correlation 연산을 수행한다.
# dst= cv.filter2D(	src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]]	)
# ----------------------------------------------------------------------------------------------------------------------
edge4 = cv.filter2D(src=img_gray, ddepth=-1, kernel=Lap4); print('edge4.dtype=', edge4.dtype)
edge8 = cv.filter2D(src=img_gray, ddepth=-1, kernel=Lap8); print('edge8.dtype=', edge8.dtype)

sharpening_mask4 = cv.filter2D(src=img_gray, ddepth=-1, kernel=smask4)
sharpening_mask8 = cv.filter2D(src=img_gray, ddepth=-1, kernel=smask8)


# ----------------------------------------------------------------------------------------------------------------------
# 단계 4
# 서브화면에 원본과 처리 결과를 출력한다. 이후 마우스를 입력할 때까지 기다린다.
# subplot(NMK)가 의미하는 것 : 화면을 rowXcolumn(NxM)개의 sub 창으로 나누고 K 번째 서브창을 지정한다.
# 이후 imshow() 함수는 지정된 sub 창에 출력한다.
# pyplt.imshow()에서는 흑백 영상은 의사 칼라 영상으로 출력되기 때문에 cmap를 gray로 정해야 한다.
# ----------------------------------------------------------------------------------------------------------------------

# 4 neighbors Laplacian 영상 및 4 neighbors Laplacian 영상
plt.figure(num='4.1: Laplacian & sharpening mask images')
maximizeWindow()
plt.suptitle("Laplacian & sharpening mask images", fontsize=12, fontweight='bold')

plt.gray()

plt.subplot(231), plt.imshow(img_gray)
plt.title('4 neighbor Laplacian edge'); plt.xticks([]), plt.yticks([])

plt.subplot(232), plt.imshow(edge4)
plt.title('4 neighbor Laplacian edge'); plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(edge8)
plt.title('8 neighbor Laplacian edge'); plt.xticks([]), plt.yticks([])

plt.subplot(235), plt.imshow(sharpening_mask4)
plt.title('4 neighbor sharpening mask image'); plt.xticks([]), plt.yticks([])
plt.subplot(236), plt.imshow(sharpening_mask8)
plt.title('8 neighbor sharpening mask image'); plt.xticks([]), plt.yticks([])




# cololr 영상에 대헤 적용한다...
plt.figure(num='Color: original ...')
maximizeWindow()
plt.imshow(img[..., ::-1])
plt.title('input image')
plt.axis('off')

sharpening_mask4 = cv.filter2D(src=img, ddepth=-1, kernel=smask4)
sharpening_mask8 = cv.filter2D(src=img, ddepth=-1, kernel=smask8)

plt.figure(num='color: mask4')
maximizeWindow()
plt.imshow(sharpening_mask4[..., ::-1])
plt.title('4 neighbor sharpening mask image')
plt.axis('off')

plt.figure(num='color: mask8')
maximizeWindow()
plt.imshow(sharpening_mask8[..., ::-1])
plt.title('8 neighbor sharpening mask image')
plt.axis('off')

plt.show()

