"""
1. 프로그램의 기능
    실험 1: 2차 미분 필터 커널을 정의하여 코릴레이션 공간 필터링을 통해 에지를 검출한다.
    실험 2: Laplacian() 함수를 이용하여 에지를 검출한다.

    * 돋보기(확대) 메뉴를 이용하여 결과를 관찰하기 바란다. 확대하면 에지가 보이는 경우가 많음.

2. 프로그램의 목표
    2차 미분에 의한 이진화 기법의 활용 가능성에 오해가 있을 수 있음을 인지하고자 한다.
    - 얼핏 보면 쓸 만한 결과가 있는 것 처럼 보인다.

3. 미션
    에지의 값을 + 값은 붉은 색상의 계조 값으로, - 값은 파란 색상의 계조 색상으로 한 화면에 표시하시오.
    - 제시된 프로그램으로는 음수와 양수 부분을 정확히 구분하여 판단하기가 어렵다.
    - 그 이유는 이들 음수, 양수부분의 에지가 근접한 부분에 밀집해 있는 경우가 있기 때문이다.
    openCV 환경으로 표현하기로 한다.

4. 질문 - 다음 절(LoG)을 학습하고 되돌아와서 확인 바람.
    1) 어떤 경우 라플라스 2차 미분의 결과 값이 음수 혹은 양수가 되는 것인지 추론 할 수 있을까?
    2) 1차 미분처럼 음수와 양수의 절대 값을 모두 합산하여 출력하는 것이 바람직할까? 어떤 문제점이 있을까?

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
Name = 'monarch.tiff'; Th=30/255.0
#Name = 'gray2.jpg'; Th=60/255.0
Name = 'GraphicShapes.png';  Th=60/255.0   # 고품질 그래픽스 영상
#Name = 'GraphicShapes.jpg';     # 50의 품질로 압축된 영상
#Name = 'graphicShapes.png'
#Name = 'circles_rects_2.png'
Name = 'lenna.tif'
Name = 'circles_rects.png'

def maximizeWindow():     # Win10에서 창크기 최대화하는 방법
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")



FullName = Path + Name
img = cv.imread(FullName)
# assert condition, message  : condition이 false이면  message 출력하면서 AssertError 발생.
assert img is not None, 'No image file....!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.


# ----------------------------------------------------------------------------------------------------------------------
# 단계 0
# interactive mode로 설정한다. plt.show() 없이 plt.imshow()만으로 출력된다.
# cv.imshow()를 사용하면 이 부분은 주석처리해야 한다.
# matplotlib를 이용해 화면에 영상을 출력하려면 이 부분을 활성화 시킨다.
# OpenCV 함수를 이용해 BGR 평면 배열을 RGB 평면 배열로 바꾼다.
# plt.ion();     # interactive mode로 설정한다. plt.show() 없이 plt.imshow()만으로 출력된다.
# ----------------------------------------------------------------------------------------------------------------------
b, g, r = cv.split(img); img = cv.merge([r,g,b]);
img_c = img.copy()      # img의 복사본을 하나 저장해 둔다.


# ----------------------------------------------------------------------------------------------------------------------
# 단계 1
# 에지 영상은 흑백 영상으로 관찰하는 것이 바람직하다.
# 입력 영상을 1채널의 흑백 gray 영상으로 변환한다.
# dst=cv.cvtColor(src, code[, dst[, dstCn]]	)
#       여기서, code= cv.COLOR_RGB2GRAY, cv.COLOR_BGR2GRAY. 주의!  적정한 것을 선택.
# ----------------------------------------------------------------------------------------------------------------------
img = cv.cvtColor(src=img, code=cv.COLOR_RGB2GRAY)/255

plt.figure(num='1: Input images', figsize=(12, 8))
plt.suptitle("1: Input images", fontsize=12, fontweight='bold')

plt.subplot(121); plt.imshow(img_c)      # 3채널로 출력
plt.title('Original(color)'); plt.xticks([]), plt.yticks([])
print("cv2: type(img)=", type(img), " img.dtype=", img.dtype, "img.shape=", img.shape)
plt.subplot(122); plt.imshow(img, cmap = 'gray')
plt.title('gray converted'); plt.xticks([]), plt.yticks([])
plt.tight_layout() # 맨 나중에 써야 영향을 미침.
plt.waitforbuttonpress()

#"""
# ======================================================================================================================
# 실험 1: 라플라스 커널 정의 후 필터링 작업에 의한 2차 미분 에지 검출
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# 단계 2
# 커널의 종류와 그 내부 계수 값을 정의한다.
# 라플라시안 연산자 - 2차 미분
# ----------------------------------------------------------------------------------------------------------------------

Laplacian4 = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]	    # 4 neighbor Laplacian Kernel
Laplacian8 = [[1, 1, 1], [1, -8, 1], [1, 1, 1]] 	# 8 neighbor Laplacian Kernel

Lap4 = np.array(Laplacian4, dtype=float );     # print(type(Lap4))
Lap8 = np.array(Laplacian8, dtype=float );       # print(type(Lap8))

#커널을 출력하되 원하는 정밀도로 출력하게 한다. >>> help(np.set_printoptions)
np.set_printoptions(precision=1, floatmode='fixed')
print('Lap4=\n', Lap4)  # 소수점 이하 1째 자리까지 출력.
print('Lap8=\n', Lap8)


# ----------------------------------------------------------------------------------------------------------------------
# 단계 3
# 2차원 correlation 연산을 수행한다.
# dst= cv.filter2D(	src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]]	)
# ----------------------------------------------------------------------------------------------------------------------
edge4 = cv.filter2D(src=img, ddepth=-1, kernel=Lap4); print('edge4.dtype=', edge4.dtype)
edge8 = cv.filter2D(src=img, ddepth=-1, kernel=Lap8); print('edge8.dtype=', edge8.dtype)


# ----------------------------------------------------------------------------------------------------------------------
# 단계 4
# 서브화면에 원본과 처리 결과를 출력한다. 이후 마우스를 입력할 때까지 기다린다.
# subplot(NMK)가 의미하는 것 : 화면을 rowXcolumn(NxM)개의 sub 창으로 나누고 K 번째 서브창을 지정한다.
# 이후 imshow() 함수는 지정된 sub 창에 출력한다.
# pyplt.imshow()에서는 흑백 영상은 의사 칼라 영상으로 출력되기 때문에 cmap를 gray로 정해야 한다.
# ----------------------------------------------------------------------------------------------------------------------
print(f'Lab 1: step 4.1- input image dtype={img.dtype}')
# 4 neighbors Laplacian 영상 및 4 neighbors Laplacian 영상
plt.figure(num='Lab 1: step 4.1- filter2D Laplacian images', figsize=(12, 8))
#maximizeWindow()
plt.suptitle("Lab 1: step 4.1- filter2D Laplacian images", fontsize=12, fontweight='bold')

plt.subplot(121),plt.imshow(edge4, cmap = 'gray')
plt.title('4 neighbor Laplacian edge'); plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edge8, cmap = 'gray')
plt.title('8 neighbor Laplacian edge'); plt.xticks([]), plt.yticks([])
plt.tight_layout() # 맨 나중에 써야 영향을 미침.
plt.waitforbuttonpress()        # 현 화면에서 버튼 입력을 기다린다.


# 4 neighbors Laplacian 영상의 이진 영상(+성분, -성분, 통합성분)
print(f'Lab 1: step 4.2- input image dtype={img.dtype}')
plt.figure(num='Lab 1: step 4.2- Binary Laplacian edge', figsize=(12, 8))
#maximizeWindow()
plt.suptitle("Lab 1: step 4.2- Binary Laplacian edge", fontsize=12, fontweight='bold')

bin4p = edge4 > Th
plt.subplot(231),plt.imshow(1-bin4p, cmap='gray')
plt.title('4 neighbor edge(+)'); plt.xticks([]), plt.yticks([])

bin4n = edge4 < - Th
plt.subplot(232),plt.imshow(1-bin4n, cmap='gray')
plt.title('4 neighbor edge(-)'); plt.xticks([]), plt.yticks([])

bin4 = bin4p | bin4n
plt.subplot(233), plt.imshow(1-bin4, cmap='gray')
plt.title('4 neighbor edge'); plt.xticks([]), plt.yticks([])

# 8 neighbors Laplacian 영상의 이진 영상(+성분, -성분, 통합성분)
bin8p = edge8 > Th
plt.subplot(234),plt.imshow(1-bin8p, cmap='gray')
plt.title('8 neighbor edge(+)'); plt.xticks([]), plt.yticks([])

bin8n = edge8 < -Th
plt.subplot(235),plt.imshow(1-bin8n, cmap='gray')
plt.title('8 neighbor edge(-)'); plt.xticks([]), plt.yticks([])

bin8 = bin8p | bin8n
plt.subplot(236),plt.imshow(1-bin8, cmap='gray')
plt.title('8 neighbor edge'); plt.xticks([]), plt.yticks([])

#cv.imshow('laplace 4', 1- 1.0 * bin4); cv.imshow('laplace 8', 1- 1.0 * bin8); cv.waitKey()
plt.tight_layout() # 맨 나중에 써야 영향을 미침.
plt.waitforbuttonpress()        # 현 화면에서 버튼 입력을 기다린다.
#plt.show()
#exit(0)
#plt.close('all')        # PyCharm에서 수행을 중지하면 모든 창이 닫힘.
#"""

# ======================================================================================================================
# 실험 2: cv.Laplacian() 함수를 사용한 2차 미분 결과 관측
# ======================================================================================================================
print(f'Lab 2: input image dtype={img.dtype}')
lap = cv.Laplacian(img, -1, ksize=1)
plt.figure(num='Lab 2: Laplacian by OpenCV', figsize=(12, 8))
#maximizeWindow()
plt.suptitle("Lab 2: Laplacian by OpenCV", fontsize=12, fontweight='bold')

plt.subplot(221); plt.imshow(img_c)      # 3채널로 출력
plt.title('Original(color)'); plt.xticks([]), plt.yticks([])
plt.subplot(222); plt.imshow(lap, cmap='gray')
plt.title('Laplacian: k=1'); plt.xticks([]), plt.yticks([])


lap = cv.Laplacian(img, -1, ksize=5)
plt.subplot(223); plt.imshow(lap, cmap='gray')
plt.title('Laplacian: k=5'); plt.xticks([]), plt.yticks([])

lap = cv.Laplacian(img, -1, ksize=9)
plt.subplot(224); plt.imshow(lap, cmap='gray')
plt.title('Laplacian: k=9'); plt.xticks([]), plt.yticks([])
plt.tight_layout() # 맨 나중에 써야 영향을 미침.
plt.waitforbuttonpress()        # 현 화면에서 버튼 입력을 기다린다.


#plt.show()
