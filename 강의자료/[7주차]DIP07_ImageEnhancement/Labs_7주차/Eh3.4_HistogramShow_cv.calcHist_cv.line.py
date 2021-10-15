"""
참고 사이트
Histograms - 2: Histogram Equalization
https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html


1. 프로그램의 기능 및 점검 포인트 - 영상의 히스토그램 분포를 다음 4가지 방법으로 보인다.
    1) numpy의 지원함수(np.histogram)를 이용하여 히스토그램 데이터를 얻는다. 그림은 plot, bar로 그린다.
       이 함수는 ndarray로 선언된 2차원 데이터를 입력 받아 그 영상의 히스토그램 데이터를 반환한다.
       이 방법이 제일 단순한 듯하다.

    2) Python 지원 함수(matplotlib.pyplot.hist)를 이용하여 영상의 히스토그램을 그린다.
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html?highlight=matplotlib%20pyplot%20hist#matplotlib.pyplot.hist
        히스토그램 분포 데이터도 구할 수 있고 영상의 분포를 그림으로 표현하는 기능은 있으나 그림을 안그리는 제어 기능은 없다.
        이 hist()함수는 1차원 flattened array를 입력으로 사용해야 한다.
            1) 이를 위해 ravel() 함수/메소드를 이용해 2차원 영상 데이터를 1차원 flattened array로 변환한다.
                   https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html
            2) 혹은 np.flatten() 함수를 이용해 2차원 영상 데이터를 1차원 flattened array로 변환한다.

    3) OpenCV 지원 함수(cv.calcHist)를 이용하여 히스토그램을 만들고
        matplotlib.pyplot.plot 함수로 영상의 히스토그램을 그린다.
        a. 히스토그램의 원리 참고 자료
            https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html
        b. OpenCV 히스토그램 함수, calcHist()의 사용법
        https://docs.opencv.org/3.4.3/d6/dc7/group__imgproc__hist.html#ga4b2b5fd75503ff9e6844cc4dcdaed35d

    4) OpenCV 지원 함수(cv.calcHist)를 이용하여 히스토그램을 만들고
        OpenCV함수(cv.line)로 히스토그램을 그리는 함수를 만들어 영상의 히스토그램을 그린다.
            OpenCV를 설치하면 다음 디렉터리에 예제 소스를 볼 수 있다.
            C:\opencv\sources\samples\python\hist.py

2. 프로그램의 목표
    1) 그래픽하게 표현된 히스토그램의 의미를 설명할 수 있다.
    2) matplotlib 혹은 OpenCV가 지원하는 기초적인 히스토그램 관련 함수의 운용법을 익힌다.


3. 미션

"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt



#=============================================================================================
# 단계 0
# 영상이 존재하는 폴더와 파일 이름을 지정하여 영상 파일 읽어 내기
# 다음 사례를 활용하여 영상 파일이 있는 폴더를 적절히 지정하시오.
#=============================================================================================
#Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
Path = '../data/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 Images 폴더.
#Path = 'D:/work/StudyImages/images/'
Name = 'lenna.tif'
#Name = 'bone(m).bmp'
#Name = 'Fig0303(a)(breast).tif'
#Name = 'Fig0306(a)(bone-scan-GE).tif'
#Name = 'CarHeadLightGlare.jpg'
#Name = 'over_exposure.jpg'          # 고조도
#Name = 'tiffany.bmp'                # 고조도
#Name = 'bukak.jpg'                 # 저조도
#Name = 'bk_gate.jpg'
#Name = 'bk_lobby.jpg'


FullName = Path + Name
img = cv.imread(FullName)       # cv2 함수는 BGR 칼라 데이터를 읽는다.
assert img is not None, 'No image file....!'

# OpenCV 함수를 이용해 BGR 평면 배열을 RGB 평면 배열로 바꾸고, GRAY 변환한다.
# matplotlib를 이용해 화면에 영상을 출력하려면 이 부분을 활성화 시킨다.
# cv.imshow()를 사용하면 이 부분은 주석처리해야 한다.

b, g, r = cv.split(img); img = cv.merge([r, g, b]);
imgC = img.copy()      # img의 복사본을 하나 저장해 둔다. RGB 순으로 재 정렬되어 있다.
imgG = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
# interactive mode로 설정한다. plt.show() 없이 plt.imshow()만으로 출력된다.
# plt.ion();

#=============================================================================================

def open_figure(win_name='', fig_size=(12, 8), sub_title='', font_size=10):
    fig = plt.figure(num=win_name, figsize=fig_size)
    if sub_title == '':
        sub_title = win_name
    plt.suptitle(sub_title, fontsize=font_size, fontweight='bold')
    #fig.patch.set_facecolor('silver')  # 화면 바탕 색상을 정한다.
    #fig.patch.set_alpha(0.3)

    # 원한다면 창을 최대화하여 그림을 출력한다.
    #figManager = plt.get_current_fig_manager()
    #figManager.window.showMaximized()
    return fig

#"""
#==============================================================================================
# 방법 3: OpenCV 지원 함수(cv.calcHist)를 이용하여 히스토그램을 만들고
#       matplotlib.pyplot.plot 함수로 영상의 히스토그램을 그린다.
#       https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html
#==============================================================================================

# ---------------------------------------------------------------------------------------------
# 1) OpenCV 함수를 이용해 BGR 평면 배열을 RGB 평면 배열로 바꾸고, GRAY 변환한다.
# matplotlib를 이용해 화면에 영상을 출력하려면 이 부분을 활성화 시킨다.
# cv.imshow()를 사용하면 이 부분은 주석처리해야 한다.
# ---------------------------------------------------------------------------------------------
#b, g, r = cv.split(img); img = cv.merge([r, g, b]);
#imgC = img.copy()      # img의 복사본을 하나 저장해 둔다.
#imgG = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
# interactive mode로 설정한다. plt.show() 없이 plt.imshow()만으로 출력된다.
# plt.ion();


# ---------------------------------------------------------------------------------------------
# 2) 원본 영상과 원본의 그레이 변환 영상을 출력한다.
# ---------------------------------------------------------------------------------------------
plt.figure(num='Method 3: Original Images & their histograms: cv.calcHist()', figsize=(12, 8))

plt.suptitle("Method 3: by cv.calcHist() & pyplot.plot()", fontsize=20, fontweight='bold')
# 창을 최대화 하여 그림을 출력한다.
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()



plt.subplot(221); plt.imshow(imgC)      # 입력받은 대로 출력
plt.title('Original(color)'); plt.axis('off')
plt.subplot(222); plt.imshow(imgG, cmap ='gray')
plt.title('Gray Image'); plt.axis('off')



# ---------------------------------------------------------------------------------------------
# 3) cv.calcHist() 함수로 gray 영상의 히스토그램을 구한 후, pyplot.plot으로 화면에 출력한다.
# ---------------------------------------------------------------------------------------------

plt.subplot(223); plt.title('Color Histogram')
color = ('b', 'g', 'r')

for i, col in enumerate(color):  # enumerate 함수는 순서가 있는 자료형(리스트, 튜플, 문자열)을 입력으로 받아 인덱스 값을 포함하는 enumerate 객체를 리턴한다.
    histr = cv.calcHist(
        [imgC], # Source arrays. 입력 영상들의 모음. CV에서는 어레이 영상의 주소로 표현된다.
        [i],    # List of the DIMS channels used to compute the histogram. 히스토그램을 구하고자 하는 채널의 번호.
        # 여기서는 0, 1, 2의 히스토그램을 3회의 looop를 통해 histr로 반환한다.
        # HSV좌표계의 영상에 대해 Hue, Saturation에 대한 각각의 히스토그램을 구한다면
        # [0 1]이라고 쓰고 1번 루프를 돌 때 H와 S의 총 2개의 히스토그램을 histr에 리스트 자료형으로 반환한다.
        # 이 경우 hist[0]에는 Hue의 히스토그램이, hist[1]에서는 S의 히스토그램이 담겨있을 것이다.
        # DIMS: Histogram dimensionality. B와 G의 히스토그램을 구한다면 DIMS=2. 파이썬에서는 DIMS는 활용 안한다.
        None,   # mask. 입력 영상과 같은 크기의 어레이 안의 0이 아닌 요소에 있는 화소에 대해서만 히스토그램을 구한다.
        # None은 마스크를 사용하지 않는다는 의미이다.
        [256],  # Array of histogram sizes in each dimension. 사실상의 bin의 개수라고 말할 수 있다.
        # 리스트 형식을 쓰는 이유는 여러 장의 영상에 따라 다르게 설정하기 위해서이다.
        # [256, 64]라고 한다면 첫 번째 영상은 256, 두 번째 영상은 64개의 크기로 나눈다는 것이다.
        [0, 256])    # Array of the dims arrays of the histogram bin boundaries in each dimension.
        # 화소의 범위라고 할 수 있다. 위에 보인 바와 같이 histogram sizes의 리스트가 [256, 64]하면
        # 이 값은 [[0, 255], [0, 63]]가 될 것이다.

    plt.plot(histr, color=col)
    plt.xlim([0, 256])
    print(i)

plt.subplot(224); plt.title('Gray Histogram')
histr = cv.calcHist([imgG],     # 입력 영상은 모노 영상 1개
                    [0],        # 채널 개념은 없지만 그래도 0번 채널만 히스토그램을 구하라 지정
                    None,       # 마스크는 지정하지 않는다.
                    [256],      # 히스토그램은 256단계의 bin으로 구분할 것이다.
                    [0, 256])    # 화소 값의 범위는 0~255이다.
plt.plot(histr)
plt.xlim([0, 256])

plt.waitforbuttonpress()        # interactive mode에서 현 화면에서 버튼 입력을 기다린다.
#plt.show()
exit(0)
#=============================================================================================
#"""



#==============================================================================================
# 방법 4: OpenCV 지원 함수(cv.calcHist)를 이용하여 히스토그램을 만들고
#         OpenCV함수(cv.line)로 히스토그램을 그리는 함수를 만들어 영상의 히스토그램을 그린다.
# OpenCV를 설치하면 다음 디렉터리에 예제 소스를 볼 수 있다.
#   C:\opencv\sources\samples\python\hist.py
#==============================================================================================

bins = np.arange(256).reshape(256, 1)  # 전역연수 bins

# ---------------------------------------------------------------------------------------------
# 히스토그램을 그리는 함수 1 - 칼라 영상의 히스토그램도 가능
# ---------------------------------------------------------------------------------------------
def hist_curve(im):
    #h = np.zeros((300, 256, 3))
    h = np.ones((300, 256, 3))
    if len(im.shape) == 2:              # mono gray이면 2차원이 될 것이다.
        color = [(255, 255, 255)]
    elif im.shape[2] == 3:              # 2채널이면 칼라이다.
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for ch, col in enumerate(color):
        hist_item = cv.calcHist([im], [ch], None, [256], [0, 256])
        cv.normalize(hist_item,hist_item, 0, 255, cv.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        pts = np.int32(np.column_stack((bins, hist)))
        cv.polylines(h, [pts], False, col)
    y = np.flipud(h)
    return y


# ---------------------------------------------------------------------------------------------
# 히스토그램을 그리는 함수 2 - only for grayscale images
# ---------------------------------------------------------------------------------------------
def hist_lines(im):
    h = np.zeros((300,256,3))
    if len(im.shape)!=2:
        print("hist_lines applicable only for grayscale images")
        #print("so converting image to grayscale for representation"
        im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
    hist_item = cv.calcHist([im],[0],None,[256],[0,256])
    cv.normalize(hist_item,hist_item,0,255,cv.NORM_MINMAX)
    hist=np.int32(np.around(hist_item))
    for x, y in enumerate(hist):
        x = int(x)
        y = int(y)
        cv.line(h, (x, 0), (x, y),(255,255,255))
    y = np.flipud(h)
    return y



# 원본 영상과 원본의 그레이 변환 영상 출력
cv.imshow("Original(Color)", img)
imgG = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Original(Gray)", imgG)
cv.waitKey(0)   # exit(0)

curve = hist_curve(img)
cv.imshow('histogram_curve', curve)
cv.waitKey(0);   #exit(0)

lines = hist_lines(imgG)
cv.imshow('histogram_lines', lines)
cv.waitKey(0);   exit(0)




