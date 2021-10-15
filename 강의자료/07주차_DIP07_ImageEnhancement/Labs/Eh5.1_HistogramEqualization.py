"""
참고 사이트
Histograms - 2: Histogram Equalization
https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html


1. 프로그램의 기능 - 영상의 히스토그램 평활화를 다음 2가지 방법으로 보인다.

    방법 1) OpenCV의 cv.equalizeHist() 함수를 이용하여 주어진 영상에 대한 평활화된 영상을 반환받는다.
        주의: 이 함수는 gray 영상에 대해서만 적용 가능하다.


    방법 2) Python 지원 함수를 이용하여 영상의 히스토그램을 그린다.
       numpy에서 제공하는 histogram() 함수로 히스토그램(DF: Distribution Function) 데이터를 반환받아서
       히스토그램과 누적분포함수를 구하고 그린다.
       누적분포함수를 정규화하여 mapping 함수, LUT를 생성한다. 이 LUT를 이용하여 intensity transform을 행할 것이다.
       LUT를 이용하여 명암 변환을 행하여 평활화한 영상을 보인다.

2. 프로그램의 목표
    1) 파이썬 기반으로 HE 알고리즘의 흐름을 이해한다.
    2) CV에서 제공하는 함수, cv.equalizeHist()를 활용할 수 있다.

3. 질문
    1) Color 영상에 대하여 HE를 행할 수 있는 아이디어는 무엇인가?
    2) cv.equalizeHist() 함수로 Color 영상에 HE를 행하는 방법은?


4. 미션
    칼라 영상에 대하여 HE를 행하시오.

4.

"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt



#=============================================================================================
# 단계 0
# 영상이 존재하는 폴더와 파일 이름을 지정하여 영상 파일 읽어 내기
# 다음 사례를 활용하여 영상 파일이 있는 폴더를 적절히 지정하시오.
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
img = cv.imread(FullName)
assert img is not None, "Failed to load image file:"

def open_figure(win_name='', fig_size=(17, 10), sub_title='', font_size=15):
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



# ============================================================================================================
# 방법 1 : OpenCV 함수, cv2.equalizeHist()를 이용하여 히스토그램 평활화를 행한다.
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html?highlight=pyplot%20hist#matplotlib.pyplot.hist
# ============================================================================================================

b, g, r = cv.split(img); img = cv.merge([r, g, b]);
imgC = img.copy()      # img의 복사본을 하나 저장해 둔다.
img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# 원본과 히스토그램 평활화 영상을 하나로 만들어 보이기. -----------------------------------------------------------------
# plt 장을 연다.
open_figure(win_name='Histogram Equalization Method 1: cv.equalizeHist()')

# OpenCV의 cv.equalizeHist() 함수를 이용하여 주어진 영상에 대한 평활화된 영상을 반환받는다.
equ = cv.equalizeHist(img_gray)
res = np.hstack((img_gray, equ)) #stacking images side-by-side
#plt.imshow(res, cmap ='gray')
plt.gray()
plt.subplot(211)
plt.imshow(res) # pseudo 의사
plt.title('Original Image and Equalized Image')
plt.axis('off')     #같은 동작 => plt.xticks([]), plt.yticks([])

# 원본 영상과 히스토그램 평활화 영상의 히스토그램을 그려보자
hist_org, bin_edges_org = np.histogram(img_gray, 256, [0, 255])     # 원본 영상의 히스토그램
hist_eq, bin_edges_eq = np.histogram(equ, 256, [0, 255])     # 원본 영상의 히스토그램
plt.subplot(212)
plt.plot(bin_edges_org[1:], hist_org, label="original", color='r')
plt.plot(bin_edges_org[1:], hist_eq, label="equalized", color='b')
#plt.xticks([0, 255])
plt.xticks(np.linspace(0, 255, 10))
plt.xlabel('pixel value')
plt.autoscale(enable=True,  # enable : True (default) turns autoscaling on,
              axis='x',  # axis : {'both', 'x', 'y'}, optional which axis to operate on
              tight=True)  # tight : bool or None, optional. If True, set view limits to data limits;
plt.legend(fontsize=15)
plt.grid(color='c', linestyle='--', linewidth=0.5)  # cyan색의 grid

# sub 화면의 간격을 조정하려면 일단 출력하고 나서 진행해야 함.
# 아래 조정 작업 중에서 나중에 쓴 것이 설정을 overwrite함.
plt.tight_layout()  # 가로, 세로 관점에서 빈틈을 줄여 그림의 크기를 키움
#plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # 상하좌우 여백의 크기를 지정

plt.waitforbuttonpress()        # interactive mode에서 현 화면에서 버튼 입력을 기다린다.
#plt.show()



# ============================================================================================================
# 방법 2 :직접 HE 처리 알고리즘을 coding 하여 처리한다.
#   1) np.histogram()를 이용해 히스토그램(DF: Distribution Function)을 구한다.
#   2) 연산을 통해 CDF(Cumulative DF)를 구하고,
#   3) 총 화소수로 나누어 NCDF(Normalized CDF)를 구하고,
#   4) 여기에 255를 구하여 mapping LUT 어레이를 구한다.
#   5) Python 함수, matplotlib.pyplot.hist를 이용하여 영상의 히스토그램을 그린다.
#       https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html?highlight=pyplot%20hist#matplotlib.pyplot.hist
# ============================================================================================================

b, g, r = cv.split(img); img = cv.merge([r, g, b])
imgC = img.copy()      # img의 복사본을 하나 저장해 둔다.
imgG = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# 단계 1: (221) 원본 보이기. 원본의 그레이 변환 영상 출력 ---------------------------------------------------------------------
open_figure(win_name='Histogram Equalization Method 2: Algorithm Coding')

plt.subplot(221); plt.imshow(imgG, cmap ='gray')
plt.title('Gray Image')
plt.axis('off')     #같은 동작 => plt.xticks([]), plt.yticks([])



# 단계 2: (222) 원본의 DF(히스토그램)과 CDF(Cumulative Distribution Function) 그리기 ---------------------------------------------------------------------
# numpy에서 제공하는 histogram() 함수로 히스토그램(DF: Distribution Function) 데이터를 반환받아서
# pyplot.hist()를 이용해서 히스토그램 그림을 그린다.
#hist, bins = np.histogram(imgG.flatten(), 256, [0, 255])    # 입력 영상을 1차원으로 펼 필요 없음.
hist, bins = np.histogram(imgG, 256, [0, 255])    # 히스토그램 데이터를 hist에 반환. bins는 용처가 없다.
print("len(hist)=", len(hist))      # print("hist=", hist)
print("len(bins)=", len(bins))      # Return the bin edges (length(hist)+1). bins는 바구니의 이름 리스트[0, 1,...,255]로 보고 싶은데 실제로는 257개로 이루어져 있다???
#print("bins.astype('uint8')=", bins.astype('uint8'))
#print("bins=", bins)

cdf = hist.cumsum()             # 누적분포함수, Cumulative Distribution Function
cdf_normalized = cdf * hist.max()/ cdf.max()    # 같은 화면에 그리려고 최대 값이 hist.max와 같게 잡았다.

plt.subplot(222)
plt.title('Histogram of Original Image')
plt.xticks([0, 255])
plt.grid(color='b', linestyle='--', linewidth=0.5)
#plt.hist(imgG.ravel(),256,[0,256], histtype='step');   # pyplot.hist()를 이용한 히스토그램 그리기

plt.plot(cdf_normalized, color='b')           # NCDF 그림으로 그리기
plt.hist(imgG.flatten(), 255, color='c')    # 히스토그램 그림으로 그리기
plt.xlim([0, 255])
plt.legend(('cdf','histogram'), loc = 'upper left', fontsize=15)

# 단계 3: (223) 히스토그램 평활화 수행하여 결과 보이기 ---------------------------------------------------------------
cdf = cdf * 255 / cdf[255]
LUT = cdf.astype('uint8')       # Look Up Table - 256바이트 크기. 화소 값 변환 테이블.
#print(LUT)


img2 = LUT[imgG]

plt.subplot(223);
plt.title('Histogram Equalized Image')
plt.imshow(img2, cmap ='gray')
plt.axis('off')     #같은 동작 => plt.xticks([]), plt.yticks([])


# 단계 4: (224) 평활화된 영상의 히스토그램 관찰하기 ---------------------------------------------------------------------

plt.subplot(224);
plt.title('Histogram of Equalized Image')

plt.xticks([0, 255])
plt.grid(color='b', linestyle='--', linewidth=0.5)

# img2.ravel()와 img2.flatten()는 같은 효과를 보임.
# step으로 관찰하면 불연속 화소 분포가 더 잘 보인다.
plt.hist(img2.flatten(), 256, [0, 256], color='c')     # 입력 영상을 1차원으로 꼭 펴야함.
# 연결선으로 관찰하면 불연속 화소 분포가 덜 눈에 띈다.
#plt.hist(img2.flatten(), 256, [0,256], histtype='step', color = 'c')

plt.xlim([0, 255])

# sub 화면의 간격을 조정하려면 일단 출력하고 나서 진행해야 함.
# 아래 조정 작업 중에서 나중에 쓴 것이 설정을 overwrite함.
plt.tight_layout()  # 가로, 세로 관점에서 빈틈을 줄여 그림의 크기를 키움
#plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # 상하좌우 여백의 크기를 지정

plt.waitforbuttonpress()        # interactive mode에서 현 화면에서 버튼 입력을 기다린다.

#plt.show(); exit(0)



