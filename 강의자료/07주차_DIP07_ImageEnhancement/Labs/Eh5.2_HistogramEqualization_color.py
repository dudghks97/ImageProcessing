"""
1. 개요
    히스토그램 평활화를 칼라 영상에 대해 처리하는 방법의 한 사례를 보인다.
    직접 HE 처리 알고리즘을 coding하여(방법 2) 히스토그램 평활화를 칼라 영상에 대해 처리하는 방법의 한 사례를 보인다

2. 처리 절차
    1) 칼라 영상을 흑백 계조 영상으로 바꾼다.
    2) 계조 영상에 대해 numpy에서 제공하는 histogram() 함수로 히스토그램(DF: Distribution Function)
        데이터를 반환받아서 히스토그램과 누적분포함수(CDF: Cumulative Distribution Function)를 구하고 그린다.
    3) 누적분포함수를 정규화하여 정규누적분포함수(NCDF: Normalized Cumulative Distribution Function), 즉 mapping 함수, LUT를 생성한다.
    4) 이 LUT를 이용하여 RGB 각 채널에 대해 intensity transform을 행한다.

3. 참고
    OpenCV의 cv.equalizeHist() 함수를 이용하여 주어진 영상에 대한 평활화된 영상을 반환받는다.
        주의: 이 함수는 gray 영상에 대해서만 적용 가능하다.
    이 함수는 영상 자체에 대해 변환된 결과를 받기 때문에 적용이 불가하다.

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
#Name = 'bone(m).bmp'
#Name = 'Fig0303(a)(breast).tif'
#Name = 'Fig0306(a)(bone-scan-GE).tif'
#Name = 'CarHeadLightGlare.jpg'
#Name = 'over_exposure.jpg'          # 고조도
#Name = 'tiffany.bmp'                # 고조도
#Name = 'bk_lobby.jpg'
#Name = 'lenna.tif'
#Name = 'low_contrast_challenge_mission.jpg'
Name = 'ele_book.jpg'
Name = 'bukak.jpg'                 # 저조도
#Name = 'foggy.jpg'

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
# 직접 HE 처리 알고리즘을 coding 하여 히스토그램 평활화를 칼라 영상에 대해 처리하는 방법의 한 사례를 보인다
#   1) np.histogram()를 이용해 히스토그램(DF: Distribution Function)을 구한다.
#   2) 연산을 통해 CDF(Cumulative DF)를 구하고,
#   3) 총 화소수로 나누어 NCDF(Normalized CDF)를 구하고,
#   4) 여기에 255를 구하여 mapping LUT 테이블를 구한다.
#   5) Python 함수, matplotlib.pyplot.hist를 이용하여 영상의 히스토그램을 그린다.
#       https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html?highlight=pyplot%20hist#matplotlib.pyplot.hist
# ============================================================================================================

b, g, r = cv.split(img); img = cv.merge([r, g, b])
imgC = img.copy()      # img의 복사본을 하나 저장해 둔다.
imgG = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

open_figure(win_name='Histogram Equalization of Color Image')       # plt 장을 연다.

# 단계 1: (221) 원본 보이기. 원본의 그레이 변환 영상 출력 ---------------------------------------------------------------------
plt.subplot(221); plt.imshow(imgC)
plt.title('Color Image', fontsize=15)
plt.axis('off')     #같은 동작 => plt.xticks([]), plt.yticks([])


# 단계 2: (222) 원본의 DF(히스토그램)과 CDF(Cumulative Distribution Function) 그리기 ---------------------------------------------------------------------
plt.subplot(222)
plt.title('Original Image Histogram(DF) & NCDF(used for mapping function)', fontsize=15)
# numpy에서 제공하는 histogram() 함수로 히스토그램(DF: Distribution Function) 데이터를 반환받아서
# pyplot.hist()를 이용해서 히스토그램 그림을 그린다.

# 2.1) 히스토그램(DF: Distribution Function)을 구한다. 최댓값이 1인 아래의 NCDF와 같은 평면에 그리기 위해 히스토그램을 정규화하여 그린다.
hist, bins = np.histogram(imgG, 256, [0, 255])    # 히스토그램 데이터를 hist에 반환. bins는 용처가 없다.
#hist, bins = np.histogram(imgG.flatten(), 256, [0, 255])    # 입력 영상을 1차원으로 펼 필요 없음.
print("len(hist)=", len(hist))      # print("hist=", hist)
print("len(bins)=", len(bins))      # Return the bin edges (length(hist)+1). bins는 바구니의 이름 리스트[0, 1,...,255]로 보고 싶은데 실제로는 257개로 이루어져 있다???
#print("bins.astype('uint8')=", bins.astype('uint8'))
#print("bins=", bins)

# 2.2) 0~1의 범위로 정규화된 히스토그램을 그린다.
#plt.hist(imgG.ravel(), 256, [0, 256], histtype='step', color='r', label='DF')  # pyplot.hist()를 이용한 히스토그램 그리기
#plt.hist(imgG.flatten(), 255, color = 'r', label='DF')    # 히스토그램 그림으로 그리기
plt.plot(hist/hist.max(), color='r', label='DF')        # 정규화된 히스토그램
#plt.plot(hist, color='r', label='DF')

# 2.3) NCDF(Normalized Cumulative Distribution Function), 정규화 누적분포함수 그리기.
cdf = hist.cumsum()             # 누적분포함수, Cumulative Distribution Function
cdf_normalized = cdf / cdf.max()    # 0~1 범위의  정규화된 누적분포함수, NCDF
#cdf_normalized = hist.max() * cdf / cdf.max()    # 다르게 출력할 용도록 NCDF의 최댓값이 hist.max이 되도록 스케일링하였다.
plt.plot(cdf_normalized, color='b', label='CDF')           # NCDF 그림으로 그리기

# 2.4) 레전드에 label로 배정된 이름 말고 다시 정해줄 수도 있다. 그린 순서대로...
#plt.legend(('ncdf', 'histogram'), loc='upper left', fontsize=15)
plt.legend(('NDF: Normalized Distribution Function', 'NCDF: Normalized Cumulative Distribution Function'),
           loc='center left', fontsize=15)
#plt.legend(loc='upper left')
#plt.legend()

# 2.5) x 축의 범위를 정하고(tick 값도 적는다) 그리드를 그린다.
#plt.xticks([0, 255])       # x축 그리드를 안그린다.
plt.xlim([0, 255])          # 그리드 별 tick 값을 출력한다.
plt.grid(color='b', linestyle='--', linewidth=0.5)


# 단계 3: (223) 히스토그램 평활화 수행하여 결과 보이기 ---------------------------------------------------------------
plt.subplot(223)
plt.title('Histogram Equalized Image', fontsize=15)

mapping = cdf * 255 / cdf[255]
LUT = mapping.astype('uint8')       # Look Up Table - 256바이트 크기. 화소 값 변환 테이블.
#print(LUT)

imgCeq = LUT[imgC]        # 3채널 칼라 영상에 대한 LUT 기반 히스토그램 평활화
imgEG = cv.cvtColor(imgCeq, cv.COLOR_RGB2GRAY)     # 히스토그램 평활화된 영상의 그레이 버전. Equalized Gray

plt.imshow(imgCeq)
plt.axis('off')     #같은 동작 => plt.xticks([]), plt.yticks([])


# 단계 4: (224) 평활화된 영상의 히스토그램 관찰하기 ---------------------------------------------------------------------

plt.subplot(224)
plt.title('Histogram of Histogram Equalized Image', fontsize=15)

#plt.xticks([0, 255])       # x축 그리드를 안그린다.
plt.xlim([0, 255])          # 그리드 별 tick 값을 출력한다.
plt.grid(color='b', linestyle='--', linewidth=0.5)

# 비교를 위해 원본 영상의 원본의 분포함수(df, 히스토그램)을 함께 보인다.
plt.hist(imgG.flatten(), bins=256, range=(0, 255), histtype='step', color='r', label='Distribution Function(DF) of Input Image')


# 히스토그램 평활화된 영상의 정규화된 분포함수를 출력한다. bar 혹은 step 중의 하나를 선택하시오.
# img2.ravel()와 img2.flatten()는 같은 효과를 보임.
# 연결선으로 관찰하면 불연속 화소 분포가 덜 눈에 띈다. bar가 default
#plt.hist(imgEG.flatten() / (imgEG.flatten()).max(), 256, [0, 255], color='g', histtype='bar', label='Normalized Distribution Function(NDF) of Equalized Image')     # 입력 영상을 1차원으로 꼭 펴야함.
# step으로 관찰하면 불연속 화소 분포가 더 잘 보인다.
plt.hist(imgEG.flatten(), bins=256, range=(0, 255), histtype='step', color='g', label='Distribution Function(DF) of Equalized Image')
#plt.legend(loc='upper left', fontsize=15)
plt.legend(fontsize=15)

# sub 화면의 간격을 조정하려면 일단 출력하고 나서 진행해야 함.
# 아래 조정 작업 중에서 나중에 쓴 것이 설정을 overwrite함.
plt.tight_layout()  # 가로, 세로 관점에서 빈틈을 줄여 그림의 크기를 키움
#plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # 상하좌우 여백의 크기를 지정

plt.waitforbuttonpress()        # interactive mode에서 현 화면에서 버튼 입력을 기다린다.
#plt.show(); exit(0)



