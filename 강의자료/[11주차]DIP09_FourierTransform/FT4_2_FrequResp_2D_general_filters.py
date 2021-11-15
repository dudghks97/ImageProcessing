"""
여러 종류의 공간 필터 커널이 주어졌을 대 이것의 주파수 응답을 그래피칼하게 표현해 본다.

Fourier Transform 기술자료- 이중에서 주파수 응답 부분(끝부분)만을 참조한다.
    url: https://docs.opencv.org/3.4.3/de/dbc/tutorial_py_fourier_transform.html


참조 소스: o_09_05_FourierResponse.m
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# ------------------------------------------------------------------------
# 여러 종류(6개)의 커널을 정의한다.
# ------------------------------------------------------------------------

# simple averaging filter without scaling parameter
N = 7     # 커널의 크기
mean_filter = np.ones((N, N))

# creating a gaussian filter
sigma = 1
x = cv.getGaussianKernel(N, sigma)
gaussian = x * x.T

# scharr in x-direction
scharr = np.array([[-3, 0, 3],
                   [-10, 0, 10],
                   [-3, 0, 3]])
# sobel in x direction
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
# sobel in y direction
sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

# y방향의 유사 가우시안 필터 1
app_gauss1_y = np.array([[0, 1, 0],
                    [0, 2, 0],
                    [0, 1, 0]])

# y방향의 유사 가우시안 필터 2
app_gauss2_y = np.array([[0, 3, 0],
                    [0, 10, 0],
                    [0, 3, 0]])

# y방향의 평균 필터
avg_y = np.array([[0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0]])


# laplacian
laplacian = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]])
# sharpening_mask
sharpening_mask = np.array([[0, 0, 0],
                      [0, 1, 0],
                      [0, 0, 0]]) - laplacian

"""
#===============================================================================================
# 실험 1: 필터의 주파수 응답을 2차 평면에 보인다. zero padding을 하지 않은 버전이다.
# 커널의 크기와 같은 크기의 주파수 응답 해상도를 보인다.
# 밝은 부분은 해당 영역의 주파수 통과 gain이 크다는 것을 의미한다.
# 어두운 부분은 그 필터가 해당 영역의 주파수를 잘 통과시킨다는 것을 의미한다.
# 실험결과: 해상도가 떨어저 주파수 응답을 제대로 확인할 수 없었다.
# 해상도를 높이기 위해 실험2의 z padding이 필요하다.
#===============================================================================================

# 단계 1 -----------------------------------------------------------------
# 테스트할 필터들의 리스트를 지정한다.
# ------------------------------------------------------------------------
filters = [mean_filter, gaussian, laplacian, sobel_x, sobel_y, scharr]
filter_name = ['mean_filter', 'gaussian','laplacian', 'sobel_x', \
                'sobel_y', 'scharr_x']

# 단계 2 -----------------------------------------------------------------
# 1) 각 필터를 푸리에 변환한 리스트 자료를 만든다.
# 2) 이를 바탕으로 대각 대칭 이동된 푸리에 변환 결과의 리스트 자료를 만든다.
# 3) log scaling된 스펙트럼 리스트를 만든다.
# ------------------------------------------------------------------------
fft_filters = [np.fft.fft2(x) for x in filters]
fft_shift = [np.fft.fftshift(y) for y in fft_filters]
mag_spectrum = [np.log10(np.abs(z)+1) for z in fft_shift]


# 단계 3 -----------------------------------------------------------------
# 결과를 그림으로 출력한다.
# ------------------------------------------------------------------------
for i in range(6):
    #plt.subplot(2, 3, i+1), plt.imshow(mag_spectrum[i], cmap='gray')
    plt.subplot(2, 3, i + 1), plt.imshow(mag_spectrum[i], cmap='jet')
    plt.title(filter_name[i])
    plt.axis('off')             # plt.xticks([]), plt.yticks([])
plt.show()
exit(0)
"""



#"""
#===============================================================================================
# 실험 2 - 커널의 주파수 응답. 역시 2차 평면에 도시한다. 단지 해상도를 높이기 위해 0 padding을 실시한다.
# 밝은 부분은 해당 영역의 주파수 통과 gain이 크다는 것을 의미한다.
# 어두운 부분은 그 필터가 해당 영역의 주파수를 잘 통과시킨다는 것을 의미한다.
#
# 충분한 해상도 확보를 위해 zero padding을 실시하여 큰 해상도의 주파수응답(푸리에 변환의 스펙트럼)을 구한다.
# 크기를 변수, bK = 255x255으로 확장하여 푸리에 변환을 행한다.
# 패딩 작업은 커널을 0으로 이루어진 어레이의 좌측 상단에 복사해 넣으면 됨.
# 사실 위치는 관계 없다. 위치를 바꿔도 주파수 응답은 변함이 없다.
#
# 미션 1- 간단 미션
#   x축에 대해 평균화 작업을 수행하는 7x7 커널의 주파수 응답을 2D 평면에 보이시오.
#
# 질문:
# 크기가 255x255으로 확장된 커널의 주파수 응답을 역변환하면 과연 공간 도메인에 사용할 수 있는 커널이 나올까?
# 이 질문의 답은 정답 5) 루틴...에 있다.
#===============================================================================================

# 단계 1 -----------------------------------------------------------------
# 여러 종류(6개)의 커널을 정의한다.
# ------------------------------------------------------------------------

# 단계 1 -----------------------------------------------------------------
# 테스트할 필터들의 리스트를 지정한다.
# ------------------------------------------------------------------------
filters = [mean_filter, gaussian, laplacian, sobel_x, sobel_y, scharr]
filter_name = ['mean_filter', 'gaussian','laplacian', 'sobel_x', \
                'sobel_y', 'scharr_x']


filters = [mean_filter, gaussian, laplacian, sobel_x, sharpening_mask, scharr]
filter_name = ['mean_filter', 'gaussian','laplacian', 'sobel_x', \
                'sharpening_mask', 'scharr_x']


# 단계 2 -----------------------------------------------------------------
# 각 필터에 대해 0 패딩된 커널을 푸리에 변환하고 스펙트럼을 출력한다.
# ------------------------------------------------------------------------
#fig = plt.figure(num='Frequency Response', figsize=(17, 8))
#plt.suptitle("Frequency Response of Filters", fontsize=14, fontweight='bold')

f_size = (14, 8)
fig = plt.figure(figsize=f_size)
plt.suptitle("Frequency Response of Filters", fontsize=14, fontweight='bold')
plt.subplots_adjust(left=0.0, right=1, top=0.9, bottom=0.05, hspace=0.12, wspace=0.0)
#plt.tight_layout()  # 가로, 세로 관점에서 빈틈을 줄여 그림의 크기를 키움


for i, item in enumerate(filters):

    # 1) 모든 원소가 0으로 선언된 크기가 확대된 빈 커널을 정의한다. => bigK(ernel)
    bK = 255
    bigK = np.zeros((bK, bK), dtype=float)
    r, c = item.shape

    # 0 padding을 실시한다. a, b 중의 하나를 선택하시오.
    # 2.a) 필터의 계수를 0 padding 확대된 커널의 좌측 상단에 복사한다.
    r_strt = 0; c_strt = 0
    # 2.b) 필터의 계수를 0 padding 확대된 커널의 임의 영역에 복사한다.
    #r_strt = 180; c_strt = 129

    bigK[r_strt:r + r_strt, c_strt:c + c_strt] = item

    # 3) 커널을 푸리에 변환하고/ 결과를 대각 대칭이동시킨 후 이것을 log scaling한다.
    y = np.fft.fft2(bigK)
    fft_shift = np.fft.fftshift(y)
    spectrum = np.log(np.abs(fft_shift)+1)
    
    # 4) 스펙트럼에 출력한다.
    #plt.subplot(2, 3, i+1), plt.imshow(spectrum, cmap='gray')
    plt.subplot(2, 3, i + 1), plt.imshow(spectrum, cmap='jet')
    plt.title(filter_name[i])
    plt.axis('off')             # plt.xticks([]), plt.yticks([])

    # 5) 역변환하여 원래의 커널의 모습이 나오는지 관찰해 보자.
    # 임베딩된 곳을 안다면 역변환 결과에 대해 해당 영역의 공간을 cropping 해 오면 될 것이다.
    ftr = np.real(np.fft.ifft2(y))
    # 영상처럼 +값만 있다면 abs() 함수를 써도 되지만, 음수가 나올 수 있는 커널이므로 실수부만 취한다.
    # 역변환한 결과의 허수부는 0으로 구성된다.
    print(f"\n{filter_name[i]} ---------- ")
    for row in range(r):
        for col in range(c):
            print(f'{ftr[row+r_strt, col+c_strt]:#8.4f}', end=' ')
        print()

plt.show()
exit(0)
#"""



