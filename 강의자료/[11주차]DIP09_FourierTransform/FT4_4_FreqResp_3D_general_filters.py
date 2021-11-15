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

# ------------------------------------------------------------------------
# 테스트할 필터들의 리스트를 지정한다.
# ------------------------------------------------------------------------
filters = [mean_filter, gaussian, laplacian, sobel_x, sharpening_mask, scharr]
filter_name = ['mean_filter', 'gaussian','laplacian', 'sobel_x', \
                'sharpening_mask', 'scharr_x']

filters = [mean_filter, gaussian, laplacian, sobel_x, sobel_y, scharr]
filter_name = ['mean_filter', 'gaussian','laplacian', 'sobel_x', \
                'sobel_y', 'scharr_x']


#===============================================================================================
# 필터의 주파수 응답을 3차원으로 보인다.
#       높이가 높으면 해당 영역의 주파수 통과 gain이 크다는 것을 의미한다.
#       이는 그 필터가 해당 영역의 주파수를 잘 통과시킨다는 것을 의미한다.
# 1 단계 : 필터 정의. 총 6종
# 2 단계 : 각 필터에 대해 0 패딩된 커널을 푸리에 변환하고 스펙트럼 리스트를 만든다.
# 3 단계 : 스펙트럼 리스트를 이용해 커널의 주파수 응답을 3D로 표현한다.
#===============================================================================================


# 단계 1 -----------------------------------------------------------------
# 블러링 커널들에 대한 주파스 응답을 살펴본다.
# ------------------------------------------------------------------------
# simple averaging filter without scaling parameter
N = 51     # 커널의 크기
mean_filter = np.ones((N, N))

# creating a gaussian filter
gaussian_kernel_lst = []
sigma_lst = [1, 3, 5, 9]
for sigma in sigma_lst:
    x = cv.getGaussianKernel(N, sigma)
    gaussian = x * x.T
    gaussian_kernel_lst.append(gaussian)


# 주어진 filters 필터(리스트 자료)에 대해 0 패딩된 커널 KxK 크기의 주파수 평면을 사용한 스펙트럼 리스트를 만든다.
def get_spectrum_list(filters, K):
    spectrum_list = []
    for i, item in enumerate(filters):
        # 1) 모든 원소가 0으로 선언된 크기가 확대된 빈 커널을 정의한다. => bigK(ernel)
        bigK = np.zeros((K, K), dtype=float)

        # 2) 필터의 계수를 0 padding 확대된 커널의 좌측 상단에 복사한다.
        r, c = item.shape  # print(r,c)
        bigK[0:r, 0:c] = item

        # 3) 커널을 푸리에 변환하고/ 결과를 대각 대칭이동시킨 후/ 이것을 log scaling한다.
        y = np.fft.fft2(bigK)
        fft_shift = np.fft.fftshift(y)
        spectrum = np.log10(np.abs(fft_shift) + 1)
        spectrum_list.append(spectrum)
    return spectrum_list




# 단계 2 -----------------------------------------------------------------
# 각 필터에 대해 0 패딩된 커널을 푸리에 변환하고 스펙트럼 리스트를 만든다.
# ------------------------------------------------------------------------


K = 31      # 주파수 응담 평면의 해상도. 홀수일 것. 크면 drawing 시간이 오래 걸림.
spectrum_list = get_spectrum_list(filters, K)

# 단계 3 -----------------------------------------------------------------
# 3D 주파수 응답을 도시한다.
# ------------------------------------------------------------------------
# 1) x, y축에 해당하는 데이터를 만든다. u, v에 해당한다.
# 실제 주파수와 매칭하는것은 시도하지 않는다.
K2 = int(K/2)
X = np.arange(-K2, K2+1, 1)
Y = np.arange(-K2, K2+1, 1)
XX, YY = np.meshgrid(X, Y)

# 2) 창을 열어 2x3 sub 창에 주파응답을 3차원으로 출력한다.
from mpl_toolkits.mplot3d import Axes3D

f_size = (14, 8)
fig = plt.figure(num='Frequency Response', figsize=f_size)
plt.suptitle("3D Frequency Response of general filters", fontsize=14, fontweight='bold')
#plt.subplots_adjust(left=0.0, right=1, top=0.9, bottom=0.05, hspace=0.4, wspace=0.0)
#plt.tight_layout()  # 가로, 세로 관점에서 빈틈을 줄여 그림의 크기를 키움


for i in range(6):
    ax = fig.add_subplot(231+i, projection='3d') # Axe3D object
    ax.set_title(filter_name[i])
    ax.plot_surface(XX, YY, spectrum_list[i], rstride=1, cstride=1, cmap='jet')
    plt.xlabel('u')
    plt.ylabel('v')

"""
# 2) 개별 창을 열어 그림을 출력한다.
from mpl_toolkits.mplot3d import Axes3D
for i in range(6):
    fig = plt.figure()
    ax = Axes3D(fig)        # subplot() 기능이 없는 듯 하다..
    ax.set_title(filter_name[i])
    ax.plot_surface(XX, YY, spectrum_list[i], rstride=1, cstride=1, cmap='jet')
"""





plt.show()
exit(0)

# 이름만 빌려 쓴다. 사실상 샤프닝 마스크
sharpening_mask = -laplacian.copy()
sharpening_mask[1, 1] += 1
print('sharpening_mask=\n', sharpening_mask)


filters = [mean_filter, gaussian, laplacian, sobel_x, sobel_y, scharr]
filter_name = ['mean_filter', 'gaussian', 'laplacian', 'sobel_x', 'sobel_y', 'scharr_x']

filters = [avg_y, gaussian, sharpening_mask, app_gauss1_y, app_gauss2_y, laplacian]
filter_name = ['avg_y', 'gaussian', 'sharpening_mask', 'app_gauss1_y', 'app_gauss2_y', 'laplacian']


filters = [mean_filter, gaussian, laplacian, sobel_x, sobel_y, sharpening_mask]
filter_name = ['mean_filter', 'gaussian', 'laplacian', 'sobel_x', 'sobel_y', 'sharpening_mask']
