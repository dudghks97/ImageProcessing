"""
Gaussian과 LoG 커널을 정의하고 이것의 주파수 응답을 구해본다.
이것의 주파수 응답을 그래피칼하게 표현해 본다.
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


#"""
# ==========================================================================================================
# 실습 1: 가우시안 블러 커널과 2차원 주파수 응답
# ==========================================================================================================
sigma_lst = [1, 3, 5]       # 편의상 3개만 설정하기로 한다.
ksize = max(sigma_lst) * 6 + 1
f_size = (11, 8)
fig = plt.figure(figsize=f_size)
plt.suptitle("Gaussian Blur Kernels and their Frequency Responses in 2D", fontsize=14, fontweight='bold')
plt.subplots_adjust(left=0.0, right=1, top=0.9, bottom=0.05, hspace=0.12, wspace=0.0)
#plt.tight_layout()  # 가로, 세로 관점에서 빈틈을 줄여 그림의 크기를 키움

for i, sigma in enumerate(sigma_lst):
    # 0) 2차원 가우시안 필터를 설계하고 이를 그림으로 출력한다.
    x = cv.getGaussianKernel(ksize, sigma)
    kernel = x * x.T
    plt.subplot(2, 3, i + 1), plt.imshow(kernel, cmap='jet')
    plt.title(f'Kernel({ksize}x{ksize}):' + r'$\sigma$' + f'={sigma_lst[i]}')
    plt.axis('off')

    # 이하 해당 커널의 주파수 응답을 2차원으로 표현하기 위한 루틴이다.
    # 1) 모든 원소가 0으로 선언된 크기가 확대된 빈 커널을 정의한다. => bigK(ernel)
    bK = 255
    bigK = np.zeros((bK, bK), dtype=float)
    r, c = kernel.shape

    # 0 padding을 실시한다. a, b 중의 하나를 선택하시오.
    # 2.a) 필터의 계수를 0 padding 확대된 커널의 좌측 상단에 복사한다.
    r_strt = 0;
    c_strt = 0
    # 2.b) 필터의 계수를 0 padding 확대된 커널의 임의 영역에 복사한다.
    # r_strt = 180; c_strt = 129

    bigK[r_strt:r + r_strt, c_strt:c + c_strt] = kernel

    # 3) 커널을 푸리에 변환하고/ 결과를 대각 대칭이동시킨 후 이것을 log scaling한다.
    y = np.fft.fft2(bigK)
    fft_shift = np.fft.fftshift(y)
    spectrum = np.log10(np.abs(fft_shift) + 1)

    # 4) 스펙트럼에 출력한다.
    # plt.subplot(2, 3, i+1), plt.imshow(spectrum, cmap='gray')
    plt.subplot(2, 3, i + 4), plt.imshow(spectrum, cmap='jet')
    plt.title(f'Freq. Respone:' + r'$\sigma$' + f'={sigma_lst[i]}')
    plt.axis('off')
#plt.show()
#exit(0)
#"""




# ==========================================================================================================
# 실습 2: LoG 필터 커널과 그 커널들의 주파수 응답
# ==========================================================================================================
def gaussian2ndDev(x, y, sigma):
    r2 = x**2 + y**2
    return( (r2-2*sigma**2)/sigma**4 * np.exp(-r2/(2*sigma**2)))

sigma_lst = [1, 3, 5]       # 편의상 3개만 설정하기로 한다.
ksize = max(sigma_lst) * 6 + 1
N = int(ksize / 2);
#print('N=', N)

fig = plt.figure(figsize=f_size)
plt.subplots_adjust(left=0.0, right=1, top=0.9, bottom=0.05, hspace=0.12, wspace=0.0)
plt.suptitle("LoG Kernels and their Frequency Responses in 2D", fontsize=14, fontweight='bold')

for i, sigma in enumerate(sigma_lst):
    # 0) 2차원 가우시안 필터를 설계하고 이를 그림으로 출력한다.
    #x = cv.getGaussianKernel(ksize, sigma); #kernel = x * x.T

    # 0) 2차원 LoG 커널을 설계하고 이를 그림으로 출력한다.
    X = np.arange(-N, N + 1, 1);
    Y = np.arange(-N, N + 1, 1)
    # print('length of X=', len(X)); print('length of Y=', len(Y))

    # x, y의 범위의 데이터를 2차원으로 확장한다.
    XX, YY = np.meshgrid(X, Y)
    kernel = gaussian2ndDev(XX, YY, sigma)

    plt.gray()
    #plt.subplot(2, 3, i + 1), plt.imshow(kernel, cmap='jet')
    plt.subplot(2, 3, i + 1), plt.imshow(kernel)
    plt.title(f'Kernel({ksize}x{ksize}):' + r'$\sigma$' + f'={sigma_lst[i]}')
    plt.axis('off')

    # 이하 해당 커널의 주파수 응답을 2차원으로 표현하기 위한 루틴이다.
    # 1) 모든 원소가 0으로 선언된 크기가 확대된 빈 커널을 정의한다. => bigK(ernel)
    bK = 255
    bigK = np.zeros((bK, bK), dtype=float)
    r, c = kernel.shape

    # 0 padding을 실시한다. a, b 중의 하나를 선택하시오.
    # 2.a) 필터의 계수를 0 padding 확대된 커널의 좌측 상단에 복사한다.
    r_strt = 0;
    c_strt = 0
    # 2.b) 필터의 계수를 0 padding 확대된 커널의 임의 영역에 복사한다.
    # r_strt = 180; c_strt = 129

    bigK[r_strt:r + r_strt, c_strt:c + c_strt] = kernel

    # 3) 커널을 푸리에 변환하고/ 결과를 대각 대칭이동시킨 후 이것을 log scaling한다.
    y = np.fft.fft2(bigK)
    fft_shift = np.fft.fftshift(y)
    spectrum = np.log10(np.abs(fft_shift) + 1)

    # 4) 스펙트럼에 출력한다.
    # plt.subplot(2, 3, i+1), plt.imshow(spectrum, cmap='gray')
    plt.subplot(2, 3, i + 4), plt.imshow(spectrum)
    plt.title(f'Freq. Response:' + r'$\sigma$' + f'={sigma_lst[i]}')
    plt.axis('off')
plt.show()
exit(0)
