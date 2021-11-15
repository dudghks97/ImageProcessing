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
from mpl_toolkits.mplot3d import Axes3D

sigma_lst = [1, 3, 5]       # 편의상 3개만 설정하기로 한다.
ksize = max(sigma_lst) * 6 + 1
N = int(ksize / 2)

bK = ksize        # 스펙트럼을 표현하기 위한 어레이, bigKernel의 크기
#bK = 129       # 실제로는 더 늘릴 필요 없다.
NN = int(bK / 2)
# x, y의 범위의 데이터를 2차원으로 확장한다.
Xe = np.arange(-NN, NN + 1, 1)
Ye = np.arange(-NN, NN + 1, 1)
XXe, YYe = np.meshgrid(Xe, Ye)
print(XXe.shape, YYe.shape)     # XX, YY extended

f_size = (14, 8)
fig = plt.figure(figsize=f_size)
plt.suptitle("Gaussian Blur Kernels and their Frequency Responses in 3D", fontsize=14, fontweight='bold')
plt.subplots_adjust(left=0.0, right=1, top=0.9, bottom=0.05, hspace=0.12, wspace=0.0)
#plt.tight_layout()  # 가로, 세로 관점에서 빈틈을 줄여 그림의 크기를 키움


for i, sigma in enumerate(sigma_lst):
    # 0) 2차원 가우시안 필터를 설계하고 이를 그림으로 출력한다.
    x = cv.getGaussianKernel(ksize, sigma)
    kernel = x * x.T

    X = np.arange(-N, N + 1, 1)
    Y = np.arange(-N, N + 1, 1)
    # print('length of X=', len(X)); print('length of Y=', len(Y))

    # x, y의 범위의 데이터를 2차원으로 확장한다.
    XX, YY = np.meshgrid(X, Y)

    ax = fig.add_subplot(231+i, projection='3d') # Axe3D object
    ax.set_title(f'Kernel({ksize}x{ksize}): ' + r'$\sigma$' + f'={sigma_lst[i]}')
    ax.plot_surface(XX, YY, kernel, rstride=1, cstride=1, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('y')

    # 이하 해당 커널의 주파수 응답을 2차원으로 표현하기 위한 루틴이다.
    # 1) 모든 원소가 0으로 선언된 크기가 확대된 빈 커널을 정의한다. => bigK(ernel)
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
    print('spectrum.shape=', spectrum.shape)

    # 4) 스펙트럼을 출력한다.
    ax = fig.add_subplot(234+i, projection='3d') # Axe3D object
    ax.set_title(f'Freq. Response: ' + r'$\sigma$' + f'={sigma_lst[i]}')
    ax.plot_surface(XXe, YYe, spectrum, rstride=1, cstride=1, cmap='jet')
    plt.xlabel('u')
    plt.ylabel('v')

#plt.show()
#exit(0)
#"""




# ==========================================================================================================
# 실습 2: LoG 필터 커널과 그 커널들의 주파수 응답
# ==========================================================================================================
from mpl_toolkits.mplot3d import Axes3D

def gaussian2ndDev(x, y, sigma):
    r2 = x**2 + y**2
    return( (r2-2*sigma**2)/sigma**4 * np.exp(-r2/(2*sigma**2)))

sigma_lst = [1, 3, 5]       # 편의상 3개만 설정하기로 한다.
ksize = max(sigma_lst) * 6 + 1
N = int(ksize / 2)

bK = ksize        # 스펙트럼을 표현하기 위한 어레이, bigKernel의 크기
#bK = 129       # 실제로는 더 늘릴 필요 없다.
NN = int(bK / 2)
# x, y의 범위의 데이터를 2차원으로 확장한다.
Xe = np.arange(-NN, NN + 1, 1)
Ye = np.arange(-NN, NN + 1, 1)
XXe, YYe = np.meshgrid(Xe, Ye)
print(XXe.shape, YYe.shape)     # XX, YY extended

fig = plt.figure(figsize=f_size)
plt.suptitle("LoG Kernels and their Frequency Responses in 2D", fontsize=14, fontweight='bold')
plt.subplots_adjust(left=0.0, right=1, top=0.9, bottom=0.05, hspace=0.12, wspace=0.0)
#plt.tight_layout()  # 가로, 세로 관점에서 빈틈을 줄여 그림의 크기를 키움


for i, sigma in enumerate(sigma_lst):
    # 0) 2차원 가우시안 필터를 설계하고 이를 그림으로 출력한다.
    #x = cv.getGaussianKernel(ksize, sigma); #kernel = x * x.T

    # 0) 2차원 LoG 커널을 설계하고 이를 그림으로 출력한다.
    X = np.arange(-N, N + 1, 1)
    Y = np.arange(-N, N + 1, 1)
    # print('length of X=', len(X)); print('length of Y=', len(Y))

    # x, y의 범위의 데이터를 2차원으로 확장한다.
    XX, YY = np.meshgrid(X, Y)
    kernel = gaussian2ndDev(XX, YY, sigma)

    # 3차원으로 도시한다.
    ax = fig.add_subplot(231+i, projection='3d') # Axe3D object
    ax.set_title(f'Kernel({ksize}x{ksize}):' + r'$\sigma$' + f'={sigma_lst[i]}')
    ax.plot_surface(XX, YY, kernel, rstride=1, cstride=1, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('y')

    # 이하 해당 커널의 주파수 응답을 2차원으로 표현하기 위한 루틴이다.
    # 1) 모든 원소가 0으로 선언된 크기가 확대된 빈 커널을 정의한다. => bigK(ernel)
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


    # 4) 스펙트럼을 3차원으로 도시한다.
    ax = fig.add_subplot(234+i, projection='3d') # Axe3D object
    ax.set_title(f'Freq. Response:' + r'$\sigma$' + f'={sigma_lst[i]}')
    ax.plot_surface(XXe, YYe, spectrum, rstride=1, cstride=1, cmap='jet')
    plt.xlabel('u')
    plt.ylabel('v')

plt.show()
exit(0)




# 단계 1 -----------------------------------------------------------------
# LoG 커널을 정의한다.
# ------------------------------------------------------------------------
# simple averaging filter without scaling parameter
sigma=3
NN = 21     # 커널의 크기
N = int(NN/2); print('N=', N)
X = np.arange(-N, N+1, 1); Y = np.arange(-N, N+1, 1)
#print('length of X=', len(X)); print('length of Y=', len(Y))

# x, y의 범위의 데이터를 2차원으로 확장한다.
XX, YY = np.meshgrid(X, Y)

# 2차원 가우시안의 2차 미분 함수를 정의하고 위의 2차원 좌표 데이터를 대입하여 값을 생성한다.
# ZZ가 커널의 크기가 된다. 가우시안 블러 함수와는 달리 정규화하지 않는다.
def gaussian2ndDev(x,y, sigma):
    r2 = x**2 + y**2
    return( (r2-2*sigma**2)/sigma**4 * np.exp(-r2/(2*sigma**2)))

#def gaussian2ndDev(x,y, sigma):
#    r2 = x**2 + y**2
#    return( ( (r2-2*sigma**2) * np.exp(-r2/(2*sigma**2)) ) / sigma**4)

# 필터 계수를 연산하고 공간 차원에서 필터링을 행한다.
ZZ = gaussian2ndDev(XX, YY, sigma)

# 단계 2 -----------------------------------------------------------------
# 각 필터에 대해 0 패딩된 커널을 푸리에 변환하고 스펙트럼을 출력한다.
# ------------------------------------------------------------------------
i = 0


# 1) 모든 원소가 0으로 선언된 크기가 확대된 빈 커널을 정의한다. => bigK(ernel)
bK = 512

# 0 padding을 실시한다. a, b 중의 하나를 선택하시오.
# 2.a) 필터의 계수를 0 padding 확대된 커널의 좌측 상단에 복사한다.
r_strt = 0; c_strt = 0
# 2.b) 필터의 계수를 0 padding 확대된 커널의 임의 영역에 복사한다.
#r_strt = 180; c_strt = 129

start_lst = [[0, 0], [180, 129]]    # [r_strt, c_strt]

frier_ker_lst = []  # fourier kernel list
for i, [r_strt, c_strt] in enumerate(start_lst):
    print(r_strt, c_strt)
    bigK = np.zeros((bK, bK), dtype=float)
    r, c = ZZ.shape
    bigK[r_strt:r + r_strt, c_strt:c + c_strt] = ZZ
    y = np.fft.fft2(bigK)
    frier_ker_lst.append(y)
    fft_shift = np.fft.fftshift(y)
    spectrum = np.log(np.abs(fft_shift)+1)      # 스펙트럼을 보인다.
    plt.subplot(220+i*2+1), plt.imshow(spectrum, cmap='gray')
    plt.title(f'Spectrum LoG: {r_strt}, {c_strt}')
    plt.axis('off')             # plt.xticks([]), plt.yticks([])
    ftr = np.real(np.fft.ifft2(y))
    plt.subplot(220+i*2+2), plt.imshow(ftr, cmap='gray')
    plt.title(f'Kernel: {r_strt}, {c_strt}')
    plt.axis('off')             # plt.xticks([]), plt.yticks([])

plt.figure(num='Spatial Domain Filtering Results')
img = cv.imread('data/lenna.tif', 0)/255.0
assert img is not None, 'No image file....!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.
img_frier = np.fft.fft2(img)

img_lst = []    # 주파수 상에서 필터링된 영상(복원)
for i, frier in enumerate(frier_ker_lst):
    g_freq = frier * img_frier       # 주파수 평면산에서의 영상에 대한 필터링
    img_freq = np.real(np.fft.ifft2(g_freq))
    img_lst.append(img_freq)
    print('img_freq:', img_freq.dtype, img_freq.shape)
    print(f'Frequency: min & max : {np.min(img_freq):#.2f}  {np.max(img_freq):#.2f}')
    plt.subplot(121+i), plt.imshow(img_freq, cmap='gray')
    plt.title(f'Kernel: {start_lst[i]}')
    plt.axis('off')             # plt.xticks([]), plt.yticks([])


plt.figure(num='Truncation')
row, col = img.shape
img_new = np.zeros(img.shape, dtype=float)
img_new[:row-N, :col-N] = img_lst[0][N:, N:]    # 우측 하단을 좌측 상단으로 보낸다.
img_new[:row-N, col-N:] = img_lst[0][N:, 0:N]    # 좌측 하단을 우측 상단으로 보낸다.
img_new[row-N:, 0:col-N] = img_lst[0][0:N, N:]    # 우측 상단을 좌측 하단으로 보낸다.
img_new[row-N:, col-N:] = img_lst[0][0:N, 0:N]    # 좌측 상단을 우측 하단으로 보낸다.

plt.imshow(img_new, cmap='gray')
plt.title(f'Truncated:')
plt.axis('off')


# [180, 129]에 대한 truncation은 생략한다.....
#for i, img_data in enumerte(img_lst):
#    img_new = np.zeros(img.shape, dtype=float)
#    img_new[:r-N, :c-N] = img_data[r_strt:, c_strt:]

#plt.waitforbuttonpress()
plt.show()
exit(0)

