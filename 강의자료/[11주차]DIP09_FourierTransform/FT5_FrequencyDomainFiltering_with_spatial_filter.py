"""
공간 필터 커널이 주어졌을 때 이를 바탕으로 공간 필터링 통한 결과와 주파수 공간 필터링 통한 결과를 서로 비교해 본다.

공간 필터링은 filter2D() 함수로 필터링한다.
주파수 공간 필터링은 커널을 영상의 크기만큼 0 padding하여 확대하여 푸리에 변환한 결과와
영상을 푸리에 변환하여 곱한 결과를 역변환하여 필터링을 행한다.

실험의 핵심 하이라이트
1. 역변환 복원된 영상이 4개 분면으로 나누어지고 이들이 서로 대각 대칭 이동된다는 사실을 발견하였다.
   따라서, 0 padding 공간의 어느 곳에 삽입하느냐가 대칭이동 복원의 중요한 포인트가 되었다.

2. 공간 필터링에서 영상의 외곽 경계를 0으로 보면 테두리가 생기는 것처럼,
   필터를 영상과의 곱셈을 위해 영상과 같은 크기로 설정하면 외곽에 공간 필터링에서와 같은 경계 테두리가 발생한다.
   이를 제거하기 위해서는 영상의 주위를 경계값이 반복되도록 필터의 반만큼 각각 확장시켜야 한다.
   여기서 이와 같은 영상 확장을 위해 다음 OpenCV 함수를 사용하였다.
    img = cv.copyMakeBorder(src, top, bottom, left, right, cv.BORDER_REPLICATE, None, value)

3. 결론적으로 다음의 절차를 거쳐 주파수 공간상에서 필터링을 행한다.
    1) 입력영상은 필터 크기의 반만큼 확장해야 하고, 상하좌우 4개의 테두리에 각 경계 값들을 확장된 부분에 채워 주어야 한다.
    2) 필터도 확장된 영상의 크기로 0 패딩해야 한다.
    3) 두 푸리에 변환 결과를 곱한 후 역변환 한 다음에는 4개 분면의 영상조각들을 대각 대칭이동해야 한다.
    4) 마지막으로 경계확장된 공간을 제거하는 truncation을 행한다.

"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt




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
def gaussian2ndDev(x, y, sigma):
    r2 = x**2 + y**2
    return( (r2-2*sigma**2)/sigma**4 * np.exp(-r2/(2*sigma**2)))

#def gaussian2ndDev(x,y, sigma):
#    r2 = x**2 + y**2
#    return( ( (r2-2*sigma**2) * np.exp(-r2/(2*sigma**2)) ) / sigma**4)

# 필터 계수를 연산하고 공간 차원에서 필터링을 행한다.
ZZ = gaussian2ndDev(XX, YY, sigma)

# 단계 2 -----------------------------------------------------------------
# 영상을 읽고 경계 영역을 확장한다.
# 확장은 상하좌우 방향으로 각각 필터 커널의 반만큼 확장하여 빈 공간을 경계 영역으로 채운다.
# ------------------------------------------------------------------------

src = cv.imread('data/lenna.tif', 0)/255.0
assert src is not None, 'No image file....!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.

# 경계를 채워 넣기
#top = int(0.05 * img.shape[0])  # shape[0] = rows
top = N
bottom = top
#left = int(0.05 * img.shape[1])  # shape[1] = cols
left = N
right = left
value = 0
#borderType = cv.BORDER_CONSTANT
borderType = cv.BORDER_REPLICATE
img = cv.copyMakeBorder(src, top, bottom, left, right, borderType, None, value)
print('2.1 src.shape=', src.shape, 'img.shape=', img.shape)


# 상하좌우 여백의 테두리를 잘라내는 함수
def truncate_image(img, top, bottom, left, right):
    row = img.shape[0]
    col = img.shape[1]
    out = np.empty((row-top-bottom, col-left-right))   # uninitialized, output may vary
    out = img[top:row-bottom, left:col-right]
    return out

ret = truncate_image(img, top, bottom, left, right)
print('2.2 ret.shape=', ret.shape)
ae = np.array_equal(src, ret)
print("2.3 array_equal(src, ret)=", ae)


# 단계 3 -----------------------------------------------------------------
# 확장된 영상 크기 만큼의 0 패딩된 영역의 일부에 커널을 복사한다.
# 여기서는 비교를 위해 좌상단과 임의 위치에 복사하였다.
# 이후 패딩된 커널을 푸리에 변환하고 스펙트럼을 출력한다.
# ------------------------------------------------------------------------
fig = plt.figure(num="1. Spectrum of 0 padding Kernels & their inverse results", figsize=(12, 8))
plt.suptitle("1. Spectrum of 0 padding Kernels & their inverse results", fontsize=14, fontweight='bold')

# 0 padding을 다음 2가지 방법으로 실시한다.
# [0, 0]: 필터의 계수를 0 padding 확대된 커널의 좌측 상단에 복사한다. r_strt = 0; c_strt = 0
# [180, 129]: 필터의 계수를 0 padding 확대된 커널의 임의 영역에 복사한다. r_strt = 180; c_strt = 129
start_lst = [[0, 0], [180, 129]]    # [r_strt, c_strt]

frier_ker_lst = []      # 영상과 같은 크기로 확장된 커널의 푸리에 변환 결과를 담는 리스트

# 영상의 크기만큼 큰 커널을 만들고 이것의 푸리에 변환을 취하고 그 결과를 출력한다.
for i, [r_strt, c_strt] in enumerate(start_lst):
    print(r_strt, c_strt)
    #bigK = np.zeros((bK, bK), dtype=float)
    bigK = np.zeros(img.shape, dtype=float)
    #print('bigK.shape=', bigK.shape)

    r, c = ZZ.shape
    bigK[r_strt:r + r_strt, c_strt:c + c_strt] = ZZ # 커널을 확장된 커널의 일부 영역에 복사해 넣는다.
    y = np.fft.fft2(bigK)
    frier_ker_lst.append(y)
    fft_shift = np.fft.fftshift(y)
    spectrum = np.log10(np.abs(fft_shift)+1)      # 스펙트럼을 보인다.
    plt.subplot(220+i*2+1), plt.imshow(spectrum, cmap='jet')
    plt.title(f'Spectrum of Kernel: {r_strt}, {c_strt}')
    plt.axis('off')             # plt.xticks([]), plt.yticks([])
    ftr = np.real(np.fft.ifft2(y))
    #plt.subplot(220+i*2+2), plt.imshow(ftr, cmap='gray')
    plt.subplot(220 + i * 2 + 2), plt.imshow(ftr)
    plt.title(f'Restored Kernel(ext): {r_strt}, {c_strt}')
    plt.axis('off')             # plt.xticks([]), plt.yticks([])

plt.figure(num='2. Raw outputs of Frequency Domain Filtering', figsize=(12, 8))
plt.suptitle("2. Raw outputs of Frequency Domain Filtering", fontsize=14, fontweight='bold')
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


plt.figure(num='3. Diagonal Shift after the frequency filtering')

# 입력 영상을 좌측 상단 NxN 되는 곳을 기준으로 대각 이동시키는 함수
# N= int(커널의 크기/2)
def diagonal_shift(img, N):
    row, col = img.shape
    img_new = np.zeros(img.shape, dtype=float)
    img_new[:row-N, :col-N] = img_lst[0][N:, N:]    # 우측 하단을 좌측 상단으로 보낸다.
    img_new[:row-N, col-N:] = img_lst[0][N:, 0:N]    # 좌측 하단을 우측 상단으로 보낸다.
    img_new[row-N:, 0:col-N] = img_lst[0][0:N, N:]    # 우측 상단을 좌측 하단으로 보낸다.
    img_new[row-N:, col-N:] = img_lst[0][0:N, 0:N]    # 좌측 상단을 우측 하단으로 보낸다.
    return img_new

img_new = diagonal_shift(img, N)
plt.imshow(img_new, cmap='gray')
plt.title(f'3. Filtered & shifted image{img_new.shape}')
plt.axis('off')



# [180, 129]에 대한 truncation은 생략한다.....
#for i, img_data in enumerte(img_lst):
#    img_new = np.zeros(img.shape, dtype=float)
#    img_new[:r-N, :c-N] = img_data[r_strt:, c_strt:]


plt.figure('4. spatial domain filtering vs frequency domain filtering', figsize=(12, 8))
plt.suptitle("4. spatial domain filtering vs frequency domain filtering", fontsize=14, fontweight='bold')

plt.subplot(121)        # spatial domain filtering
result = cv.filter2D(src, cv.CV_64F, ZZ, borderType=cv.BORDER_REPLICATE)
#result = cv.filter2D(img, cv.CV_64F, ZZ, borderType=cv.BORDER_REFLECT)
#result = cv.filter2D(src, cv.CV_64F, ZZ, cv.BORDER_CONSTANT)
plt.imshow(result, cmap='gray')
plt.title(f'Result after spatial filtering{result.shape}')
plt.axis('off')

plt.subplot(122)    # frequency domain filtering
ret = truncate_image(img_new, top, bottom, left, right)
print('ret.shape=', ret.shape)
plt.imshow(ret, cmap='gray')
plt.title(f'Truncated image after frequency filtering{ret.shape}')
plt.axis('off')


# 공간 필터링과 주파수 필터링 결과의 MSE를 구해 본다.
mse = 255 * np.sum(abs(result-ret) ** 2) / (ret.shape[0] * ret.shape[1])
print(f'\nMSE(Mean Square Error) in 8 bit depth ={mse:.3e}')

#print(result.dtype, ret.dtype)
#print(result.max(), result.min())
#print(ret.max(), ret.min())

#val = cv.PSNR()

#plt.waitforbuttonpress()
plt.show()
exit(0)

