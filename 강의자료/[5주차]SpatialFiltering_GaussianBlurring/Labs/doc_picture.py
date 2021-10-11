"""
 교과서를 만들위한 예제 프로그램
"""

import numpy as np
import cv2 as cv
# 만약 matplotlib 모듈이 없다면 윈도 컴맨드 창에서 "pip install matplotlib" 명령을 수행한다.
from matplotlib import pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
# 공통 섹션:  영상이 존재하는 폴더와 파일 이름을 지정하고 읽어내기
# ----------------------------------------------------------------------------------------------------------------------
# 다음 사례를 활용하여 영상 파일이 있는 폴더를 적절히 지정하시오.
#Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
Path = '../data/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 Images 폴더.
#Path = 'd:/work/StudyImages/images/'
Name = 'col_bar.png'
Name = 'Mondrian.jpg'
#Name = "Resolution chart.tiff"
Name = 'lenna.tif'


FullName = Path + Name

img = cv.imread(FullName)

# assert condition, message  : condition이 false이면  message 출력하면서 AssertError 발생.
assert img is not None, 'No image file....!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.


#"""
# ======================================================================================================================
# 실습 1: OpenCV 기반의 간략형 실험 - averaging kernel을 사용한 filtering
# ======================================================================================================================

# 넘파이 2차원 행렬을 행으로 줄을 나누어 정해준 형식으로 출력하는 함수
def prt_mtx(a, fmt):     # 어레이를 출력하는 함수
    print(f"\nshape of the array={a.shape}")
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            print(f"{a[r, c]:{fmt}}", end="")
        print()



cv.imshow('img', img)

N = 5
kernel = np.ones((N, N), np.float32)/(N * N)         # N*N으로 나누어 정규화한다.
print(np.sum(kernel))           # 커널의 모든 원소의 값을 합하면 1이 된다.
#np.set_printoptions(precision=2)        # numpy 배열의 정밀도를 소수 이하 3자리로 제한하여 출력하는 것으로 설정한다.
#print(kernel)

prt_mtx(kernel, '#6.2f')

#exit()
# img 영상을 kernel로 필터링(코릴레이션 연산)하여 결과 dst를 반환한다.
# 출력 영상, dst는 shape가 입력 영상과 같다. => dst.shape=src.shape.
# -1은 출력 영상의 depth가 입력 영상과 같음을 의미한다. => dst.dtype=src.dtype.
dst = cv.filter2D(img, -1, kernel)
print('type(dst)=', type(dst), ' | dst.shape=', dst.shape, ' | dst.dtype=', dst.dtype )
cv.imshow('dst', dst)
cv.waitKey()
cv.imwrite("tmp_blur.jpg", dst)


# 소벨 연산  ---------------------------------------
from matplotlib import pyplot as plt

img = img[..., ::-1]

# 소벨 에지 연산자(밝은 곳으로 올때 + 에지 발생, OpenCV Sobel() 함수가 사용)
SobelHorizontal = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]	    # 수평방향의 1차 미분
SobelVertical = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]] 	    # 수직방향의 1차 미분
kerH = np.array(SobelHorizontal, dtype=float)     # print(type(kerH))
kerV = np.array(SobelVertical, dtype=float)       # print(type(kerV))

print(f"kerH.shape={kerH.shape}")
print(f"kerV.shape={kerV.shape}")


np.set_printoptions(precision=3)        # numpy 배열의 정밀도를 소수 이하 3자리로 제한하여 출력하는 것으로 설정한다.


#print(SobelHorizontal)
#print(SobelVertical)

prt_mtx(kerH, '#6.2f')
prt_mtx(kerV, '#6.2f')

edgeH = cv.filter2D(src=img, ddepth=-1, kernel=kerH); print(edgeH.dtype)
edgeV = cv.filter2D(src=img, ddepth=-1, kernel=kerV); print(edgeV.dtype)



edge = abs(edgeH) + abs(edgeV)


plt.figure(num='input and edge images')
#plt.subplot(221); plt.imshow(img, cmap = 'gray')
#plt.subplot(221); plt.imshow(img_c, cmap = 'gray')
#plt.subplot(221); plt.imshow(img)      # 3채널로 출력
plt.subplot(141); plt.imshow(img)      # 3채널로 출력
plt.title('a) original input', fontsize=20); plt.xticks([]), plt.yticks([])

#plt.subplot(222),plt.imshow(edge, cmap = 'gray')
#plt.title('combined edge'); plt.xticks([]), plt.yticks([])


plt.subplot(142), plt.imshow(edgeH, cmap='gray')
plt.title('b) horizontal edge', fontsize=20); plt.xticks([]), plt.yticks([])

plt.subplot(143),plt.imshow(edgeV, cmap = 'gray')
plt.title('3) vertical edge', fontsize=20); plt.xticks([]), plt.yticks([])

plt.subplot(144); plt.imshow(dst[..., ::-1])      # 3채널로 출력
plt.title('4) blur by averaging', fontsize=20); plt.xticks([]), plt.yticks([])


plt.waitforbuttonpress()        # 현 화면에서 버튼 입력을 기다린다.


exit(0)
#"""


"""
# ======================================================================================================================
# 실습 2: matplotlib, OpenCV 기반의 실험 - averaging kernel을 사용한 filtering
# 영상 처리는 OpenCV 함수를 이용. 영상 출력은 matplotlib를 이용
# 3채널에 대해 적용하니, 칼라 영상을 사용하기 바랍니다.
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# 단계 1 - BGR to RGB 배열을 바꾼다.
# matplotlib를 이용해 화면에 영상을 출력하려면 영상 평면을 바꾸어야 한다.
# OpenCV 함수를 이용해 BGR 배열을 RGB 배열로 바꾼다.
# ----------------------------------------------------------------------------------------------------------------------

b, g, r = cv.split(img)   # img파일을 b,g,r로 분리
img = cv.merge([r, g, b]) # b, r을 바꿔서 Merge

# ----------------------------------------------------------------------------------------------------------------------
# 단계 2 - Average filter의 Kernel을 선언한다.
# 내부 요소가 1로만 이루어진 NxN 형태의 커널을 만든다.
# 커널의 계수합이 1이 되도록 정규화 해야 한다.
# ----------------------------------------------------------------------------------------------------------------------
N = 31

# 1) 커널 정의 방법 1
#kernel = np.ones((N, N), np.float32) / (N * N)      # N*N으로 나누어 커널의 합이 1이 되도록 정규화한다.

# 2) 커널 정의 방법 2
kernel = np.ones((N, N), np.float32)
kernel /= np.sum(kernel)   # 커널의 합으로 나누어 커널의 합이 1이 되도록 정규화한다.


np.set_printoptions(precision=2)    # 소수 이하 2자리까지 출력. 이하 0이면 출력 안한다.
#print('kernel=\n', kernel)          # 커널의 원소 값을 출력
print('np.sum(kernel)=', np.sum(kernel))    # 커널의 총합을 확인.

# ----------------------------------------------------------------------------------------------------------------------
# 단계 3 - correlation 연산을 행한다.
# 위의 커널으로 영상에 대해 correlation 연산을 수행한다.
# 이런 동작을 spatial filtering(공간 필터링)이라고 말한다.
# filter2D - 2차원 코릴레이션 연산을 행하는 함수
# python shell에서 help(cv.filter2D) 명령어를 입력하면 여러가지 정보를 얻을 수 있다.
# The function(filter2D) does actually compute correlation, not the convolution
# ----------------------------------------------------------------------------------------------------------------------
#dst = cv.filter2D(img, -1, kernel, borderType=cv.BORDER_DEFAULT)        # gfedcb|abcdefgh|gfedcba. 거울 반전
dst = cv.filter2D(img, -1, kernel, borderType=cv.BORDER_ISOLATED )      # 경계처리 안함.
#dst = cv.filter2D(img, -1, kernel, borderType=cv.BORDER_REPLICATE )     # 맨 바깥 화소의 값이 중복된 것으로 간주.
print('type(img)=', type(img), ' | img.shape=', img.shape, ' | img.dtype=', img.dtype )
print('type(dst)=', type(dst), ' | dst.shape=', dst.shape, ' | dst.dtype=', dst.dtype )

# 단계 3 --------------------------------------
# 서브화면에 원본과 처리 결과를 출력한다. 이후 마우스를 입력할 때까지 기다린다.
# subplot(NMK)가 의미하는 것 : 화면을 rowXcolumn(NxM)개의 sub 창으로 나누고 K 번째 서브창을 지정한다.
# 이후 imshow() 함수는 지정된 sub 창에 출력한다.
plt.subplot(121)        # 1x2 서브 화면에서 1 번을 지정.
plt.imshow(img)
plt.title('Original')   # 지정된 sub 창에 그림 제목을 출력한다.
#plt.xticks([]), plt.yticks([])      # x축, y축에 label을 출력하지 않는다.
plt.axis('off')

plt.subplot(122),plt.imshow(dst),plt.title('Averaging: N=' + str(N)) # 1x2 서브 화면에서 2 번을 지정.
plt.axis('off')                     # x축, y축에 label을 출력하지 않는다.

#plt.show()      # 창을 닫아야 프로그램이 종료됨.
plt.waitforbuttonpress()        # 키 혹은 버튼 입력을 기다린다.
# 이후 프로그램이 종료되므로 모든 창이 닫힌다.

exit(0)
"""





# ======================================================================================================================
# 실습 3: matplotlib 기반의 실험 - averaging kernel을 사용한 filtering
# 1 x N, row kernel
# N x 1, column kernel
# interactive mode로 동작한다. 아무런 키나 마우스를 입력하면 종료한다.
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
# 단계 1 - BGR to RGB 배열을 바꾼다.
# matplotlib를 이용해 화면에 영상을 출력하려면 영상 평면을 바꾸어야 한다.
# OpenCV 함수를 이용해 BGR 배열을 RGB 배열로 바꾼다.
# ----------------------------------------------------------------------------------------------------------------------
#b, g, r = cv.split(img); img = cv.merge([r, g, b])   # img파일을 b,g,r로 분리하고, b, r을 바꿔서 Merge
img = img[..., ::-1]

# ----------------------------------------------------------------------------------------------------------------------
# 단계 2 - Average filter의 Kernel을 선언한다.
# 내부 요소가 1로만 이루어진 NxN 형태의 커널을 만든다.
# 커널의 계수합이 1이 되도록 정규화 해야 한다.
# ----------------------------------------------------------------------------------------------------------------------
N = 51

kernel_list = []         # 다수의 커널로 list 자료로 기록한다.

# 1) 1xN 커널 정의
kernel = np.ones((1, N), np.float32) / (N)      # 주의!!! N으로 나누어 커널의 합이 1이 되도록 정규화한다.
print('np.sum(kernel)=', np.sum(kernel))    # 커널의 총합을 확인.
kernel_list.append(kernel)

# 2) Nx1 커널 정의
kernel = np.ones((N, 1), np.float32)
kernel /= np.sum(kernel)   # 커널의 합으로 나누어 커널의 합이 1이 되도록 정규화한다.
print('np.sum(kernel)=', np.sum(kernel))    # 커널의 총합을 확인.
kernel_list.append(kernel)

print(type(kernel_list))

#np.set_printoptions(precision=2)    # 소수 이하 2자리까지 출력. 이하 0이면 출력 안한다.
#print('kernel=\n', kernel)          # 커널의 원소 값을 출력


# ----------------------------------------------------------------------------------------------------------------------
# 단계 3 - correlation 연산을 행하고 그 결과를 pyplot 결과창에 출력한다.
# 위의 커널 list로 영상에 대해 correlation 연산을 수행한다.
# ----------------------------------------------------------------------------------------------------------------------
#plt.ion()
fig = plt.figure(num='interactive mode', figsize=(16, 5))
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.85)       # 상하좌우 여분 공백 제어하기
fig.patch.set_facecolor('green')            # 화면 바탕 색상을 정한다.
#fig.patch.set_facecolor('lightblue')       # 화면 바탕 색상을 정한다.
fig.patch.set_alpha(0.1)        # Alpha, transparency: 0이면 투명, 1이면 완전 불투명.

plt.suptitle("Averaging Filtering or Box Filtering", fontsize=14, fontweight='bold')

plt.subplot(1, 3, 1)    # 1x3 창의 첫번째 창 선택
plt.imshow(img)     # 원본 영상
plt.axis('off')
plt.title('Original')

i = 0   # kernel index
for knl in kernel_list:
    print(type(knl), knl.shape, knl.dtype)      # <class 'numpy.ndarray'> (1, 51) float32
    dst = cv.filter2D(img, -1, knl, borderType=cv.BORDER_DEFAULT)
    #print(f"dst.dtype={dst.dtype}")     # dst.dtype=uint8
    plt.subplot(130 + (i+2))      # 1x3 창의  (i+2)번째 창 선택
    plt.imshow(dst)     # filtering 결과 영상
    plt.axis('off')
    plt.title('Kernel=' + str(knl.shape))
    i += 1

#plt.show()      # interactive mode 일 때는 필요 없음
plt.waitforbuttonpress()        # 키 혹은 버튼 입력을 기다린다.
# 이후 프로그램이 종료되므로 모든 창이 닫힌다.
exit(0)




# ======================================================================================================================
# 미션 1: 실습 3을 2x2 서브 화면으로 나누고, 4번째 화면에는 양방향(가로/세로)으로 평균필터링한 영상을 출력하시오.
# ======================================================================================================================

# ======================================================================================================================
# 미션 2: 커널의 크기가 5x5, 15x15, 31x31의 average filtering 결과를 원본 영상과 함께 2x2 화면에 출력하시오
# ======================================================================================================================
