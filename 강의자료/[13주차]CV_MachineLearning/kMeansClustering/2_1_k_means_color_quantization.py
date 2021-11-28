"""
K-means clustering algorithm applied to color quantization

k-means clustering은 지정된 K개의 그룹으로 나누기 때문에 24비트 칼라를 K개로 그룹핑할 수 있다.

"""

# Import required packages:
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

path = '../data/'
f_name = 'landscape_2.jpg'
#f_name = 'bts3.jpg'
full_name = path + f_name


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def color_quantization(image, k):
    """Performs color quantization using K-means clustering algorithm"""
    # Transform image into 'data':

    # 문제의 올바른 접근을 검토해 보자:
    # 1) RGB 3채널별로 K개의 칼라 군집화를 하는 것이 맞을까?
    # 2) 아니면 화면 전체의 색상에 대해 K 군집화를 하는 것이 맞을까?
    # -> 1은 사실상 Kx3개로 클러스링이 이루어지는 것이고, 색상의 부조화가 일어날 가능성 있다.
    #    마치 컬러히스토그램 평활화에서 채널별로 평활화를 하면 안되는 것처럼..
    # -> 2의 방법이 옳다. 가로x세로의 화소에 대해 3개 채널의 중심점을 K개 찾아내는 것이 맞다.
    #
    # 영상 데이터를 (가로x세로, 3)의 2차원 배열을 가진 데이터 형태로 나열하여 입력한다.
    # 마치 (x,y) 좌표로 주어진 샘플들의 군집화를 행하는 것처럼
    # 샘플 갯수 x 3채널 상의 데이터 군집화 문제를 해결하는 것이다.
    #
    # kmeans() 함수를 통해 군집화를 행하고,
    # center.shape=(k, 3)를 반환받는다. RGB의 중점으로 반환한다.
    # labels.shape=(가로x세로, 1)의 배열로 반환된다. 각 원소에는 센터의 인덱스 번호가 int32 자료로 담겨있다.
    # 라벨에서 지정하는 정보를 센터에 대한 인덱스로 활용하여, 대표 색상을을 알아내고 해당 화소의 RGB를 바꾼다.
    # 레이블의 한 원소가 3채널로 구성된 centers(사실상 팔레트)의 주소를 indexing하여 컬러값을 받아,
    # 그 레이블에 해당하는 화소의 RGB를 바꾼다.
    #
    # k=256이면, 24비트를 8비트로 줄인 효과를 얻을 수 있다.
    data = np.float32(image).reshape((-1, 3))   # 맨 뒤자리 차원 3은 고정하고 나머지는 정리하여 화솟수x3채널 자료로 만든다.
    print(f'\nk={k}: data.shape={data.shape}')       # data.shape=(400000, 3). 영상크기(가로x세로) x 3채널

    # Define the algorithm termination criteria (the maximum number of iterations and/or the desired accuracy):
    # In this case the maximum number of iterations is set to 20 and epsilon = 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    s_time = time.time()
    # Apply K-means clustering algorithm:
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    print(f"label: type={type(label)}, shape={label.shape}, len={len(label)}")
    # label: type=<class 'numpy.ndarray'>, shape=(400000, 1), len=400000
    print(f"center: shape={center.shape}, dtype={center.dtype}")
    # K일 때 center: shape=(K, 3), dtype=float32

    e_time = time.time() - s_time

    # At this point we can make the image with k colors
    # 센터에는 K개의 군집을 대표하는 3채널 RGB 값이 들어 있다.
    # 센터의 자료형이 dtype=float32이므로 컬러값으로 쓰기 위해 uint8형의 바꾼다.
    center = np.uint8(center)   # k개의 센터를 확보함.

    # Replace pixel values with their center value:
    result = center[label.flatten()]

    # 영상과 같은 shape로 만든다.
    result = result.reshape(img.shape)

    # 원본과의 유사도를 나타내는 PSNR 품질을 확인한다.
    psnr = cv2.PSNR(image, result)
    print(f'k={k}: time={e_time:#.2f}, PSNR={psnr:#.2f}')
    return result


# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(12, 6))
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.85)


plt.suptitle("Color quantization using K-means clustering algorithm", fontsize=14, fontweight='bold')
#fig.patch.set_facecolor('silver')

# Load BGR image:
img = cv2.imread(full_name)
show_img_with_matplotlib(img, "original image", 1)

# Apply color quantization:


# 화면 구성이 2x3이므로 원본 영상을 sub 화면 1번으로 하면 최대 5개의 K값을 리스트 자료에 설정할 수 있습니다.
for i, k in enumerate([8, 4]):          # 확인용으로 적절
#for i, k in enumerate([[2, 3, 4, 5, 6]):
#for i, k in enumerate([64, 32, 16, 8, 4]):  # 최대 5개
#for i, k in enumerate([32, 64, 128, 256]):  # 21초
    color_2 = color_quantization(img, k)
    show_img_with_matplotlib(color_2, f"color quantization (k = {k})", i+2)



# Show the Figure:
plt.show()



"""
기록 보관용
@@@@@@@@@@@@@@@@  과년도 과제     @@@@@@@@@@@@@@@@@@@@@@@@@@@@
1. 본 예제 프로그램에 대하여 제공되는 파일 Figure_bts2.png, Figure_sku1.png와 같은 내용이 출력되도록 프로그램을 수정하시오.
    1) 타이틀에는 k 값과 PSNR이 출력되어야 한다.
    2) 다음과 같은 정보가 파이썬 콘솔창에 출력되어야 한다.(k와 PSNR 정보가 어느 정도 유사하게 일치해야 할 것임. 시간은 다르겠지만...)
        k=2 | PSNR=16.12[dB] | time=  0.58[sec.]
        k=4 | PSNR=20.86[dB] | time=  1.01[sec.]
        k=8 | PSNR=24.31[dB] | time=  2.87[sec.]
        k=16 | PSNR=27.17[dB] | time=  5.46[sec.]
        k=32 | PSNR=29.76[dB] | time= 10.14[sec.]
        k=64 | PSNR=31.89[dB] | time= 19.62[sec.]
        k=128 | PSNR=33.57[dB] | time= 40.72[sec.]

2. K=256로 그룹핑한다면 모든 24비트 칼라를 8비트 칼라로 군집화하는 것이다.
   이 때는 (R, G, B)로 이루어진  256개의 중심 값을 cv2.kmeans() 함수가 centers 변수로 반환한다.
   PNG, BMP 파일은 파일 내부의 팔레트(palette) 헤더에 이 정보를 수록하고,
   화소값은 8비트만 기록하여 파일의 크기를 1/3로 줄일 수 있는 기능을 제공한다.
   1) 이 기능을 이용하여 K개로 줄인 색상을 PNG 파일 형식으로 저장하고,
   2) 다시 파일을 읽어 들어 24비트 원본 영상과 8비트로 줄인 영상의 화질을 PSNR 지표를 사용하여 비교하는 프로그램을 작성하시오.
   - 방법 제공되는 파일(Figure_bts2.png, Figure_sku1.png)의 내용과 같이 화면에 출력하면 됨.
   - 본 과정의 순조로운 프로그램을 위하여 몇 개의 선행 학습용 프로그램을 제공합니다..


"""