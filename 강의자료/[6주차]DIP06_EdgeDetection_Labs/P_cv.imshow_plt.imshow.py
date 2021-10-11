"""
1. 개요
    cv.imshow와 plt.imshow()의 화면 출력 특징을 아래 여러가지 조건에 따라 어떻게 달라지는지 확인한다.
    입력 파일: color,    mono
    데이터형: uint8, floating point(0~1)
    음수 발생: -128,   -0.5
    범위 초과: +128, +0.5


2. 결론
1. cv.imshow(): 칼라, 모노 영상에 관계 없다.
1) 부동소수 영상의 경우
    ① 0~1의 부동소수 영역에 대해서만 화면에 표시된다.
    ② 0보다 작은 음수에 대해서는 0으로 clipping 된다. 즉, 0 이하는 0으로 간주한다.
    ③ 1을 넘어서는 데이터에 대해서는 1로 clipping 된다. 따라서 정규화가 안된 부동소수 영상은 흰색 화면을 만들 수 있다.

2) uint8 영상의 경우
    ① 범위를 넘어서는 데이터는 최하위 8 비트만 취한다.
    ② 화솟값이 256이면 1로 표현된다. 그래서 밝은 부분이 오히려 검게 나온다.
    ③ 음수가 발생하면 그것을 unsignd char 형으로 보고 하위 7비트만 취하기 때문에 오히려 큰 값으로 표현될 수 있다. 이 때문에 검은 부분이 오히려 밝게 나온다. 예를 들어 화소 값이 –1이면 0xFF=255로 표현된다.
    ④ 영상에 연산을 가할 때 uint8 입력이 부동소수로 바뀔 수 있음에 유의해야 한다. 예: uint8 * 1.3 => 부동소수, uint8/2 => 부동소수
    ⑤ 음수가 나오면 8 비트를 넘어서는 연산 결과가 나올 수 있다. 음수는 two’s compliment형으로 표현되는데 하위 8 비트만 취해서 반환하기 때문에 큰 값(밝은 화소)으로 표현될 수 있다.
    ⑥ 8 비트를 넘어서는 큰 양의 정수가 나와도 역시 하위 8비트트만취해서 반환하기 때문에 작은 값(어두운 화소)으로 표현될 수 있다.


2. plt.imshow(): 칼라와 모노 영상에 대한 처리가 다르다.
1) 3 채널(칼라), 1 채널(모노) uint8 영상의 경우
    3 채널 부동소수, 1 채널 uint8 영상에 대해서는 cv.imshow()와 동일한 방식으로 처리된다.

2) 1채널 부동소수 영상의 경우
    영상의 최솟값, 최댓값을 찾아 이를 기준으로 0~1의 값으로 정규화한다. 이 때문에 음수가 있건, 최댓값이 1을 넘건 가리지 않고 충실하게 영상을 표현한다. 원본이 혹은 0~1의 범위에 분포하는 영상이라면 여기에 어떤 값을 곱하거나 나누어도 원본과 같은 영상을 출력한다.
    수식으로 표현애 본다면; np.min(img) ~ np.max(img) 범위의 영상을 0~1의 범위로 화면에 출력한다. 원본의 범위를 좁히거나, 넓혀서 출력하는 경우가 될 수도 있다.

3. 목표
    1. 칼라 영상의 RGB 평면의 배열, 영상 데이터의 자료형 및 값의 범위의 따른 출력 특성을 익힌다.
    2. 모노 영상의 경우 cmap의 선정, 영상 데이터 자료형 및 값의 범위의 따른 특징을 익힌다.

4. 실험
    Lab 1: openCV:  cv.imshow()의 영상 데이터 출력 범위 알아보기
    Lab 2: pyplot.imshow()의 칼라 평면 해석 및 cmap 파라미터 용도 점검하기
    Lab 3: 영상에 대한 가산/감산, 곱셈/나눗셈에 대한 pyplot.imshow()의 출력 특성 알아보기
        3a - 3채널 영상에 대한 pyplot.imshow()의 영상 데이터 출력 범위 알아보기
        3b - 1채널 영상에 대한 pyplot.imshow()의 영상 데이터 출력 범위 알아보기



"""


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
# 공통 부분 - 영상이 존재하는 폴더와 파일 이름을 지정하기
# 다음 사례를 활용하여 영상 파일이 있는 폴더를 적절히 지정하시오.
# ----------------------------------------------------------------------------------------------------------------------
#Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
Path = '../data/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 Images 폴더.
#Path = 'd:/work/@@DIP/LectureMaterials/Images/'
#Name = 'monarch.bmp'
#Name = 'road.png'
#Name = 'steps.png'
#Name = 'circles_rects.png'
Name = 'lenna.tif'
#Name = 'graphicShapes.png'

FullName = Path + Name


image = cv.imread(FullName)           # 칼라 영상. 칼라와 모노 영상의 출력 특성이 같다.
#img = cv.imread(FullName, 0)        # 모노 영상. 칼라와 모노 영상의 출력 특성이 같다.

# assert condition, message  : condition이 false이면  message 출력하면서 AssertError 발생.
assert image is not None, f'No image file [{Name}]....!'  # 입력 영상을 제대로 읽어오지 못하면 NULL을 반환.

print("type(image)=", type(image), " image.dtype=", image.dtype, "image.shape=", image.shape)

img_c_8 = image.copy()        # color 8 bit
img_c_f = image/255           # color float
img_m_8 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
img_m_f = img_m_8 / 255

img8_lst = [img_c_8, img_m_8]
imgf_lst = [img_c_f, img_m_f]

imgC_lst = [img_c_f, img_c_8]
imgM_lst = [img_m_f, img_m_8]


# ---------------------------------------------------------------------------------------------------
# Lab 1: openCV:  cv.imshow()의 영상 데이터 출력 범위 알아보기
# ---------------------------------------------------------------------------------------------------
# 목표: cv.imshow() 함수의 영상 데이터의 자료형 및 값의 범위에 대한 출력 특성을 익힌다.
# cv.imshow()를 이용한 칼라/모노 영상의 출력 특성
#   1. 칼라와 모노 영상의 출력 특성에 대해 실험
#   2. 데이터 형은 uint8 혹은 0~1 범위의 flaot 형만 표현 가능하다.
#       1) uint8의 경우: 이 범위를 넘어서는 것들에 대한 처리
#           0~255를 넘어서면 32비트로 확장된 후 하위 8비트만 취한다.
#       2) float의 경우: 이 범위를 넘어서는 것들에 대한 처리
#           + 값은 1보다 크면 1로 clipping 처리된다.
#           - 값은 0으로 처리된다.
# ---------------------------------------------------------------------------------------------------

"""
def title_info(img2):
    return(f" => OUT:{str(img2.dtype).upper()}, {np.min(img2):#.1f}~{np.max(img2).astype(np.float64):#.1f}")
    
#for img in img8_lst:
for img in [img_c_f, img_m_f, img_c_8, img_m_8 ]:        # [img_c_8, img_c_f, img_m_8, img_m_f]
    if img.dtype=='uint8': ofst = 128
    else: ofst = 0.5
    if img.ndim == 3:
        ttle = 'IN: COLOR'
    else:
        ttle = 'IN: MONO'
    title = ttle + f", {str(img.dtype).upper()} => "
    # type(img) = <class 'numpy.dtype[uint8]'>. str()을 취해야 함.

    img2 = img + ofst
    cv.imshow(title + f"img+{ofst}" + title_info(img2), img2)
    img2 = img - ofst
    cv.imshow(title + f"img-{ofst}" + title_info(img2), img2)

    img2 = img * 2
    cv.imshow(title + f"img*2" + title_info(img2), img2)
    img2 = img / 2
    cv.imshow(title + f"img/2" + title_info(img2), img2)
    cv.waitKey()
    cv.destroyAllWindows()

cv.waitKey()
#cv.destroyAllWindows()

"""


# ---------------------------------------------------------------------------------------------------
# Lab 2: pyplot.imshow()의 칼라 평면 해석 및 cmap 파라미터 용도 점검하기
# ---------------------------------------------------------------------------------------------------

fig = plt.figure("Lab 2: Color planes & Color map", figsize=(14, 8))
plt.suptitle(f"Lab 2: 1)~2) uint8 color inputs, 3)~6) uint8 mono inputs", fontsize=18, fontweight='bold')
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.85)       # 상하좌우 여분 공백 제어하기
#fig.patch.set_facecolor('silver')  # 화면 바탕 색상을 정한다.
#fig.patch.set_alpha(0.4)

color_map = ['none', 'none', 'viridis', 'gray', 'Purples', 'jet']
for i, img in enumerate([img_c_8, img_c_8, img_m_8, img_m_8, img_m_8, img_m_8]):
    plt.subplot(2, 3, i + 1)
    #print(f"drawing row={i}, shape={img.shape}....")
    if img.ndim == 3:
        channel = 'C:'
    else:
        channel = 'M:'
    if i == 0:
        img = img[..., ::-1]    # BGR -> RGB 변환
        channel += 'RGB '
    elif i == 1:
        channel += 'BGR '
    elif i > 1:
        channel = channel + f'cmap={color_map[i]} '
    data_type = str(img.dtype).upper()
    if data_type == 'FLOAT64':
        data_type = 'F64'
    data_range = f" {np.min(img):#.1f}~{np.max(img):#.1f}"
    ttle = f"{i+1}) " + channel + data_type + data_range

    plt.title(ttle)    # , loc='right', verticalalignment='bottom'
    plt.xticks([]), plt.yticks([])  # 검은 테두리 있음
    if i == 0 or i == 1 or i == 2:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=color_map[i])

#plt.show()

#exit()

# ---------------------------------------------------------------------------------------------------
# Lab 3: 영상에 대한 가산/감산, 곱셈/나눗셈에 대한 pyplot.imshow()의 출력 특성 알아보기
#     Lab 3a: 3채널 영상에 대한 pyplot.imshow()의 영상 데이터 출력 범위 알아보기
#     Lab 3b: 1채널 영상에 대한 pyplot.imshow()의 영상 데이터 출력 범위 알아보기
# ---------------------------------------------------------------------------------------------------
def title_info(img2):
    #return(f"->{str(img2.dtype).upper()}, {np.min(img2):#.1f}~{np.max(img2).astype(np.float64):#.1f}")
    data_type = str(img2.dtype).upper()
    if data_type == 'FLOAT64':
        data_type = 'F64'
    return (f"->{data_type}, {np.min(img2):#.1f}~{np.max(img2).astype(np.float64):#.1f}")

def display(img2):
    if img.ndim == 3:
        plt.imshow(img2[..., ::-1])
    else:
        plt.imshow(img2, cmap='gray')


for ii, lst in enumerate([imgC_lst, imgM_lst]): # [lab3a, Lab3b]
    tle_page = 'Lab 3a: 3 channel image' if ii == 0 else 'Lab 3b: 1 channel image'
    fig = plt.figure(tle_page, figsize=(16, 8))
    plt.suptitle(f"{tle_page} = 1st row: float inputs, 2nd row: uint8 inputs", fontsize=18, fontweight='bold')
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.85)       # 상하좌우 여분 공백 제어하기
    #fig.patch.set_facecolor('silver')  # 화면 바탕 색상을 정한다.
    #fig.patch.set_alpha(0.4)
    print(f"Selected list pack num={ii} ---------------------------------------------")

    for i, img in enumerate(lst):
        data_range = f" {np.min(img):#.1f}~{np.max(img):#.1f}"      # 입력 영상의 데이터 범위
        print(f".... drawing row={i}, shape={img.shape}, range:{data_range}")
        if img.dtype=='uint8': ofst = 128
        else: ofst = 0.5
        if img.ndim == 3:
            ttle = 'C:'
        else:
            ttle = 'M:'
        data_type = str(img.dtype).upper()
        if data_type == 'FLOAT64':
            data_type = 'F64'
        title = ttle + data_type + '->'
        # type(img.dtype) = <class 'numpy.dtype[uint8]'>. str()을 uint8을 반납받음.

        img2 = img + ofst
        plt.subplot(2, 4, i*4 + 1)
        head_line = f'{i*4 + 1}) ' + title + f"img+{ofst}" + title_info(img2)
        print(head_line)
        plt.title(head_line)    # , loc='right', verticalalignment='bottom'
        display(img2)
        plt.xticks([]), plt.yticks([])  # 검은 테두리 있음
        # plt.axis('off')                     # 테두리 없음.
        #print(4, 4, i*4 + 1)

        img2 = img - ofst
        plt.subplot(2, 4, i*4 + 2)
        head_line = f'{i*4 + 2}) ' + title + f"img-{ofst}" + title_info(img2)
        print(head_line)
        plt.title(head_line)    # , loc='right', verticalalignment='bottom'
        display(img2)
        plt.xticks([]), plt.yticks([])  # 검은 테두리 있음

        img2 = img * 2
        plt.subplot(2, 4, i*4 + 3)
        head_line = f'{i*4 + 3}) ' + title + f"img*2" + title_info(img2)
        print(head_line)
        plt.title(head_line)    # , loc='right', verticalalignment='bottom'
        display(img2)
        plt.xticks([]), plt.yticks([])  # 검은 테두리 있음

        img2 = img / 2
        plt.subplot(2, 4, i*4 + 4)
        head_line = f'{i*4 + 4}) ' + title + f"img/2" + title_info(img2)
        print(head_line)
        plt.title(head_line)    # , loc='right', verticalalignment='bottom'
        display(img2)
        plt.xticks([]), plt.yticks([])  # 검은 테두리 있음

plt.show()
exit()
