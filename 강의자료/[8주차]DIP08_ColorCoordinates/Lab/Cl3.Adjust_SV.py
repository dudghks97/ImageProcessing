"""

1. 프로그램의 기능 - 채도와 밝기를 변환시킨다.

2. 프로그램의 목표
    Hue는 고정하고 S, V를 변화시킨 후 역 변환하여 S, V의 변화가 어떤 결과를 낳는지 고찰한다.

3. 프로그램에서 점검 포인트
    실습 1, 2 : uint8 FULL, float32 - 2가지 데이터 타입으로 이루어진 HSV 좌표계의 S값의 변화영향을 고찰한다.
    실습 3: 부동소수에 대해 HSV 좌표계의 S, V값의 동시 변화영향을 고찰한다.
            RGB역변환 과정에 값의 범위 초과 경고가 발생하기도 한다.

4. 미션
    1) 원본 영상의 S와 V 성분을 약화 혹은 강화를 꾀하는 프로그램을 작성하시오.
    2) 이를 OpenCV Trackbar를 적용하여 구현하시오.

5. 질문

"""


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#=============================================================================================
# 영상이 존재하는 폴더와 파일 이름을 지정하여 영상 파일 읽어 내기
# 다음 사례를 활용하여 영상 파일이 있는 폴더를 적절히 지정하시오.
#Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
Path = '../data/'
Name = 'lenna.tif'
Name = 'monarch.bmp'
Name = 'fruits.jpg'
#Name = 'bone.jpg'
#Name = 'woman_in_scarf(c).jpg'
#Name = 'man_woman.jpg'

#Name = 'over_exposure.jpg'
#Name = 'bukak.jpg'
#Name = 'Macbeth_NoColorProfile.jpg'
#Name = 'colorbar_chart.jpg'
#Name = 'RGB.jpg'
#Name = 'RGBcolors.jpg'


FullName = Path + Name
img = cv.imread(FullName)
# assert condition, message  : condition이 false이면  message 출력하면서 AssertError 발생.
assert img is not None, 'No image file....!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.



# RGB <-> BGR 간의 영상 평면의 배열을 바꾸는 함수를 정의한다.
# 1) cv의 BGR 배열을 python의 RGB 배열로 바꾸어 반환한다.
def cv2py(cvBGR):
    b, g, r = cv.split(cvBGR)
    return cv.merge([r,g,b])            # Python RGB 배열

# 2) python의 RGB 배열을 cv의 BGR 배열로 바꾸어 반환한다.
def py2cv(pyRGB):
    r, g, b = cv.split(pyRGB)
    return cv.merge([b,g,r])            # cv RGB 배열

# ============================================================================================



"""
# 방법 1 ======================================================================================
# 채도만 변화를 주어 다시 역변환해서 보인다.
# uint8 정수 버전 
# HSV중 H는 0~255의 범위에 있다. 이른바 FULL
# =============================================================================================

# 원본 영상 출력
plt.figure(num='Original Image & its satuartion adjustments: uint8 type image(FULL)', figsize=(13, 8))
imgP = cv2py(img)       # Python RGB 배열
#imgP = cv.cvtColor(img, cv.COLOR_BGR2RGB)       # OpenCV 함수가 있음.
plt.subplot(231)
plt.imshow(imgP)
plt.title('original')
plt.xticks([]), plt.yticks([])


# 원본 영상의 HSV 좌표 변환. BGR을 HSV 좌표계로 변환한다.
imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL )
#print('sat max=', np.max(np.max(imgHSV[:,:,1])))   # 255인 것으로 판명됨
print('imgHSV.dtype=', imgHSV.dtype)
# 처리될 영상의 생성
imgNew = imgHSV.copy()

# 6단계의 채도 변화 관찰. 0~255이 넘는 Hue에 대해 자동 절환이 일어나는지 살핀다.
i=1
scale_list = np.linspace(0.2, 2, 5)  # (start, stop, num). stop까지 포함.
for scale in scale_list:
    print('scale={:5.2f}'.format(scale))
    imgSat = imgHSV[:, :, 1].astype("float32") * scale
    print('imgSat max=', np.amax(imgSat))

    # unit8 타입에 대해서는 원래 수동 clipping이 필요하다.
    # 아래의 수동 변환 작업(방법1 혹은 방법2)이 필요하다.

    # clipping 방법1
    #imgGE = imgSat >= 256
    #imgLess = imgSat < 256
    #imgNew[:, :, 1] = imgGE * 255 + imgSat * imgLess    # 255 넘는 것(이것은 255로 clipping한다) + 255 넘지 못하는 것.

    # clipping 방법2
    imgNew[:, :, 1] = np.clip(imgSat, 0, 255)

    # 단순기법 (8비트로 절환) ????? -> 실패
    #imgNew[:, :, 1] = imgSat.astype(np.uint8)
    #tmp = imgSat.astype(np.int16)
    #imgNew[:, :, 1] = tmp.astype(np.uint8)

    # clipping 방법3 - 잘못된 방법
    #imgNew[:, :, 1] = imgSat.astype("uint8")   # 만약 이렇게 하면 하위 8비트만 assign 할 것이다. => 오류


    imgDisplay = cv.cvtColor(imgNew, cv.COLOR_HSV2RGB_FULL)
    plt.subplot(231+int(i))
    plt.imshow(imgDisplay)
    str_sft = '{:5.2f}'.format(scale)
    plt.title('scale: *'+str_sft)
    plt.xticks([]), plt.yticks([])
    i +=1
plt.show()
exit()
#=============================================================================================
"""


"""
# 방법 2 ======================================================================================
# 채도만 변화를 주어 다시 역변환해서 보인다.
# 부동소수 버전
# 주의 사항 : cvtColor() 함수의 입력 영상의 부동소수 지원 범위. 0 to 1 for CV_32F images
# 입력 영상이 부동 소수일 때 HSV로 변환되고 난 이후는 H의 범위는 0~360의 범위를 가진다.
#=============================================================================================

img = img.astype("float32") /255

# 원본 영상 출력
plt.figure(num='Original Image & its satuartion adjustments: floating point type', figsize=(14, 8))
imgP = cv2py(img)       # Python RGB 배열
#imgP = cv.cvtColor(img, cv.COLOR_BGR2RGB)       # OpenCV 함수가 있음. float64는 지원안됨.
plt.subplot(231)
plt.imshow(imgP)
plt.title('original')
plt.xticks([]), plt.yticks([])


# 원본 영상의 HSV 좌표 변환. BGR을 HSV 좌표계로 변환한다.
imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV )

# 처리될 영상의 생성
imgNew = imgHSV.copy()
print("imgNew.dtype=", imgNew.dtype)

i=1
scale_list = np.linspace(0.2, 2, 5)  # (start, stop, num). stop까지 포함.
for scale in scale_list:
    print('scale={:5.2f}'.format(scale))
    imgSat = imgHSV[:, :, 1] * scale
    print('imgSat max=', np.amax(imgSat))

    # uint8 타입과는 달리 부동소수에서는 clipping 작업은 적어도 OpenCV에서는 불필요한 것으로 판단되었다.
    # 부동소수에서는 saturation 값이 1이 넘겠지만 HSV -> RGB에 영향을 주지 않았다.
    # 아마도 HSV2RGB에서 RGB에 영향을 주어 Hue, Value에 부작용을 미쳤을 것으로 예상된다.
    
    imgNew[:, :, 1] = imgSat
    imgDisplay = cv.cvtColor(imgNew, cv.COLOR_HSV2RGB)
    plt.subplot(231+int(i))
    plt.imshow(imgDisplay)
    str_sft = '{:5.2f}'.format(scale)
    plt.title('Saturation Shift: *'+str_sft)
    plt.xticks([]), plt.yticks([])
    i += 1
plt.show()
exit()
"""



# 실험 3 ======================================================================================
# S, V의 값을 바꾼 후 다시 역변환해서 출력하면 영상이 어떤 효과가 나타나는지 보인다.
# 부동소수 버전 - 간략해서 학생 학습용으로 추천
# HSV 좌표계에서 채도와 명도 정보를 조정해서 역변환한다.
# 주의 사항 : cvtColor() 함수의 입력 영상의 부동소수 지원 범위. 0 to 1 for CV_32F images
# 입력 영상이 부동 소수일 때 HSV로 변환되고 난 이후는 H의 범위는 0~360의 범위를 가진다.
# =============================================================================================

img = img.astype("float32") / 255
#img = img/255           # float64는 지원안됨.

# 원본 영상 출력
plt.figure(num='Original Image & its SV adjustments: floating point type', figsize=(13, 8))
#imgP = cv2py(img)       # Python RGB 배열
imgP = cv.cvtColor(img, cv.COLOR_BGR2RGB)       # OpenCV 함수가 있음. float64는 지원안됨.
plt.subplot(231)
plt.imshow(imgP)
plt.title('original')
plt.xticks([]), plt.yticks([])


# 원본 영상의 HSV 좌표 변환. BGR을 HSV 좌표계로 변환한다.
imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV )

# 처리될 영상의 생성
imgNew = imgHSV.copy()
#print("imgNew.dtype=", imgNew.dtype)


# 변화시킬 채도와 밸류의 스케일값을 조합으로 리스트로 만든다.
scale_list = [[0.1, 0.1], [0.4, 0.4], [1.5, 1.5], [2.0, 2.0], [3.0, 3.0]]
scale_list = [[0.2, 1.0], [0.4, 1.5], [1.5, 1.5], [2.0, 1.5], [3.0, 1.4]]
for i, (Sscale, Vscale) in enumerate(scale_list, start=1) :
    print(f'{i}: Sscale={Sscale:1.2f}, Vscale={Vscale:1.2f}')
    imgSat = imgHSV[:, :, 1] * Sscale
    imgVal = imgHSV[:, :, 2] * Vscale
    #print('imgSat max={:4.2f}'.format(np.max(np.max(imgSat))))
    print(f'imgSat max={np.max(imgSat):4.2f}')

    # uint8 타입과는 달리 부동소수에서는 clipping 작업은 적어도 OpenCV에서는 불필요한 것으로 판단되었다.
    # 부동소수에서는 SV 값이 1이 넘어도 HSV -> RGB 역변환에는 영향을 주지 않았다.

    imgNew[:, :, 1] = imgSat
    imgNew[:, :, 2] = imgVal
    imgDisplay = cv.cvtColor(imgNew, cv.COLOR_HSV2RGB)
    print('imgDisplay.dtype=', imgDisplay.dtype, np.amax(imgDisplay[:, :, 0]),
          np.amax(imgDisplay[:, :, 1]), np.amax(imgDisplay[:, :, 2]))
    plt.subplot(231 + i)

    # clipping을 안하면 범위초과 경고오류 발생.
    imgDisplay = np.clip(imgDisplay, 0, 1)  # min=0, max=1

    plt.imshow(imgDisplay)
    plt.title(f'S * {Sscale:4.2f}, V * {Vscale:4.2f}')
    plt.xticks([]), plt.yticks([])

plt.show()

