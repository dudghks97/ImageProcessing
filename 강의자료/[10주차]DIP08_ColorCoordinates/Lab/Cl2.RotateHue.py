"""
참고 URL:
    Changing Colorspaces
    https://docs.opencv.org/3.4.3/df/d9d/tutorial_py_colorspaces.html

    cvtColor
    https://docs.opencv.org/3.4.3/d7/d1b/group__imgproc__misc.html#ga8e873314e72a1a6c0252375538fbf753

    Color conversions
    https://docs.opencv.org/3.4.3/de/d25/imgproc_color_conversions.html

    ColorConversionCodes
    https://docs.opencv.org/3.4.3/d7/d1b/group__imgproc__misc.html#ga4e0972be5de079fed4e3a10e24ef5ef0

dst	=cv.cvtColor(src, code[, dstCn])
    src:	input image: 8-bit unsigned, 16-bit unsigned ( CV_16UC... ),
            or single-precision floating-point.
    dst:	output image of the same size and depth as src.
    code:	color space conversion code (see ColorConversionCodes).
    dstCn:	number of channels in the destination image;
            if the parameter is 0, the number of the channels is derived automatically 
            from src and code.

0. 실습의 내용
    실습 1a: uint8 정수 버전- HSV중 H는 0~179의 범위에 있다.
    실습 1b: uint8 정수 버전- HSV중 H는 0~255의 범위에 있다. 이른바 FULL
    실습 2: 0~1의 CV_32F 부동소수 버전

1. 프로그램의 기능 - 색조를 다단계로 변화(즉, Hue를 회전)하여 출력한다.
    색조, Hue를 현재의 값에 60도씩 증가시켜 색조의 변화를 관찰한다.

2. 프로그램의 목표
    1) 색조 데이터의 표현 방법[각도]의 개념을 이해한다.
    2) cvtColor() 함수의 입력 데이터형에 따른 HSV 데이터의 처리 방법을 이해한다.

3. 프로그램에서 점검 포인트
    1) 색조 데이터의 특정 각도(예를 들어 60도) 회전시키는 방법
    2) 영상데이터의 데이터 형에 따른 Hue의 범위
    3) 색조 데이터가 360도가 넘었을 경우의 처리 방법 - 영상 데이터의 데이터 형에 따른다.

4. 미션
    1) -5도 ~ +5도 범위의 색조 화소는 그대로 남겨두고 나머지 화소는 흑백으로 표현하시오.
    2) 특정 화소를 클릭하면 이 색조의 -5도 ~ +5도 범위의 색조 화소는 그대로 남겨두고 나머지 화소는 흑백으로 표현하시오.
    3) 특정 화소를 2개이상 선정가능하도록 설계하시오.

5. 질문
    1) 영상 데이터가 0~1의 부동소수로 구성되어 있다. Hue를 반시계 방향으로 30도 회전하려 위해 HSV 데이터 중
       H에 변화를 어떻게 주어야 하는지 설명하시오.
    2) uint8 영상 데이터를 COLOR_BGR2HSV로 HSV 좌표계로 변환하였다.
       Hue를 120도 시계 방향으로 회전하고자 한다.
       HSV 좌표계의 어떤 채널을 어떤 방식으로 가공해야 할지 설명하시오.
    3) uint8 영상 데이터를 COLOR_BGR2HSV_FULL로 HSV 좌표계로 변환하였다.
       Hue를 120도 시계 방향으로 회전하고자 한다.
       HSV 좌표계의 어떤 채널을 어떤 방식으로 가공해야 할지 설명하시오.


6. 실험 결과의 정리 : cvtColor()을 이용한 RGB ↔ HSV 변환에 대하여
    1) 입력 RGB 데이터가 0~1의 부동소수이면 HSV 결과도 부동소수이다.
        => 사실은 부동소수는 정규화된 입력 값을 사용하기로 가정되어 있다.
        예: HSV = cv.cvtColor(img/255, cv.COLOR_BGR2HSV)
        S와 V는 모두 0~1의 범위를 갖는데 반해, H는 0~360(도) 값의 범위를 갖는다.
        H를 회전하여 각도를 더한 값이 360도를 넘으면 OpenCV에서 자동으로 360도를 뺀 값으로 절환하는 기능이 작동한다.
    2) 입력 RGB 데이터가 uint8로 되어 있으면 변환 과정을 거치면 HSV 모두 uint8로 된다.
        2a) RGB2HSV, BGR2HSV 등 _FULL이 붙지 않은 변환에서는 H는 0~179로 스케일링된다.
            Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255].
            H를 회전하여 각도를 더한 값이 180을 넘으면 수동으로 180을 뺀 값을 연산해서 사용해야 한다.
        2b) 아래와 같이 _FULL이 붙은 변환에서는 H는 0~255의 값의 범위를 가진다.
            H를 회전하여 각도를 더한 값이 255를 넘어도 uint8 변수의 특성상 상위 비트는 데이터는 사라지고
            하위 8비트 값만 남는다. 따라서 특별한 처리가 필요없는 장점이 있다.
            "_FULL" suffix has a larger H range
  cv::COLOR_BGR2HSV_FULL = 66,
  cv::COLOR_RGB2HSV_FULL = 67,
  cv::COLOR_BGR2HLS_FULL = 68,
  cv::COLOR_RGB2HLS_FULL = 69,
  cv::COLOR_HSV2BGR_FULL = 70,
  cv::COLOR_HSV2RGB_FULL = 71,
  cv::COLOR_HLS2BGR_FULL = 72,
  cv::COLOR_HLS2RGB_FULL = 73,

"""


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#=============================================================================================
# 영상이 존재하는 폴더와 파일 이름을 지정하여 영상 파일 읽어 내기
# 다음 사례를 활용하여 영상 파일이 있는 폴더를 적절히 지정하시오.
#Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
Path = '../data/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 Images 폴더.
Name = 'lenna.tif'
#Name = 'monarch.bmp'
#Name = 'fruits.jpg'

#Name = 'over_exposure.jpg'
#Name = 'bukak.jpg'
Name = 'Macbeth_NoColorProfile.jpg'
#Name = 'colorbar_chart.jpg'
#Name = 'RGB.jpg'
Name = 'red_rose.jpg'        # 추천
Name = 'RGBcolors.jpg'      # 추천
#Name='Hue_color_wheel_by_degree.png'

FullName = Path + Name
img = cv.imread(FullName)

# assert condition, message  : condition이 false이면  message 출력하면서 AssertError 발생.
assert img is not None, 'No image file....!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.


"""
# 실습 1a ======================================================================================
# uint8 정수 버전- HSV중 H는 0~179의 범위에 있다.

plt.figure(num='[0<Hue<179, by cv.COLOR_BGR2HSV, uint8] Original & Color Shift')
print('1a: [0<Hue<179, by cv.COLOR_BGR2HSV, uint8] Original & Color Shift')

#plt.subplot(231); plt.imshow(imgP)      # 입력받은 대로 출력
#plt.title('Original(color)'); plt.xticks([]), plt.yticks([])


# 원본 영상의 HSV 좌표 변환. BGR을 HSV 좌표계로 변환한다.
imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV )
#imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL )
#imgHSV = cv.cvtColor(imgP, cv.COLOR_RGB2HSV )
print('imgHSV.dtype=', imgHSV.dtype)
print('Hue: min & max=', np.amin(imgHSV[:, :, 0]), np.amax(imgHSV[:, :, 0]))     # 제일 작은/큰 Hue 값
print('Saturation: min & max=', np.amin(imgHSV[:, :, 1]), np.amax(imgHSV[:, :, 1]))
print('Value: min & max=', np.amin(imgHSV[:, :, 2]), np.amax(imgHSV[:, :, 2]))

# 처리될 영상의 생성
imgNew = imgHSV.copy()

i = 0
#for hue_shift in range(0, 720, 120):   # 0~720까지 120도씩 증가. 0~180이 넘는 Hue에대해 자동 절삭이 일어나는지 살핀다.
for hue_shift in range(0, 360, 60):     # 0~360까지 60도씩 증가
    shift_value = int((hue_shift/360) * 180)
    print(f'hue_shift in degree={hue_shift}, shift_value={shift_value}/180')
    imgHue = imgHSV[:, :, 0].astype("int16") + shift_value

    # 부동소수 형에 대해서는 Hue의 자동 클리핑이 일어나지만, unit8 타입에 대해서는 수동 절환 작업이 필요하다.
    # 180 이상이 되는 값에 대해서는 180을 감산한다. 즉, 360도가 넘게 회전하면 다시 0도부터 시작해야 한다.
    # 0~180 범위를 가지는 unit8의 Hue 데이터에 대해 Octave, OpenCv-Python에서는 자동 교정이 없다.
    # 아래 변환식은 2바퀴 회전할 때까지만 유효하다. 2바퀴를 넘어서 회전하면 또 180을 빼는 과정을 적용해야 한다.
    imgGE = imgHue >= 180       # 180을 넘지 않으면 해당 원소 값은 0이 되고, 넘으면 1이 된다.
    imgHue = (imgHue - imgGE.astype('int16') * 180).astype('uint8')
    imgNew[:, :, 0] = imgHue    # 16비트 값을 8비트 변수 어레이에 배정한다.
    imgDisplay = cv.cvtColor(imgNew, cv.COLOR_HSV2RGB)
    plt.subplot(231+int(i))
    plt.imshow(imgDisplay)
    plt.title('Hue Shift: +'+str(hue_shift)+", value="+str(shift_value))
    plt.xticks([]), plt.yticks([])
    i +=1
"""


# 실습 1b ======================================================================================
# uint8 정수 버전- HSV중 H는 0~255의 범위에 있다. 이른바 FULL
plt.figure(num='[0<Hue<255, by cv.COLOR_BGR2HSV_FULL, uint8] Original & Color Shift', figsize=(14, 8))
print('1b: [0<Hue<255, by cv.COLOR_BGR2HSV_FULL, uint8] Original & Color Shift')

# 원본 영상의 HSV 좌표 변환. BGR을 HSV 좌표계로 변환한다.
imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)
print('imgHSV.dtype=', imgHSV.dtype)
print('min & max:', np.amin(imgHSV[:, :, 0]), np.amax(imgHSV[:, :, 0]))     # 제일 작은/큰 Hue 값

# 처리될 영상의 생성
imgNew = imgHSV.copy()

i=0
#for hue_shift in range(0, 720, 120):     # 0~720까지 120도씩 증가. 0~255가 넘는 Hue에대해 자동 clipping이 일어나는지 살핀다.
for hue_shift in range(0, 360, 60):     # 0~360까지 60도씩 증가
    shift_value = int((hue_shift/360) * 255)
    print(f'hue_shift in degree={hue_shift}, shift_value={shift_value}/255')
    imgHue = imgHSV[:, :, 0].astype("int16") + shift_value

    # 부동소수 형에 대해서는 Hue의 자동 클리핑이 일어나지만, unit8 타입에 대해서는 원래 수동 절환작업이 필요하다.
    # 0~255 범위를 가지는 unit8의 Hue 데이터에 대해서 OpenCv-Python는
    # 그 범위를 넘어서는 데이터에 대해 1바이트의 범위를 넘어서는 데이터에 대해 하위 8비트만 취하게 된다.
    # 이러한 동작은 265 => 1로 받아들이는데 이는 Hue 데이터 해석에 올바른 결과를 미친다.
    # 따라서 아래의 수동 절환 작업이 필요없다.
    # 이는 C++처럼 16비트 데이터를 8비트 데이터 형의 변수에 assign 할 때와 같은 동작이다.
    #imgGE = imgHue >= 255
    #imgHue = (imgHue - imgGE.astype('int16') * 255).astype('uint8')


    imgNew[:, :, 0] = imgHue        # 16비트를 8비트로 절환하였다.
    imgDisplay = cv.cvtColor(imgNew, cv.COLOR_HSV2RGB_FULL)
    plt.subplot(231+int(i))
    plt.imshow(imgDisplay)
    plt.title('Hue Shift: +'+str(hue_shift)+", value="+str(shift_value))
    plt.xticks([]), plt.yticks([])
    i += 1

print('imgNew.dtype=', imgNew.dtype)
plt.show()
exit()
#"""




# 실습 2 ======================================================================================
# 0~1의 CV_32F 부동소수 버전
# cvtColor() 함수의 입력 영상의 부동소수 지원 범위: 0 to 1 for CV_32F images
# 입력 영상이 부동 소수일 때 HSV로 변환되고 난 이후는 H의 범위는 0~360의 범위를 가진다.

# 32비트 단정도 0~1 범위의 영상 데이터로 바꾼다.
img = img.astype("float32") / 255
#img = img/255        # <=== 이것은 안됩니다. 왜냐하면 32비트 단정도 부동소수가 필요하기 때문이다.

plt.figure(num='2: [0<Hue<1.0, by cv.COLOR_BGR2HSV, float] Original & Color Shift', figsize=(10, 8))

# 원본 영상의 HSV 좌표 변환. BGR을 HSV 좌표계로 변환한다.
imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV )        # H의 범위는 0~360이다.
print('imgHSV.dtype=', imgHSV.dtype)
print('min & max:', np.amin(imgHSV[:, :, 0]), np.amax(imgHSV[:, :, 0]))     # 제일 작은/큰 Hue 값

# 처리될 영상의 생성
imgNew = imgHSV.copy()

i = 0
#for hue_shift in range(0, 720, 120):     # 0~720까지 120도씩 증가
for hue_shift in range(0, 360, 60):     # 0~360까지 60도씩 증가
    print(f'hue_shift in degree={hue_shift:5.2f}')
    #print('hue_shift={:5.2f}'.format(hue_shift))

    # 부동 소수를 변환하면 이미 Hue는 360도 단위를 사용하므로 바로 더해도 된다.
    # 데이터 절환 작업은 OpenCV에서 지원하는 듯 하다.
    imgHue = imgHSV[:, :, 0] + hue_shift

    imgNew[:, :, 0] = imgHue
    imgDisplay = cv.cvtColor(imgNew, cv.COLOR_HSV2RGB)
    plt.subplot(231 + int(i))
    plt.imshow(imgDisplay)
    str_deg = '{:5.2f}'.format(hue_shift)
    plt.title('Hue Shift: +' + str_deg )
    plt.xticks([]), plt.yticks([])
    i += 1
plt.show()

