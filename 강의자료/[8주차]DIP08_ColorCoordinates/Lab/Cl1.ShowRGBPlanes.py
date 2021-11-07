"""
1. 프로그램의 기능
    입력 영상의 색상 구성 성분인 RGB 정보를 각 채널의 해당 색상으로 표현한다.
    이를 3가지 방법을 사용하여 구현한다.

2. 프로그램의 목표
    영상 정보에서 특성 색상 정보를 추출할 수 있다.

3. 점검 포인트
    img2[:,:,i] = img[:,:,i]
    # img의 i번째 채널의 모든 데이터를 img2의 해당 채널에 복사한다.

4. 미션


공통 주의 사항
    matplotlib를 이용해 화면에 영상을 출력하려면 RGB 평면 배열을 사용해야 한다.
    OpenCV 함수를 이용해 BGR 평면 배열을 RGB 평면 배열을 사용해야 한다.

"""


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt



#=============================================================================================
# 단계 0
# 영상이 존재하는 폴더와 파일 이름을 지정하여 영상 파일 읽어 내기
# 다음 사례를 활용하여 영상 파일이 있는 폴더를 적절히 지정하시오.
#Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
Path = '../data/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 Images 폴더.
Name = 'lenna.tif'
#Name = 'barns.jpg'
#Name = 'monarch.bmp'
#Name = 'fruits.jpg'
#Name = 'hats.bmp'
#Name = 'bikes.bmp'
#Name = 'parrots.bmp'
#Name = 'Macbeth_NoColorProfile.jpg'
#Name = 'colorbar_chart.jpg'
#Name = 'RGB.jpg'
Name = 'RGBcolors.jpg'
#Name = 'YCM.png'

FullName = Path + Name

#===============================================================================
# print("영상 파일을 열어 화면에 출력하기")
#===============================================================================
ImreadMode = cv.IMREAD_COLOR       # 영상 데이터의 반환 모드를 결정
#   IMREAD_COLOR = 1            # default. 생략가능. 모노 영상도 3채널(RGB) 영상이 된다.
#   IMREAD_GRAYSCALE = 0        # 칼라 영상도 모노로 변환하여 연다. 1채널 영상이 됨.
#   IMREAD_UNCHANGED = -1       # 있는 그대로 열기.
img = cv.imread(FullName, ImreadMode)      # OpenCV에서는 BGR 배열임
# assert condition, message  : condition이 false이면  message 출력하면서 AssertError 발생.
assert img is not None, 'No image file....!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.

# 입력 파일이 지나치게 크면 크기를 조정하시오.
#img = cv.resize(img, dsize=(0, 0), fx=1/2, fy=1/2, interpolation=cv.INTER_LINEAR)


#"""
# 방법 1 ======================================================================================
# 가장 손쉬운 방법
# print("RGB 3개 영상(imgR, imgG, imgB)을 미리 정의하여 두고 화면에 출력한다.")

b, g, r = cv.split(img);
img = cv.merge([r, g, b]);    # 영상을 RGB 순으로 배열한다.
imgBlank = np.zeros(img.shape, dtype='uint8')   # 원영상과 같은 크기의 빈 영상을 준비한다.

# 해당 되는 채널의 화면 정보를 화면 출력을 위한 어레이에 복사한다.
imgR = imgBlank.copy(); imgR[:, :, 0] = img[:, :, 0]    # r
imgG = imgBlank.copy(); imgG[:, :, 1] = img[:, :, 1]    # g
imgB = imgBlank.copy(); imgB[:, :, 2] = img[:, :, 2]    # b


plt.figure(num='Original Image & their color components', figsize=(14,8))
plt.subplot(221); plt.imshow(img)      # 입력받은 대로 출력
plt.title('1) Original', fontsize='x-large')
plt.axis('off')             # plt.xticks([]), plt.yticks([])

ii = 0
for color, array in [ ['2) Red', imgR], ['3) Green', imgG], ['4) Blue', imgB]]:
    plt.subplot(220+ii+2)
    plt.imshow(array)
    plt.axis('off')             # plt.xticks([]), plt.yticks([])
    plt.title(color, fontsize='x-large')
    ii += 1

plt.show()
#exit()
#=============================================================================================
#"""



"""
# 방법 2 ======================================================================================
# print("방법 1과는 달리 RGB 3개 평면 어레이(imgR, imgG, imgB)를 만들지 않는다.")

b, g, r = cv.split(img); img = cv.merge([r, g, b]);

# interactive mode로 설정한다. plt.show() 없이 plt.imshow()만으로 출력된다.
# plt.ion();

imgBlank = np.zeros(img.shape, dtype='uint8')

plt.figure(num='Original Image & their color components')
plt.subplot(221); plt.imshow(img)      # 입력받은 대로 출력
plt.title('Original(color)')
plt.axis('off')                 # plt.xticks([]), plt.yticks([])

ii=0
for color in ['Red', 'Green', 'Blue']:
    imgShow = imgBlank.copy()
    imgShow[:, :, ii] = img[:, :, ii]
    plt.subplot(220+ii+2)
    plt.imshow(imgShow)
    plt.axis('off')             # plt.xticks([]), plt.yticks([])
    plt.title(color)
    ii += 1

plt.show()
exit(0)
#=============================================================================================
"""



# 방법 3 ======================================================================================
# 방법 1과는 달리 RGB 3개 평면 어레이(imgR, imgG, imgB)를 만들지 않는다.
# print("RGB 분리된 영상을 기록용으로 저장한다.)

imgBlank = np.zeros(img.shape, dtype='uint8')

plt.figure(num='Original Image & their color planes')

#plt.subplot(221); plt.imshow(img)      # 오류! BGR 영상이라 올바로 출력되지 않는다.

# ['Color Title', channel_num_RGB, channel_num_BGR]
for color, i, ii in [['Red', 0, 2] , ['Green', 1, 1], ['Blue', 2, 0]]:
    imgShow = imgBlank.copy()
    imgShow[:, :, i] = img[:, :, ii]
    plt.subplot(220+i+2)        # 화면 번호 2번부터 R, G, B 순번.
    plt.imshow(imgShow)
    plt.axis('off')  # plt.xticks([]), plt.yticks([])
    plt.title(color)
    cv.imwrite(color+'.jpg', imgShow, (cv.IMWRITE_JPEG_QUALITY, 97) )   # 품질 97(1~100)로 저장

imgShow[:, :, 1] = img[:, :, 1]     # imgShow의 Green에는 OpenCV 영상 img의 1번채널
imgShow[:, :, 0] = img[:, :, 2]     # imgShow의 Red에는 OpenCV 영상 img의 2번채널. Blue에는 이미 데이터가 들어가 있음.
plt.subplot(221); plt.imshow(imgShow)      # 1번 화면에는 원본 영상을 보인다.
plt.title('Original')
plt.axis('off')                 # plt.xticks([]), plt.yticks([])

# OpenCV로 파일에 저장할 때는 BGR 순서이므로 imgShow 영상을 저장하면 안된다.
cv.imwrite('rgb.jpg', img, (cv.IMWRITE_JPEG_QUALITY, 97) )   # 읽은 그대로 저장(img). 품질 97(1~100)로 저장
plt.show()
exit()
#=============================================================================================


