"""
1. 프로그램의 기능
    1차 미분 필터 커널을 정의하여 코릴레이션 공간 필터링을 통해 에지를 검출한다.
    interactive mode로 동작한다.

    edgeV는 가로 방향으로 소벨 미분을 구하는 kerH 커널로,
    edgeH는 세로 방향으로 소벨 미분을 구하는 verV 커널로 2차원 선형 필터링으로 구해서 에지 영상을 만들어 낸다.
    가로 방향의 미분은 세로 방향의 에지를 만들어 내고,
    세로 방향의 ㅁ미분은 가로 방향의 미분을 만들어 낸다.

2. 프로그램의 목표
    1) 에지 검출하는 과정에 수반된 데이터형 관리 문제를 이해한다.
    - 커널에 음수가 있으면 결과 데이터도 음수가 나올 수 있다. 따라서 unit8 형을 쓸 수 없다.
    2) 검출된 에지 결과를 화면에 출력하는 문제에 대해 이해한다.
    - 음수와 양수 부분을 효과적으로 출력하는 방법을 이해한다.

3. 프로그램에서 점검 포인트

주의사항(영상처리 관점)
    1) 필터커널은 np.array()로 정의한다. ()안에 2차원 list 혹은 tuple을 정의하면 된다.
    2) 커널에 음수가 있으면 코릴레이션 결과에 음수가 발생할 수 있다.
        따라서 에지의 연산결과는 signed 정수형 혹은 부동소수형이어야 한다.
        이를 위해 입력 영상을 0~1의 범위의 부동소수형으로 정의하는 것을 추천한다.
    3) 에지의 출력은 음수 값은 128 이하, 양수 값은 128이상으로 처리하는 것이 바람직하다.
        pyplot.imshow()는 이러한 처리를 내재하고 있다.

주의사항(matplotlib.pyplot 관점)
    1) pyplot.imshow()는 1채널 영상에 대해 cmap을 지정하지 않으면 디폴트로 pseudo color로 표현한다.
    2) pyplot.imshow()는 'cmap=gray' 옵션에서는 음수가 포함된 데이터를 화소값 128을 0점으로 하여 구분하여 표시한다.
    3) ndarray 데이터는 copy() 메소드를 지원한다. astype()과 함께 새로운 영상 객체를 복사해 낸다.

4. 미션
    1) 화소값 128을 기준으로 +에지 성분과 - 에지 성분을 cv.imshow() 함수를 사용하여 표현하시오.
        pyplot.imshow()는 'cmap=gray' 옵션에서는 음수 성분이 있을 때 자동화 처리한다.
        openCV에서 이를 흉내내어 처리해 보자는 의도이다.
    2) 에지의 값을 + 값은 붉은 색상의 계조 값으로, - 값은 파란 색상의 계조 색상으로 한 화면에 표시하시오.
        openCV 환경으로 표현하기로 한다.
        1) 번보다 명확하게 음수와 양수 부호를 갖는 에지를 색상으로 표현해 보자는 의도이다.
"""


import numpy as np
import cv2 as cv
# 만약 matplotlib 모듈이 없다면 윈도 컴맨드 창에서 "pip install matplotlib" 명령을 수행한다. 
from matplotlib import pyplot as plt



# ----------------------------------------------------------------------------------------------------------------------
# 영상이 존재하는 폴더와 파일 이름을 지정하기.
# 다음 사례를 활용하여 영상 파일이 있는 폴더를 적절히 지정하시오.
# ----------------------------------------------------------------------------------------------------------------------
#Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
Path = '../data/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 Images 폴더.
#Path = 'd:/work/@@DIP/LectureMaterials/Images/'
#Name = 'monarch.bmp'
#Name = 'graphicShapes.png'
#Name = 'circles_rects.png'
#Name = 'circles_rects_2.png'
Name = 'lenna.tif'
Name = 'circles_rects.png'

FullName = Path + Name
img = cv.imread(FullName)

# assert condition, message  : condition이 false이면  message 출력하면서 AssertError 발생.
assert img is not None, 'No image file....!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.




# ----------------------------------------------------------------------------------------------------------------------
# 단계 0
# cv.imshow()를 사용하면 이 부분은 주석처리해야 한다.
# matplotlib를 이용해 화면에 영상을 출력하려면 이 부분을 활성화 시킨다.
# OpenCV 함수를 이용해 BGR 평면 배열을 RGB 평면 배열로 바꾼다.
# interactive mode로 설정한다. plt.show() 없이 plt.imshow()만으로 출력된다.
# ----------------------------------------------------------------------------------------------------------------------
b, g, r = cv.split(img); img = cv.merge([r,g,b]); #plt.ion();
img_c = img.copy()      # img의 복사본을 하나 저장해 둔다.
plt.figure(num='Input images', figsize=(10, 8))
plt.subplot(121); plt.imshow(img_c)      # 3채널로 출력
plt.title('Original(color)'); plt.xticks([]), plt.yticks([])



# ----------------------------------------------------------------------------------------------------------------------
# 단계 1
# 에지 영상은 흑백 영상으로 관찰하는 것이 바람직하다.
# 입력 영상을 1채널의 흑백 gray 영상으로 변환한다.
# dst=cv.cvtColor(src, code[, dst[, dstCn]]	)
#       여기서, code= cv.COLOR_RGB2GRAY, cv.COLOR_BGR2GRAY. 주의!  적정한 것을 선택.
# ----------------------------------------------------------------------------------------------------------------------
img = cv.cvtColor(src=img, code=cv.COLOR_BGR2GRAY)/255
print("cv2: type(img)=", type(img), " img.dtype=", img.dtype, "img.shape=", img.shape)
plt.subplot(122); plt.imshow(img, cmap = 'gray')
plt.title('gray converted'); plt.xticks([]), plt.yticks([])


plt.waitforbuttonpress()


# ----------------------------------------------------------------------------------------------------------------------
# 단계 2
# 커널의 종류와 그 내부 계수 값을 정의한다.
# ----------------------------------------------------------------------------------------------------------------------

# 로베르츠 에지 연산자
RobertsSlash = [[0, 0, 0], [0, -1, 0], [0, 0, 1]]       # / 방향의 에지를 찾아내는 로베르츠 연산자.
RobertsBackSlash = [[0, 0, 0], [0, 0, -1], [0, 1, 0]]   # \ 방향의 에지를 찾아내는 로베르츠 연산자.

# 프리위트 에지 연산자
#PrewittHorizontal = [1 0 -1; 1 0 -1; 1 0 -1] ;	# 수평방향의 1차 미분
#PrewittVertical = [1 1 1; 0 0 0; -1 -1 -1];	# 수직방향의 1차 미분

# 소벨 에지 연산자(어두운 곳으로 올때 + 에지 발생)
#SobelHorizontal = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]	    # 수평방향의 1차 미분
#SobelVertical = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]] 	    # 수직방향의 1차 미분


# 소벨 에지 연산자(밝은 곳으로 올때 + 에지 발생, OpenCV Sobel() 함수가 사용)
SobelHorizontal = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]	    # 수평방향의 1차 미분
SobelVertical = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]] 	    # 수직방향의 1차 미분


# 프라이첸 에지 연산자
#FriHorizontal = [1 0 -1; sqrt(2) 0 -sqrt(2); 1 0 -1] ;   # 수평방향의 1차 미분
#FriVertical = [1 sqrt(2) 1; 0 0 0; -1 -sqrt(2) -1] ;      # 수직방향의 1차 미분


kerH = np.array(SobelHorizontal, dtype=float )     # print(type(kerH))
kerV = np.array(SobelVertical, dtype=float )       # print(type(kerV))

#커널을 출력하되 원하는 정밀도로 출력하게 한다. >>> help(np.set_printoptions)
np.set_printoptions(precision=1, floatmode='fixed')
print('kerH=\n', kerH)  # 소수점 이하 1째 자리까지 출력.
print('KerV=\n', kerV)

# ----------------------------------------------------------------------------------------------------------------------
# 단계 3
# 2차원 correlation 연산을 수행한다.
# dst= cv.filter2D(	src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]]	)
# ----------------------------------------------------------------------------------------------------------------------
# ddepth=-1은 조심해서 사용해야 한다. 음수가 나올 수 있기 때문에 부동소수 입력일 때만 사용할 수 있다.
edgeV = cv.filter2D(src=img, ddepth=-1, kernel=kerH); print(edgeV.dtype)    # 가로 방향 미분으로 세로 에지가 탄생.
edgeH = cv.filter2D(src=img, ddepth=-1, kernel=kerV); print(edgeH.dtype)    # 세로 방향 미분으로 가로 에지가 탄생.



edge = abs(edgeH) + abs(edgeV)


# ----------------------------------------------------------------------------------------------------------------------
# 단계 4
# 서브화면에 원본과 처리 결과를 출력한다. 이후 마우스를 입력할 때까지 기다린다.
# subplot(NMK)가 의미하는 것 : 화면을 rowXcolumn(NxM)개의 sub 창으로 나누고 K 번째 서브창을 지정한다.
# 이후 imshow() 함수는 지정된 sub 창에 출력한다.
# pyplt.imshow()에서는 흑백 영상은 의사 칼라 영상으로 출력되기 때문에 cmap를 gray로 정해야 한다.
# ----------------------------------------------------------------------------------------------------------------------
plt.figure(num='input and edge images', figsize=(10, 8))
#plt.subplot(221); plt.imshow(img, cmap = 'gray')
#plt.subplot(221); plt.imshow(img_c, cmap = 'gray')
plt.subplot(221); plt.imshow(img_c)      # 3채널로 출력
plt.title('Original'); plt.xticks([]), plt.yticks([])

plt.subplot(222),plt.imshow(edge, cmap='gray')
plt.title('combined edge'); plt.xticks([]), plt.yticks([])


plt.subplot(223),plt.imshow(edgeH, cmap='gray')
plt.title('edgeH'); plt.xticks([]), plt.yticks([])

plt.subplot(224),plt.imshow(edgeV, cmap='gray')
plt.title('edgeV'); plt.xticks([]), plt.yticks([])

plt.tight_layout() # 맨 나중에 써야 영향을 미침.

plt.waitforbuttonpress()        # 현 화면에서 버튼 입력을 기다린다.

# 이후 프로그램이 종료되므로 모든 창이 닫힌다.