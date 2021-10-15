"""

개요
    [*** 입력된 칼라 영상의 R, G, B 칼라 채널의 영상을 해당 색상으로 화면에 출력한다. ***]
    칼라 영상 img는 img.shape=(높이, 넓이, 3). 3차원 어레이이다. => 3 channel
    OpenCV 함수에서는 Blue(ch=0), Green(ch=1), Red(ch=2)의 순서로 배열되는 것으로 간주하여 처리한다.
    ==> 결론적으로; Blue=img[:, :, 0], Green=img[:, :, 1], Red=img[:, :, 2]
    [:, :, 0]에서 :의 의미는 for all이다. 따라서 '첫 번째 :'는 '모든 row에 대해',
        '두 번째 :'는 '모든 colulm에 대해'를 뜻한다.
    [:, :, 0]는 편의상 [..., 0]으로 기술할 수 있다. '...,'은 '기술한 인덱스외 나머지는 모두'의 의미를 가진다고 할 수 있다.
    matplotlib, PIL 등의 함수들은 RGB 순서로 배열된다. 따라서 이들 모듈을 혼용할 때는 주의를 요한다.

    이 소스는 아래와 같이 2개의 실습 예제로 구성된다. -> 배포 버전은 실습 1만 수행합니다.

    실습 1: R, G, B 칼라 채널의 영상을 해당 색상으로 화면에 출력하기
           0으로 초기화된 빈 영상 어레이를 선언한다. 입력영상과 같은 크기에 3채널 빈 어레이가 필요하다.
           이곳에 특정 영상의 칼라 채널을 복사해서 해당 색상으로 그 채널 영상을 보인다.

    실습 2: array1 = array2 명령이 수행되면 단지 포인터만 일치시키는지 아니면,
           두 개의 전용 메모리 공간이 새로 만들어지는지 영상을 통해 확인한다.

질문
    각 단계에서의 질문에 답하시오.

"""
import cv2  as cv
import numpy as np


#===============================================================================
# 실습1, 실습2 공통 단계: 영상이 존재하는 폴더와 파일 이름을 지정하기.
# 질문: 영상 데이터의 여러 속성을 제시하고 그 의미를 설명하시오.
#===============================================================================
Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
#Path = '../../Images/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 Images 폴더.
Path = '../data/'
Name = 'RGBColors.JPG'
#Name= 'colorbar_chart.jpg'
#Name = 'lenna.bmp'
#Name = 'monarch.bmp'
FullName = Path + Name

# 주어진 스텝 번호(step_num)와 스트링 문자열(string)로 주어지 어레이 변수의 속성을 출력한다.
def printImgAtt (step_num, string):
    print("\n"+ f'Step {step_num}:' + string)
    data = eval(string)
    print(' type :', type(data))           # imge type =  <class 'numpy.ndarray'>
    print(' shape = ', data.shape)      # 영상 어레이의 크기 알아내기. image shape =  (512, 768, 3). (행, 열, 채널)
    print(' data type = ', data.dtype) # 어레이 요소의 데이터 타입 알아내기. uint8 = unsigned 8비트.
    if len(data.shape) >=2:
        print(' row = ', data.shape[0])  # shape는 tuple이다. 원소는 []로 지정. 리스트도 []로 지정한다.
        print(' column = ', data.shape[1])  # 열의 수.
        if len(data.shape) ==3:
            print(' channel = ', data.shape[2])  # 채널의 수. 칼라 영상이면 3. 모도 영상이면 1.


# ========================================================================================================
# 실습1, 실습2 공통 단계: 영상 파일을 열어 화면에 출력한다
# ========================================================================================================
# ImreadMode: 영상 데이터의 반환 모드를 결정
#   IMREAD_COLOR = 1            # default. 모노 영상도 3채널(RGB) 영상이 된다.
#   IMREAD_GRAYSCALE = 0        # 칼라 영상도 모노로 변환하여 연다. 1채널 영상이 됨.
#   IMREAD_UNCHANGED = -1       # 있는 그대로 열기.
image = cv.imread(FullName)     # read as 3 channel. default.

# assert condition, message  : condition이 false이면  message 출력하면서 AssertError 발생.
assert image is not None, f'No image file: [{FullName}]....! '  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.

cv.imshow('input image', image)
printImgAtt('공통', 'image')
cv.waitKey()
#cv.destroyWindow('input image')


#"""
# ========================================================================================================
print("실습 1: R, G, B 칼라 채널의 영상을 해당 색상으로 화면에 출력하기")
#            0으로 초기화된 빈 영상 어레이를 선언한다. 입력영상과 같은 크기에 3채널 빈 어레이가 필요하다.
#            이곳에 특정 영상의 칼라 채널을 복사해서 해당 색상으로 그 채널 영상을 보인다.
# ========================================================================================================

# Yellow = Green + Red
# Magenta = Red + Blue
# Cyan = Blue + Green
# 질문 1: 아래 문장이 의미하는 바를 설명하시오.
#           img = np.zeros(image.shape, dtype=np.uint8)
#           img[:, :, 0] = image[:, :, 0]
#           img[:, :, 2] = 0
# 질문 2: 아래 방법 3)이 불가한 이유를 설명하시오. 이를 copy()를 활용하여 해결하시오.
# --------------------------------------------------------------------------------------------------------

# 입력 영상과 같은 크기의 모든 요소의 값이 0으로 초기화된 영상 어레이 img를 선언한다.
#img = np.zeros(image.shape, dtype='uint8')     # 방법 1) 이것도 됨.
img = np.zeros(image.shape, dtype=np.uint8)    # 방법 2) 이것도 됨.
#img = image; img[:,:,:] = 0                     # 방법 3) 이것은 안되는 이유가 무엇인가? image도 같이 0이 된다.

printImgAtt(2, 'img')
cv.imshow('img - at the beginning ', img)
cv.waitKey()

#img[:, :, 0] = image[:, :, 0]       # copy B channel. 이미 0으로 채워져 있므므로 필요없다.
img[..., 0] = image[..., 0]       # B 채널만 0으로 만든다.: ..., 이런 표현도 가능하다. for all의 의미, 나머지만 빼고...
cv.imshow('B', img)
cv.waitKey()

img[:, :, 0] = 0                  # clear B channel
img[:, :, 1] = image[...,1]       # copy G channel
cv.imshow('G', img)
cv.waitKey()

img[:, :, 1] = 0                  # clear G channel
img[..., 2] = image[:, :, 2]       # copy R channel
cv.imshow('R', img)
cv.waitKey()
exit()
#"""

# ========================================================================================================
print("\n실습 2: array1 = array2 명령이 수행되면 두 개의 전용 메모리 공간이 새로 만들어지는지 영상을 통해 확인한다.")
# array1 = array2 명령이 수행되면 단지 포인터만 일치시키는지 아니면, 두 개의 전용 메모리 공간이 새로 만들어지는지 영상을 통해 확인한다.
#
# 잠정 결론: ndarray에 관한 한 명백해 보인다.
#   array1 = array2 명령은 array1 포인터를 array2가 가리키는 데이터 영역을 지정하도록 조정된다.
#   이후 array2=value 동작은 array1도 변경이 일어나는 것과 동일한 영향을 나타낸다.
# ========================================================================================================
# image는 공통 단계에서 사용된 변수이다. 이를 이용하여 같은 크기의 빈 공간 어레이를 생성한다.
img = np.zeros(image.shape, dtype=np.uint8)
# 실습 1과 분리해서, 실습 2만 개별 술행이 가능하도록 실습 1의 최종 단계와 같도록 만든다.
img[..., 2] = image[:, :, 2]
# 윗 동작은 별도의 공간에 대한 수정 작업이다. 따라서 현재 상황에서는 두 공간은 명백히 분리되어 있다.


# 1) img에 대해 assign을 행하고, image가 변하는지 살쳐본다.
# 원래 image가 온전히 보관되는지 다음 중 하나를 선택하여 확인해 보기로 한다. Yes/No
#img[:, :, :] = 0       # 1) Yes: 이미 img와 image는 분리되어 있었다. img는 모두 0으로 채워져 있다.
#img = image             # 2) Yes: img의 데이터 포인터가 image의 데이터 영역을 지정하도록 수정되었을 것이다.
                        # 사실상 본래의 img는 없어지고, img를 새로 정의하는 것이다.
#img = image; img = 0   # 3) Yes: 주의!! img는 ndarray가 아니라 새로 정의된 int 데이터이다.

# 아래 4, 5, 6)에 해당: img는 image가 보유한 데이터를 가리키는 포인터를 갖고 있다.
# 그러므로 img를 바꾸면 image도 바뀐다.
#img = image; img[:] = 0        # 4) No: Blnak.
#img = image; img[:, :, :] = 0  # 5) No: Blnak.
#img = image; img[..., :] = 255     # 6) No: White.


# 2) image에 대해 assign을 행한다. 이후 img에 대해서만 수정을 가한다.
# 원래 image가 온전히 보관되는지 다음 중 하나를 선택하여 확인해 보기로 한다. Yes/No

# 아래 7~10)에 해당: image의 데이터 포인터가 img의 것과 같도록 바뀐다.
# 따라서, image는 img의 데이터 내용을 공유한다.
# image는 img의 데이터 내용을 공유하기 때문에 img의 데이터가 바뀌면 img의 내용도 바뀐다.
#image = img                    # 7) No: Red 영상이 나온다.
#image = img[:]                 # 8) No. Red 영상이 나온다.
#image = img; img[..., :] = 255     # 9) No. White 영상이 나온다.
#image = img[:]; img[..., 0] = 255  # 10) No. 파란색 바탕에 Magenta(R+B) 영상이 나온다.
cv.imshow('image, the original', image)
cv.waitKey()


#===============================================================================
# 질문
# 아래 동작이 copy를 실행하여 독립적인 변수로 동작하는지 아니면
# 단지 포인터만 일치시키고, 두 변수의 데이터는 공유하는지 확인하는 방법을 제시하시오.
# img[:, :, 0] = image[:, :, 0]
# 예제 프로그램을 만들어 channel 0에 대한 복사가 시행되었는지,
# 아니면 같은 데이터의 내용을 공유하는지 설명
#===============================================================================



