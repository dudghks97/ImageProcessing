"""

개요
    OpencCV 함수 기반으로 영상 파일을 읽어들여 화면에 출력한다.
    함수 용법에 대한 참조 링크 https://docs.opencv.org/
    영상 파일을 읽는 imread(파일명, flags)는 flags에 따라 파일을 읽어내는 영상 어레이를 반환하는 방식이 달라진다.
        flags=-1: 있는 그대로.. 모노 그레이면 1채널, 칼라영상이면 3채널[BGR]을 반환한다.
        flags=0: 원래 영상에 관계 없이 모노 그레이, 1채널로 변환하여 반환한다.
        flags=1: 디폴트. 칼라 영상 뿐만 아니라 모노 그레이 영상도 3채널[BGR] 영상으로 변환하여 반환한다.

동작
    1. 디폴드 모드(flags=1)인 3채널 영상 파일로 읽어 영상을 화면에 출력한다.
    2. 모노 그레이모드(flags=0)  1채널 영상 파일을 읽어 영상을 화면에 출력한다. 저장한다.
    3. 저장된 영상을 읽어 모노(1채널)인지 칼라 영상(3채널)인지 확인한다.

중요 함수

1. 영상 파일을 읽기
    retval = cv.imread(filename[, flags]	)
        본 함수를 통해 지정된 영상을 읽어내어 영상 데이터를 numpy.ndarray 객체로 반환한다.
        영상 파일("abc.jpg") 읽어 영상 어레이(img)에 반환하기: => img=cv.imread("abc.jpg", 열기_모드)
            https://docs.opencv.org/4.1.1/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
        filename = 영상 파일의 이름. 거의 모든 파일 양식(bmp. tif, jpg, png 등)을 지원. path도 함께 지정가능.
        flags = 읽어 내는 모드. 다음 3가지 중 하나.
        IMREAD_COLOR = 1            default. 칼라영상을 row x column x channel의 3차원 ndarrary로 읽어 반환한다.
                                    칼라 채널 배열 순서는 B, G, R 순이다.
                                    모노 영상도 3채널(BGR) 영상으로 확장하여 읽는다. 이 경우 각 채널에는 같은 값이 들어 있다.
        IMREAD_UNCHANGED = -1       있는 그대로 열기.
        retval = numpy.ndarray 타입의 영상 데이터 객체.

2. 영상을 화면에 보이기
    None = cv.namedWindow(winname[, flags])
           winname(Name of window) 창을 생성하여 화면에 보인다. 아직 내용물(영상)은 안보임.
           필요에 따라 생략해도 상관없다.
    None = cv.imshow(winname, array)              # array 영상을 winname 창에 출력한다.
        영상 어레이(img)를 지정된 이름("win")의 타이틀에 출력하기: imshow("win", img)
        아래 함수들은 cv.waitKey() 함수가 있어야 화면에 내용물을 출력한다.
            열려져 있는 winname 창이 없으면 생성한 후 영상을 출력한다.
            winname = Name of window
            array = Image to be shown
    waitKey() = waitKey(0) or waitKey(500)
        키입력을 무한(0) 혹은 지정된 시간(500[ms])동안 기다려서 입력한 키를 반환한다.

3. 화면 창을 닫기
    destroyWindow("abc")
        타이틀 이름("abc")으로 지정된 창을 닫는다.
    destroyAllWindows()
        열려진 모든 창을 닫는다. 실행하지 않아도 종료시에는 모든 창이 자동으로 닫힌다.

질문
    1. 영상 파일을 읽을 때 칼라 파일을 1채널의 모노 파일로 읽어 들이는 방법은?
    2. 1채널의 모노 영상을 3채널의 칼라 데이터로 읽어 들이는 방법은?

"""

import cv2 as cv


# ----------------------------------------------------------------------------------------------------------------------
# 섹션 1, 정의 부분: 영상이 존재하는 폴더와 파일 이름을 지정하기.
# ----------------------------------------------------------------------------------------------------------------------
#Path = 'd:\\Work\\StudyImages\\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
#Path = 'd:/Work/StudyImages/Images/'
Path = '../data/'       # . 현재 폴더, .. 바로 윗 폴더, ../data/ => 현재 폴더 바로 위 폴더 아래에 있는 data 폴더 아래
Name = 'lenna_mono.jpg'     # mono gray 영상. 1채널 영상
Name = 'RGBColors.JPG'
Name = 'colorbar_chart.jpg'
Name = 'lenna.tiff'
Name = 'smooth.jpg'         # mono gray 영상. 1채널 영상
Name = 'monarch.bmp'
FullName = Path + Name
print("프로그램 시작.. 파일과 폴더의 이름은 한글로 쓸 수 없습니다.")
print("파일의 경로가 포함된 파일이름:", FullName)


# ----------------------------------------------------------------------------------------------------------------------
# 섹션 2: numpy.ndarray 데이터의 상세 내용을 출력하는 함수를 정의한다.
# 데이터 변수 객체를 문자열로 입력한다.
# numpy.ndarray
#   https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
# ----------------------------------------------------------------------------------------------------------------------
def printImgAtt(string):
    print("\n" + string)
    data = eval(string)             # 전달받은 스트링 문자열을 프로그램 소스의 변수에 반영한다.
    print(' type :', type(data))           # imge type =  <class 'numpy.ndarray'>
    print(' shape = ', data.shape)      # 영상 어레이의 크기 알아내기. image shape =  (512, 768, 3). (행, 열, 채널)
    print(' data type = ', data.dtype) # 어레이 요소의 데이터 타입 알아내기. uint8 = unsigned 8비트.
    if len(data.shape) >= 2:
        print(' row = ', data.shape[0])  # shape는 tuple이다. 원소는 []로 지정. 리스트도 []로 지정한다.
        print(' column = ', data.shape[1])  # 열의 수.
        if len(data.shape) == 3:
            print(' channel = ', data.shape[2])  # 채널의 수. 칼라 영상이면 3. 모도 영상이면 1.
    print(' itemsize = ', data.itemsize)    # 한 개의 원소를 이루는데 필요한 바이트의 수



# ----------------------------------------------------------------------------------------------------------------------
# 단계 1: 영상 파일의 영상 읽기.
# cv.imread(파일, flags)에서; -1=있는 그대로 읽기. 0=모노(1채널)로 읽기, 1=칼라(3채널)로 읽기(default)
# ----------------------------------------------------------------------------------------------------------------------
read_mode = cv.IMREAD_COLOR         # 파일 읽어내는 모드 = 칼라 = 1. default
imgC = cv.imread(FullName, flags=read_mode)         # 칼라모드로 읽어낸다.

# assert condition, message  : condition이 false이면 message 출력하면서 AssertError 발생.
# assert(가정 설정문)은 어떤 조건이 True임을 보증한다. True가 아니면 Assert exception이 발생하면서 중지한다.
# 현재 읽은 영상 어레이가 None 아님을 보증한다. 입력 영상을 제대로 읽어오지 못하여 None을 반환한다.
assert imgC is not None, f'No image file [{Name}] ....!' # 현재 None 아님을 보증한다.

printImgAtt('imgC')
# imgC
#  type : <class 'numpy.ndarray'>
#  shape =  (512, 768, 3)
#  data type =  uint8
#  row =  512
#  column =  768
#  channel =  3

# ----------------------------------------------------------------------------------------------------------------------
# 단계 2: 영상 데이터 화면에 출력하기.
# ----------------------------------------------------------------------------------------------------------------------

winname_img ="imgC: ImReadMode=" + str(cv.IMREAD_COLOR)     # =1. default. BGR color. 3 channel.
cv.namedWindow(winname_img)
key = cv.waitKey(0)       # 이곳에서 키보드 입력을 무한정 기다린다.
print(f"The 1st key={key:x}, chr='{chr(key)}'")

cv.imshow(winname_img, imgC)
key = cv.waitKey(0)       # 이곳에서 키보드 입력을 무한정 기다린다.
print(f"The 2nd key={key:x}, chr='{chr(key)}'")


# ----------------------------------------------------------------------------------------------------------------------
# 단계 4: 영상 파일의 영상 읽기. 다음 함수를 통해 지정된 영상을 읽어내어 영상 데이터를 numpy.ndarray 객체로 반환한다.

imgM = cv.imread(FullName, 0)      # IMREAD_GRAYSCALE = 0. Read in mono.

printImgAtt('imgM')
# imgM
#  type : <class 'numpy.ndarray'>
#  shape =  (512, 768)
#  data type =  uint8
#  row =  512
#  column =  768
winname_imgM = "'imgM': ImReadMode=" + str(cv.IMREAD_GRAYSCALE)  # =0. mono gray. 1 channel.
cv.imshow(winname_imgM, imgM)
key = cv.waitKey(0)       # 이곳에서 키보드 입력을 무한정 기다린다.
print(f"The 3rd key={key:x}, chr='{chr(key)}'")



cv.destroyWindow(winname_imgM)     # 지정한 이름의 창을 닫는다.

key = cv.waitKey(0)       # 이곳에서 키보드 입력을 무한정 기다린다.
print(f"The 4th key={key:x}, chr='{chr(key)}'")

cv.destroyAllWindows()      # 모든 창을 닫는다. 수행 안해도 종료시에는 모두 닫는다.


#====================================================================================================================
# 미션 과제(1)
# 변수 fname로 주어진 영상 파일에 대해 주어진 미션을 완수히시오.
# fname = "../data/colorbas_chart.jpg"      # 단, 파일의 실제 이름은 바뀔 수도 있음.
# 0. fname으로 지정된 파일을 화면에 출력하시오.. 종료
# 1. 가로(column, 열) 및 세로(row, 행) 정보를 변수 a, b에 넣으시오.
#     a.shape
#    이를 이용하여 영상의 가로, 세로 정보를 출력하시오. => print(f'가로={a}, 세로={b}')
# 2. 영상의 면적을 출력하시오. => 면적 = 23674834
# 3. 파일이 칼라인지 모노 영상인지를 검사하여 그 결과를 출력하시오. => 영상은 '칼라 혹은 모노'입니다.
# 4. 해당 영상을 화면에 출력하시오. 이때 파일 이름이 타이틀 바에 출력되어야 합니다. 경로(path)를 제외하면 가점
#====================================================================================================================



