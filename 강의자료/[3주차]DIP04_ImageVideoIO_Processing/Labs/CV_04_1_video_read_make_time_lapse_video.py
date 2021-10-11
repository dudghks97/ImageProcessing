"""
1. 개요
    동영상 파일을 읽어 화면에 재생하고 이를 다른 이름으로 비디오 파일의 속성(fps)를 바꾸어 저장한다.
    정상적인 속도로 촬영된 이름(파일 명: FullName)의 비디오 파일을 2배 저속/고속 재생하는 동영상을 만든다.
    이 프로그램을 수행하기 위해서는 현 폴더의 상위에 있는 data 폴더 밑에 지정된 입력 동영상이 있어야 한다.
    생성된 재생될 파일의 이름:  원본 영상의 앞에 'tmp_'가 붙는다. 단계 4에서 정한다.
    생성하는 동영상의 fps와는 상관없이 화면에는 정상속도와 가깝게 처리중에도 재생된다.
        - 너무 영상이 크거나, 처리 내용이 복잡하다면 원본의 속도로 재생할 수 없을 것이다.
    방법: 지정한 비디오 파일의 속성 중 fps 정보를 1/2배 혹은 2배로 조정하여 저장한다.

2. 동작
    단계 1 : 읽기용의 비디오 객체를 생성한다.
    단계 2 : 읽을 비디오 파일의 속성을 출력한다.
    단계 3 : 저장할 비디오 파일 코덱을 지정한다.
    단계 4 : 쓰기용의 비디오 객체를 생성한다. 이때 fps를 입력 비디오 영상의 2배 혹은 1/2배로 지정한다.
    단계 5 : 첫 번째 프레임을 읽어보고 영상의 속성을 출력한다.
    단계 6 : 직전의 읽기 동작이 성공하였다면 화면에 영상 데이터를 출력하면서 저장한다.

3. 함수
    videoCapture=cv.VideoCapture(): 비디오 객체(읽기용)를 생성한다.
    cv.VideoCapture.get(): 읽을 비디오 파일(비디오 객체)의 속성을 출력한다.
    cv.VideoWriter_fourcc(): 쓰기 용의 비디오 영상 코덱을 지정한다.
    쓰기용 객체=cv.VideoWriter(): 쓰기용의 비디오 객체를 생성한다.
    frame=videoCapture.read(): 한 프레임의 영상을 반환한다.

4. 영상의 주요 정보을 알아내거나 설정하는 함수와 파라미터: videoCapture.get(), videoCapture.set()
    CAP_PROP_FPS - frame per second
    CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT - 영상의 가로, 세로 정보
    CAP_PROP_FOURCC - 4문자로 이루어진 CODEC.비디오 압축 및 해제 알고리즘
    CAP_PROP_FRAME_COUNT - 파일 안에 담긴 총 프레임 수

5. 질문
    영상을 1.5배 밝게 하여 출력하거나, 저장하고자 하는데 아래 # 주석문의 방법은 왜 오류가 발생할까?

6. 미션
    위 오류 사항을 해결하여 프로그램을 완성하시오..
"""

import cv2 as cv
import time
import numpy as np

# numpy.ndarray 데이터의 상세 내용을 출력하는 함수
# 데이터 변수를 문자열로 입력한다.
def printImgAtt (string):
    print("\n" + string)
    data = eval(string)
    print(' type :', type(data))           # imge type =  <class 'numpy.ndarray'>
    print(' shape = ', data.shape)      # 영상 어레이의 크기 알아내기. image shape =  (512, 768, 3). (행, 열, 채널)
    print(' data type = ', data.dtype) # 어레이 요소의 데이터 타입 알아내기. uint8 = unsigned 8비트.
    if len(data.shape) >= 2:
        print(' row = ', data.shape[0])  # shape는 tuple이다. 원소는 []로 지정. 리스트도 []로 지정한다.
        print(' column = ', data.shape[1])  # 열의 수.
        if len(data.shape) == 3:
            print(' channel = ', data.shape[2])  # 채널의 수. 칼라 영상이면 3. 모도 영상이면 1.


# 입력 동영상 --------------------------
Path = '../data/'
#Name = 'CV_Bullet Time and The Matrix.mp4'
#Name = 'matrix.mp4'    # 44초
#Name = 'frozen.avi'
#Name = 'car_race1.mp4'
#Name = 'car_race2.mp4'

Name = 'the_return_of_the_king.avi'
Name = 'frozen.mp4'

FullName = Path + Name

# ========================================================================================================
# 단계 1 : 읽기용의 비디오 객체를 생성한다. 파일 혹은 캠카메라에서 읽을 수 있다.
#         비디오 카메라는 파일명 대신 카메라 번호를 지정하면 된다. 예) cv.VideoCapture(0)
#   Class for video capturing from video files, image sequences or cameras.
#       https://docs.opencv.org/4.1.1/d8/dfe/classcv_1_1VideoCapture.html
# ========================================================================================================
videoCapture = cv.VideoCapture(FullName)


# ========================================================================================================
# 단계 2 : 처리할 비디오 영상 파일의 속성을 확인한다.
# 	retval	=	cv.VideoCapture.get(propId)
#       propId = VideoCapture generic properties identifier.
#       https://docs.opencv.org/4.1.1/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
# ========================================================================================================
fps = videoCapture.get(cv.CAP_PROP_FPS)
print ('fps of input file:' + Name + '=', fps)
dly_ms = 1000/(fps)       # dly_ms: ms로 표시한 프레임간의 간격[ms]
print(f'delay time between frames={int(dly_ms)}[ms]')

size = (int(videoCapture.get(cv.CAP_PROP_FRAME_WIDTH)),
    int(videoCapture.get(cv.CAP_PROP_FRAME_HEIGHT)))
print ('frame size=', size)

fourcc = videoCapture.get(cv.CAP_PROP_FOURCC)
print ('FOURCC =', fourcc)             # 압축 코덱 4글자를 숫자로 치환한 값. 부동소수
print ('int(FOURCC) =', int(fourcc))   # 압축 코덱 4글자를 숫자로 치환한 값. 정수=828601953
print (f'int(FOURCC) ={int(fourcc):#x}')    # hexa로 표현: 0x31637661 => '1', 'c', 'v'. 'a'

# see VideoWriter::fourcc
#     https://docs.opencv.org/4.0.0/dd/d9e/classcv_1_1VideoWriter.html#afec93f94dc6c0b3e28f4dd153bc5a7f0
# see Video Codecs by FOURCC: 각 코덱에 따른 4글자의 four CC 값 목록. 사용하기전 미리 해당 코덱을 설치해야 하는 경우가 많음.
#     https://www.fourcc.org/codecs.php

# 숫자로 표현된 fourcc 숫자 정보를 4개의 문자 fourcc 정보로 치환하는 함수
# 맨 뒤의 8비트부터 시작해서 이를 ASCII 문자로 만들어 list 자료로 만든 다음 이를 join() method로 스트링으로 만든다.
def decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])

print(decode_fourcc(fourcc))
# FOURCC = 828601953 일 경우 'avc1'인 것으로 함수 적용 결과 알게 되었다.
# => Apple's version of the MPEG4 part 10/H.264 standard apparently


number_of_total_frames = videoCapture.get(cv.CAP_PROP_FRAME_COUNT)
print ('Number of total frames, CAP_PROP_FRAME_COUNT=', int(number_of_total_frames))



# ========================================================================================================
# 단계 3 : 저장할 비디오 영상 코덱을 지정한다.
# FourCC(Four Character Code)
#   그대로 "4글자 코드"라는 뜻이며, 4 바이트로 된 문자열은 데이터 형식을 구분하는 고유 글자가 된다.
#   FourCC는 주로 AVI 파일의 영상 코덱을 구분할 때 사용된다
# 참조 URL:   http://www.fourcc.org/
# ========================================================================================================
#CODEC = -1      # 윈도에서 지원하는 코덱의 다이얼박스 목록에서 선택한다. Intel YUV=70MB. MS RLE 80=실패.
#CODEC = cv.VideoWriter_fourcc('D','I','V','X')
#CODEC = cv.VideoWriter_fourcc('I','4','2','0')         # 파일 용량 큼
CODEC = cv.VideoWriter_fourcc('F', 'M', 'P', '4')          # 파일 용량 적음

# 아래 코덱은 저장할 때 다음 오류를 유발하므로 미리 다운 받아야 함.
#Failed to load OpenH264 library: openh264-1.7.0-win64.dll
# https://github.com/cisco/openh264/releases
# 주의사항: 오류 메시지에서 제시된 버전(1.7.0)을 다운 받아야함. 이 버전은 파이썬 설치 버전에 따라 달라질 수 있음.
# dll 파일을 본 프로그램의 폴더에 복사해 넣거나 PATH=C:\Python\Scripts\;C:\Python\ 등으로 제시된 path 폴더 중의 하나에 넣어도 됨.
#CODEC = cv.VideoWriter_fourcc('X','2','6','4')

print('CODEC=', CODEC); print('type(CODEC)=', type(CODEC))


# ========================================================================================================
# 단계 4 : 쓰기용의 비디오 객체를 생성한다.
# 그리고, 새로 만들 동영상의 fps를 조정한다.
# 그리고, 저장될 동영상의 이름도 정한다.
# ========================================================================================================
#new_fps = int(fps/2)        # 1/2 배속. 실제 녹화된 동영상보다 초당 재생하는 프레임의 수가 수가 적어지므로 저속으로 움직인다.
new_fps = int(2*fps)        # 2배속. 실제 녹화된 동영상보다 초당 재생하는 프레임의 수가 수가 많아지므로 고속으로 움직인다.
print(f"new_fps={new_fps}")

# 출력 저장용 동영상 -----------------------------
fname = Name[0:-4]      # 파일 이름에서 확장자(.bmp, .jpg, .png 등 점 포함)를 제거한다.
SaveFileName = 'tmp_' + fname + f'(fps={new_fps}).avi'

videoWriter = cv.VideoWriter(
    #SaveFileName, CODEC, fps, size)         # 정상 속도. 똑 같은 파일을 만든다.
    SaveFileName, CODEC, new_fps, size)



# ========================================================================================================
# 단계 5 : 첫 번째 프레임을 읽어보고 영상의 속성을 출력한다.
# ========================================================================================================
success, frame = videoCapture.read()
printImgAtt('frame')



# ========================================================================================================
# 단계 6 : 직전의 읽기 동작이 성공하였다면 화면에 영상 데이터를 출력하면서 저장한다.
# 질문 : 1.5배 밝게하여 저장하고자 하는 코드가 오류가 발생하는 이유(오류 1, 2)를 설명하시오.
# 미션 : 영상을 1.5배 밝게 만들어 저장하는 코드를 작성하시오.
# ========================================================================================================

"""
# play 방법 1: 프레임과 프레임 간의 간격을 정해진 시간 만큼 지연시켜서 재생시간을 확보하는 방안
scale =1            # play time will be decreased by this amount.
count = 0
margin = 4      # 재생외의 작업에 투여될 것으로 예상되는 시간

s_time = time.time()
while success: # Loop until there are no more frames.
    count += 1
    #frame = 1.5 * frame                  # 오류 1- 오류
    #frame= (1.5 * frame).astype(dtype='uint8')  # 오류 2- 영상 잡음. why?: np.clip()를 써야 했음.
    cv.imshow(Name, frame)
    videoWriter.write(frame)               # 정상 작동
    success, frame = videoCapture.read()
    print("\rCount = ", count, end=' ')
    cv.waitKey(int(1000 / (fps * scale))-margin)  # fps 만큼의 시간 지연
e_time = time.time() - s_time
print(', Total Count = ', count)

playing_sec = number_of_total_frames/fps  # 상영시간[sec]
print(f'Play time in spec. ={playing_sec:#.2f}[sec]')
print(f'Play time in real = {e_time:#.2f}[sec]')
videoCapture.release()

"""

#"""
# play 방법 2: 프레임과 프레임 간의 간격을 실제로 시간을 측정해서 필요한 만큼의 지연시간을 보충하는 방안.
# - time() 함수를 사용. 영상처리에 시간이 많이 소요되어도 일정 시간 간격을 유지할 수 있음.
# 단, 영상을 출력하는 시간이 FPS의 간격보다 적게 소모되어야 적용하는데 의미가 있다.

count = 0
margin = 1      # 순수한 영상출력(재생) 외의 다른 작업에 소비되는 예상 추정시간[ms]. 경험치

s_time = time.time()            # ms 단위의 현재 tick count을 반환
while success:          # Loop until there are no more frames.
    s = time.time()     # start. time in sec.
    cv.imshow(Name, frame)          # 처리하지 않은 원본 영상을 그대로 보인다.
    count += 1

    # 이왕 저장하는 김에 원본 영상에 처리를 가하여 보다 밝게 저장해 봅시다.
    #frame = 1.5 * frame                  # 오류 1- 오류
    #frame= (1.5 * frame).astype(dtype='uint8')  # 오류 2- 영상 잡음. why?: np.clip()를 써야 했음.
    # frame = np.clip((1.5 * frame).astype(dtype='uint8'), 0, 255)  # 오류 3- 잘못 처리된 방법
    frame = (np.clip(255 * (frame / 255 * 1.5), 0, 255)).astype('uint8')

    videoWriter.write(frame)               # 저장한다...
    success, frame = videoCapture.read()    # 다음 프레임을 읽어온다.
    cv.waitKey(1)
    print("\rCurrent frame number = ", count, end=' ')
    while ( (time.time() - s) * 1000 ) < (dly_ms - margin): # dly_ms: ms로 표시한 프레임간의 간격[ms]
        pass
e_time = time.time() - s_time
playing_sec = number_of_total_frames/fps  # 상영시간[sec]
print(f'\n\nExpected play time={playing_sec:#.2f}[sec]')
print(f'Real play time={e_time:#.2f}[sec]')

videoCapture.release()


#"""