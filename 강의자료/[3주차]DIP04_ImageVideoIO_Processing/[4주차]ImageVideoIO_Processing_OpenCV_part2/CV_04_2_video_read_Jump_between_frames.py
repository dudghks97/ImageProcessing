"""
1. 개요
    입력 비디오 파일의 시간 상의 특정 위치에 있는 영상 프레임(0, 1, 중간)을 화면에 출력한다.

2. 함수
    cv.VideoCapture.get(): 읽을 비디오 파일의 속성을 출력한다.
    cv.VideoCapture.set(): 비디오 파일의 속성을 지정한다.

    총 프레임 수 알아내기
        total_count = int(VideoCapture_객체.get(cv2.CAP_PROP_FRAME_COUNT))
    현재의 프레임 번호 알아내기
        cnt = int(VideoCapture_객체.get(cv2.CAP_PROP_POS_FRAMES))
    프레임 카운터를 원하는 위치로 설정하기
        VideoCapture_객체.set(cv2.CAP_PROP_POS_FRAMES, mid)


"""

import cv2

Path = '../data/'
Name = 'the_return_of_the_king.avi'
#Name = 'CV_Bullet Time and The Matrix.mp4'
Name = 'car_race1.mp4'
Name = 'car_race2.mp4'
Name = 'frozen.mp4'
#Name = 'matrix.mp4'

FullName = Path + Name
#FullName = 'test.avi'           # 테스트용
#FullName = 'CamVid.avi'


# 단계 1: 비디오 파일을 읽고 총 프레임수를 알아낸다.
capture = cv2.VideoCapture(FullName)
total_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
print('total frame count:', total_count)

# 단계 2: 현재의 프레임 번호를 알아낸다. 맨 처음은 0을 가리킨다.
cnt = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
_, frame = capture.read()
cv2.imshow(f'current frame number:{cnt}', frame)

# 단계 3: 현재의 프레임 번호를 알아낸다. 한번 읽었기 때문에 1을 가리킨다.
cnt = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
_, frame = capture.read()
cv2.imshow(f'current frame number:{cnt}', frame)

# 단계 4: 프레임 번호를 지정하여 읽어 낸다. 중간 지점을 읽기로 한다.
mid = int(total_count/2)
capture.set(cv2.CAP_PROP_POS_FRAMES, mid)
cnt = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
_, frame = capture.read()
cv2.imshow(f'setting number={mid}, current number={cnt}', frame)

cv2.waitKey()
cv2.destroyAllWindows()