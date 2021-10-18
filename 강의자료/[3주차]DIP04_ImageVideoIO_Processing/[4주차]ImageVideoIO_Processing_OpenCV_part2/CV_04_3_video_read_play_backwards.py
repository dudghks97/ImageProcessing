"""
1. 개요: 동영상을 거꾸로 재생하는 프로그램
    맨 마지막 프레임부터 화면을 출력하기 시작하여 역순으로 화면에 출력한다.
        1) 비디오 파일의 프레임 번호를 지정하여 해당 프레임의 영상을 가져 올 수 있다.
            capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)   # 프레임의 현재 번호를 frame_index으로 지정
        2) 이것을 이용하여 영상의 맨 끝 프레임 번호부터 하나씩 감소시켜 가면서 영상 정보를 가져와 역순으로 재생한다.
    재생 도중에 'q'를 입력하면 종료한다.
    rate 변수를 이용하여 거꾸로 재생하기의 지연 배속을 지정할 수 있다.
        1보다 크면 저속 재생이 되고, 1보다 작으면 고속 재생된다.
        예를 들어 2로 지정하면 2배로 저속 재생되고, 0.5이면 2배 빠르게 재생된다.
        이는 한 프레임의 화면 재생 유지 시간을 제어하여 달성한 것이다.
     
2. 개선할 점
    재생을 위해서는 화면에 출력하는 시간 간격이 일정해야 하는데 현재의 코드는 이에 대한 고려가 없다.

3. Mission: 다음 조건을 모두 만족하는 코드를 만드시오.
    1) 등간격(프레임 간격이 동일)으로 재생하는 방안을 제시하시오. (아래 둘 중의 하나. 방법 a는 강의 중 방법이 제시되었음)
        방법 a: time.time() 함수를 활용
        방법 b: threading을 이용한 방법

    2) 거꾸로 재생한 영상을 저장하는 루틴을 추가하시오.

    3) 화면의 좌측 반면은 화소에 1.5배를 곱한 값으로 보인다. 즉, 1.5배 밝아지는 처리를 수행한다.

"""

# Import the required packages:
import cv2  as cv
import argparse
import time



Path = '../data/'
Name = 'the_return_of_the_king.avi'
Name = 'CV_Bullet Time and The Matrix.mp4'
#Name = 'car_race1.mp4'
Name = 'car_race2.mp4'
Name = 'frozen.mp4'
#Name = 'matrix.mp4'

#FullName = 'reverse_tmp.avi'
FullName = Path + Name

capture = cv.VideoCapture(FullName)

# Check if camera opened successfully:
if capture.isOpened() is False:
    print(f"Error opening video stream or file: {Name}")
    exit(0)

fps = capture.get(cv.CAP_PROP_FPS)
print ('1. fps of input file:' + Name + ' =', fps)
dly_ms = 1000/(fps)       # ms로 표시한 프레임간의 간격[ms]
print('2. regular delay time between frames =', int(dly_ms))

# 영상 파일의 맨 뒤부터 읽어옴
# We get the index of the last frame of the video file:
frame_index = capture.get(cv.CAP_PROP_FRAME_COUNT) - 1
print(f"3. starting frame number = {int(frame_index)}")

number_of_total_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
print(f'4. number of total  frames = {number_of_total_frames}')

rate = 2        # 재생 시간이 rate 배로 저속 지생된다.
#rate = 1/2        # 재생 시간이 1/2 배로 저속 지생된다. => 2배 고속 재생된다.
print(f"5. rate = {rate:#.2f}, play time multiplication rate = {rate}")
dly_between_frames = rate*1000 / fps
# dly_between_frames = 50
print(f'6. delay time between frames = {dly_between_frames:#.2f}[ms]')

# 프레임간의 시간지연 시간 조정. +이면 증가. -이면 감소. [ms] 단위,.
playing_sec = rate * number_of_total_frames/fps           # 상영시간[sec]
print(f"7.          expected playing time = {playing_sec:#.2f}[sec.]")

s_time = time.time()


# 방법 1: 프레임과 프레임 간의 간격을 정해진 시간 만큼 지연시켜서 재생시간을 확보하는 방안

while capture.isOpened() and frame_index >= 0:
    t_frame_s = time.time()
    capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)    # We set the current frame position
    ret, frame = capture.read()     # Capture frame-by-frame from the video file
    if ret is True:
        # Print current frame number per iteration
        # print("CAP_PROP_POS_FRAMES : '{}'".format(capture.get(cv.CAP_PROP_POS_FRAMES)))
        # Get the timestamp of the current frame in milliseconds
        # print("CAP_PROP_POS_MSEC : '{}'".format(capture.get(cv.CAP_PROP_POS_MSEC)))
        cv.imshow('Original frame', frame)                  # Display the resulting frame
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Convert the frame to grayscale
        cv.imshow('Grayscale frame', gray_frame)            # Grayscale로 출력
        frame_index = frame_index - 1                       # 프레임 감소 => 역재생
        print(f"\r  next index to read: {frame_index}", end='  ')
        if cv.waitKey(1) == ord('q'):      # Press q on keyboard to exit the program
            break        # Break the loop
        else:
            while((time.time() - t_frame_s) * 1000 < dly_between_frames):   pass
                #continue
    else:
        break

e_time = time.time() - s_time
print(f'\n8.          play time in actual test = {e_time:#.2f}[sec]')

# 시간 차이 = 전체 이론적 상영시간(초) - 실제 상영시간(초)
# 한 frame 재생에 필요한 교정 시간[ms] = 1000 * 시간차이(초)  / 총 프레임의 수
proper_delay = (playing_sec-e_time) * 1000/number_of_total_frames
print(f'9. proper delay adjustment per frame = {proper_delay:#.2f}[ms]')
if  proper_delay < 0:   print("   Your CPU's processing power may not be sufficient as required.")
print(f'10. adjusted dly_between_frames for 6 = {dly_between_frames+proper_delay:#.6f}[ms]')
# Release everything:
capture.release()
cv.destroyAllWindows()





# 미션 수행
# play 방법 2: 프레임과 프레임 간의 간격을 실제로 시간을 측정해서 필요한 만큼의 지연시간을 보충하는 방안.
# 아래 2가지 방법 중에서 여러분이 선택해서 도전해 보세요. (b) 방법이 성공하면 더 우대하여 평가하겠습니다.
#   (a) time() 함수를 사용. 영상처리에 시간이 많이 소요되어도 일정 시간 간격을 유지할 수 있음.
#       단, FPS시간 이내 작업이 이루어진다는 전제입니다. 예를 들어, 간단한 영상 처리만 한다는 전제입니다.
#       프레임과 프레임 간격 안에 필요한 처리을 못한다면 이 방법은 사용이 불가하겠죠.
#   (b) threading을 이용한 방안도 있을 것으로 보임. 강의 내용에 포함도지 않았음으로 자율적인 학습이 필요함.
#       '파이썬 스레드 프로그래밍' 등의 키워드로 검색하여 예제 프로그램을 활용하면 될 것으로 보임.
