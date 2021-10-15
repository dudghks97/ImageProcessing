"""
1. 개요
    USB 캠카메라의 영상을 정해진 시간(run_time)만큼 촬영하여 이를 파일(CamVid(size, fps=xx.yy).avi)로 저장한다.
    과정 중 해상도와 FPS(Frame Per Second)를 바꾸고 제대로 반영되었는지 확인한다.
    1) 현재 사용하는 USB CAM " 라이프캠 HD-3000. USB 2.0
        https://www.microsoft.com/accessories/ko-kr/products/webcams/lifecam-hd-3000/t3h-00014
    2) 힉과 보유 USB CAM : oCam - 5MP USB 3.0 Color Camera
        https://github.com/withrobot/oCam/tree/master/Products/oCam-5CRO-U-M
2. 동작
    단계 1: 비디오 객체(읽기용)를 생성한다.
    단계 2: 읽을 비디오 카메라(비디오 객체)의 속성을 출력한다.
    단계 3: resolution, fps를 변경하고 설정한 대로 변경되었는지 확인한다.
    단계 4: 카메라의 해상도를 다시 설정하고 설정한 대로 변경되었는지 확인한다.
    단계 5: 비디오 객체(쓰기용)을 생성한다.
    단계 6: 메인 루틴 - 카메라에서 읽어서 비디오 파일에 저장한다.

3. 사용된 함수
    cv.VideoCapture(): 비디오 객체(읽기용)를 생성한다.
    cv.videoCapture.read(): 한 프레임의 영상을 반환한다.
    cv.VideoCapture.get(): 비디오 카메라(비디오 객체)의 속성을 출력한다.
    cv.VideoCapture.set(): 비디오 카메라(비디오 객체)의 속성을 지정한다.
    cv.VideoWriter_fourcc(): 쓰기 용의 비디오 영상 코덱을 지정한다.
    cv.VideoWriter(): 쓰기용의 비디오 객체를 생성한다.
    cv.VideoWriter.write(): 쓰기용의 비디오 객체에 쓰기 동작을 수행한다.

4. oCam-5CRO-U-M, link: http://withrobot.com/camera/ocam-5cro-u-m/
    2592 (H) x 1944 (V) pixels @15, 7.5, 3.75 fps
    1920 (H) x 1080 (V) pixels @30, 15, 7.5 fps
    1280 (H) x 960 (V) pixels @45, 30, 15 fps
    1280 (H) x 720 (V) pixels @60, 30, 15 fps
    640 (H) x 480 (V) pixels @90, 60, 30 fps    -> default인 듯.
    320 (H) x 240 (V) pixels @120, 100, 90, 60, 30 fps

5. 미션
    하단부의 미션 프로그램을 작성하시오.


* warning error에 대하여...
cap_msmf.cpp (438) `anonymous-namespace' => backend를 direct show로 사용한다고 선언하면 없어진다.
그러나, 속도가 떨어진다고 말도 있기도 하고, 여러 해결책이 있으나 ...
카메라 객제를 선언할 때 backend를 DirectShow backend로 변경하면 경고 오류가 없어진다.
    cameraCapture = cv.VideoCapture(0) => cameraCapture = cv.VideoCapture(0, cv.CAP_DSHOW)


* 정확한 소요시간을 재기 위한 다른 방안에 대하여...
time.time()의 정밀도는 아직 파악 중이다...
time.clock_gettime_ns(clk_id)의 경우 int clock_gettime()과 비슷하지만, 시간을 나노초로 반환합니다.


"""


import cv2 as cv
import time

# ========================================================================================================
# 단계 0: 비디오 객체(읽기용)를 생성한다.
# 장치 설정에 따라 카메라 번호를 바꾸어야할 때도 있다.
# ========================================================================================================
#cameraCapture = cv.VideoCapture(0)
cameraCapture = cv.VideoCapture(0, cv.CAP_DSHOW)   # backend를 DirectShow backend로 변경하면 경고 오류가 없어진다.


# ========================================================================================================
# 단계 1: 읽을 비디오 카메라(비디오 객체)의 속성을 출력한다.
# ========================================================================================================
size = (int(cameraCapture.get(cv.CAP_PROP_FRAME_WIDTH)),
    int(cameraCapture.get(cv.CAP_PROP_FRAME_HEIGHT)))
print('\n1. Default Setting:')
print('resolution : WIDTH*HEIGHT=', size)

fps_cam = int(cameraCapture.get(cv.CAP_PROP_FPS))
print('fps=', fps_cam)          # 의외로 잘 안읽힌다. 그러나 정상 작동.



# ========================================================================================================
# 단계 2: 카메라의 해상도를 다시 설정하고 설정한 대로 변경되었는지 확인한다.
# 저급 카메라는 고해상도 설정에 실패한다.
# ========================================================================================================
#N_width = 1280; N_height = 720         # 저급 카메라는 안됨.
#N_width = 1920; N_height = 1080         # OK. @30, 15, 7.5 fps for oCam
#N_width = 1920; N_height = 1080         # OK. @30, 15, 7.5 fps for oCam
#N_width = 1280; N_height = 720          # @60, 30, 15 fps for oCam
N_width = 320; N_height = 240           # 120fps. 대다수 작동.
cameraCapture.set(cv.CAP_PROP_FRAME_WIDTH, N_width)
cameraCapture.set(cv.CAP_PROP_FRAME_HEIGHT, N_height)
size = (int(cameraCapture.get(cv.CAP_PROP_FRAME_WIDTH)),
    int(cameraCapture.get(cv.CAP_PROP_FRAME_HEIGHT)))
print('\n2. Resolution Setting & reading:')
print('Changed Resolution : WIDTH*HEIGHT=', size)


# ========================================================================================================
# 단계 3: fps를 변경하고 설정한 대로 변경되었는지 확인한다.
# 실험에 의하면 설정이 바꾸지 않는 경우가 많음.
# 저장할 때 FPS를 바꾸면 되기 때문에 심각한 문제는 아니라고 판단됨.
# ========================================================================================================
print('\n3. FPS Change:')
fps_set = float(60)             # 희망하는 FPS(Frame Per Second)
print(f'setting fps= {fps_set}')

# 카메라에 따라 설정이 실패할 수도 있다. MS Camera HD-3000의 경우 성공사례: 15, 30, 60, 120
# fps 프레임수가 높아지면 초당 많은 프레임을 저장해야 하므로 loop가 빠른 속도로 돌지 못할 수도 있다.
# 적절한 제안:
#   카메라 세팅을 2배 정도로 높히고, 실제 저장은 그 반 정도의 속도로 처리한다.
#   화면 취득하는데 속도가 걸리기 때문에 카메라 세팅 속도와 저장 속도가 같으면 좋지 않다.

cameraCapture.set(cv.CAP_PROP_FPS, float(fps_set))
fps_rd = cameraCapture.get(cv.CAP_PROP_FPS)
print(f'fps after change= {fps_rd}')            # 따라서 설정한대로 읽히지 않을 수도 있음.
if fps_set != fps_rd:
    print('FPS setting error ???...')
    #exit(0)

fps_actual = 30         # 저장용의 비디오 파일에 사용할 실제 fps

# ========================================================================================================
# 단계 4: 비디오 객체(쓰기용)을 생성한다.
# ========================================================================================================
print('\n4. Create a video object...')
#str_sc = '%4.2f'% fps_sc      # 카메라의 fps
str_sc = f'{fps_actual:#04.2f}'      # 카메라의 fps

print(f'fps_actual for writing video file= {fps_actual}')
videoWriter = cv.VideoWriter(
#    'CamVid.avi', cv.VideoWriter_fourcc('I','4','2','0'), fps_actual, size)
    f"CamVid({size}, fps={str_sc}).avi", cv.VideoWriter_fourcc('F', 'M', 'P', '4'), fps_actual, size)


# ========================================================================================================
# 단계 5: 메인 루틴 - 카메라에서 읽어서 비디오 파일에 저장한다.
# ========================================================================================================
print('\n5. Writing into file...\n')

run_time = 8         # 녹화시간[초]
numTotal = run_time * fps_actual           # 촬영할 프레임의 갯수
numFramesRemaining = numTotal           # 남아있는 촬영할 프레임의 개수
print('Total frame count: numTotal=', numTotal)
cv.namedWindow('CAM')

# display fps는 writing으로 고정한다.
dly_ms = 1000/fps_actual - 0.5      # ms로 표시한 delay time.
#dly_ms = int(1000/fps_actual)       # ms로 표시한 delay time

print('delay time[ms] between frames=', dly_ms)

print("looping while decrementing numFramesRemaining....")
success = True
ss = time.time()
while success and numFramesRemaining > 0:
    s = time.time()  # start. time in sec.
    success, frame = cameraCapture.read()
    if success == True:
        if frame is None:
            print('_', end='')
        else:
            cv.imshow('CAM', frame)
    else:
        print('.', end='')
    videoWriter.write(frame)
    numFramesRemaining -= 1
    print('\r', numFramesRemaining, '/', numTotal, end='')
    cv.waitKey(1)
    while (time.time() - s) * 1000 < dly_ms:
        pass
sl = time.time() - ss       # total recording time
print(f'\nrecording time= {sl:#.2f}')
cameraCapture.release()
cv.destroyAllWindows()
print('END')


# ========================================================================================================
# 미션 : 320x240의 영상을 촬영하여 그 영상과 그 영상의 모노의 영상을 가로로 이어 붙이고
# 이를 파일로 저장하시오.
# ========================================================================================================
