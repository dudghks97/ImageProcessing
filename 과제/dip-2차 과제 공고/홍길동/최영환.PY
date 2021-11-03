# 영상처리 2차 과제

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

Path = './'
Name = 'matrix.mp4'
#Name = 'polor_express.avi'
#Name = 'the_return_of_the_king.avi'


# 트랙바의 콜백 함수
def onChange(x):
    pass


FullName = Path + Name      # 재생 할 파일의 전체 경로

# 영상 객체 선언 및 로드
cap = cv.VideoCapture(FullName)

success, frame = cap.read()        # 영상 로드

fps = cap.get(cv.CAP_PROP_FPS)     # 가져온 영상의 fps
#fps = 60
number_of_total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)  # 영상의 총 프레임 개수
dly_ms = 1000/(fps)       # dly_ms: ms로 표시한 프레임간의 간격[ms]
print('fps of input file:' + Name + '=', fps)
print('Number of total frames, CAP_PROP_FRAME_COUNT=', int(number_of_total_frames))
print(f'delay time between frames={int(dly_ms)}[ms]')

cv.namedWindow(Name)    # 윈도우 생성
# 트랙바 생성
cv.createTrackbar('HE wgt', Name, 100, 100, onChange)
cv.createTrackbar('sigma', Name, 0, 7, onChange)
cv.createTrackbar('scale', Name, 0, 5, onChange)
cv.createTrackbar('position', Name, 0, int(number_of_total_frames), onChange)

# 영상 재생부
margin = 1      # 순수한 영상출력(재생) 외의 다른 작업에 소비되는 예상 추정시간[ms]. 경험치
count = 0
s_time = time.time()            # ms 단위의 현재 tick count을 반환
while success:          # 영상의 끝까지 재생
    s = time.time()     # 시작 시간
    count += 1

    # 트랙바 값 획득
    sigma = cv.getTrackbarPos('sigma', Name)
    scale = cv.getTrackbarPos('scale', Name)
    pos = cv.getTrackbarPos('position', Name)
    weight = cv.getTrackbarPos('HE wgt', Name)

    pos += 1
    cv.setTrackbarPos('position', Name, pos)    # 현재 트랙바의 위치 설정
    cap.set(cv.CAP_PROP_POS_FRAMES, pos)        # 현재 프레임 = 현재 트랙바의 위치
    success, frame = cap.read()                 # 다음 프레임을 읽어온다

    # 히스토그램 평활화(HE) 실행 : 가중치 값 설정은 했으나 맞는지는 모르겟음
    if frame is not None:
        frameC = frame.copy()                               # 프레임 복사
        frameG = cv.cvtColor(frameC, cv.COLOR_BGR2GRAY)     # 그레이로 변환

        # 히스토그램(분산함수, DF) 를 구함
        hist, bins = np.histogram(a=frameG, bins=256, range=[0, 255])
        cdf = hist.cumsum()     # 누적분포함수

        # 매핑 후 LUT(Look Up Table) 생성
        mapping = cdf * 255 / cdf[255]
        LUT = mapping.astype('uint8')

        # LUT 기반 HE 시행, 컬러
        frameCeq = LUT[frameC]

        # 원본영상 반영비율 조절(가중 평균 미적용) => HE 만 적용한 결과와의 차이 : 밝아짐
        #frameCeq = np.clip((255*(weight * 0.01 * frameCeq/255)) + frameC, 0, 255).astype('uint8')

        # 원본영상 반영비율 조절(가중 평균 미적용) => HE 만 적용한 결과와의 차이 : 없어짐
        w = weight * 0.01
        frameCeq = np.clip((255 * (w * frameCeq/255)) + (255 * ((1-w) * frameC/255)), 0, 255).astype('uint8')

        # 원본영상 반영 비율 조절(가중 평균 값 적용)
        #w = 0.01 * weight
        #weighted_avg = (np.sum(w * frameCeq)) / (w * 255)   # 가중평균 값
        #frameCeq = np.clip((255 * (weighted_avg * frameCeq / 255)), 0, 255).astype('uint8')

        # 원본 영상에 텍스트 추가
        org = (0, 30)
        cv.putText(frameC, f"org_index = {pos}", org, 0, 0.8, (0, 0, 255), 2)

        # 언샤프 마스킹(UM) 미적용. 히스토그램 평활화(HE)만 적용
        if (scale == 0) or (sigma == 0):
            # 평활화 영상에 텍스트 추가
            cv.putText(frameCeq, f"this_index = {count}", org, 0, 0.8, (0, 0, 255), 2)
            cv.putText(frameCeq, f"sigma={sigma}, scale={scale}, weight={weight}", (0, 180), 0, 0.8, (0, 0, 255), 2)
            res = cv.hconcat([frameC, frameCeq])  # 출력될 결과 영상
        # 언샤프 마스킹(UM) 적용.
        else:
            frameCeq = frameCeq / 255
            k = sigma * 6 + 1
            blur = cv.GaussianBlur(src=frameCeq, ksize=(k, k), sigmaX=sigma)
            um = frameCeq + scale * (frameCeq - blur)
            um = np.clip(um * 255, 0, 255).astype(frameC.dtype)
            # 마스킹 영상에 텍스트 추가
            cv.putText(um, f"this_index = {count}", org, 0, 0.8, (0, 0, 255), 2)
            cv.putText(um, f"sigma={sigma}, scale={scale}, weight={weight}", (0, 180), 0, 0.8, (0, 0, 255), 2)
            res = cv.hconcat([frameC, um])                 # 출력될 결과 영상

        cv.imshow(Name, res)                    # 영상 출력

        # 키 입력 대기
        key = cv.waitKey(1)
        if key == 27:           # esc 입력 시 종료
            print('\nTerminate Program...')
            exit(0)
        elif key == ord('s'):   # 저장
            saveName = f'tmp_{int(pos):02d}_matrix.png'
            cv.imwrite(Path + saveName, res)
            if frameCeq is not None:
                print(f"\nFile name is {saveName}")
                print('\nImage write success!')
            else:
                print('\nError Occured!')
        elif key == 32:     # 스페이스바 입력 시 정지
            print(f'\nPause!')
            cv.waitKey(0)

    print(f"\rCurrent frame number = {pos} \t playing count = {count}", end=' ')
    # 영상을 실제 재생시간에 맞추어 재생하기 위한 코드
    while ((time.time() - s) * 1000) < (dly_ms - margin):   # dly_ms: ms로 표시한 프레임간의 간격[ms]
        pass

e_time = time.time() - s_time               # 예상 재생 시간
playing_sec = number_of_total_frames/fps    # 실제 재생시간
print(f'\n\nExpected play time={playing_sec:#.2f}[sec]')
print(f'Real play time={e_time:#.2f}[sec]')

cap.release()               # 객체 반환
cv.destroyAllWindows()      # 모든 창 종료
