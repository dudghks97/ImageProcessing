# 사용된 라이브러리
import cv2 as cv
import numpy as np

# 변수 선언
Path = '../'
Name = 'dark1.png'


# 트랙바의 콜백함수
def onChange(x):
    pass


# 영상 로드
FullName = Path + Name             # 원본 파일의 전체 경로
img = cv.imread(FullName)          # 원본 파일 로드
assert img is not None, 'Failed to load image file!'    # 입력영상 로드 실패. NULL 반환

imgC = img.copy()               # 원본영상
out = img.copy()                # 변환영상
dst = cv.hconcat([imgC, out])   # 출력영상

# 윈도우 생성
cv.namedWindow('image')

# 트랙바 생성
cv.createTrackbar('scale', 'image', 0, 30, onChange)

# 영상 출력 및 밝기 제어
while True:
    cv.imshow('image', dst)     # 출력 영상
    outC = out.copy()           # 변환 영상 복사

    # 트랙바 위치 획득
    scale = cv.getTrackbarPos("scale", "image")

    # 영상의 밝기 변환
    s = 1 + (scale * 0.1)       # 원영상에 곱해질 곱셈 배수
    outC = (np.clip(255*(outC/255 * s), 0, 255)).astype('uint8')

    # 키 입력 대기
    key = cv.waitKey(1)
    # s 입력 시 지정된 형식의 이름으로 영상 저장
    if key == ord('s'):
        saveName = f'tmp{int(scale):02d}_{Name}'  # 저장 파일의 이름
        savePath = './' + saveName                # 저장 파일의 전체 경로
        cv.imwrite(savePath, outC)                # 파일 저장
        print(f"File name is {saveName}")
        print(f'Image successfully wrote to {savePath}')
    # esc 입력 시 프로그램 종료
    elif key == 27:
        print('Terminating program...')
        cv.destroyAllWindows()
        exit(0)
    # 영상에 글자 삽입
    else:
        org = (50, 50)
        cv.putText(imgC, "Original", org, 0, 1, (0, 0, 255), 2)
        cv.putText(outC, "Brighter", org, 0, 1, (0, 0, 255), 2)

    # 최종 출력 영상
    dst = cv.hconcat([imgC, outC])
