# 디지털 영상처리 1차 과제
# 레포트에 각 문제를 밝히고 달성 여부를 설명하는 부분 포함
# 1. 영상을 읽어 들여 이 영상의 밝기를 트랙바로 제어
# 2. 원본 영상과 밝게 만든 영상을 나란히 배열하여 출력. (좌측 : 원본 영상, 우측 : 밝게 만든 영상)  => cv.hconcat()
# 3. 트랙바는 0~30의 값을 가짐. 다음과 같이 원본 영상에 대한 곱셈 배수를 곱해서 구현 (10% => 1+0.1, 20% => 1+0.2, 30% => 1+0.3, ..., 90% => 1+0.9, 300% => 1+3.0)  => np.clip(), .astype(np.uint8)
# 4. 트랙바를 오른쪽으로 움직였다가 다시 왼쪽으로 움직여 어둡게 만들면 최종 단계에서 0을 선택하면 원본 영상과 같아야함.
# 5. 각 화면의 좌측 상단(50, 50)에 'Original' 과 'Brighter' 문자를 출력해야함. 밝기에 영향 x
# 6. key 's' 입력 시 밝아진 영상을 현재의 폴더에 지정된 파일 이름으로 저장함. 문자 정보는 포함 x.     ex) tmpxx_dark1.png => fstring 사용
# xx는 영상의 밝기
# 7. esc 입력 시 종료
# ========================================================================================================================================================= #
import cv2 as cv
import numpy as np

# 변수 선언
Path = './'
Name = 'dark1.png'


# 트랙바의 콜백함수
def onChange(x):
    pass


# 영상 로드
fname = Path + Name             # 원본 파일의 전체 경로
img = cv.imread(fname)          # 원본 파일

imgC = img.copy()               # 원본영상
out = img.copy()                # 변환영상
dst = cv.hconcat([imgC, out])   # 출력영상

# 윈도우 생성
cv.namedWindow('image')

# 트랙바 생성
cv.createTrackbar('scale', 'image', 0, 30, onChange)

# 키 입력 부분, s를 누르면 지정된 이름으로 변환 영상 저장
# esc 입력 시 종료.
while True:
    cv.imshow('image', dst)     # 출력 영상
    outC = out.copy()           # 변환 영상 복사

    # 트랙바 위치 획득
    scale = cv.getTrackbarPos("scale", "image")

    # 영상 밝게
    s = 1 + (scale * 0.1)
    outC = (np.clip(255*(outC/255 * s), 0, 255)).astype('uint8')
    imgS = outC.copy()  # 저장에 쓰일 복사본 생성

    # 영상에 글자 삽입
    org = (50, 50)
    cv.putText(imgC, "Original", org, 0, 1, (0, 0, 255), 2)
    cv.putText(outC, "Brighter", org, 0, 1, (0, 0, 255), 2)

    # 영상 새로 출력
    dst = cv.hconcat([imgC, outC])

    # 키 입력 대기
    key = cv.waitKey(1)
    # s 입력 시 지정된 형식의 이름으로 영상 저장
    if key == ord('s'):
        sname = f'{Path}tmp{int(scale):02d}_dark1.png'
        cv.imwrite(sname, imgS)
        print(f"File name is {sname}")
        print('Image write success!')
    # esc 입력 시 프로그램 종료
    elif key == 27:
        cv.destroyAllWindows()
        print('Terminating program...')
        exit(0)
