
import cv2 as cv
import time


cameraCapture = cv.VideoCapture(0, cv.CAP_DSHOW)         # 0은 비디오 입력 장치의 번호이므로 바뀔 수도 있음.

# 다른 VideoCaptureProperties도 살펴 보고자 한다.
#   링크: https://docs.opencv.org/4.5.2/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d


"""
# ========================================================================================================
# 실습 1: cam camera의 특정 속성을 바꾸는 것을 지원하는지 확인해 보자.
# ========================================================================================================
# CAP_PROP_CONTRAST default=5, CAP_PROP_CONTRAST after adjusting=7
# CAP_PROP_SATURATION default=124, CAP_PROP_SATURATION after adjusting=186
# CAP_PROP_EXPOSURE  default=-6, CAP_PROP_EXPOSURE  after adjusting=-9
#ppty_lst = ["CAP_PROP_CONTRAST", "CAP_PROP_SATURATION", "CAP_PROP_EXPOSURE "]
ppty_lst = ["CAP_PROP_BRIGHTNESS"]
ppty_lst = ["CAP_PROP_AUTO_WB"]

for ppty in ppty_lst:
    str_ppty = "cv." + ppty
    val = int(cameraCapture.get(eval(str_ppty)))
    print(f"{ppty} default={val}", end="")
    #cameraCapture.set(eval(str_ppty), int(val * 1.5))
    #val = int(cameraCapture.get(eval(str_ppty)))
    #print(f", {ppty} after adjusting={val}")

    cameraCapture.set(eval(str_ppty), 1)
    val = int(cameraCapture.get(eval(str_ppty)))
    print(f", {ppty} after adjusting={val}")

exit()

# 상태 복구 종료
cameraCapture.set(cv.CAP_PROP_BRIGHTNESS, 80)
cameraCapture.set(cv.CAP_PROP_CONTRAST, 5)
cameraCapture.set(cv.CAP_PROP_SATURATION, 124)
cameraCapture.set(cv.CAP_PROP_EXPOSURE, -6)


exit()
"""


# ========================================================================================================
# 실습 2: 트랙바를 이용하여 카메라의 속성을 바꾸면서 화면이 어떻게 바뀌는지 관찰해 보자.
# ========================================================================================================
def callBack_BR(x):
    cameraCapture.set(cv.CAP_PROP_BRIGHTNESS, x)

cv.namedWindow('camera')             # createTrackbar를 호출하기 위해 imshow() 대신 써도 됨. 나중에 어차피 imshow()가 호출됨
val = int(cameraCapture.get(cv.CAP_PROP_BRIGHTNESS))
print(f"CAP_PROP_BRIGHTNESS: current setting={val}")

cv.createTrackbar ('Brightness', 	# 트랙바 앞에 표시될 트랙바의 이름
    'camera',	# 트랙바가 나타날 창의 이름
    val,			# 시작 당시의 슬라이더의 초기 위치
    512,		# 슬라이더의 최댓값. 최솟값은 0으로 고정.
    callBack_BR)	    # 슬라이더가 움직일 때 호출될 콜백 함수의 이름.
                    # 첫 번째 파라미터: 트랙 바 위치. 두 번째 파라미터: 사용자 데이터.

success = True
while success and cv.waitKey(1) == -1:
    success, frame = cameraCapture.read()
    cv.imshow('camera', frame)

cameraCapture.set(cv.CAP_PROP_BRIGHTNESS, 80)       # 원상 복귀
cv.destroyWindow('camera')
cameraCapture.release()
