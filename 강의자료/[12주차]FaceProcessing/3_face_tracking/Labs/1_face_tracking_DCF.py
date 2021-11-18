"""
Face tracking using dlib frontal face detector for initialization
and dlib discriminative correlation filter DCF tracker

USB 캠 카메라 필요
동작 설명
    처음 얼굴이 인식되는 사람을 목표로 추적을 시작한다.
    추적 대상은 녹색 사각형으로 표시한다.
    다시 사람을 인식하게 하고 싶으면 스페이스바를 입력한다.
    종료할 때는 esc 키를 입력한다.

프로그램 수행 절차
# 1) dlib 함수로 frontal face detector를 생성한다.
# 2) dlib 함수로 correlation tracker를 생성한다.
3.1) 얼굴 추적 중이 아니라면...
        # 영상을 모노로 바꾸어 얼굴을 다시 검출하여 검출된 얼굴이 있으면 이를 목표 영상으로 설정한다.
3.2) 얼굴 추적 중이라면...
        # 추적기를 업데이크하고 검출 신뢰 점수를 출력하고, 위치을 반환받아 해당 위치에 4각형을 그린다.
    # 4) 키를 점검하여 재 탐색 여부와 중지를 결정한다.
# 5) 영상을 출력하고, 파일이라면 잠시(1/30 초) 출력지연시간을 유지한다.
    # 이 지연시간 때문에 키보드 입력을 놓치는 경우가 많다.


미션 - 현재 프로그램에 일부 불완전하게 구현하였음. 키 입력이 잘 안되는 문제가 가장 심각.
    본 프로그램은 USB 캠 영상을 기반으로 한다.
    1) 이를 비디오 파일 기반으로 바꾸어 재 작성하시오(주의!! 프레임 처리 속도)
    -> 정상 속도로 화면에 출력되어야 한다.
    2) 현재는 1인만 추적하는데 원하는 사람의 수가 나올 때 스페이스바로 정지하게 하여 다수인을 추적하도록 확장하시오.
    3) 이때 검출된 인물 마다 다른 색상으로 출력한다..==> 미션 아님: ==> 사실 여러 사람에게 적용하기 불합리한 구조를 갖고 있다.


"""






# Import required packages:
import cv2
import dlib

path = '../../data/'            # 동영상이 존재하는 곳
file = '러닝맨 오프닝.mp4'        # 다수인이라 프로그램의 댜폭적인 수정이 필요함
file = '이소라.mp4'
full_file_name = path + file


def draw_text_info():
    """Draw text information"""

    # We set the position to be used for drawing text and the menu info:
    menu_pos_1 = (10, 20)
    menu_pos_2 = (10, 40)

    # Write text:
    cv2.putText(frame, "Use space bar to re-initialize tracking", menu_pos_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    if tracking_face:
        cv2.putText(frame, "tracking the face", menu_pos_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    else:
        cv2.putText(frame, "detecting a face to initialize tracking...", menu_pos_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255))


# Create VideoCapture object to get images from the webcam:
#capture = cv2.VideoCapture(0)            # USB CAM camera
capture = cv2.VideoCapture(full_file_name)    # video file


# 1) dlib 함수로 frontal face detector를 생성한다.
detector = dlib.get_frontal_face_detector()

# 2) dlib 함수로 correlation tracker를 생성한다.
tracker = dlib.correlation_tracker()

# This variable will hold if we are currently tracking the face:
tracking_face = False

while True:
    # Capture frame from webcam:
    ret, frame = capture.read()

    # We draw basic info:
    draw_text_info()

    if tracking_face is False:      # 3.1) 얼굴 추적 중이 아니라면...
        # 영상을 모노로 바꾸어 얼굴을 다시 검출하여 검출된 얼굴이 있으면 이를 목표 영상으로 설정한다.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Try to detect a face to initialize the tracker:
        rects = detector(gray, 0)
        # Check if we can start tracking (if we detected a face):
        if len(rects) > 0:
            # Start tracking:
            tracker.start_track(frame, rects[0])    # 검출된 얼굴 중에 첫번재 얼굴을 추척 대상으로 삼는다.
            tracking_face = True

    if tracking_face is True:       # 3.2) 얼굴 추적 중이라면...
        # 추적기를 업데이크하고 검출 신뢰 점수를 출력하고, 위치을 반환받아 해당 위치에 4각형을 그린다.
        # Update tracking and print the peak to side-lobe ratio (measures how confident the tracker is):
        score = tracker.update(frame)
        #print(f"{score:#.1f}")
        # Get the position of the tracked object:
        pos = tracker.get_position()
        # Draw the position:
        cv2.rectangle(frame, (int(pos.left()), int(pos.top())), (int(pos.right()), int(pos.bottom())), (0, 255, 0), 3)

    # We capture the keyboard event
    key = 0xFF & cv2.waitKey(1)

    # 4) 키를 점검하여 재 탐색 여부와 중지를 결정한다.
    # Press ' ' to re-initialize tracking (it will detect the face again):
    if key == ord(" "):
        print('Detecting new face')
        cv2.waitKey()
        tracking_face = False

    # To exit, press 'q':
    if key == 0x1b:         # 0x1b=esc key  # ord('q'):
        break

    # 5) 영상을 출력하고, 파일이라면 잠시(1/30 초) 출력지연시간을 유지한다.
    # 이 지연시간 때문에 키보드 입력을 놓치는 경우가 많다.
    cv2.imshow("Face tracking using dlib frontal face detector and correlation filters for tracking", frame)
    cv2.waitKey(1000//30 - 3)   # 파일을 사용할 때만 임시로 사용하시오..
    # 이곳에서 loop를 돌기 때문에 키보드 이벤트 처리를 못하는 문제가 있다.
    # 임시용이므로 이점 잘 인지하고 양해 바랍니다.

# Release everything:
capture.release()
cv2.destroyAllWindows()
