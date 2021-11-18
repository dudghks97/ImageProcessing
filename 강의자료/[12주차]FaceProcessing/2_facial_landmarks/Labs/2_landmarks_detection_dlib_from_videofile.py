"""
Detecting facial landmarks using dlib from a video file

개요
    동영상에서 취득한 영상을 화면에 출력하면서 검출한 얼굴 영역과 랜드마크를 계속 표시한다.
    dlib 함수로 얼굴을 검출하고, dlib 함수로 얼굴의 주요 랜드마크를 검출한다.
    q를 입력하면 프로그램을 종료한다.
    스페이스 바를 누르면 다시 키를 누를 때까지 잠시 정지한다.

설치상 주의점
    동영상이 필요하다. => path = '../../data/'  아래에 있어야 한다.
    shape detector 데이터 파일이 필요하다. => path + "dlib_shape_predictors/" 아래에 있음

동작 - 주의!! 얼굴을 검출하고 검출된 얼굴 영역내에서 랜드마크를 검출한다.
    1) 얼굴 객체 생성: detector = dlib.get_frontal_face_detector()
    2) 랜드마크 검출 객체 생성: predictor = dlib.shape_predictor(p)
        이때 p 파일은 랜드마크 객체를 생성하기 위해 정보를 담고 있는 shape predictor 파일이다.
    3) 얼굴을 검출한다. => 4각형 영역(rect)을 반환한다: rects = detector(gray, 0)
    4) 검출된 얼굴 안에서 랜드마크를 검출한다. => 랜드마크 shape descriptor를 반환한다.: shape = predictor(gray, rect)
    5) 랜드마크 디스크립터를 numpy array 정보로 바꾼다: shape = shape_to_np(shape)
    6) 검출된 얼굴 영역과 랜드마크를 화면에 출력한다.

"""

# Import required packages:
import cv2
import dlib
import numpy as np

file = "test_face_detection.jpg"
file = 'face_test.png'      # 한 사람만 존재

path = '../../data/'            # 동영상이 존재하는 곳
file = '러닝맨 오프닝.mp4'
full_file_name = path + file

# Name of the two shape predictors: shape dector detector 파일의 위치와 이름
p = path + "dlib_shape_predictors/shape_predictor_68_face_landmarks.dat"
# p = path + "dlib_shape_predictors/shape_predictor_5_face_landmarks.dat"


# Define what landmarks you want:
JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_BRIDGE_POINTS = list(range(27, 31))
LOWER_NOSE_POINTS = list(range(31, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))
ALL_POINTS = list(range(0, 68))


def draw_shape_lines_all(np_shape, image):
    """Draws the shape using lines to connect between different parts of the face(e.g. nose, eyes, ...)"""

    draw_shape_lines_range(np_shape, image, JAWLINE_POINTS)
    draw_shape_lines_range(np_shape, image, RIGHT_EYEBROW_POINTS)
    draw_shape_lines_range(np_shape, image, LEFT_EYEBROW_POINTS)
    draw_shape_lines_range(np_shape, image, NOSE_BRIDGE_POINTS)
    draw_shape_lines_range(np_shape, image, LOWER_NOSE_POINTS)
    draw_shape_lines_range(np_shape, image, RIGHT_EYE_POINTS, True)
    draw_shape_lines_range(np_shape, image, LEFT_EYE_POINTS, True)
    draw_shape_lines_range(np_shape, image, MOUTH_OUTLINE_POINTS, True)
    draw_shape_lines_range(np_shape, image, MOUTH_INNER_POINTS, True)


def draw_shape_lines_range(np_shape, image, range_points, is_closed=False):
    """Draws the shape using lines to connect the different points"""

    np_shape_display = np_shape[range_points]
    points = np.array(np_shape_display, dtype=np.int32)
    cv2.polylines(image, [points], is_closed, (255, 255, 0), thickness=1, lineType=cv2.LINE_8)


def draw_shape_points_pos_range(np_shape, image, points):
    """Draws the shape using points and position for every landmark filtering by points parameter"""

    np_shape_display = np_shape[points]
    draw_shape_points_pos(np_shape_display, image)


def draw_shape_points_pos(np_shape, image):
    """Draws the shape using points and position for every landmark"""

    for idx, (x, y) in enumerate(np_shape):
        # Draw the positions for every detected landmark:
        cv2.putText(image, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))

        # Draw a point on every landmark position:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)


def draw_shape_points_range(np_shape, image, points):
    """Draws the shape using points for every landmark filtering by points parameter"""

    np_shape_display = np_shape[points]
    draw_shape_points(np_shape_display, image)


def draw_shape_points(np_shape, image):
    """Draws the shape using points for every landmark"""

    # Draw a point on every landmark position:
    for (x, y) in np_shape:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)


def shape_to_np(dlib_shape, dtype="int"):
    """Converts dlib shape object to numpy array"""

    # Initialize the list of (x,y) coordinates
    coordinates = np.zeros((dlib_shape.num_parts, 2), dtype=dtype)

    # Loop over all facial landmarks and convert them to a tuple with (x,y) coordinates:
    for i in range(0, dlib_shape.num_parts):
        coordinates[i] = (dlib_shape.part(i).x, dlib_shape.part(i).y)

    # Return the list of (x,y) coordinates:
    return coordinates





# ====================================================================================================
# 프로그램의 시작 -->
# ====================================================================================================

# Initialize frontal face detector and shape predictor:
detector = dlib.get_frontal_face_detector()     # 얼굴검출 객체 생성
predictor = dlib.shape_predictor(p)             # 랜드마크 검출 객체 생성. p는 맨위에 정의함.

# Create VideoCapture object to get images from the webcam:
#video_capture = cv2.VideoCapture(0)            # USB CAM camera
video_capture = cv2.VideoCapture(full_file_name)    # video file



while True:

    # Capture frame from the VideoCapture object:
    ret, frame = video_capture.read()

    # Just for debugging purposes:
    # frame = test_face.copy()

    # Convert frame to grayscale:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces:
    rects = detector(gray, 0)   # 0: 영상을 upsampling하지 않고, 현재의 해상도를 사용. (1: 2배, 2: 4배)

    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Draw a box around the face:
        cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 1)

        # Get the shape using the predictor:
        shape = predictor(gray, rect)

        # Convert the shape to numpy array:
        shape = shape_to_np(shape)

        # Draw all lines connecting the different face parts:
        # draw_shape_lines_all(shape, frame)

        # Draw jaw line:
        # draw_shape_lines_range(shape, frame, JAWLINE_POINTS)

        # Draw all points and their position:
        # draw_shape_points_pos(shape, frame)
        # You can also use:
        # draw_shape_points_pos_range(shape, frame, ALL_POINTS)

        # Draw all shape points:
        draw_shape_points(shape, frame)

        # Draw left eye, right eye and bridge shape points and positions
        # draw_shape_points_pos_range(shape, frame, LEFT_EYE_POINTS + RIGHT_EYE_POINTS + NOSE_BRIDGE_POINTS)

    # Display the resulting frame
    cv2.imshow("Landmarks detection using dlib", frame)

    # Press esc key to exit
    key_in = cv2.waitKey(1) & 0xff

    if key_in == 0x1b:        # esckey =  0x1b,  #ord('q'):
        break
    elif key_in == ord(' '):     # 스페이스면 잠시 정지
        cv2.waitKey(0)


# Release everything:
video_capture.release()
cv2.destroyAllWindows()
