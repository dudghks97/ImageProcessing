"""
Detecting facial landmarks using dlib
 - 캠카메라 버전

개요
    캠카메라(혹은 동영상)을 입력으로 취득한 영상을 화면에 출력하면서 검출한 얼굴 영역과 랜드마크를 계속 표시한다.
    dlib 함수로 얼굴을 검출하고, 그 안에서 dlib 함수로 얼굴의 주요 랜드마크를 검출한다.
    esc를 입력하면 프로그램을 종료한다.

설치상 주의점
    1) 카메라가 필요하다. 영상 파일 필요 없음.
    2) shape detector 데이터 파일이 필요하다. => path + "dlib_shape_predictors/" 아래에 있음

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

# Name of the two shape predictors: shape dector detector 파일의 위치와 이름
p = "../../data/dlib_shape_predictors/shape_predictor_68_face_landmarks.dat"
# p = "../../data/dlib_shape_predictors/shape_predictor_5_face_landmarks.dat"


# Define what landmarks you want:
# shape = predictor(gray, rect)를 수행하여 반환받은 68x2의 dlib 랜드마크 객체 중에서
# 어떤 범위의 인덱스가 얼굴의 주요 성분을 지칭하는가를 변수로 정의하였다.
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


# dlip 검출기가 반환한 point 정보를 numpy array 스타일로 바꾼다.
# 반환되는 coordinates는 <class '_dlib_pybind11.full_object_detection'> 형으로 다루기가 곤란하다.

def shape_to_np(dlib_shape, dtype="int"):
    """Converts dlib shape object to numpy array"""

    # Initialize the list of (x,y) coordinates
    coordinates = np.zeros((dlib_shape.num_parts, 2), dtype=dtype)

    # Loop over all facial landmarks and convert them to a tuple with (x,y) coordinates:
    for i in range(0, dlib_shape.num_parts):
        coordinates[i] = (dlib_shape.part(i).x, dlib_shape.part(i).y)

    # Return the list of (x,y) coordinates:
    return coordinates




# 프로그램의 시작 ======================================================================================


# 1) frontal_face_detector() 함수를 통해 얼굴 검출 객체를 생성한다.
#    shape_predictor(p)를 함수를 통해 랜드마크 검출 객체를 생성한다.
#       이때 랜드마크 프레딕터 데이터 파일(p)을 지정하여 로드해야 한다.
detector = dlib.get_frontal_face_detector()     # 얼굴검출 객체 생성
predictor = dlib.shape_predictor(p)             # 랜드마크 검출 객체 생성. p 상단부에 정의됨

# 2) 캠 카메라로부터 영상을 취득하기 위한 VideoCapture object를 생성한다.
video_capture = cv2.VideoCapture(0)

once = True
while True:

    # Capture frame from the VideoCapture object:
    ret, frame = video_capture.read()   # ret는 성공여부(True/False)

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
        # 반환받는 shape는 <class '_dlib_pybind11.full_object_detection'>형이다.
        # shape.num_parts 길이(68 혹은 9)의 랜드마크 정보를 담고 있다.
        # (shape.(i).x, shape.(i).y) 좌표 정보를 shape.num_parts 만큼 갖고 있다.

        # Convert the shape to numpy array:
        shape = shape_to_np(shape)      # 이제 shape는 numpy arrary가 되었다. shape.shape = (68, 2)

        draw_shape_lines_range(shape, frame, RIGHT_EYE_POINTS, True)
        draw_shape_lines_range(shape, frame, LEFT_EYE_POINTS, True)
        draw_shape_lines_range(shape, frame, MOUTH_OUTLINE_POINTS, True)
        draw_shape_lines_range(shape, frame, MOUTH_INNER_POINTS, True)

        # Draw all lines connecting the different face parts:
        # draw_shape_lines_all(shape, frame)

        # Draw jaw line:
        draw_shape_lines_range(shape, frame, JAWLINE_POINTS)

        # Draw all points and their position:
        # draw_shape_points_pos(shape, frame)
        # You can also use:
        # draw_shape_points_pos_range(shape, frame, ALL_POINTS)

        # Draw all shape points:
        # draw_shape_points(shape, frame)

        # Draw left eye, right eye and bridge shape points and positions
        # draw_shape_points_pos_range(shape, frame, LEFT_EYE_POINTS + RIGHT_EYE_POINTS + NOSE_BRIDGE_POINTS)

    # Display the resulting frame
    cv2.imshow("Landmarks detection using dlib", frame)

    # Press 'q' key to exit
    if cv2.waitKey(1) & 0xFF == 0x1b:    #ord('q')
        break


# Release everything:
video_capture.release()
cv2.destroyAllWindows()
