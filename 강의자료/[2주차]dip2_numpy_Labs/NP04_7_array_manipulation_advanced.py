"""
numpy 배열 조작에 관계된 함수 혹은 메서드
NumPy Link : https://docs.scipy.org/doc/numpy/index.html

    1. arange: 연속적인 원소로 이루어진 1차원 배열을 반환한다.
    2. reshape: shape를 지정한 대로 바꾸어 반환한다. 데이터를 원본과 공유한다.
    3. resize: 원본 자체의 모양을 바꾼다. 반환 값이 없다.
    4. ravel: 임의의 배열을 1차원 배열로 만들어 반환한다. 데이터를 원본과 공유한다.
    5. flatten: 임의의 배열을 1차원 배열로 만들어 반환한다. ravel과는 달리 데이터를 원본과 공유하지 않는다.
    6. squeeze: 어레이의 디멘전을 가장 최소화하여 표현할 수 있는 수준으로 줄인다.

"""

import numpy as np



# ---------------------------------------------------------------------------------------------
# print_att()
#   입력 파라미터 변수(ndarray)에 대한 정보를 출력하는 함수
#   입력 파라미터는 변수의 스트링을 취한다.
# ---------------------------------------------------------------------------------------------
def print_att(str_a):

    # str_a를 스트링으로 간주하고 실제 값을 evaluate 한다.
    a = eval(str_a)
    print(f"{str_a}={a}")

    # type(): 자료형을 반환한다.
    print(f"type({str_a})={type(a)}")

    # shape: 행렬의 모양(행x열)을 지정하는 멤버변수
    print(f"{str_a}.shape={a.shape}")

    # ndim: 행렬의 차원을 지정하는 멤버변수
    print(f"{str_a}.ndim={a.ndim}")

    # dtype: 행렬의 원소의 데이터형을 지정하는 멤버변수. 행렬은 모두 같은 데이터형으로 구성된다.
    print(f"{str_a}.dtype={a.dtype}")

    # 한 원소를 표현하기 위해 필요한 데이터의 바이트 수
    print(f"{str_a}.itemsize={a.itemsize}")

    # array에 들어 있는 총 원소의 수
    print(f"{str_a}.size={a.size}")


# ---------------------------------------------------------------------------------------
# 실습 7: reshape() 메서드의 응용 고급,  advanced topic
# ---------------------------------------------------------------------------------------

print("실습 7: reshape() 메서드의 응용 고급,  advanced topic")
# reshape() 함수는 지정한 크기로 재배열하는 함수이다.
r = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
print(f'\n1) r.shape={r.shape}, r=\n{r}')
r43 = np.reshape(r, (4, 3))

# 여기서 파라미터가 -1로 지정될 때가 있는데 이것의 의미는 다음과 같다.
# One shape dimension can be -1.
# In this case, the value is inferred from the length of the array and remaining dimensions.
# 즉, -1이 아닌 배열을 먼저 구성한 다음에 남은 원소로 나머지 배열을 이루라는 뜻이다.

rv = np.reshape(r, (-1, 1))     # X x 1 어레이. 12개의 원소이므로 (12x1)이 된다.
print(f'\n2) rv.shape={rv.shape}, rv=\n{rv}')
# rv.shape=(12, 1), rv=
# [[ 1]
#  [ 2]
#  [ 3]
#  [ 4]
#  [ 5]
#  [ 6]
#  [ 7]
#  [ 8]
#  [ 9]
#  [10]
#  [11]
#  [12]]

r1 = np.reshape(r, (1, -1))     # 1 x X 어레이. 12개의 원소이므로 (1x12)이 된다.
print(f'\n3) r1.shape={r1.shape}, r1=\n{r1}')
# r1.shape=(1, 12), r1=
# [[ 1  2  3  4  5  6  7  8  9 10 11 12]]

r2 = np.reshape(r, (2, 3, -1))     # (2, 3) 어레이로 총 12개의 원소가 되려면 (2, 3) 어레이가 2개ㅔ 있어야 한다.
print(f'\n4) r2.shape={r2.shape}, r2=\n{r2}')
# r2.shape=(2, 3, 2), r2=
# [[[ 1  2]
#   [ 3  4]
#   [ 5  6]]
#
# [[ 7  8]
#   [ 9 10]
#   [11 12]]]

print(f'\n5) reshape(i, -1)에서 i=1, 2, 3, 4, 6 일 때의 반환 값을 살펴본다.')
for i in [1, 2, 3, 4, 6]:
    rr = r.reshape(i, -1)
    print(f'\nr.reshape({i}, -1): ', rr.shape, '\n', rr)

