"""
numpy 배열 조작에 관계된 함수 혹은 메서드
NumPy Link : https://docs.scipy.org/doc/numpy/index.html

    1. arange: 연속적인 원소로 이루어진 1차원 배열을 반환한다.
    2. reshape: shape를 지정한 대로 바꾸어 반환한다. 데이터를 원본과 공유한다.
    3. resize: 원본 자체의 모양을 바꾼다. 반환 값이 없다.
    4. ravel: 임의의 배열을 1차원 배열로 만들어 반환한다. 데이터를 원본과 공유한다.
    5. flatten: 임의의 배열을 1차원 배열로 만들어 반환한다. ravel과는 달리 데이터를 원본과 공유하지 않는다.

"""

import numpy as np

print("실습 3_4_5: arange, reshape, resize, ravel, flatten 사용법 비교 검토 및 ndarray 동일성 검토")

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

print()

print('\n사례 1 ----- reshape()를 사용하여 원본 A의 모양을 바꾼 바꾼 경우. 바뀐 객체 B는 A와 자료를 공유한다.')
n1 = 3; n2 = 2
A = np.arange(n1 * n2)
print("A =", A)
# A = [0 1 2 3 4 5]

B = A.reshape(n1, n2)       # 3x2 array로 변환
print("B =\n", B)
# B =
#  [[0 1]
#  [2 3]
#  [4 5]]

# 다른 배열의 데이터를 공유하고 있는 뷰라면 해당 배열이 출력됩니다.
print("A.base =", A.base)      # A는 자체 생성하였으므로 base는 다른 객체를 지정하지 않는다.
# A.base = None

print("B.base =", B.base)  # 배열 B의 base를 출력한다. 어떤 값을 참조하는지 보인다.
# B.base = [0 1 2 3 4 5]       다른 객체를 지정한다.

# B.base와 A가 동일한 객체인지 확인해봅니다.
if B.base is A:  # is는 똑같은 객체인지 확인할 때 사용합니다.
    print("B는 A의 데이터를 공유한다.")
# 출력: B는 A의 데이터를 공유한다.

# A를 수정했는데 B도 같이 바뀌었다.
A[0] = 100
print("A =\n", A)
print("B =\n", B)
# B.base = [100   1   2   3   4   5]


print(f"참고: id(A) = {id(A)}, id(B) = {id(B)}")
# id(object)는 객체를 입력받아 객체의 고유 주소 값(레퍼런스)을 돌려주는 파이썬 함수이다.
# 같은 객체를 가리킨다면 id도 같을 것이다.
# id는 다르다.
# id(A) = 2160717135104, id(B) = 2161101985632
# base는 파이썬의 id()와 유사한 역할을 수행하는 numpy의 메서드이다.

A[0] = 9                    # A를 바꾸었는데 B의 내용도 같이 변한다.
print("B =\n", B)           # B[0]도 바뀐다.
# B =
#  [[9 1]
#  [2 3]
#  [4 5]]


print('\n사례 2 ------- copy()를 사용하여 객체를 복사할 경우. 자료가 copy되어 새로운 객체가 생성된다.')
A = np.arange(n1 * n2)
B = A.reshape(n1, n2).copy()        # 3x2 array로 변환한 후 copy()하여 객체 B를 생성한다.

print("A.base =", A.base)          # 자체 생성
# A.base = None

print("B.base =", B.base)          # 자체 생성
# B.base = None

if B.base is A:
    print("B는 A의 데이터를 공유한다.")
else:
    print("B는 A의 데이터를 공유하지 않는다.")
# 출력: B는 A의 데이터를 공유하지 않는다.

A[0] = 9                      # A를 바꾸어도 B는 바뀌지 않는다.
print("B =\n", B)
# B =
#  [[0 1]               B[0]가 바뀌지 않았다.
#  [2 3]
#  [4 5]]


print('\n사례 3 ---------- 원본 자체의 모양을 바꾸는 메서드 resize()')
C = A.resize(n2, n1)        # resize메서드는 반환 값이 없음에 유의. n1 = 3; n2 = 2
print("A =\n", A)           # resize는 해당 어레이의 원본 자체를 수정하여 크기를 조정한다.
# A =
#  [[9 1 2]
#  [3 4 5]]

print("C =\n", C)           # resize()에서는 반환하는 값이 없다. 원본 객체를 바꾼다.
# C =
#  None



print('\n사례 4 ------------ ravel: n차 배열을 1차원 배열로 만들어 반환한다. 데이터는 원본과 공유한다.')
AA = B.ravel()                    # AA는 B를 공유.
print("AA =", AA)
# AA = [0 1 2 3 4 5]
AA[0] = 100
print("AA.base =\n", AA.base)    # 공유하므로 base는 B의 위치에 있는 값을 가리킨다.
# AA.base =
# [[100 1]
#  [2 3]
#  [4 5]]
print("B.base =", B.base)
# B.base = None
print('B=\n', B)
# B=
#  [[100   1]
#  [  2   3]
#  [  4   5]]

# ravel() 메서드를 4차원 데이터에 적용해 본다.
AA.resize([n1, n2, 1, 1])     # n1 = 3; n2 = 2. 4차원으로 만들어 본다.
print(f'AA.shape={AA.shape}, AA.ndim={AA.ndim}')
CC = AA.ravel()         # ravel()은 차원을 가리지 않고 1차원으로 만든다.
print(f'CC.shape={CC.shape}', CC)


print('\n사례 5 ------------ flatten: n차 배열을 1차원 배열로 만들어 반환한다.\
 ravel()과는 달리 반환 값은 원본과 데이터를 공유하지 않는다.')
AA = B.flatten()                    #
print("AA =", AA)
# AA = [100 1 2 3 4 5]
print("AA.base =", AA.base)    # 공유하지 않고 새로 생성된 개체이기 때문에 base는 None의 값을 갖는다.
# AA.base = None
AA[0] = -1
print(f'AA=\n{AA}')
# [-1  1  2  3  4  5]
print(f'B=\n{B}')               # 공유하지 않기 때문에 A를 바꾸었다고 같이 바뀌지는 않는다.
# [[100   1]
#  [  2   3]
#  [  4   5]]


