"""
실습 1: 두 행렬의 각 원소별 가감산 및 논리 연산 등..
실습 2 - 행렬의 원소의 min/max/sum/average를 numpy 함수로 구하기, 난수 어레이에 대한 기본 연산
실습 3 - linspace(start, stop, num): 시작점부터 종료지점까지를 num 단계로 나눈 ndarray를 반환한다.
        양쪽 단의 경계 값을 포함한다.
실습 4 - universal functions, exp, sqrt

"""

import numpy as np

# ---------------------------------------------------------------------------------------------
# print_n_att()
#   출력문의 번호를 먼저 프린트 한 후,
#   입력 파라미터 변수(ndarray)에 대한 정보를 출력한다.
#   입력 파라미터는 변수의 스트링을 취한다.
# ---------------------------------------------------------------------------------------------


def print_n_att(num, str_a):
    print()
    print(num)

    # str_a를 스트링으로 간주하고 실제 값을 evaluate 한다.
    a = eval(str_a)
    print(f"\n{str_a}={a}")

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
    print(f"{str_a}.size={a.size}\n")


def print_simple(num, str_a):        # num = 실험 번호 스트링, str_a = 변수 스트링
    print()
    a = eval(str_a)
    print("{} {}={}".format(num, str_a, a))
    print("type({})={}".format(str_a, type(a)))
    print("{}.shape={}".format(str_a, a.shape), end='    ')
    print("{}.dtype={}".format(str_a, a.dtype))


def print_n(num, str_a):           # num=실험 번호, str_a = 객체를 지정하는 스트링
    print()
    a = eval(str_a)
    print("{}) {}={}".format(num, str_a, a))
    print("{}.shape={}".format(str_a, a.shape), end='    ')
    print(" {}.dtype={}".format(str_a, a.dtype))

#--------------------------------------------------------------------------
print("실습 4: universal functions, exp, sqrt")
# NumPy provides familiar mathematical functions such as sin, cos, and exp.
# In NumPy, these are called “universal functions”(ufunc).
# Within NumPy, these functions operate elementwise on an array, producing an array as output.
#--------------------------------------------------------------------------
B = np.arange(3.0)
B = [[1.0, 2], [3, 4]]
B = np.array(B)
print_n('1', 'B')
# 1) B=[[1. 2.]
#  [3. 4.]]
# B.shape=(2, 2)     B.dtype=float64


eB = np.exp(B)        # exponential
print_n('2', 'eB')
# 2) eB=[[ 2.71828183  7.3890561 ]
#  [20.08553692 54.59815003]]
# eB.shape=(2, 2)     eB.dtype=float64


sB = np.sqrt(B)       # square root
print_n('3', 'sB')
# 3) sB=[[1.         1.41421356]
#  [1.73205081 2.        ]]
# sB.shape=(2, 2)     sB.dtype=float64


A = np.array([[1, 2], [3, 4]])
print_n('4', 'A')
# 4) A=[[1 2]
#  [3 4]]
# A.shape=(2, 2)     A.dtype=int32

B = np.array([[1.0, 2], [3, 4]])
print_n('5', 'B')
# 5) B=[[1. 2.]
#  [3. 4.]]
# B.shape=(2, 2)     B.dtype=float64


ApB = np.add(A, B )
print_n('6', 'ApB')
# 6) ApB=[[2. 4.]
#  [6. 8.]]
# ApB.shape=(2, 2)     ApB.dtype=float64

AmB = A - B
print_n('7', 'AmB')
# 7) AmB=[[0. 0.]
#  [0. 0.]]
# AmB.shape=(2, 2)     AmB.dtype=float64

Asq = A ** 2          # 제곱
print_n('8', 'Asq')
# 8) Asq=[[ 1  4]
#  [ 9 16]]
# Asq.shape=(2, 2)     Asq.dtype=int32

exit(0)





