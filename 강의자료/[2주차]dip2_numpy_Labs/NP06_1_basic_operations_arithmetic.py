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
print("실습 1: 두 행렬의 각 원소별 가감산 및 논리 연산 등..")
#--------------------------------------------------------------------------
a = np.array([20, 30, 40, 50])
b = np.arange(4, dtype=float)

print_simple('1)', 'a')
# 1) a=[20 30 40 50]
# type(a)=<class 'numpy.ndarray'>
# a.shape=(4,)    a.dtype=int32

print_simple('2)', 'b')
# 2) b=[0. 1. 2. 3.]
# type(b)=<class 'numpy.ndarray'>
# b.shape=(4,)    b.dtype=float64

c = a + b
print_n(3, 'c')
# 3) c=[20. 31. 42. 53.]
# c.shape=(4,)     c.dtype=float64

d = a ** 2      # element-wise square
print_n(4, 'd')
# 4) d=[ 400  900 1600 2500]
# d.shape=(4,)     d.dtype=int32

s = 10 * np.sin(a)      # scalar multiplication. np.sin()함수는 어레이의 각 요소에 대해 sin()을 취한 결과를 어레이로 반환한다.
print_n(5, 's')
# 5) s=[ 9.12945251 -9.88031624  7.4511316  -2.62374854]
# s.shape=(4,)     s.dtype=float64

v1 = a > 35       # greater than. a=[20 30 40 50]
# 어레이에 대한 로직 연산은 어레이로 구성된 bool 값을 반환한다. 결과 값 v1도 ndarray 자료형.
print_n_att('6) Grater than', 'v1')
print('a=', a)
# 6) Grater than
# v1=[False False  True  True]
# type(v1)=<class 'numpy.ndarray'>
# v1.shape=(4,)    v1.ndim=1
# v1.dtype=bool    v1.itemsize=1    v1.size=4


v2 = [1, 0, 1, 0]   # v2는 list
print('type(v2)=', type(v2), 'v2=', v2)
# type(v2)= <class 'list'>

v = v1 & v2     # Logical AND. v1=[False False  True  True]
# ndarray와 list간의 연산이 가능하다. 연산 결과는 ndarray, dtype=int32 이다.
print_n_att('7) v = v1 & v2', 'v')
# 7) v = v1 & v2
# v=[0 0 1 0]
# type(v)=<class 'numpy.ndarray'>
# v.shape=(4,)    v.ndim=1
# v.dtype=int32    v.itemsize=4    v.size=4

v3 = [True, False, True, False]   # v1=[False False  True  True]
v = v1 & v3     # Logical AND. ndarray와 list간의 연산이 가능하다. 연산 결과는 ndarray, dtype=bool 이다.
print_n_att('8) v = v1 & v3', 'v')
# 8) v = v1 & v3
# v=[False False  True False]
# type(v)=<class 'numpy.ndarray'>
# v.shape=(4,)    v.ndim=1
# v.dtype=bool    v.itemsize=1    v.size=4

v = v1 | v3     # Logical OR. ndarray와 list간의 연산이 가능하다. 연산 결과는 bool 이다.
print_n_att('9) v = v1 | v3', 'v')
# 9) v = v1 | v3
# v=[ True False  True  True]
# type(v)=<class 'numpy.ndarray'>
# v.shape=(4,)    v.ndim=1
# v.dtype=bool    v.itemsize=1    v.size=4

# array_equal - 두 어레이가 모두 같은 값을 갖고 있는지 확인하는 함수
a = np.array([1, 2, 3, 4])
b = np.array([1.0, 2.0, 3.0, 4.0])
c = np.array([1.0, 2.0, 3.0, 4.01])
ab = np.array_equal(a, b)
print("\n10) array_equal(a, b)=", ab)
# array_equal(a, b)= True

ac = np.array_equal(a, c)
print("\n11) array_equal(a, c)=", ac)
# array_equal(a, c)= False

