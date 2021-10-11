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
print("실습 3: linspace(start, stop, num): 시작점부터 종착점까지를 num 단계로 나눈 ndarray를 반환한다.\n")
# numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
# Return evenly spaced numbers over a specified interval.
# Returns num evenly spaced samples, calculated over the interval [start, stop].
# If dtype is not given, infer the data type from the other input arguments.
# The endpoint of the interval can optionally be excluded.
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html?highlight=linspace#numpy.linspace
#--------------------------------------------------------------------------

print("참고: 간격 =", 2/(9-1) )
# 참고: 간격 = 0.25

# 0~2까지를 9개로 나눈 어레이를 반환한다.
# 양쪽 경계가 모두 포함된다.
x = np.linspace(start=0, stop=2, num=9)
print_simple('1)', 'x')
# 1) x=[0.   0.25 0.5  0.75 1.   1.25 1.5  1.75 2.  ]
# type(x)=<class 'numpy.ndarray'>
# x.shape=(9,)    x.dtype=float64

for i in x:     # x는 ndarray이지만 sequence data이다.
    print(i, end=' ')
print()

from numpy import pi
b = np.linspace(0, pi, 5)                 # 0~pi까지를 5개로 나눈다.
print_simple('2)', 'b')
# 2) b=[0.         0.78539816 1.57079633 2.35619449 3.14159265]
# type(b)=<class 'numpy.ndarray'>
# b.shape=(5,)    b.dtype=float64
