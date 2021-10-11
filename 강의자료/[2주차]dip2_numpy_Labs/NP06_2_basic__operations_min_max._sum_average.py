"""

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
print("실습 2 - 행렬의 원소의 min/max/sum/average 구하기, 난수 어레이에 대한 기본 연산")
#
# 참고: random 메소드 검색 방법
#       https://docs.scipy.org/doc/numpy/reference/routines.html 에서 random을 검색
#       검색된 메소드가 맨 위에 올라온다.
#
# average()
#   https://docs.scipy.org/doc/numpy/reference/generated/numpy.average.html?highlight=average#numpy.average
#--------------------------------------------------------------------------
b = np.random.random((100, 100))     # 0~1까지의 균일한 분포를 갖는 난수를 발생하여  100 x 100 어레이에 배정한다.
print_n(0, 'b') # ㅠsms 100 x 100의 어레이
# b.shape=(100, 100)     b.dtype=float64

print('1) b.max =', b.max(), ' np.amax(b)=', np.amax(b))    # max value. 10000개의 원소 중에 가장 큰 값.
print('2) b.min =', b.min(), ' np.amin(b)=', np.amin(b))    # min value. 10000개의 원소 중에 가장 작은 값.
print('3) b.sum =', b.sum(), ' np.sum(b)=', np.sum(b))  # sum value. 10000개의 원소들의 평균 값. 균일 분포이므로 5000에 가까운 값을 갖는다.
print('4) b.sum/b.size =', b.sum()/b.size)      # avg value. b.size는 원소의 개수.
print('5) np.average(b) =', np.average(b))      # avg value. 원소가 많을 수록 0.5 근처의 값을 가진다.

a = np.ones((2, 6), dtype=int)
a *= 3      # a = a*3
print_n('\n6', 'a')
# 6) a=[[3 3 3 3 3 3]
#  [3 3 3 3 3 3]]
# a.shape=(2, 6)     a.dtype=int32


b = np.random.random((2, 6))     # 0~1까지의 균일한 분포를 갖는 난수를 발생하여 어레이에 배정한다.
print_n('7', 'b')
# 7) b=[[0.82678395 0.17724314 0.77842821 0.57034831 0.99443973 0.33542468]
#  [0.95118171 0.49611969 0.89826231 0.08730827 0.7700454  0.3618515 ]]
# b.shape=(2, 6)     b.dtype=float64


b += a
print_n('8', 'b')
# 8) b=[[3.82678395 3.17724314 3.77842821 3.57034831 3.99443973 3.33542468]
#  [3.95118171 3.49611969 3.89826231 3.08730827 3.7700454  3.3618515 ]]
# b.shape=(2, 6)     b.dtype=float64


#a += b     # Error!!! b is not automatically converted to integer type
a = a + b   # 이것은 이상 없음.
print_n('9', 'a')
# 9) a=[[6.99100543 6.34942707 6.93232202 6.98464507 6.72567318 6.68008706]
#  [6.94506372 6.60858167 6.50497092 6.95599134 6.51161018 6.82064767]]
# a.shape=(2, 6)     a.dtype=float64



