"""

    In linear algebra, a column vector is a column of entries.
    Similarly, a row vector is a row of entries. [위키피디어]

개요:
   1차원 어레이와 numpy scalar를 정의하는 방법에 대하여 살핀다.
    numpy에서 말하는 1차원 어레이는 방향성이 없다.
    1차원 numpy array는 선형대수에서 말하는 열벡터, 행벡터와 다르다.
        선형대수에서 말하는 행벡터는 넘파이에서는 열의 개수가 1개인 2차원 어레이로 표현된다.
        선형대수에서 말하는 열벡터는 넘파이에서는 행의 개수가 1개인 2차원 어레이로 표현된다.

numpy에서는;
   행벡터 혹은 열벡터는 배열의 방향성이 있는, 즉 row/column 벡터는 2차원 매트릭스이다.
       shape = (1, n): 2차원 matrix, row vector
       shape = (n, 1): 2차원 matrix, column vector
   1차원 데이터 배열은 그 방향성은 없다.
        n개의 원소를 갖는 1차원 매트릭스, a의 shape는 a.shape=(n,), a.ndim=1이다.
        shape=(n,): 1차원 매트릭스, 방향이 없으므로 아래의 (1, n)/(n, 1) 매트릭스가 사용되는 연산에 사용할 수 있다.
        1차원 배열은 len() 함수를 이용해 그 원소의 개수를 셀 수 있다.

1차원 매트릭스:

    a.shape=(3,)        # 1x3 row vector가 3x1 column vector도 아닌 방향성이 없는 1차원 데이터이다.
                        # 1x3 row vector와 3x1 column vector는 모두 2차원(ndim=2)이다.
    a.ndim=1            # 1차원 matrix이다.

   참고: python의 sequence 자료형과 유사. string, list, tuple은 sequence 자료형이라 한다.
    그러나, 연산에 있어서는 sequence 자료형과 다른다.
    사례:  sequence 자료형에 대한 +연산은 python에서는 concatenation으로 실현되지만, numpy에서는 각 요소에 대한 개별 덧셉으로 이루어진다.





"""

import numpy as np
import sys      # sys.getsizeof() 함수는 객체 전체가 차지하고 있는 용량(바이트)를 반환한다.


# ---------------------------------------------------------------------------------------------
# print_att()
#   입력 파라미터 변수(ndarray)에 대한 정보를 출력하는 함수
#   입력 파라미터는 변수가 아니라 변수의 이름을 담은 스트링이다.
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

    # itemsize: 한 원소를 표현하기 위해 필요한 데이터의 바이트 수
    print(f"{str_a}.itemsize={a.itemsize}")

    # size: array에 들어 있는 총 원소의 수
    print(f"{str_a}.size={a.size}")

    import sys  # sys.getsizeof() 함수는 객체 전체가 차지하고 있는 용량(바이트)를 반환한다.
    print(f"sys.getsizeof({str_a})={sys.getsizeof(a)}")


# --------------------------------------------------------------------------------------------------
#
# parm = ((1, 2, 3))      # (3,) array. 1차원
# parm = (1, 2, 3)      # (3,) array. 1차원
# parm = [1, 2, 3]      # (3,) array. 1차원
# parm = [[1, 2, 3]]      # (1, 3) array. 2차원
# parm = [(1, 2, 3)]      # (1, 3) array. 2차원
# parm = [((1, 2, 3))]      # (1, 3) array. 2차원
# --------------------------------------------------------------------------------------------------

# 1) 1차원 매트릭스 선언
print("\n1) 1차원 매트릭스 선언")
parm = [2, 3, 4]            # a.dtype=int32, a.itemsize=4
a = np.array(parm)
print_att('a')
# a=[2 3 4]
# type(a)=<class 'numpy.ndarray'>
# a.shape=(3,)          # 1x3 row vector가 아님에 유의. 그러나 대개의 경우 큰 무리없이 활용가능하다.
# 1x3 혹은 3x1벡터가 필요한 곳에 (3,) sequence를 활용가능하다.
# a.ndim=1
# a.dtype=int32
# a.itemsize=4
# a.size=3
# number of elements=3

# 2) 1차원 매트릭스는 치환행렬을 취해도 그 내용이 변하지 않는다. 방향이 없기 때문이다..
print("2) 1차원 매트릭스는 치환행렬을 취해도 그 내용이 변하지 않는다. 방향이 없기 때문이다..")
a_T = a.T            # a_T=a.transpose() 도 같은 표현이다.
print_att('a_T')

# a와 a_T가 같은지 확인해 본다.
print(type(a == a.T), a == a.T, (a == a.T).all)
if (a == a.T).all:
    print(f"Every element is equal to that of a.T")
print()

# a_T=[2 3 4]           # 1차원이면 a = a.T
# type(a_T)=<class 'numpy.ndarray'>
# a_T.shape=(3,)        # 3x1 column vector가 아님에 유의.
# a_T.ndim=1
# a_T.dtype=int32
# a_T.itemsize=4
# a_T.size=3
# sys.getsizeof(a_T)=104


# --------------------------------------------------------------------------------------------------
# 0차원 매트릭스.
# 즉, scalar. 파이썬의 int나 float와 다른 점은 여기서는 numpy.ndarray 형이라는 것이다.
# 이 경우 0차원 ndarray 데이터는 shape=()가 된다.
# --------------------------------------------------------------------------------------------------

print("3) 0차원 매트릭스 선언")
parm = 2 ** 128
print('type(parm)=', type(parm))
a = np.array(parm)
print_att('a')

"""
# 참고 --------------
# 64비트 이하의 데이터에 대해서는 8비트 truncation 가능.
# 위의 parm에 대해서는 처리 불가능. 비트가 크면 안됨
parm2 = 0x1234567890abcdef
b = (np.array(parm2)).astype(np.uint8)
print_att('b')
print(f'b={b:#x}')
"""