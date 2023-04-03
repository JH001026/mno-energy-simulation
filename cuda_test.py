import math
import random

import numba
import numpy.typing
from numba import jit, njit, types, vectorize, cuda
import time
import numpy as np
import numpy.typing as npt
import base.util as util

@jit(nopython=True)
def monte_carlo_pi(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples


def monte_carlo_pi_no_numba(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples


@vectorize
def scalar_computation(num):
    if num % 2 == 0:
        return 2
    else:
        return 1


@jit
def jitted_function(input_list):
    output_list = []
    for item in input_list:
        if item % 2 == 0:
            output_list.append(2)
        else:
            output_list.append('1')
    return output_list


def original_function(input_list):
    output_list = []
    for item in input_list:
        if item % 2 == 0:
            output_list.append(2)
        else:
            output_list.append('1')
    return output_list


def add_one_unoptimized(A):
    for i in range(len(A)):
        A[i] = A[i] + 1


##############################################################
#                                                            #
#                      Python x CUDA                         #
#                                                            #
##############################################################

@cuda.jit
def kernel_add_one_with_some_func(A):
    pos = cuda.grid(1)
    if pos < A.size:
        A[pos] = A[pos] + 1
        some_func(A)


@cuda.jit
def kernel_add_one(A):
    pos = cuda.grid(1)
    if pos < A.size:
        A[pos] = A[pos] + 1


@cuda.jit
def kernel_test(u_ids, u_xs, u_ys, BS_IS, bs_xs, bs_ys, RES):

    pos = cuda.grid(1)
    if pos < u_ids.size:
        RES[pos][0] = 1
        # for i in range(len(bs_xs)):
        #     RES[pos][i] = 1
        #
        # for i in range(10):
        #     for j in range(len(bs_xs)):
        #         BS_IS[pos][j] = 1

    # def is_in(item, l):
    #     for e in l:
    #         if item == e:
    #             return True
    #     return False
    #
    # pos = cuda.grid(1)
    # if pos < u_ids.size:
    #     u_id, u_x, u_y = u_ids[pos], u_xs[pos], u_ys[pos]
    #
    #     for i in range(len(bs_xs)):
    #         # Fill array with distances
    #         RES[pos][i] = math.sqrt(pow((bs_xs[i] - u_x), 2) + pow((bs_ys[i] - u_y), 2))
    #
    #     # Argsort the 10 closest base stations and store the result in BS_IS[pos]
    #     for i in range(10):
    #         m_i = 0
    #         m = np.inf
    #         for j in range(len(bs_xs)):
    #             if RES[pos][j] < m and not is_in(j, BS_IS[pos][:i]):
    #                 m = RES[pos][j]
    #                 m_i = j
    #         BS_IS[pos][i] = m_i

@njit
def some_func(A):
    pass


def kernel_find_bs_close_nc(u_ids, u_xs, u_ys, bs_xs, bs_ys, RES):
    for i in range(len(u_ids)):
        u_id, u_x, u_y = u_ids[i], u_xs[i], u_ys[i]
        for i in range(len(bs_xs)):
            RES[i][i] = pow((bs_xs[i] - u_x), 2) + pow((bs_ys[i] - u_y), 2)




@cuda.jit
def kernel_find_bs_close(u_ids, u_xs, u_ys, BS_IS, bs_xs, bs_ys, RES):
    """
    Kernel that finds the closest BS to all users
    :param bs_is: indices (NOT IDs) in RES that correspond to lowest 10 distanced base stations
    """
    def is_in(item, l):
        for e in l:
            if item == e:
                return True
        return False

    pos = cuda.grid(1)
    if pos < u_ids.size:
        u_id, u_x, u_y = u_ids[pos], u_xs[pos], u_ys[pos]

        for i in range(len(bs_xs)):
            # Fill array with distances
            RES[pos][i] = math.sqrt(pow((bs_xs[i] - u_x), 2) + pow((bs_ys[i] - u_y), 2))

        # Argsort the 10 closest base stations and store the result in BS_IS[pos]
        for i in range(10):
            m_i = 0
            m = np.inf
            for j in range(len(bs_xs)):
                if RES[pos][j] < m and not is_in(j, BS_IS[pos][:i]):
                    m = RES[pos][j]
                    m_i = j
            BS_IS[pos][i] = m_i

"""
Now for these base stations, we find the best channel for each user
We need:
    SHARED: bs.id, bs.x, bs.y, bs.radio, bs.area_type
    For sinr: u.id, u.x, u.y, c.height, c.frequency, c.bandwidth, c.bs_interferers
"""



def print_timing(f, args, kind='unknown'):
    start = time.time()
    f(*args)
    print(f'{kind} run took {(time.time() - start) * 1000} ms')


def test_find_link_bs_kernel():
    u_ids_cpu = np.random.randint(0, 10000, 100000, dtype=np.int32)
    u_ids_gpu = cuda.to_device(u_ids_cpu)

    u_xs_cpu = np.asarray(np.random.uniform(0, 200000, 100000), dtype=np.float64)
    u_xs_gpu = cuda.to_device(u_xs_cpu)
    u_ys_cpu = np.asarray(np.random.uniform(0, 200000, 100000), dtype=np.float64)
    u_ys_gpu = cuda.to_device(u_ys_cpu)

    BS_IS_cpu = np.asarray(np.zeros(shape=(100000, 10)), dtype=np.int32)
    BS_IS_gpu = cuda.to_device(BS_IS_cpu)

    bs_xs_cpu = np.asarray(np.random.uniform(0, 200000, 1000), dtype=np.float64)
    bs_xs_gpu = cuda.to_device(bs_xs_cpu)
    bs_ys_cpu = np.asarray(np.random.uniform(0, 200000, 1000), dtype=np.float64)
    bs_ys_gpu = cuda.to_device(bs_ys_cpu)

    RES_cpu = np.asarray(np.zeros(shape=(100000, 1000)), dtype=np.float64)
    RES_gpu = cuda.to_device(RES_cpu)

    threadsperblock = 32
    blockspergrid = u_ids_cpu.size + (threadsperblock - 1)
    print_timing(kernel_test[blockspergrid, threadsperblock], (u_ids_gpu, u_xs_gpu, u_ys_gpu, BS_IS_gpu
                                                                        , bs_xs_gpu, bs_ys_gpu, RES_gpu), 'TESTING')

    # print_timing(kernel_find_bs_close[blockspergrid, threadsperblock], (u_ids_gpu, u_xs_gpu, u_ys_gpu, BS_IS_gpu
    #                                                                     , bs_xs_gpu, bs_ys_gpu, RES_gpu)
    #              , 'Compiling linking kernel')
    # print_timing(kernel_find_bs_close[blockspergrid, threadsperblock], (u_ids_gpu, u_xs_gpu, u_ys_gpu, BS_IS_gpu
    #                                                                     , bs_xs_gpu, bs_ys_gpu, RES_gpu)
    #              , 'Running linking kernel')

    # print_timing(kernel_add_one[blockspergrid, threadsperblock], [RES_gpu], "SEOCOND")


    # print_timing(kernel_find_bs_close_nc, (u_ids_cpu, u_xs_cpu, u_ys_cpu, bs_xs_cpu, bs_ys_cpu, RES_cpu)
    #              , 'Running linking no cuda')

    # print(RES_gpu.copy_to_host()[0][:100])
    # print(RES_gpu.copy_to_host())

    # Test if these two are equal...
    # print(BS_IS_gpu.copy_to_host()[0])
    # print(np.argsort(RES_gpu.copy_to_host()[0])[:10])

def shift_right_insert(value, li):
    start = -1
    for i in range(len(li)):  # Loop through indices of array
        if value > li[i]:  # Sorted array, so if value > li[i], we have insert index
            start = i  # Set start to index
            break  # Break out of loop

    index = len(li) - 1
    if start > index or start < 0:  # Impossible shift
        return li
    while True:
        print(index)
        if index == start:
            li[index] = value  # Insert value
            return li
        li[index] = li[index - 1]  # Shift values right
        index = index - 1  # Reduce index


if __name__ == '__main__':
    print('huh')
    l1 = [4,3,2,1]
    print(shift_right_insert(-1, l1))

    # print_timing(monte_carlo_pi, [1], 'Compilation')
    #
    # print_timing(monte_carlo_pi_no_numba, [1000000], 'nojit')
    #
    # print_timing(monte_carlo_pi, [1000000], 'jit')
    #
    # test_array = np.arange(100000)
    # print_timing(scalar_computation, [1], 'Vectorize compile')
    # print_timing(scalar_computation, [test_array], 'vectorized with array')
    # # print_timing(jitted_function, test_array, 'jitted function')
    # print_timing(original_function, [test_array], 'original function')
    #
    # # Python x CUDA
    array_dummy_cpu = np.random.randint(0, 1, size=1)
    array_dummy_gpu = cuda.to_device(array_dummy_cpu)

    array_cpu = np.random.randint(0, 100, size=4000000)
    array_gpu = cuda.to_device(array_cpu)

    threadsperblock = 32
    blockspergrid = array_cpu.size + (threadsperblock - 1)

    # Make sure functions are compiled
    kernel_add_one[blockspergrid, threadsperblock](array_dummy_gpu)
    kernel_add_one_with_some_func[blockspergrid, threadsperblock](array_dummy_gpu)

    print_timing(add_one_unoptimized, [array_cpu], 'Unoptimized run')
    #
    # # Time compiled cuda run
    print_timing(kernel_add_one_with_some_func[blockspergrid, threadsperblock], [array_gpu], 'GPU add_one with some_func')

    print_timing(kernel_add_one[blockspergrid, threadsperblock], [array_gpu], 'GPU add_one without nested function call')
    #
    # test_find_link_bs_kernel()
    #
    # print(array_gpu.copy_to_host())

    # test_find_link_bs_kernel()

