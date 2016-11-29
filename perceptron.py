#!/usr/bin/env python3

import numpy as np


def learn_arr(p_arr, check_err_arr):
    learn_factor = 0.2

    delta_weight = np.outer(p_arr, check_err_arr) * learn_factor
    delta_bias = check_err_arr * learn_factor
    return delta_weight, delta_bias

p = [[1, -1, -1],
    [1, 1, -1],
    [-1, 1, -1],
    [1, -1, 1],
]

target = [[-1, -1],
    [-1, 1],
    [1, -1],
    [1, 1],
]

weight = [[0, 0],
    [0, 0],
    [0, 0],
]

bias = [0, 0]

p_arr = np.array(p)
wei_arr = np.array(weight).astype(float)
tar_arr = np.array(target)
bias_arr = np.array(bias).astype(float)

has_err = True
while has_err:
    has_err = False

    for num in range(len(p_arr)):
        # classify
        a_arr = np.where(np.dot(p_arr[num], wei_arr) + bias_arr >= 0, 1, -1)
        check_err_arr = tar_arr[num] - a_arr
        print("num: {}, check_err_arr: {}".format(num, check_err_arr))

        if (abs(check_err_arr) == 2).any():
            has_err = True
            delta_wei, delta_bias = learn_arr(p_arr[num], check_err_arr)
            np.add(wei_arr, delta_wei, wei_arr)
            np.add(bias_arr, delta_bias, bias_arr)
            print("dw: {},\nnew w: {}".format(delta_wei, wei_arr))
            print("db: {}, new b: {}".format(delta_bias, bias_arr))

    print("==========")

print("Final weight: {}".format(wei_arr))
print("Final bias: {}".format(bias_arr))

print("Test: {}".format(np.where(np.dot([-1, 1, 1], wei_arr) + bias_arr >= 0, 1, -1)))
