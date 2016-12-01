#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

#網路上找的dataset 可以線性分割

dataset1 = np.array([
((1, -0.4, 0.3), -1),
((1, -0.3, -0.1), -1),
((1, -0.2, 0.4), -1),
((1, -0.1, 0.1), -1),
((1, 0.9, -0.5), 1),
((1, 0.7, -0.9), 1),
((1, 0.8, 0.2), 1),
((1, 0.4, -0.6), 1)])

#PLA演算法實作

def naive_pla(dataset):

    #判斷有沒有分類錯誤，並列印錯誤率

    def check_error(w, dataset):
        result = None, None
        has_err = False
        for x, s in dataset:
            x = np.array(x)
            print("w: {}. data: {}".format(w, x))
            if int(np.sign(w.T.dot(x))) != s:
                has_err = True
                result = x, s
                print("error")
                break

        return result, has_err

    w = np.zeros(3)
    
    while True:
        (x, s), has_err = check_error(w, dataset)
        if not has_err:
            break
        w += s * x

    return w

dataset2 = np.array([
((1, -0.4, 0.3), -1),
((1, -0.3, -0.1), -1),
((1, -0.2, 0.4), -1),
((1, -0.1, 0.1), -1),
((1, 0.9, -0.5), -1),
((1, 0.7, -0.9), 1),
((1, 0.8, 0.2), 1),
((1, 0.4, -0.6), 1)])

def pocket_pla(dataset):

    def check_err(w, dataset):
        result = None, None
        err = 0
        for x, s in dataset:
            x = np.array(x)
            if int(np.sign(w.T.dot(x))) != s:
                result = x, s
                err += 1
        print("error: {} / {}".format(err, len(dataset)))

        return result, err

    w = np.random.rand(3)
    min_err = len(dataset)
    min_w = np.copy(w)
    num = 0
    while True:
        num += 1
        if num > 10:
            break

        (x, s), err = check_err(w, dataset)
        if min_err > err:
            min_err = err
            min_w = np.copy(w)

        print("min_w: {}, w: {}, min_err: {}, err: {}".format(min_w, w, min_err, err))
        if err == 0:
            break
        w += s * x

    return min_w

dataset = dataset2

#執行
w = pocket_pla(dataset)
print("Final w: {}".format(w))

#畫圖

ps = [v[0] for v in dataset]
fig = plt.figure()
ax1 = fig.add_subplot(111)

#dataset前半後半已經分割好 直接畫就是

ax1.scatter([v[1] for v in ps[:4]], [v[2] for v in ps[:4]], s=10, c='b', marker="o", label='O')
ax1.scatter([v[1] for v in ps[4:]], [v[2] for v in ps[4:]], s=10, c='r', marker="x", label='X')
l = np.linspace(-2,2)
a,b = -w[1]/w[2], -w[0]/w[2]
ax1.plot(l, a*l + b, 'b-')
plt.legend(loc='upper left');
plt.show()


