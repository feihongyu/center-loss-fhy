import os
import sys

import cv2
import numpy
import torch

from models import BackboneByResNet18


def t1():
    import os
    path1 = r"E:\datasets\stanford car\concat"  # 输入一级文件夹地址
    files1 = os.listdir(path1)  # 读入一级文件夹
    num1 = len(files1)  # 统计一级文件夹中的二级文件夹个数
    num2 = []  # 建立空列表
    all = 0
    for i in range(num1):  # 遍历所有二级文件夹
        path2 = path1 + '//' + files1[i]  # 某二级文件夹的路径
        files2 = os.listdir(path2)  # 读入二级文件夹
        num2.append(len(files2))  # 二级文件夹中的文件个数
        all += len(files2)

    print("所有二级文件夹名:")
    print(files1)  # 打印二级文件夹名称
    print("所有二级文件夹中的文件个数:")
    print(num2)  # 打印二级文件夹中的文件个数

    print("对应输出:")
    xinhua = dict(zip(files1, num2))  # 将二级文件夹名称和所含文件个数组合成字典
    for key, value in xinhua.items():  # 将二级文件夹名称和所含文件个数对应输出
        print('{key}:{value}'.format(key=key, value=value))
    print(all)

def cuttensor():
    a = torch.randint(10,(2,5))
    print(a)
    b = a[:, 0:2]
    print(b)

def fun1():
    print(sys.platform)

def fun2():
    a = numpy.array(
        [
            [1,2],
            [3,4],
            [5,6]
        ],dtype=numpy.int32
    )
    numpy.savetxt("1.txt", a)
    b = numpy.loadtxt("1.txt")
    print(b)

def fun3():
    a = numpy.array(
        [
            [1,2],
            [2,3],
            [1,1]
        ]
    )
    b = numpy.array(
        [
            1,1,1
        ]
    )
    b = b[:, None]

    print(numpy.concatenate((a,b), axis=-1))

if __name__ == '__main__':
    fun3()