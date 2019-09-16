# -*- coding: utf-8 -*-
# -*- coding: big5 -*-
import numpy as np

def NumNode_(x, w, b):
    return np.dot(x, w) + b

def NumReLU_(x):
    return np.maximum(x, 0)

def NumSigmoid_(x):
    return 1/(1 + np.exp(-x))

def NumTanh_(x):
    return (1 - np.exp(-2 * x))/(1 + np.exp(-2 * x))

def NumSoftmax_(x):
    '''
    回傳陣列機率分佈 [0, 1]
    '''
    return x/(np.sum(x) + 1E-9)

def NumCrossEntropy_(x, y):
    '''
    回傳陣列交叉熵
    '''
    return - np.sum(x * np.log10(y))

def NumMeanSquaredError(x, y):
    '''
    回傳陣列均方誤差
    '''
    return np.sum(np.square(x - y)) / x.size

def NumArgMax_(x, axis= None):
    '''
    回傳陣列最大值之索引值
    '''
    return np.argmax(x, axis)

def NumArgMin(x, axis= None):
    '''
    回傳陣列最小值之索引值
    '''
    return np.argmin(x, axis)

def NumL1Regularization(x):
    '''
    回傳L1正規化
    '''
    return np.sum(np.abs(x))

def NumL2Regularization(x):
    '''
    回傳L2正規化
    '''
    return np.sum(np.abs(np.square(x)))

def NumL1L2Regularization(x, apha):
    '''
    回傳L1與L2正規化
    '''
    return np.sum(apha * np.abs(x) + (1 - apha) * np.square(x))

def NumShadow(x, y, decay):
    '''
    回傳新的影子變數
    '''
    return decay * y + (1 - decay) * x

if __name__ == "__main__":
    pass