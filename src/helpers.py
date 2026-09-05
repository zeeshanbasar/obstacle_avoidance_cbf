import numpy as np

def heading_wrap(a):

    return (a + np.pi) % (2*np.pi) - np.pi

def path_length(x, y):
    n = len(x)
    lv = [np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2) for i in range(1,n)]
    L = sum(lv)

    return L