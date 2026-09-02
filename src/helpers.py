import numpy as np

def heading_wrap(a):

    return (a + np.pi) % (2*np.pi) - np.pi