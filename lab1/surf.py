import matlab.engine
import numpy as np

def surfFeatures(img, nfeatures = 300):
    eng = matlab.engine.start_matlab()
    ret1, ret2=eng.surf(img,nargout=2)
    dp1 = np.array(ret1)
    kp1 = np.array(ret2)
    return kp1, dp1