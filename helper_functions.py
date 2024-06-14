
import numpy as np
def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError('Object of type ndarray is not JSON serializable')