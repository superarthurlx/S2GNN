import numpy as np 

def time_in_day(data_norm, steps_per_day, tod=None):
    l, n, f = data_norm.shape
    # numerical time_of_day
    if tod is None:
        tod = [i % steps_per_day / steps_per_day for i in range(data_norm.shape[0])]
        tod = np.array(tod)
    tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
    return tod_tiled

def day_in_week(data_norm, steps_per_day, dow=None):
    l, n, f = data_norm.shape
    if dow is None:
        dow = [(i // steps_per_day) % 7 / 7 for i in range(data_norm.shape[0])]
        dow = np.array(dow)
    dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
    return dow_tiled

def day_in_month(data_norm, steps_per_day, dom=None):
    l, n, f = data_norm.shape
    if dom is None:
        raise Exception("dom index is none")
    dom_tiled = np.tile(dom, [1, n, 1]).transpose((2, 1, 0))
    return dom_tiled

def day_in_year(data_norm, steps_per_day, doy=None):
    l, n, f = data_norm.shape
    if doy is None:
        raise Exception("doy index is none")
    doy_tiled = np.tile(doy, [1, n, 1]).transpose((2, 1, 0))
    return doy_tiled
