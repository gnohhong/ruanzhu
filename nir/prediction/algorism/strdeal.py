import numpy as np

def splitstr(str, symbol, types):
    lists = str.split(symbol)

    if (types == 'float'):
        nlist = list(map(float, lists))
    elif (types == 'int'):
        nlist = list(map(int, lists))

    return nlist
