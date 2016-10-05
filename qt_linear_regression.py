from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

"""
The origin data format:
    | date | time(s) | x1 | y1 | z1 | a1 | b1 | c1 | d1 |
    | date | time(s) | x2 | y2 | z2 | a2 | b2 | c2 | d2 |
    ......
    | date | time(s) | xn | yn | zn | an | bn | cn | dn |

The target data format:
    | time(ms) | x1 | y1 | z1 | a1 | b1 | c1 | d1 | ... | x8 | y8 | z8 | a8 | b8 | c8 | d8 |
    | time(ms) | x9 | y9 | z9 | a9 | b9 | c9 | d9 | ........................................
    | time(ms) ........................................ | xk | yk | zk | ak | bk | ck | dk |
    where k = (n/8*8)
"""

def convertDataFormat(ipath, opath):
    df = pd.read_csv(ipath, sep=' ')
    print(df.shape)

    # record the start and end time, and then calculate the time period
    time_bgn = df.iat(0, 2)
    time_end = df.iat(df.count-1, 2)
    period = time_bgn - time_end
    #
    df.to_csv(opath)


def main():
    ipath = '/Users/qiaotian/Downloads/dataset/sample1/resp.txt'
    opath = '/Users/qiaotian/Downloads/dataset/sample1/std_resp.txt'
    convertDataFormat(ipath, opath)

    # read csv file directly from local directory
    resp_data = pd.read_csv(opath)

    # display the first 5 row
    # print(resp_data.head())

if __name__ == '__main__':
    main()
