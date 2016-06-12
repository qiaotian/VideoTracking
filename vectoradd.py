# @Author: Tian Qiao <qiaotian>
# @Date:   2016-06-06T10:54:12+08:00
# @Email:  qiaotian@me.com
# @Last modified by:   root
# @Last modified time: 2016-06-13T00:25:32+08:00
# @License: DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE



import numpy as np
import tensorflow as tf
from timeit import default_timer as timer

def main():
    N = 32000000

    A = tf.ones((N,1), tf.int32)
    B = tf.ones((N,1), tf.int32)
    C = tf.mul(A, B)
    sess = tf.Session()

    start = timer()
    sess.run(C)
    vectoradd_time = timer() - start

    print("VectorAdd took %f seconds" % vectoradd_time)

if __name__ == '__main__':
    main()
