#!/usr/bin/env python3

from .MNIST_make_all import make_MNIST

def main():

    print('Testing MNIST module:')

    make_MNIST('/tmp/foo')

    print('\tMNIST_make_all class test passed')

if __name__ == '__main__':
    main()
