#!/usr/bin/env python3

def main():

    print('Testing MNIST module:')

    import MNIST_make_all

    MNIST_make_all.make_MNIST()

    print('\tMNIST_make_all class test passed')

if __name__ == '__main__':
    main()
