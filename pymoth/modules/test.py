def main():

    import os, sys
    sys.path.append(os.getcwd() + os.sep + 'pymoth')

    from MNIST_all import test_MNIST
    import test_classify, test_generate, test_params

    test_MNIST.main()

    test_classify.main()

    test_generate.main()

    test_params.main()

if __name__ == '__main__':
    main()
