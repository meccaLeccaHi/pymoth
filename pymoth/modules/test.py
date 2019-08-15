def main():

    import os, sys


    start_dir = os.getcwd()
    print(start_dir)

    sys.path.append(start_dir + os.sep + 'MNIST_all')

    import test_MNIST, test_classify, test_generate, test_params

    test_MNIST.main()
    
    test_classify.main()

    test_generate.main()

    test_params.main()

if __name__ == '__main__':
    main()
