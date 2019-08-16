from ..MNIST_all import test_MNIST
from . import test_classify, test_generate, test_params

def main():

    test_MNIST.main()

    test_classify.main()

    test_generate.main()

    test_params.main()

if __name__ == '__main__':
    main()
