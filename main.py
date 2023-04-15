from model import MNISTModel
from test import predict_random_image


def main():
    model = MNISTModel()
    model.load_latest()

    predict_random_image(model)


if __name__ == '__main__':
    main()
