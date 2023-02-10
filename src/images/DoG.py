import cv2
import sys
import matplotlib.pyplot as plt
from ImageReport import ImageReport


def DoG(img, lower_res = (3,3), higher_res = (5,5)):
    """Performs Difference of Gaussian.
    
    """
    low_gauss = cv2.GaussianBlur(img, lower_res, 0)
    high_gauss = cv2.GaussianBlur(img, higher_res, 0)
    return low_gauss - high_gauss


def main():
    if len(sys.argv) < 2:
        print("USAGE:\nPlease provide an image path as first argument.")
        return 1

    path = sys.argv[1]
    img = cv2.imread(path)

    dog = DoG(img, low_gauss, high_gauss)
    ImageReport.report_cv2_mat(dog, filename="dog")
    
    plt.imshow(dog)
    plt.show()

    return 0



if __name__ == "__main__":
    sys.exit(main())