import matplotlib.pyplot as plt
import cv2
def show_image(im):
    print(im.shape)
    plt.imshow(im,cmap='gray')
    plt.show()


def resize_image(image, new_width):
    # we need to keep in mind aspect ratio so the image does
    # not look skewed or distorted -- therefore, we calculate
    # the ratio of the new image to the old image
    r = new_width / image.shape[1]
    dim = (new_width, int(image.shape[0] * r))

    # perform the actual resizing of the image and show it
    resized = cv2.resize(image, (220,155), interpolation = cv2.INTER_AREA)
    return resized


def preprocess_image(image):
    im= cv2.imread(image)
    im_resize = resize_image(im, 128)
    gray_image = cv2.cvtColor(im_resize, cv2.COLOR_BGR2GRAY)
    inv_image = cv2.bitwise_not(gray_image)
    norm_imge = inv_image/255
    return norm_imge
