#Importing Libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt


def g_kernel(length, sigma):
    g = np.exp(-0.5 * np.square(np.linspace((1-length) / 2, (length - 1) / 2, length)) / np.square(sigma))
    kernel = np.outer(g, g)
    res = kernel / np.sum(kernel)
    return res


def grad_magnitude(img):
    return np.sqrt(convolve(img, np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])) ** 2
                   + convolve(img, np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])) ** 2)


def threshold(gradient_img, e):
    curr = gradient_img.mean()
    old = 1 + curr + e
    while abs(curr - old) > e:
        old = curr
        curr = (gradient_img[gradient_img < curr].mean() + gradient_img[gradient_img >= curr].mean())*0.5
    edge = np.zeros(gradient_img.shape, int)
    edge[gradient_img >= curr] = 255

    return edge


def convolve(image, filter):
    filter = np.flip(filter, 0)
    filter = np.flip(filter, 1)
    k1, k2 = filter.shape
    output = np.zeros(image.shape)

    pad_h = int((k1 - 1) / 2)
    pad_w = int((k2 - 1) / 2)

    padded_image = np.pad(image, [(pad_h, pad_w), (pad_w, pad_h)])
    for u in range(pad_h, padded_image.shape[0] - pad_h):
        for v in range(pad_w, padded_image.shape[1] - pad_w):
            output[u - pad_h, v - pad_w] = np.sum(padded_image[u - pad_h: u + pad_h + 1, v - pad_w: v + pad_w + 1]
                                                  * filter)
    return output

def step1():
    print('Step 1')
    # make the visualization plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    plot1 = ax1.imshow(g_kernel(5, 1))
    plot2 = ax2.imshow(g_kernel(15, 8))

    fig.colorbar(plot1, ax=ax1)
    fig.colorbar(plot2, ax=ax2)

    plt.show()


def step2():
    print("Step 2")
    kernel = g_kernel(15, 8)
    img1 = cv2.imread('image1.jpg', 0)
    img2 = cv2.imread('image2.jpg', 0)
    img3 = cv2.imread('image3.jpg', 0)


    print("Working on blurred1")
    blurred1 = convolve(img1, kernel)
    cv2.imwrite("blurred1.jpg", blurred1)
    print("Working on blurred2")
    blurred2 = convolve(img2, kernel)
    cv2.imwrite("blurred2.jpg", blurred2)
    print("Working on blurred3")
    blurred3 = convolve(img3, kernel)
    cv2.imwrite("blurred3.jpg", blurred3)



def step3():
    print('Step 3')
    blurred1 = cv2.imread('blurred1.jpg',0)
    blurred2 = cv2.imread('blurred2.jpg',0)
    blurred3 = cv2.imread('blurred3.jpg',0)

    print("Working On Edges 1")
    grad_img = grad_magnitude(blurred1)
    output = threshold(grad_img, 1e-2)
    cv2.imwrite("output1.jpg", output)

    print("Working On Edges 2")
    grad_img = grad_magnitude(blurred2)
    output = threshold(grad_img, 1e-2)
    cv2.imwrite("output2.jpg", output)

    print("Working On Edges 3")
    grad_img = grad_magnitude(blurred3)
    output = threshold(grad_img, 1e-2)
    cv2.imwrite("output3.jpg", output)





#Running Through Steps (STEP 4)
step1()
step2()
step3()