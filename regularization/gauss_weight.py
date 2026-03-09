import numpy as np
import torch


def colorforward_difference_x_direction(array, dy=1):
    imgb = array[:, :, 0]
    imgg = array[:, :, 1]
    imgr = array[:, :, 2]
    image=np.zeros(array.shape)
    b = forward_difference_x_direction(imgb)
    g = forward_difference_x_direction(imgg)
    r = forward_difference_x_direction(imgr)
    image[:, :, 0] = r
    image[:, :, 1] = g
    image[:, :, 2] = b
    return image
def colorbackward_difference_x_direction(array, dy=1):
    imgb = array[:, :, 0]
    imgg = array[:, :, 1]
    imgr = array[:, :, 2]
    image=np.zeros(array.shape)
    b = backward_difference_x_direction(imgb)
    g = backward_difference_x_direction(imgg)
    r = backward_difference_x_direction(imgr)
    image[:, :, 0] = r
    image[:, :, 1] = g
    image[:, :, 2] = b
    return image
def colorforward_difference_y_direction(array, dy=1):
    imgb = array[:, :, 0]
    imgg = array[:, :, 1]
    imgr = array[:, :, 2]
    image=np.zeros(array.shape)
    b = forward_difference_y_direction(imgb)
    g = forward_difference_y_direction(imgg)
    r = forward_difference_y_direction(imgr)
    image[:, :, 0] = r
    image[:, :, 1] = g
    image[:, :, 2] = b
    return image
def colorbackward_difference_y_direction(array, dy=1):
    imgb = array[:, :, 0]
    imgg = array[:, :, 1]
    imgr = array[:, :, 2]
    image=np.zeros(array.shape)
    b = backward_difference_y_direction(imgb)
    g = backward_difference_y_direction(imgg)
    r = backward_difference_y_direction(imgr)
    image[:, :, 0] = r
    image[:, :, 1] = g
    image[:, :, 2] = b
    return image


def forward_difference_y_direction(array, dy=1):

    array = np.float32(array)

    gradient_forward = np.diff(array, axis=0) / dy

    gradient_wraparound = -(array[-1, :] - array[0, :]) / dy
    gradient_wraparound = gradient_wraparound[np.newaxis, :]  


    gradient_full = np.concatenate((gradient_forward,gradient_wraparound), axis=0)

    return -gradient_full
def backward_difference_y_direction(array, dy=1):

    array = np.float32(array)

    gradient_forward = -np.diff(array, axis=0) / dy


    gradient_wraparound = -(array[0, :] - array[-1, :]) / dy
    gradient_wraparound = gradient_wraparound[np.newaxis, :]  


    gradient_full = np.concatenate((gradient_wraparound,gradient_forward), axis=0)

    return -gradient_full

def forward_difference_x_direction(array, dx=1):

    array = np.float32(array)
    # array=np.float32(array)
    gradient_backward = np.diff(array) / dx
    gradient_wraparound = (array[:, 0] - array[:, -1]) / dx
    gradient_wraparound = gradient_wraparound[:, np.newaxis]
    gradient_forward = np.concatenate((gradient_backward, gradient_wraparound), axis=1)
    return -gradient_forward

def backward_difference_x_direction(array, dx=1):

    array = np.float32(array)
    gradient_backward = -np.diff(array) / dx
    gradient_wraparound = -(array[:, 0] - array[:, -1]) / dx
    gradient_wraparound = gradient_wraparound[:, np.newaxis]
    gradient_forward = np.concatenate((gradient_wraparound,gradient_backward ), axis=1)
    return -gradient_forward
def update_weights(x, p, k):

    D_h_x = forward_difference_x_direction(x)
    D_v_x = forward_difference_y_direction(x)

    w_h = 1 / (np.power(abs(D_h_x), 2 - p) + k)
    w_v = 1 / (np.power(abs(D_v_x), 2 - p) + k)

    w_h[np.isinf(w_h)] = 0
    w_v[np.isinf(w_v)] = 0

    return w_h, w_v
def colorupdate_weights(x, p, k):

    D_h_x = colorforward_difference_x_direction(x)
    D_v_x = colorforward_difference_y_direction(x)

    w_h = 1 / (np.power(abs(D_h_x), 2 - p) + k)
    w_v = 1 / (np.power(abs(D_v_x), 2 - p) + k)


    w_h[np.isinf(w_h)] = 0
    w_v[np.isinf(w_v)] = 0

    return w_h, w_v

def gauss_weights(x, sigama):

    w = torch.exp(- x * x / (2 * sigama * sigama))

    return w