import numpy as np
import math

from .functions import construct_volume


def translate_axes(size, axes):
    return [axis if axis > 0 else size + axis for axis in axes]


def calculate_max_shape(shapes: list):
    max_volume = construct_volume(shapes[0])
    max_shape = tuple()
    
    for shape in shapes:
        tmp_volume = construct_volume(shape)
        if tmp_volume > max_volume:
            max_shape = shape
            max_volume = tmp_volume
    return max_shape


def calculate_max_shape_ndarray(mndarray: list[np.ndarray]):
    max_shape = mndarray[0].shape
    max_volume = construct_volume(max_shape)
    for ndarray in mndarray:
        tmp_volume = construct_volume(ndarray.shape)
        if tmp_volume > max_volume:
            max_shape = ndarray.shape
            max_volume = tmp_volume
        elif tmp_volume == max_volume:
            if len(ndarray.shape) > len(max_shape):
                max_shape = ndarray.shape
    
    return max_shape


def calculate_conv_output_shape(input, padding, kernel_size, stride, dilation):
    return int((input + padding - dilation * (kernel_size - 1) - 1) / stride + 1)


def calculate_convtranspose_output_shape(input, output_padding, padding, kernel_size, stride, dilation):
    return stride * (input - 1) + output_padding + ((kernel_size - 1) * dilation + 1) - padding


def calculate_pooling_shape(input_shape, padding, kernel_shape, dilation, stride, use_ceil = True):
    output_shape = (input_shape + padding - ((kernel_shape - 1) * dilation + 1)) / stride + 1
    return math.ceil(output_shape) if use_ceil else math.floor(output_shape)


def calculate_sparsity(ndarray):
    if len(ndarray.shape) not in [2, 4]:
        return None

    if ndarray.dtype in [np.float32, np.float64, np.int32, np.int8]:
        flag = ndarray == 0
        return flag.sum() / ndarray.size
    if ndarray.dtype == np.uint8:
        flag = ndarray == 128
        return flag.sum() / ndarray.size


def zero_flag(ndarray):
    if ndarray.dtype in [np.float32, np.float64, np.int32, np.int8]:
        return ndarray == 0
    if ndarray.dtype == np.uint8:
        return ndarray == 128
