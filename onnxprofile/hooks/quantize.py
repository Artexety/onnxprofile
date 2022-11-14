import numpy as np
import math

from .common.utilities import *
from .common.functions import *
from .common.constants import MACS, ZERO_OP

from .pointwise import PointwiseBase
from .elementwise import (Conv, Gemm)


class DequantizeLinear(PointwiseBase):
    """ Defines dequantize_linear module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.MUL


class QuantizeLinear(PointwiseBase):
    """ Defines quanzize_linear module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.MUL


class QuantizeLinearMatMul(Gemm):
    """ Defines quanzize_linear_mat_mul module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        attribute_names = ['transA', 'transB']
        self.transA = None
        self.transB = None
        set_attributes(proto.attribute, attribute_names, self)

    def profile(self, inputs: list[np.ndarray], outputs: list[np.ndarray]):
        x_shape = inputs[0].shape
        weight_shape = inputs[3].shape
        macs = construct_volume(x_shape)
        if self.__class__ == Gemm:
            macs *= weight_shape[0]
        else:
            macs *= weight_shape[-1]
        return (macs, ZERO_OP)

    def inference_shape(self, inputs: list[np.ndarray]):
        x = inputs[0]
        w = inputs[3]

        if self.__class__ == Gemm:
            if self.transA is not None and self.transA > 0:
                x_shape = x.shape[::-1]
            else:
                x_shape = x.shape
            if self.transB is not None and self.transB > 0:
                yshape = x_shape[:-1] + (w.shape[-2],)
            else:
                yshape = x_shape[:-1] + (w.shape[-1],)
        else:
            yshape = x.shape[:-1] + (w.shape[-1],)

        return [construct_ndarray(yshape, dtype=np.float32)]


class QuantizeLinearConv(Conv):
    """ Defines quantize_linear_conv module hook. """

    def profile(self, inputs: list[np.ndarray], outputs: list[np.ndarray]):
        if self.dimensions["output"] == 1:
            kernel_shape = inputs[3].shape
            params = construct_volume(kernel_shape)
            if self.dimensions["inputs"] == 3:
                params += kernel_shape[0]

            if len(kernel_shape) > 3:
                output_volume = construct_volume(outputs[0].shape)
                macs = output_volume * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]
            elif len(kernel_shape) == 3:
                output_volume = construct_volume(outputs[0].shape)
                macs = output_volume * kernel_shape[1] * kernel_shape[2]

            if self.dimensions["inputs"] == 9:
                macs += (output_volume * MACS.ADD)

        return (macs, params)

    def inference_shape(self, inputs: list[np.ndarray]):
        x_tensor = inputs[0]
        w_tensor = inputs[3]
        x_shape = x_tensor.shape
        w_shape = w_tensor.shape

        shape = []

        if self.auto_padding is None or self.auto_padding == b'NOTSET':
            if len(x_shape) == 4:
                output_h = calculate_conv_output_shape(
                    x_shape[2], 
                    self.paddings[0] + self.paddings[2], 
                    w_shape[2], 
                    self.strides[0], 
                    self.dilations[0]
                )
                output_w = calculate_conv_output_shape(
                    x_shape[3], 
                    self.paddings[1] + self.paddings[3], 
                    w_shape[3], 
                    self.strides[1], 
                    self.dilations[1]
                )
                shape = (x_shape[0], w_shape[0], output_h, output_w)
            elif len(x_shape) == 3:
                output_h = calculate_conv_output_shape(
                    x_shape[2], 
                    self.paddings[0] + self.paddings[1], 
                    w_shape[2], 
                    self.strides[0],
                    self.dilations[0]
                )
                shape = (x_shape[0], w_shape[0], output_h)
        elif self.auto_padding in [b'SAME_LOWER', b'SAME_UPPER']:
            shape = (x_shape[0], w_shape[0], math.ceil(x_shape[2] / self.strides[0]))
            if len(x_shape) == 4:
                shape += (math.ceil(x_shape[3] / self.strides[1]),)
        return [construct_ndarray(shape, dtype=np.float32)]