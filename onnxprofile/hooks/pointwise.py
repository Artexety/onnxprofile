import numpy as np

from .common.utilities import *
from .common.functions import *
from .common.constants import MACS, ZERO_OP

from .elementwise import ElementwiseBase


class PointwiseBase(ElementwiseBase):
    """ Defines parent class for pointwise module hooks. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = ZERO_OP
        self.ratio = max(1, self.dimensions["inputs"] - 1)

    def profile(self, inputs: list[np.ndarray], outputs: list[np.ndarray]):
        return (construct_volume(outputs[0].shape) * self.ratio * self.operation_mac_count, ZERO_OP)

    def inference_shape(self, inputs: list[np.ndarray]):
        return [construct_ndarray(calculate_max_shape_ndarray(inputs), dtype=np.float32)]


class CategoryMapper(PointwiseBase):
    """ Defines category_mapper module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = ZERO_OP


class LocalResponseNormalization(PointwiseBase):
    """ Defines local_response_normalization module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        for a in proto.attribute:
            if a.name == 'size':
                self.size = a.i

    def profile(self, inputs: list[np.ndarray], outputs: list[np.ndarray]):
        output_volume = construct_volume(outputs[0].shape)
        output_volume *= (MACS.DIV + MACS.EXP + MACS.ADD + self.size * MACS.MUL)
        return (output_volume, 0)


class Add(PointwiseBase):
    """ Defines add module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.ADD

    def inference_shape(self, inputs: list[np.ndarray]):
        return [inputs[0] + inputs[1]]


class Sum(PointwiseBase):
    """ Defines sum module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.ADD

    def inference_shape(self, inputs: list[np.ndarray]):
        y = inputs[0]
        for i in range(1, len(inputs)):
            y += inputs[i]
        return [y]


class Abs(PointwiseBase):
    """ Defines abs module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.ADD

    def inference_shape(self, inputs: list[np.ndarray]):
        return [np.abs(inputs[0])]


class Neg(PointwiseBase):
    """ Defines neg module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.ADD

    def inference_shape(self, inputs: list[np.ndarray]):
        return [-inputs[0]]


class Sub(PointwiseBase):
    """ Defines sub module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.ADD

    def inference_shape(self, inputs: list[np.ndarray]):
        return [inputs[0] - inputs[1]]


class Min(PointwiseBase):
    """ Defines min module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.CMP
        self.ratio = self.dimensions["inputs"] - 1

    def inference_shape(self, inputs: list[np.ndarray]):
        result = inputs[0]
        for i in range(1, self.dimensions["inputs"]):
            result = np.minimum(result, inputs[i])
        return [result]


class Max(Min):
    """ Defines max module hook. """

    def inference_shape(self, inputs: list[np.ndarray]):
        result = inputs[0]
        for i in range(1, self.dimensions["inputs"]):
            result = np.maximum(result, inputs[i])
        return [result]


class Hardmax(PointwiseBase):
    """ Defines hardmax module hook. """
    pass


class Relu(PointwiseBase):
    """ Defines relu module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.CMP
        self.ratio = 1


class PRelu(PointwiseBase):
    """ Defines prelu module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.CMP + MACS.MUL
        self.ratio = 1


class Clip(PointwiseBase):
    """ Defines clip module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.CMP * 2
        self.ratio = 1


class Relu6(Clip):
    """ Defines relu6 module hook. """
    pass


class Exp(PointwiseBase):
    """ Defines exp module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.EXP
        self.ratio = 1


class Log(PointwiseBase):
    """ Defines log module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.LOG
        self.ratio = 1


class CumSum(PointwiseBase):
    """ Defines cumulative_sum module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.ADD
        self.ratio = 1


class Softmax(Exp):
    """ Defines softmax module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.EXP + MACS.DIV
        self.ratio = 1


class Sigmoid(Exp):
    """ Defines sigmoid module hook. """
    pass


class Tanh(PointwiseBase):
    """ Defines tanh module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.EXP
        self.ratio = 2


class Mul(PointwiseBase):
    """ Defines mul module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.MUL

    def inference_shape(self, inputs: list[np.ndarray]):
        return [inputs[0] * inputs[1]]


class InstanceNormalization(PointwiseBase):
    """ Defines instance_normalization module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = 2 * MACS.ADD + MACS.MUL + MACS.DIV


class Sqrt(PointwiseBase):
    """ Defines sqrt module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.SQRT


class Pow(PointwiseBase):
    """ Defines pow module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.POW


class Sin(PointwiseBase):
    """ Defines sin module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.SIN


class Cos(PointwiseBase):
    """ Defines cos module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.COS


class Div(PointwiseBase):
    """ Defines div module hook. """
    
    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.DIV

    def inference_shape(self, inputs: list[np.ndarray]):
        return [inputs[0] / (inputs[1])]


class Range(PointwiseBase):
    """ Defines range module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = 1

    def inference_shape(self, inputs: list[np.ndarray]):
        return [np.arange(inputs[0], inputs[1], inputs[2], dtype=np.float32)]


class Reciprocal(PointwiseBase):
    """ Defines reciprocal module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.DIV


class HardSigmoid(PointwiseBase):
    """ Defines hard_sigmoid module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.MUL + MACS.ADD + MACS.CMP * 2


class LeakyRelu(PointwiseBase):
    """ Defines leaky_relu module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.MUL + MACS.CMP