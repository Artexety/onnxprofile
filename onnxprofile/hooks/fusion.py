import numpy as np

from .common.utilities import *
from .common.functions import *

from .elementwise import ElementwiseBase

from copy import deepcopy
from collections import defaultdict


class FusionBase(ElementwiseBase):
    """ Defines parent class for fused module hooks. """

    def __init__(self, proto):
        super().__init__(proto)

    def inference_shape(self, inputs: list[np.ndarray]):
        return [inputs[0]]


class Identity(FusionBase):
    """ Defines identity module hook. """
    pass


class Erf(FusionBase):
    """ Defines erf module hook. """
    pass


class Dropout(FusionBase):
    """ Defines dropout module hook. """
    pass


class BatchNormalization(FusionBase):
    """ Defines batch_normalization module hook. """
    pass


class Slice(FusionBase):
    """ Defines slice module hook. """
    
    def __init__(self, proto):
        super().__init__(proto)
        attribute_names = ['axes', 'end_position', 'start_position']
        set_attributes(proto.attribute, attribute_names, self)

    def inference_shape(self, inputs: list[np.ndarray]):
        if len(inputs) == 1:
            data = inputs[0]
            start_position = self.start_position
            end_position = self.end_position
            axes = self.axes
            steps = defaultdict(lambda: 1)
        elif len(inputs) == 3:
            data, start_position, end_position = inputs[:3]
            return [data[start_position[0]:end_position[0]]]
        elif len(inputs) == 4:
            data, start_position, end_position, axes = inputs[:4]
            steps = defaultdict(lambda: 1)
        elif len(inputs) == 5:
            data, start_position, end_position, axes, steps = inputs[:5]

        index, storage = (0, deepcopy(data))
        for i in range(len(storage.shape)):
            if i in axes:
                if i == 0:
                    storage = storage[start_position[index]:end_position[index]:steps[index], ...]
                elif i == 1:
                    storage = storage[:, start_position[index]:end_position[index]:steps[index], ...]
                elif i == 2:
                    storage = storage[:, :, start_position[index]:end_position[index]:steps[index], ...]
                elif i == 3:
                    storage = storage[:, :, :, start_position[index]:end_position[index]:steps[index], ...]
                elif i == 4:
                    storage = storage[:, :, :, :, start_position[index]:end_position[index]:steps[index], ...]
                index += 1
        return [storage]


class Cast(FusionBase):
    """ Defines cast module hook. """

    def inference_shape(self, inputs: list[np.ndarray]):
        return [inputs[0]]


class Floor(FusionBase):
    """ Defines floor module hook. """

    def inference_shape(self, inputs: list[np.ndarray]):
        return [np.floor(inputs[0])]


class Ceil(FusionBase):
    """ Defines ceil module hook. """

    def inference_shape(self, inputs: list[np.ndarray]):
        return [np.ceil(inputs[0])]