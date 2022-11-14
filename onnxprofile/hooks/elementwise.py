import numpy as np
import math

from .common.utilities import *
from .common.functions import *
from .common.constants import MACS, ZERO_OP


class ElementwiseBase(object):
    """ Defines parent class for elementwise module hooks. """

    def __init__(self, proto):
        self.dimensions = {
            "inputs": len(proto.input),
            "outputs": len(proto.output),
        }
        self.module_name = proto.name

    def profile(self, inputs: list[np.ndarray], outputs: list[np.ndarray]):
        return (ZERO_OP, ZERO_OP)
    
    def inference_shape(self, inputs: list[np.ndarray]):
        return [(ZERO_OP), ]


class Conv(ElementwiseBase):
    """ Defines conv module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        attribute_names = ['pads', 'strides', 'dilations', 'auto_pad']
        self.auto_pad = None
        self.pads = (0, 0, 0, 0)
        self.strides = (1, 1)
        self.dilations = (1, 1)
        set_attributes(proto.attribute, attribute_names, self)
    
    def profile(self, inputs: list[np.ndarray], outputs: list[np.ndarray]):
        if self.dimensions["outputs"] == 1 and self.dimensions["inputs"] in [2, 3]:
            kernel_shape = inputs[1].shape
            params = construct_volume(kernel_shape)
            if self.dimensions["inputs"] == 3:
                params += kernel_shape[0]

            output_volume = construct_volume(outputs[0].shape)
            if len(kernel_shape) > 3:
                macs = output_volume * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]
                macs += output_volume * MACS.ADD
            elif len(kernel_shape) == 3:
                macs = output_volume * kernel_shape[1] * kernel_shape[2]
                macs += output_volume * MACS.ADD
        return (macs, params)

    def inference_shape(self, inputs: list[np.ndarray]):
        shape = []
        x_tensor, w_tensor = inputs[:2]
        x_shape = x_tensor.shape
        w_shape = w_tensor.shape
        if self.auto_pad is None or self.auto_pad == b'NOTSET':
            if len(x_shape) == 4:
                oh = calculate_conv_output_shape(
                    x_shape[2], 
                    self.pads[0] + self.pads[2], 
                    w_shape[2], 
                    self.strides[0], 
                    self.dilations[0]
                )
                ow = calculate_conv_output_shape(
                    x_shape[3], 
                    self.pads[1] + self.pads[3], 
                    w_shape[3], 
                    self.strides[1], 
                    self.dilations[1]
                )
                shape = (x_shape[0], w_shape[0], oh, ow)
            elif len(x_shape) == 3:
                oh = calculate_conv_output_shape(
                    x_shape[2], 
                    self.pads[0] + self.pads[1], 
                    w_shape[2], 
                    self.strides[0],
                    self.dilations[0]
                )
                shape = (x_shape[0], w_shape[0], oh)
        elif self.auto_pad in [b'SAME_LOWER', b'SAME_UPPER']:
            shape = (x_shape[0], w_shape[0], math.ceil(x_shape[2] / self.strides[0]))
            if len(x_shape) == 4:
                shape += (math.ceil(x_shape[3] / self.strides[1]), )
        return [construct_ndarray(shape, dtype=np.float32)]


class ConvTranspose(ElementwiseBase):
    """ Defines conv_transpose module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        attribute_names = ['pads', 'strides', 'dilations', 'output_padding', 'output_shape', 'group']
        self.pads = (0, 0, 0, 0)
        self.strides = (1, 1)
        self.dilations = (1, 1)
        self.output_padding = (0, 0, 0, 0)
        self.output_shape = (0, 0)
        self.group = 1
        set_attributes(proto.attribute, attribute_names, self)

    def profile(self, inputs: list[np.ndarray], outputs: list[np.ndarray]):
        if self.dimensions["outputs"] == 1 and self.dimensions["inputs"] in [3, 2]:
            kernel_shape = inputs[1].shape
            params = output_volume(kernel_shape)
            if self.dimensions["inputs"] == 3:
                params += kernel_shape[0]

            output_volume = construct_volume(inputs[0].shape)
            if len(kernel_shape) > 3:
                macs = output_volume * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]
                macs += output_volume * MACS.ADD
            elif len(kernel_shape) == 3:
                macs = output_volume * kernel_shape[1] * kernel_shape[2]
                macs += output_volume * MACS.ADD
        return (macs, params)

    def inference_shape(self, inputs: list[np.ndarray]):
        x_tensor, w_tensor = inputs[:2]
        x_shape = x_tensor.shape
        w_shape = w_tensor.shape
        shape = []
        channels = self.group * w_shape[1]
        if len(x_shape) == 4:
            output_w = calculate_convtranspose_output_shape(
                x_shape[2], 
                self.output_padding[0], 
                self.pads[0] + self.pads[2], 
                w_shape[2], 
                self.strides[0], 
                self.dilations[0]
            )
            output_h = calculate_convtranspose_output_shape(
                x_shape[3], 
                self.output_padding[1],
                self.pads[1] + self.pads[3], 
                w_shape[3], 
                self.strides[1],
                self.dilations[1]
            )
            shape = [x_shape[0], channels, output_w, output_h]
            if construct_volume(self.output_shape) != 0:
                shape[2:] = self.output_shape
        elif len(x_shape) == 3:
            output_w = calculate_convtranspose_output_shape(
                x_shape[2], 
                self.output_padding[0], 
                self.pads[0] + self.pads[1],
                w_shape[2], 
                self.strides[0], 
                self.dilations[0]
            )
            shape = [x_shape[0], channels, output_w]
            if construct_volume(self.output_shape) != 0:
                shape[2] = self.output_shape[0]
        return [construct_ndarray(shape, dtype=np.float32)]


class Unsqueeze(ElementwiseBase):
    """ Defines unsqueeze module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.axes = None
        set_attributes(proto.attribute, ['axes'], self)

    def inference_shape(self, inputs: list[np.ndarray]):
        outputs = inputs[0]
        axes = inputs[1] if self.axes is None else self.axes
        for axis in axes:
            outputs = np.expand_dims(outputs, axis=axis)
        return [outputs]


class Squeeze(ElementwiseBase):
    """ Defines squeeze module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.axes = [0]
        for a in proto.attribute:
            if a.name == 'axes':
                self.axes = get_attribute(a)

    def inference_shape(self, inputs: list[np.ndarray]):
        outputs = inputs[0]
        if self.dimensions["inputs"] == 2:
            self.axes = inputs[1]
        for idx, axis in enumerate(self.axes):
            outputs = np.squeeze(outputs, axis=axis - idx)
        return [outputs]


class Resize(ElementwiseBase):
    """ Defines resize module hook. """

    def __init__(self, proto):
        attribute_names = ['mode']
        set_attributes(proto.attribute, attribute_names, self)

    def profile(self, inputs: list[np.ndarray], outputs: list[np.ndarray]):
        output_volume = construct_volume(outputs[0].shape)
        if self.mode == b'nearest':
            output_volume *= 0
        elif self.mode == b'linear':
            output_volume *= 4
        elif self.mode == b'cubic':
            output_volume *= 8
        return (output_volume, ZERO_OP)

    def inference_shape(self, inputs: list[np.ndarray]):
        x = inputs[0]
        sizes, output_shape = ([], [])

        if len(inputs) == 2:
            scales = inputs[1]
        elif len(inputs) >= 3:
            scales = inputs[2]
            if len(inputs) >= 4:
                sizes = inputs[3]

        if validate_ndarray(sizes):
            if len(sizes) == 4:
                output_shape = sizes
            if len(sizes) == 2:
                output_shape = x.shape[:2] + sizes
        elif validate_ndarray(scales):
            for src, scale in zip(x.shape, scales):
                output_shape.append(math.floor(src * scale))

        if validate_ndarray(output_shape) and output_shape.dtype != np.int64:
            output_shape = output_shape.astype(dtype=np.int64)

        return [construct_ndarray(output_shape, dtype=np.flaot32)]


class ScatterND(ElementwiseBase):
    """ Defines scatter_nd module hook. """

    def inference_shape(self, inputs: list[np.ndarray]):
        data, indices, updates = inputs[:3]
        assert indices.shape[-1] <= len(data.shape)
        assert updates.shape == indices.shape[:-1] + data.shape[indices.shape[-1]:]

        outputs = np.copy(data)
        for i in np.ndindex(indices.shape[:-1]):
            outputs[indices[i]] = updates[i]
        return [outputs]


class Pool(ElementwiseBase):
    """ Defines pool module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        attribute_names = ['kernel_shape', 'pads', 'strides', 'ceil_mode']
        self.kernel_shape = (3, 3)
        self.ceil_mode = 0
        self.pads = (0, 0, 0, 0)
        self.strides = (1, 1)
        self.dilations = (1, 1)
        set_attributes(proto.attribute, attribute_names, self)
        self.operation_mac_count = MACS.CMP

    def profile(self, inputs: list[np.ndarray], outputs: list[np.ndarray]):
        output_volume = construct_volume(outputs[0].shape)
        macs = output_volume * MACS.CMP * self.kernel_shape[0]
        if len(self.kernel_shape) == 2:
            macs *= self.kernel_shape[1]
        return (macs, 0)

    def inference_shape(self, inputs: list[np.ndarray]):
        if len(self.kernel_shape) == 1:
            input_shape = inputs[0].shape
            output_shape = input_shape[:2] + (
                calculate_pooling_shape(
                    input_shape[2], 
                    self.pads[0] + self.pads[1], 
                    self.kernel_shape[0], 
                    self.dilations[0], 
                    self.strides[0], 
                    self.ceil_mode
                ),
            )
            return [construct_ndarray(output_shape, dtype=np.float32)]
        if len(self.kernel_shape) == 2:
            input_shape = inputs[0].shape
            output_shape = input_shape[:2] + (
                calculate_pooling_shape(
                    input_shape[2], 
                    self.pads[0] + self.pads[2], 
                    self.kernel_shape[0], 
                    self.dilations[0], 
                    self.strides[0], 
                    self.ceil_mode
                ),
                calculate_pooling_shape(
                    input_shape[3],
                    self.pads[1] + self.pads[3], 
                    self.kernel_shape[1], 
                    self.dilations[1], 
                    self.strides[1], 
                    self.ceil_mode
                ),
            )
            return [construct_ndarray(output_shape, dtype=np.float32)]


class MaxPool(Pool):
    """ Defines max_pool module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.CMP


class AveragePool(Pool):
    """ Defines average_pool module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.operation_mac_count = MACS.ADD


class GlobalAveragePool(ElementwiseBase):
    """ Defines global_average_pool module hook. """

    def inference_shape(self, inputs: list[np.ndarray]):
        input_shape = inputs[0].shape
        shape = input_shape[:2] + tuple([1] * (len(input_shape) - 2))
        return [construct_ndarray(shape, dtype=np.float32)]

    def profile(self, inputs: list[np.ndarray], outputs: list[np.ndarray]):
        return (construct_volume(inputs[0].shape) * MACS.ADD + construct_volume(outputs[0].shape) * MACS.DIV, ZERO_OP)


class Gather(ElementwiseBase):
    """ Defines gather module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.axis = 0
        for a in proto.attribute:
            if a.name == 'axis':
                self.axis = a.i

    def inference_shape(self, inputs: list[np.ndarray]):
        return [np.take(inputs[0], inputs[1].astype(dtype=np.int32), axis=self.axis)]


class Constant(ElementwiseBase):
    """ Defines constant module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.value = 0
        for a in proto.attribute:
            if a.name == 'value':
                self.value = get_attribute(a)

    def inference_shape(self, inputs: list[np.ndarray]):
        return [self.value]


class Concat(ElementwiseBase):
    """ Defines concat module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        self.axis = 0
        for a in proto.attribute:
            if a.name == 'axis':
                self.axis = get_attribute(a)

    def inference_shape(self, inputs: list[np.ndarray]):
        return [np.concatenate(inputs, self.axis)]


class Reshape(ElementwiseBase):
    """ Defines reshape module hook. """

    def __init__(self, proto):
        super().__init__(proto)

    def inference_shape(self, inputs: list[np.ndarray]):
        input_shape = inputs[0].shape
        shape = inputs[1]
        output_shape = []
        for i in range(len(shape)):
            if shape[i] == 0:
                output_shape.append(int(input_shape[i]))
            else:
                output_shape.append(int(shape[i]))
        try:
            outputs = (inputs[0].reshape(output_shape))
        except Exception:
            outputs = np.zeros(shape.astype(np.int64), inputs[0].dtype)
        return [outputs]


class ArgMax(ElementwiseBase):
    """ Defines argmax module hook. """

    def __init__(self, n):
        super().__init__(n)
        attribute_names = ['axis', 'keepdims']
        self.keepdims = 1
        set_attributes(n.attribute, attribute_names, self)

    def inference_shape(self, inputs: list[np.ndarray]):
        return [construct_ndarray(inputs[0], self.axis, self.keepdims)]


class Expand(ElementwiseBase):
    """ Defines argmax module hook. """

    def __init__(self, proto):
        super().__init__(proto)

    def inference_shape(self, inputs: list[np.ndarray]):
        return [inputs[0] * np.ones(inputs[1].astype(np.int64), dtype=np.float32)]


class Tile(ElementwiseBase):
    """ Defines argmax module hook. """

    def __init__(self, proto):
        super().__init__(proto)

    def inference_shape(self, inputs: list[np.ndarray]):
        return [np.tile(inputs[0], inputs[1])]


class GRU(ElementwiseBase):
    """ Defines argmax module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        attribute_names = ['hidden_size']
        set_attributes(proto.attribute, attribute_names, self)

    def profile(self, inputs: list[np.ndarray], outputs: list[np.ndarray]):
        w, r, b = inputs[1:4]
        params = construct_volume(w.shape) + construct_volume(r.shape) + construct_volume(b.shape)
        batch = inputs[0].shape[1]
        macs = construct_volume(w.shape) + construct_volume(r.shape) + construct_volume(b.shape) * MACS.ADD
        return (macs * batch, params)

    def inference_shape(self, inputs: list[np.ndarray]):
        x, w = inputs[:2]
        seq_len, batch = x.shape[:2]
        num_dir = w.shape[0]
        h_len = w.shape[1] // 3
        return [construct_ndarray((seq_len, num_dir, batch, h_len), dtype=np.float32), 
                construct_ndarray((num_dir, batch, h_len), dtype=np.float32)]


class TopK(ElementwiseBase):
    """ Defines argmax module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        attribute_names = ['axis']
        self.axis = None
        set_attributes(proto.attribute, attribute_names, self)

    def inference_shape(self, inputs: list[np.ndarray]):
        x = inputs[0]
        k = inputs[1][0]
        newshape = []
        for i in range(len(x.shape)):
            if i == self.axis:
                newshape.append(k)
            else:
                newshape.append(x.shape[i])
        return [construct_ndarray(newshape, dtype=np.float32), construct_ndarray(newshape, dtype=np.int64)]


class LSTM(ElementwiseBase):
    """ Defines lstm module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        attribute_names = ['direction', 'hidden_size']
        self.direction = None
        self.hidden_size = None
        set_attributes(proto.attribute, attribute_names, self)

    def profile(self, inputs: list[np.ndarray], outputs: list[np.ndarray]):
        w, r, b = inputs[1:4]
        params = construct_volume(w.shape) + construct_volume(r.shape) + construct_volume(b.shape)
        batch = inputs[0].shape[1]
        macs = construct_volume(w.shape) + construct_volume(r.shape) + construct_volume(b.shape) * MACS.ADD
        return (macs * batch, params)

    def inference_shape(self, inputs: list[np.ndarray]):
        x, w = inputs[:2]
        seq_len, batch = x.shape[:2]
        num_dir = w.shape[0]
        h_len = w.shape[1] // 4
        return [construct_ndarray((seq_len, num_dir, batch, h_len), dtype=np.float32),
                construct_ndarray((num_dir, batch, h_len), dtype=np.float32)]


class Compress(ElementwiseBase):
    """ Defines compress module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        attribute_names = ['axis']
        self.axis = None
        set_attributes(proto.attribute, attribute_names, self)

    def inference_shape(self, inputs: list[np.ndarray]):
        return [np.compress(inputs[1], inputs[0], self.axis)]


class RoiAlign(ElementwiseBase):
    """ Defines roi_align module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        attribute_names = ['mode', 'output_height', 'output_width', 'sampling_ratio', 'spatial_scale']
        set_attributes(proto.attribute, attribute_names, self)

    def inference_shape(self, inputs: list[np.ndarray]):
        if len(inputs[0].shape) == 4 and self.output_height is not None and self.output_width is not None:
            newshape = inputs[0].shape[:2] + (self.output_height, self.output_width)
        else:
            raise NotImplementedError()
        return [construct_ndarray(newshape, dtype=np.float32)]


class ScatterElements(ElementwiseBase):
    """ Defines scatter_elements module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        attribute_names = ['axis']
        set_attributes(proto.attribute, attribute_names, self)

    def inference_shape(self, inputs: list[np.ndarray]):
        data, indices, updates = inputs[:3]
        assert indices.shape[-1] <= len(data.shape)
        assert updates.shape == indices.shape[:-1] + data.shape[indices.shape[-1]:]

        outputs = np.copy(data)
        for i in np.ndindex(indices.shape[:-1]):
            if self.axis == 'add':
                outputs[indices[i]] += updates[i]
            elif self.axis == 'mul':
                outputs[indices[i]] *= updates[i]
            else:
                outputs[indices[i]] = updates[i]
        return [outputs]


class Gemm(ElementwiseBase):
    """ Defines gemm module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        attribute_names = ['transA', 'transB']
        self.transA = None
        self.transB = None
        set_attributes(proto.attribute, attribute_names, self)

    def profile(self, inputs: list[np.ndarray], outputs: list[np.ndarray]):
        x_shape = inputs[0].shape
        if self.dimensions["inputs"] < 2:
            raise NotImplementedError()
        weight_shape = inputs[1].shape
        params = construct_volume(weight_shape)
        if self.dimensions["inputs"] == 3:
            params += construct_volume(inputs[2].shape)

        macs = construct_volume(x_shape)
        if self.__class__ == Gemm:
            macs *= weight_shape[0]
        else:
            macs *= weight_shape[-1]

        if self.dimensions["inputs"] == 3:
            macs += construct_volume(inputs[0].shape) * MACS.ADD
        return (macs, params)

    def inference_shape(self, inputs: list[np.ndarray]):
        x, w = inputs[:2]

        if self.__class__ == Gemm:
            if self.transA is not None and self.transA > 0:
                x_shape = x.shape[::-1]
            else:
                x_shape = x.shape
            if self.transB is not None and self.transB > 0:
                y_shape = x_shape[:-1] + (w.shape[-2],)
            else:
                y_shape = x_shape[:-1] + (w.shape[-1],)
        else:
            y_shape = x.shape[:-1] + (w.shape[-1],)

        return [construct_ndarray(y_shape, dtype=np.float32)]


class MatMul(Gemm):
    """ Defines mat_mul module hook. """
    pass


class MatMulInteger(Gemm):
    """ Defines mat_mul_integer module hook. """
    pass


class Shape(ElementwiseBase):
    """ Defines shape module hook. """

    def inference_shape(self, inputs: list[np.ndarray]):
        return [np.array(inputs[0].shape, np.int32)]


class OneHot(ElementwiseBase):
    """ Defines one_hot module hook. """    

    def __init__(self, proto):
        super().__init__(proto)
        attribute_names = ['axis']
        self.axis = -1
        set_attributes(proto.attribute, attribute_names, self)

    def inference_shape(self, inputs: list[np.ndarray]):
        indices, depth = inputs[:2]
        values = np.asarray(indices)
        rank = len(values.shape)
        depth_range = np.arange(depth)
        if axis < 0:
            axis += (rank + 1)
        ls = values.shape[:self.axis]
        rs = values.shape[self.axis:rank]
        targets = np.reshape(depth_range, (1,) * len(ls) + depth_range.shape + (1,) * len(rs))
        values = np.reshape(np.mod(values, depth), ls + (1,) + rs)
        return [np.asarray(targets == values, dtype=np.float32)]


class Pad(ElementwiseBase):
    """ Defines pad module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        attribute_names = ['pads', 'value']
        self.pads = None
        self.value = 0
        set_attributes(proto.attribute, attribute_names, self)

    def inference_shape(self, inputs: list[np.ndarray]):
        data = inputs[0]
        newshape = []
        if self.pads is None:
            if self.dimensions["inputs"] > 1:
                pads = inputs[1]
                newshape.extend(v + pads[i] + pads[i + len(data.shape)] for i, v in enumerate(data.shape))
        else:
            newshape.extend(v + self.pads[i] + self.pads[i + len(data.shape)] for i, v in enumerate(data.shape))
        return [construct_ndarray([int(val) for val in newshape], dtype=np.float32)]


class Split(ElementwiseBase):
    """ Defines split module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        attribute_names = ['axis', 'split']
        self.axis = None
        self.split = None
        set_attributes(proto.attribute, attribute_names, self)

    def inference_shape(self, inputs: list[np.ndarray]):
        split = []
        end = 0
        if self.split is None:
            if len(inputs) == 2:
                self.split = inputs[1]
            else:
                self.split = [inputs[0].shape[self.axis] // 2]

        self.axis = translate_axes(len(inputs[0].shape), [self.axis])[0]
        for v in self.split:
            if end + v >= inputs[0].shape[self.axis]:
                break
            split.append(end + v)
            end += v
        return np.split(inputs[0], split, self.axis)


class ReduceMean(ElementwiseBase):
    """ Defines reduce_mean module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        attribute_names = ['axes', 'keepdims']
        self.axes = None
        self.keepdims = 1
        set_attributes(proto.attribute, attribute_names, self)
        self.axes = tuple(self.axes) if self.axes is not None else None

    def profile(self, inputs: list[np.ndarray], outputs: list[np.ndarray]):
        return (construct_volume(inputs[0].shape) * MACS.ADD, ZERO_OP)

    def inference_shape(self, inputs: list[np.ndarray]):
        return [np.mean(inputs[0], axis=self.axes, keepdims=self.keepdims == 1)]


class ReduceProd(ReduceMean):
    """ Defines reduce_prod module hook. """

    def profile(self, inputs: list[np.ndarray], outputs: list[np.ndarray]):
        return (construct_volume(inputs[0].shape) * MACS.MUL, ZERO_OP)
    
    def inference_shape(self, inputs: list[np.ndarray]):
        return [np.prod(inputs[0], axis=self.axes, keepdims=self.keepdims == 1)]


class ReduceL2(ElementwiseBase):
    """ Defines reduce_l2 module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        attribute_names = ['axes', 'keepdims']
        self.axes = None
        self.keepdims = 1
        set_attributes(proto.attribute, attribute_names, self)
        self.axes = tuple(self.axes) if self.axes is not None else None

    def profile(self, inputs: list[np.ndarray], outputs: list[np.ndarray]):
        return (construct_volume(inputs[0].shape) * (MACS.ADD + MACS.SQRT), ZERO_OP)

    def inference_shape(self, inputs: list[np.ndarray]):
        return [np.sqrt(np.sum(inputs[0], axis=self.axes, keepdims=self.keepdims == 1))]


class ReduceSum(ReduceMean):
    """ Defines reduce_sum module hook. """

    def inference_shape(self, inputs: list[np.ndarray]):
        return [np.sum(inputs[0], axis=self.axes, keepdims=self.keepdims == 1)]


class ReduceMin(ReduceMean):
    """ Defines reduce_min module hook. """

    def profile(self, inputs: list[np.ndarray], outputs: list[np.ndarray]):
        return (construct_volume(inputs[0].shape) * MACS.CMP, ZERO_OP)

    def inference_shape(self, inputs: list[np.ndarray]):
        return [np.minimum.reduce(inputs[0], axis=self.axes, keepdims=self.keepdims == 1)]


class ReduceMax(ReduceMin):
    """ Defines reduce_max module hook. """

    def inference_shape(self, inputs: list[np.ndarray]):
        return [np.maximum.reduce(inputs[0], axis=self.axes, keepdims=self.keepdims == 1)]


class NonZero(ElementwiseBase):
    """ Defines non_zero module hook. """

    def profile(self, inputs: list[np.ndarray], outputs: list[np.ndarray]):
        return (construct_volume(outputs[0].shape) * MACS.CMP, ZERO_OP)

    def inference_shape(self, inputs: list[np.ndarray]):
        result = np.array(np.nonzero(inputs[0]), dtype=np.int64)
        if construct_volume(result.shape) == 0:
            result = np.array(np.nonzero(np.ones_like(inputs[0])), dtype=np.int64)
        return [result]


class Transpose(ElementwiseBase):
    """ Defines transpose module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        attribute_names = ['perm']
        self.perm = None
        set_attributes(proto.attribute, attribute_names, self)

    def inference_shape(self, inputs: list[np.ndarray]):
        return [np.transpose(inputs[0], self.perm)]


class ConstantOfShape(ElementwiseBase):
    """ Defines constants_of_shape module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        attribute_names = ['value']
        self.value = None
        set_attributes(proto.attribute, attribute_names, self)

    def inference_shape(self, inputs: list[np.ndarray]):
        arr = np.zeros(inputs[0].astype(np.int64), dtype=np.float32)
        if self.value is not None and len(self.value) == 1:
            arr.fill(self.value[0])
        return [arr]


class Flatten(ElementwiseBase):
    """ Defines flatten module hook. """

    def __init__(self, node):
        super().__init__(node)
        attribute_names = ['axis']
        self.axis = None
        set_attributes(node.attribute, attribute_names, self)

    def inference_shape(self, inputs: list[np.ndarray]):
        if self.axis is None:
            return [inputs[0].reshape((inputs[0].shape[0], -1))]
        tmp_volume = 1
        for i in range(self.axis):
            tmp_volume *= inputs[0].shape[i]
        return [inputs[0].reshape((tmp_volume, -1))]


class Einsum(ElementwiseBase):
    """ Defines einsum module hook. """

    def __init__(self, proto):
        super().__init__(proto)
        set_attributes(proto.attribute, ['equation'], self)
        strings = self.equation.split(b',')
        self.a_shape = strings[0].replace(b' ', b'')
        strings = strings[1].split(b'->')
        self.b_shape = strings[0].replace(b' ', b'')
        self.c_shape = strings[1].replace(b' ', b'')

    def profile(self, inputs: list[np.ndarray], outputs: list[np.ndarray]):
        macs = 1
        maps = {self.ashape[i]: v for i, v in enumerate(inputs[0].shape)}
        for i, v in enumerate(inputs[1].shape):
            maps[self.bshape[i]] = v
        for value in maps.values():
            macs *= value
        return (macs, ZERO_OP)

    def inference_shape(self, inputs: list[np.ndarray]):
        maps = {self.ashape[i]: v for i, v in enumerate(inputs[0].shape)}
        for i, v in enumerate(inputs[1].shape):
            maps[self.bshape[i]] = v
        shape = [maps[k] for k in self.cshape]
        return [construct_ndarray(shape, dtype=np.float32)]


class Less(ElementwiseBase):
    """ Defines less module hook. """

    def profile(self, inputs: list[np.ndarray], outputs: list[np.ndarray]):
        return (construct_volume(outputs[0].shape) * MACS.CMP, ZERO_OP)

    def inference_shape(self, inputs: list[np.ndarray]):
        result = np.less(inputs[0], inputs[1])
        return [result]


class LessOrEqual(Less):
    """ Defines less_or_equal module hook. """

    def inference_shape(self, inputs: list[np.ndarray]):
        return [np.less_equal(inputs[0], inputs[1])]


class Not(ElementwiseBase):
    """ Defines not module hook. """

    def inference_shape(self, inputs: list[np.ndarray]):
        result = np.logical_not(inputs[0].astype(np.bool))
        return [result]


class And(ElementwiseBase):
    """ Defines and module hook. """

    def inference_shape(self, inputs: list[np.ndarray]):
        return [np.logical_and(inputs[0].astype(np.bool), inputs[1].astype(np.bool))]


class Where(ElementwiseBase):
    """ Defines where module hook. """

    def inference_shape(self, inputs: list[np.ndarray]):
        return [np.where(inputs[0], inputs[1], inputs[2])]


class Equal(ElementwiseBase):
    """ Defines equal module hook. """

    def inference_shape(self, inputs: list[np.ndarray]):
        return [np.equal(inputs[0], inputs[1])]


class Greater(ElementwiseBase):
    """ Defines greater module hook. """

    def inference_shape(self, inputs: list[np.ndarray]):
        return [np.greater(inputs[0], inputs[1])]