import numpy as np


def get_tensor_shape(x):
    output_shape = []
    for d in x.type.tensor_type.shape.dim:
        assert (d.dim_value != None)
        output_shape.append(d.dim_value)
    return output_shape


def get_initializer_shape(x):
    return list(x.dims)


def construct_volume(shape):
    volume = 1 if len(shape) > 0 else 0
    return volume * np.prod(shape)


def construct_ndarray(shape, dtype=np.float32):
    if dtype == np.float32:
        return np.ones(shape, dtype=dtype)
    elif dtype == np.int64:
        return np.zeros(shape, dtype=dtype)
    else:
        raise NotImplementedError(
            f"dtype: {dtype} currently not implemented"
        )


def validate_ndarray(ndarray):
    if isinstance(ndarray, (list, tuple)) and len(ndarray) == 0 or ndarray is None:
        return False
    elif isinstance(ndarray, np.ndarray):
        return bool(ndarray.size) if construct_volume(ndarray.shape) == 0 else True
    return False


def get_dtype(x):
    if x.data_type == x.FLOAT:
        return np.float32
    elif x.data_type == x.INT64:
        return np.int64
    elif x.data_type == x.INT32:
        return np.int32
    elif x.data_type == x.INT16:
        return np.int16
    elif x.data_type == x.INT8:
        return np.int8
    elif x.data_type == x.UNIT8:
        return np.uint8
    elif x.data_type == x.BOOL:
        return np.bool
    

def transform_to_ndarray(x):
    shape = get_initializer_shape(x)
    dtype = get_dtype(x)

    if x.raw_data != b'':
        return np.frombuffer(x.raw_data, dtype=dtype).reshape(shape)
    arr = np.zeros(shape, dtype=dtype).reshape((-1))
    if dtype == np.float64:
        for i in range(len(x.double_data)):
            arr[i] = x.double_data[i]
    elif dtype == np.float32:
        for i in range(len(x.float_data)):
            arr[i] = x.float_data[i]
    elif dtype == np.int64:
        for i in range(len(x.int64_data)):
            arr[i] = x.int64_data[i]
    elif dtype == np.int32:
        for i in range(len(x.int32_data)):
            arr[i] = x.int32_data[i]
    return arr.reshape(shape)


def get_attribute(a):
    if a.type == a.INTS:
        return list(a.ints)
    elif a.type == a.FLOATS:
        return list(a.floats)
    elif a.type == a.INT:
        return a.i
    elif a.type == a.FLOAT:
        return a.f
    elif a.type == a.STRING:
        return a.s
    elif a.type == a.TENSOR:
        return transform_to_ndarray(a.t)


def set_attributes(attributes, names, obj):
    for a in attributes:
        if a.name in names:
            obj.__setattr__(a.name, get_attribute(a))


def remove_initialisers(model):
    model.graph.ClearField("initializer")


def remove_constants(model):
    valid_nodes = [node for node in model.graph.node if node.op_type != "Constant"]
    model.graph.ClearField("node")
    for node in valid_nodes:
        model.graph.node.append(node)