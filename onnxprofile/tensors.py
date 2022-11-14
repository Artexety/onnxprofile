import onnx

from .hooks.common.functions import (transform_to_ndarray, get_attribute, construct_volume)


def set_inputs(graph: onnx.GraphProto, dynamic_tensors: dict, tensor_map: dict) -> onnx.GraphProto:
    """ 
    Sets values for dynamic graph inputs.
    
    Args:
        graph (onnx.GraphProto): graph to set inputs for
        dynamic_tensors (dict): dictionary of dynamic tensors
        tensor_map (dict): dictionary of tensors
    Returns:
        Updated graph
    """
    for x in graph.input:
        if dynamic_tensors.keys().__contains__(x.name):
            tensor_map[x.name] = dynamic_tensors[x.name]
            dim = x.type.tensor_type.shape.dim
            for nb, dnb in zip(dim, dynamic_tensors[x.name].shape):
                nb.dim_value = dnb
    return graph


def add_outputs(graph: onnx.GraphProto, output_names: list, tensor_map: dict) -> onnx.GraphProto:
    """
    Creates new outputs for the given graph.

    Args:
        graph (onnx.GraphProto): graph to add outputs to.
        output_names (list): names of the outputs to add to the graph.
        tensor_map (dict): dictionary of tensors
    Returns:
        Updated graph
    """
    for name in output_names:
        if tensor_map is not None and name in tensor_map:
            new_output = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, tensor_map[name].shape)
        else:
            new_output = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, ())
        graph.output.append(new_output)
    return graph


def update_statics(graph: onnx.GraphProto, tensor_map: dict, params_map: dict) -> int:
    """
    Updates static inputs of the given graph.

    Args:
        graph (onnx.GraphProto): graph to update statics for
        tensor_map (dict): dictionary of tensors
        params_map (dict): dictionary of parameters
    Returns:
        Total amount of parameters present in the graph
    """
    total_params = 0
    for x in graph.initializer:
        ndarray = transform_to_ndarray(x)
        tensor_map[x.name] = ndarray
    for n in graph.node:
        if n.op_type == 'Constant':
            for a in n.attribute:
                if a.name == 'value':
                    tensor_map[n.output[0]] = get_attribute(a)
    for k in tensor_map:
        params_map[k] = construct_volume(tensor_map[k].shape)
        total_params += params_map[k]
    return total_params


def remove_unused_tensors(graph : onnx.GraphProto) -> None:
    """
    Removes unused tensors from the graph.

    Args:
        graph (onnx.GraphProto): graph to remove unused tensors from
    """
    consumer = {}
    producer = {initial.name: 0 for initial in graph.initializer}
    for node in graph.node:
        for input in node.input:
            consumer[input] = 0
        for output in node.output:
            producer[output] = 0
    inputs = [key for key in consumer if key not in producer]
    outputs = [key for key in producer if key not in consumer]
    valid_inputs = [input for input in graph.input if input.name in inputs]
    valid_outputs = [output for output in graph.output if output.name in outputs]
    graph.ClearField('input')
    for input in valid_inputs:
        graph.input.append(input)
    graph.ClearField('output')
    for output in valid_outputs:
        graph.output.append(output)