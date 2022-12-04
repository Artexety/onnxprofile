import numpy as np
import onnx

from typing import Union
from os.path import abspath, join

from .commons import (Registry, AttributeDict, SimpleTimer, profile_to_console)
from .hooks.common.functions import (construct_volume, get_tensor_shape, validate_ndarray)
from .hooks.common.constants import METRICS

from .tensors import (remove_unused_tensors, set_inputs, update_statics)
from .operations import SUPPORTED_OPERATIONS
from .sparsity import SparsitySearch

import logging
import logging.config

# logging.config.fileConfig(join(abspath("."), 'logging.conf')) FIXME
logger = logging.getLogger() # FIXME


class Profiler(object):
    def __init__(self) -> None:
        self.registry_instance = Registry()
        self.crawler = SparsitySearch()
        
        for op in SUPPORTED_OPERATIONS:
            self.registry_instance.register(SUPPORTED_OPERATIONS[op], op)

        self.globals = AttributeDict()
        self.globals.tensor_map = {}
        self.globals.params_map = {}
        self.globals.sparse_map = {}
        self.globals.node_map = {}
        self.globals.shared_nodes = {}

        self.measurements = AttributeDict()
        self.measurements.macs = 0.0
        self.measurements.params = 0
        self.measurements.memory = 0

    def profile(self, config: dict = None, model : Union[str, onnx.ModelProto] = None, dynamic_inputs : dict = None, 
            hidden_operations : dict = None, stdout : bool = False, verbose: bool = False) -> Union[tuple, dict]:
        if config is not None:
            return self._profile_models_from_config(hidden_operations=hidden_operations, 
                verbose=verbose, config=config, stdout=stdout)
        elif model is not None and dynamic_inputs is not None:
            return self._profile_single_model(hidden_operations=hidden_operations, 
                dynamic_inputs=dynamic_inputs, verbose=verbose, stdout=stdout, model=model)
        raise ValueError("model and config arguments cannot be None at the same time!")                               # TODO

    def _profile_models_from_config(self, config: dict, hidden_operations: dict = None, stdout: bool = False, 
            verbose: bool = False) -> dict:
        if config is None:
            raise ValueError("config cannot be None!")                                                              # TODO
        response = {}
        for e in config:
            dynamic_inputs = {x: np.zeros(shape=x["shape"], dtype=x["dtype"]) for x in config[e]["dynamic_inputs"]}
            response[e] = self._profile_single_model(
                model=config[e]["path"],
                dynamic_inputs=dynamic_inputs,
                hidden_operations=hidden_operations,
                stdout=stdout,
                verbose=verbose
            )
        return response

    def _profile_single_model(self, model : Union[str, onnx.ModelProto], dynamic_inputs : dict, 
            hidden_operations : dict = None, stdout : bool = False, verbose: bool = False):
        if isinstance(model, str):
            model = onnx.load(model)
        
        if isinstance(model, onnx.ModelProto):
            remove_unused_tensors(model.graph)
            self._profile(model.graph, dynamic_inputs, hidden_operations, verbose)
            if stdout: 
                profile_to_console(
                    messurements={
                        "macs": self.measurements.macs,
                        "memory": self.measurements.memory,
                        "params": self.measurements.params,
                    }, 
                    maps={
                        "sparse": self.globals.sparse_map,
                        "node": self.globals.node_map,
                    },
                    operation_metric=METRICS.MACs 
                )
            return (self.measurements.macs, self.measurements.params)

    def _profile_single_node(self, proto: onnx.NodeProto, inputs: list[np.ndarray], 
            outputs: list[np.ndarray]) -> tuple:
        node_class = self.registry_instance.lookup(proto.op_type)
        if node_class is not None:
            worker = node_class(proto)
            return worker.profile(inputs, outputs)
        raise NotImplementedError()                                                         # TODO

    def _retrieve_single_shape(self, proto: onnx.NodeProto, inputs: list[np.ndarray]) -> tuple:
        node_class = self.registry_instance.lookup(proto.op_type)
        if node_class is not None:
            worker = node_class(proto)
            return worker.inference_shape(inputs)
        raise NotImplementedError()                                                         # TODO

    def _profile_inputs(self, graph: onnx.GraphProto) -> None:
        for _input in graph.input:
            tensor = self.globals.tensor_map[_input.name]
            memory = construct_volume(tensor.shape) * 4
            node_data = {
                "macs" : 0,
                "params": 0,
                "memory": memory,
                "input_shape": tensor.shape,
                "output_shape": tensor.shape,
                "block_size" : (1, 1),
                "ratio" : 0, 
                "block_ratio": 0
            }
            self.globals.node_map[_input.name] = node_data
            self.measurements.memory += memory

    def _profile_sparse_node(self, node: onnx.NodeProto, node_data: dict) -> None:
        is_sparse = False
        for input in node.input:
            if input == '':
                continue
            if input in self.globals.sparse_map.keys():
                node_data |= self.globals.sparse_map[input]
                is_sparse = True
                break
        if not is_sparse:
            node_data.update(
                {
                    'block_size': (1, 1), 
                    'block_ratio': 0,
                    'ratio': 0, 
                }
            )

    def _collect_node_inputs(self, node: onnx.NodeProto, flags: dict) -> tuple:
        memory, params, inputs = (0, 0, [])
        for _input in node.input:
                if _input == '':
                    continue
                inputs.append(self.globals.tensor_map[_input])
                if _input in self.globals.params_map.keys():
                    if flags[_input] == 0:
                        params += self.globals.params_map[_input]
                        memory += self.globals.params_map[_input]
                    flags[_input] += 1
        return (memory, params, inputs)

    def _collect_node_outputs(self, node: onnx.NodeProto, input_memory: int) -> tuple:
        memory, outputs = (input_memory, [])
        for _output in node.output:
            if self.globals.tensor_map.keys().__contains__(_output):
                outputs.append(self.globals.tensor_map[_output])
                if node.op_type == 'Constant':
                    continue
                memory += construct_volume(self.globals.tensor_map[_output].shape)
        return (memory, outputs)

    def _profile_nodes(self, graph: onnx.GraphProto, sparse_model: bool, 
            hidden_operations: dict, flags: dict) -> None:
        profile_position = 0
        for _node in graph.node:
            if hidden_operations is not None and _node.op_type in hidden_operations:
                continue
            
            memory, params, inputs = self._collect_node_inputs(_node, flags)
            memory, outputs = self._collect_node_outputs(_node, memory)
            macs, _ = self._profile_single_node(_node, inputs, outputs)
            output_shape, input_shape = ((0,), (0,))
            
            if outputs:
                output_shape = outputs[0].shape
                output_shape = (0,) if len(output_shape) == 0 else output_shape
            
            if inputs:
                input_shape = inputs[0].shape
                input_shape = (0,) if len(input_shape) == 0 else input_shape
            
            if len(_node.name) == 0:
                _node.name = f'{_node.op_type}_{profile_position}'

            profile_position += 1
            memory *= 4

            node_data = {'macs': macs, 'params': params, 'memory': memory, 
                'input_shape': input_shape, 'output_shape': output_shape}

            if sparse_model:
                self._profile_sparse_node(_node, node_data)

            self.globals.node_map.update({_node.name: node_data})
            self.measurements.macs += macs
            self.measurements.params += params
            self.measurements.memory += memory

    def _profile(self, graph: onnx.GraphProto, dynamic_inputs: dict = None, 
            hidden_operations: dict = None, verbose: bool = False) -> None:
        if hidden_operations is None:
            hidden_operations = {'Identity', 'Constant'}
        internal_timer = SimpleTimer()
        self._retrieve_single_shapes(graph, dynamic_inputs)

        if verbose:
            logger.info(f"retrivial of all tensor shapes took {internal_timer.stop():3f} sec's")

        sparse_model = len(self.globals.sparse_map.keys()) > 0
        params_flags = {key: False for key in self.globals.params_map.keys()}

        self._profile_inputs(graph)
        self._profile_nodes(graph, sparse_model, hidden_operations, params_flags)

        if verbose:
            logger.info(f"profiling of all nodes took {internal_timer.stop():.3f} sec's")

        for node in graph.node:
            for _input in node.input:
                if _input == '':
                    continue
                if _input in self.globals.params_map.keys() and params_flags[_input] > 1 \
                             and construct_volume(self.globals.tensor_map[_input].shape) > 128:
                    if _input in self.globals.shared_nodes:
                        self.globals.shared_nodes[_input].append(node.name)
                    else:
                        self.globals.shared_nodes[_input] = [node.name]

        if verbose:
            count = sum(construct_volume(self.globals.tensor_map[t].shape) for t in self.globals.tensor_map) * 4
            difference = abs(self.measurements.memory - count) / count
            logger.info(f"globals -> tensor_map: {count}")
            logger.info(f"globals -> nodememory: {self.measurements.memory}")
            logger.info(f"globals -> difference: {difference:.3%}")
            assert (difference < 0.01), f"diference to large! delta: {difference}"

    def _retrieve_single_shapes(self, graph: onnx.GraphProto, dynamic_inputs: dict) -> None:
        self.globals.tensor_map = {}
        self.globals.params_map = {}

        update_statics(graph, self.globals.tensor_map, self.globals.params_map)
        self.globals.sparse_block_map = self.crawler.search(self.globals.tensor_map)

        if dynamic_inputs is not None:
            set_inputs(graph, dynamic_inputs, self.globals.tensor_map)

        for _input in graph.input:
            shape = get_tensor_shape(_input)
            for dim in shape:
                if dim < 0:
                    raise ValueError()                                                          # TODO
            if _input.name not in self.globals.tensor_map:
                self.globals.tensor_map[_input.name] = np.zeros(shape, dtype=np.float32)

            if not validate_ndarray(self.globals.tensor_map[_input.name]):
                raise ValueError()                                                              # TODO

        for node in graph.node:
            inputs = [self.globals.tensor_map[x] for x in node.input if x != '']
            outputs = [y for y in node.output if y != '']
            output_tensors = self._retrieve_single_shape(node, inputs)
            for tensor, name in zip(output_tensors, outputs):
                self.globals.tensor_map[name] = tensor

        for key in self.globals.tensor_map:
            shape = self.globals.tensor_map[key].shape
            if len(shape) == 0:
                shape = (0,)
            value_info = onnx.helper.make_tensor_value_info(key, onnx.TensorProto.FLOAT, shape)
            graph.value_info.append(value_info)

        for output in graph.output:
            dim = output.type.tensor_type.shape.dim
            for nb, dnb in zip(dim, self.globals.tensor_map[output.name].shape):
                nb.dim_value = dnb
