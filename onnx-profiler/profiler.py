import numpy as np
import onnx

import core
import hooks

from core.tensors import update_statics, set_inputs, remove_unused_tensors
from tabulate import tabulate

from hooks.common.functions import construct_volume, get_tensor_shape, validate_ndarray, construct_ndarray
from hooks.common.utilities import calculate_sparsity, zero_flag


class Profiler(object):

    def __init__(self):
        self.registry_instance = core.collections.Registry()
        self.__construct_base_registry()

        self.globals = core.collections.AttributeDict()
        self.globals.tensor_map = {}
        self.globals.params_map = {}
        self.globals.sparse_map = {}
        self.globals.node_map = {}
        self.globals.shared_nodes = {}

        self.measurements = core.collections.AttributeDict()
        self.measurements.macs = 0.0
        self.measurements.params = 0
        self.measurements.memory = 0

    def profile(self, model, dynamic_shapes, hidden_operations, stdout, verbose):
        if isinstance(model, str):
            model = onnx.load(model)
        
        if isinstance(model, onnx.ModelProto):
            remove_unused_tensors(model.graph)
            self.__profile(model.graph, dynamic_shapes, hidden_operations, verbose)
            if stdout: 
                self.__to_stdout(metric='FLOPs')
            return (self.measurements.macs, self.measurements.params)

    def __profile_node(self, proto, inputs, outputs):
        node_class = self.registry_instance.lookup(proto.op_type)
        if node_class is not None:
            worker = node_class(proto)
            return worker.profile(inputs, outputs)
        raise NotImplementedError()                                                         # TODO

    def __retrieve_shape(self, proto, inputs):
        node_class = self.registry_instance.lookup(proto.op_type)
        if node_class is not None:
            worker = node_class(proto)
            t = worker.inference_shape(inputs)
            return worker.inference_shape(inputs)
        raise NotImplementedError()                                                         # TODO

    def __construct_base_registry(self):
        supported_operations = {
            'Conv' : hooks.elementwise.Conv,
            'ConvTranspose' : hooks.elementwise.ConvTranspose,
            'Unsqueeze' : hooks.elementwise.Unsqueeze,
            'Squeeze' : hooks.elementwise.Squeeze,
            'Resize' : hooks.elementwise.Resize,
            'ScatterND' : hooks.elementwise.ScatterND,
            'ScatterElements' : hooks.elementwise.ScatterElements,
            'Pool' : hooks.elementwise.Pool,
            'MaxPool' : hooks.elementwise.MaxPool,
            'AveragePool' : hooks.elementwise.AveragePool,
            'GlobalAveragePool' : hooks.elementwise.GlobalAveragePool,
            'Gather' : hooks.elementwise.Gather,
            'Constant' : hooks.elementwise.Constant,
            'Concat' : hooks.elementwise.Concat,
            'Reshape' : hooks.elementwise.Reshape,
            'ArgMax' : hooks.elementwise.ArgMax,
            'Expand' : hooks.elementwise.Expand,
            'Tile' : hooks.elementwise.Tile,
            'GRU' : hooks.elementwise.GRU,
            'TopK' : hooks.elementwise.TopK,
            'LSTM' : hooks.elementwise.LSTM,
            'Compress' : hooks.elementwise.Compress,
            'RoiAlign' : hooks.elementwise.RoiAlign,
            'Gemm' : hooks.elementwise.Gemm,
            'MatMul' : hooks.elementwise.MatMul,
            'MatMulInteger' : hooks.elementwise.MatMulInteger,
            'Shape' : hooks.elementwise.Shape,
            'OneHot' : hooks.elementwise.OneHot,
            'Pad' : hooks.elementwise.Pad,
            'Split' : hooks.elementwise.Split,
            'ReduceMean' : hooks.elementwise.ReduceMean,
            'ReduceProd' : hooks.elementwise.ReduceProd,
            'ReduceL2' : hooks.elementwise.ReduceL2,
            'ReduceMax' : hooks.elementwise.ReduceMax,
            'ReduceMin' : hooks.elementwise.ReduceMin,
            'ReduceSum' : hooks.elementwise.ReduceSum,
            'NonZero' : hooks.elementwise.NonZero,
            'Transpose' : hooks.elementwise.Transpose,
            'ConstantOfShape' : hooks.elementwise.ConstantOfShape,
            'Flatten' : hooks.elementwise.Flatten,
            'Einsum' : hooks.elementwise.Einsum,
            'Less' : hooks.elementwise.Less,
            'LessOrEqual' : hooks.elementwise.LessOrEqual,
            'Not' : hooks.elementwise.Not,
            'And' : hooks.elementwise.And,
            'Where' : hooks.elementwise.Where,
            'Greater' : hooks.elementwise.Greater,
            'Equal' : hooks.elementwise.Equal,
            'CategoryMapper' : hooks.pointwise.CategoryMapper,
            'LRN' : hooks.pointwise.LocalResponseNormalization,
            'Add' : hooks.pointwise.Add,
            'Sub' : hooks.pointwise.Sub,
            'Mul' : hooks.pointwise.Mul,
            'Sum' : hooks.pointwise.Sum,
            'Abs' : hooks.pointwise.Abs,
            'Neg' : hooks.pointwise.Neg,
            'Min' : hooks.pointwise.Min,
            'Max' : hooks.pointwise.Max,
            'Pow' : hooks.pointwise.Pow,
            'Exp' : hooks.pointwise.Exp,
            'Log' : hooks.pointwise.Log,
            'Sin' : hooks.pointwise.Sin,
            'Cos' : hooks.pointwise.Cos,
            'Div' : hooks.pointwise.Div,
            'Hardmax' : hooks.pointwise.Hardmax,
            'Relu' : hooks.pointwise.Relu,
            'PRelu' : hooks.pointwise.PRelu,
            'Clip' : hooks.pointwise.Clip,
            'Relu6' : hooks.pointwise.Relu6,
            'CumSum' : hooks.pointwise.CumSum,
            'Sigmoid' : hooks.pointwise.Sigmoid,
            'Softmax' : hooks.pointwise.Softmax,
            'Tanh' : hooks.pointwise.Tanh,
            'Sqrt' : hooks.pointwise.Sqrt,
            'InstanceNormalization' : hooks.pointwise.InstanceNormalization,
            'Range' : hooks.pointwise.Range,
            'Reciprocal' : hooks.pointwise.Reciprocal,
            'HardSigmoid' : hooks.pointwise.HardSigmoid,
            'LeakyRelu' : hooks.pointwise.LeakyRelu,
            'Identity' : hooks.fusion.Identity,
            'Erf' : hooks.fusion.Erf,
            'Dropout' : hooks.fusion.Dropout,
            'BatchNormalization' : hooks.fusion.BatchNormalization,
            'Slice' : hooks.fusion.Slice,
            'Cast' : hooks.fusion.Cast,
            'Floor' : hooks.fusion.Floor,
            'Ceil' : hooks.fusion.Ceil,
            'DequantizeLinear' : hooks.quantize.DequantizeLinear,
            'QuantizeLinear' : hooks.quantize.QuantizeLinear,
            'QuantizeLinearMatMul' : hooks.quantize.QuantizeLinearMatMul,
            'QuantizeLinearConv' : hooks.quantize.QuantizeLinearConv,
        }

        for name in supported_operations:
            self.registry_instance.register(supported_operations[name], name)

    def __profile_inputs(self, graph):
        for x in graph.input:
            tensor = self.globals.tensor_map[x.name]
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
            self.globals.node_map[x.name] = node_data
            self.measurements.memory += memory

    def __profile_nodes(self, graph, sparse_model, hidden_operations, flags):
        profile_position = 0

        for x in graph.node:
            inputs, outputs = ([], [])
            macs, params, memory = (0, 0, 0)

            if hidden_operations is not None and x.op_type in hidden_operations:
                continue

            for input in x.input:
                if input == '':
                    continue

                inputs.append(self.globals.tensor_map[input])
                if input in self.globals.params_map.keys():
                    if flags[input] == 0:
                        params += self.globals.params_map[input]
                        memory += self.globals.params_map[input]
                    flags[input] += 1

            for output in x.output:
                if self.globals.tensor_map.keys().__contains__(output):
                    outputs.append(self.globals.tensor_map[output])
                    if x.op_type == 'Constant':
                        continue
                    
                    memory += construct_volume(self.globals.tensor_map[output].shape)
            
            macs, _ = self.__profile_node(x, inputs, outputs)
            output_shape, input_shape = ((0,), (0,))
            
            if outputs:
                output_shape = outputs[0].shape
                output_shape = (0,) if len(output_shape) == 0 else output_shape
            if inputs:
                input_shape = inputs[0].shape
                input_shape = (0,) if len(input_shape) == 0 else input_shape
            if len(x.name) == 0:
                x.name = f'{x.op_type}_{profile_position}'

            profile_position += 1
            memory *= 4

            note_data = {
                'macs': macs, 
                'params': params, 
                'memory': memory, 
                'input_shape': input_shape,
                'output_shape': output_shape,
            }

            if sparse_model:
                is_sparse = False
                for input in x.input:
                    if input == '':
                        continue
                    if input in self.globals.sparse_map.keys():
                        note_data |= self.globals.sparse_map[input]
                        is_sparse = True
                        break
                if not is_sparse:
                    note_data.update(
                        {
                            'block_size': (1, 1), 
                            'block_ratio': 0,
                            'ratio': 0, 
                        }
                    )

            self.globals.node_map.update({x.name: note_data})

            self.measurements.macs += macs
            self.measurements.params += params
            self.measurements.memory += memory

    def __profile(self, graph, dynamic_shapes = dict, hidden_operations=None, verbose = False):
        if hidden_operations is None:
            hidden_operations = {'Identity', 'Constant'}
        internal_timer = core.collections.Timer()
        self.__retrieve_shapes(graph, dynamic_shapes)

        if verbose:
            print(f"infered all tensor shapes in {internal_timer.stop():3f} sec's")

        sparse_model = len(self.globals.sparse_map.keys()) > 0
        params_flags = {key: False for key in self.globals.params_map.keys()}

        self.__profile_inputs(graph)
        self.__profile_nodes(graph, sparse_model, hidden_operations, params_flags)

        if verbose:
            print(f"profiled all nodes in {internal_timer.stop():.3f} sec's")  

        for node in graph.node:
            for x in node.input:
                if x == '':
                    continue
                if x in self.globals.params_map.keys() and params_flags[x] > 1 \
                             and construct_volume(self.globals.tensor_map[x].shape) > 128:
                    if x in self.globals.shared_nodes:
                        self.globals.shared_nodes[x].append(node.name)
                    else:
                        self.globals.shared_nodes[x] = [node.name]

        if verbose:
            count = sum(construct_volume(self.globals.tensor_map[t].shape) for t in self.globals.tensor_map)
            count *= 4
            difference = abs(self.measurements.memory - count) / count
            print(f"globals->tensor_map: {count} globals->self.globals.node_map: {self.measurements.memory}, delta: {difference:.3%}")
            assert (difference < 0.01)

    def __retrieve_shapes(self, graph, dynamic_tensors):
        self.globals.tensor_map = {}
        self.globals.params_map = {}

        total = update_statics(graph, self.globals.tensor_map, self.globals.params_map)
        self.__perform_sparsity_search()

        if dynamic_tensors is not None:
            set_inputs(graph, dynamic_tensors, self.globals.tensor_map)

        for x in graph.input:
            shape = get_tensor_shape(x)
            for dim in shape:
                if dim < 0:
                    raise ValueError()                                                          # TODO
            if x.name not in self.globals.tensor_map:
                self.globals.tensor_map[x.name] = np.zeros(shape, dtype=np.float32)

            if not validate_ndarray(self.globals.tensor_map[x.name]):
                raise ValueError()                                                              # TODO

        for node in graph.node:
            inputs = [self.globals.tensor_map[x] for x in node.input if x != '']
            outputs = [y for y in node.output if y != '']
            output_tensors = self.__retrieve_shape(node, inputs)
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

    def __validate_block_1d(self, ndarray, x, size, ratio, axis, threshold):
        if x and ndarray.shape[axis] % size == 0:
            if axis == 0:
                flag = zero_flag(ndarray.reshape(-1, size, ndarray.shape[1]))
                _sum = np.sum(flag, 1)
            elif axis == 1:
                flag = zero_flag(ndarray.reshape(ndarray.shape[0], -1, size))
                _sum = np.sum(flag, -1)
            _ratio_x = (_sum == size).sum() / _sum.size
            return (True, _ratio_x) if _ratio_x > ratio - threshold else (False, ratio)
        else:
            return (False, ratio)
    
    def __validate_block_nd(self, ndarray, x, size, ratio, axis, threshold):
        if x and ndarray.shape[axis] % size == 0:
            if axis == 0:
                flag = zero_flag(ndarray.reshape(-1, size, *ndarray.shape[1:]))
                _sum = np.sum(flag, 2)
            elif axis == 1:
                flag = zero_flag(ndarray.reshape(ndarray.shape[0], -1, size, *ndarray.shape[2:]))
                _sum = np.sum(flag, 1)
            _ratio_x = (_sum == size).sum() / _sum.size
            return (True, _ratio_x) if _ratio_x > ratio - threshold else (False, ratio)
        else:
            return (False, ratio)

    def __search_sparse_1d(self, ndarray, ratio, threshold=0.1):
        init_size, valid_size = 2, 1
        p, q = True, True
        valid_ratio = ratio
        while True:
            is_valid_q, valid_ratio = self.__validate_block_1d(
                ndarray=ndarray, 
                x=q, 
                size=init_size, 
                axis=1,
                ratio=valid_ratio, 
                threshold=threshold,
            )

            is_valid_p, valid_ratio = self.__validate_block_1d(
                ndarray=ndarray, 
                x=p, 
                size=init_size,
                axis=0, 
                ratio=valid_ratio, 
                threshold=threshold
            )

            if not is_valid_q and not is_valid_p:
                break

            valid_size = init_size
            init_size *= 2
            p = is_valid_p
            q = is_valid_q

        if q and p:
            temp = ndarray.reshape(
                ndarray.shape[0] // valid_size, 
                valid_size,
                ndarray.shape[1] // valid_size, 
                valid_size
            )
            flag = zero_flag(temp)
            _sum = np.sum(flag, axis=(1, -1))
            _ratio_s = (_sum == (valid_size * valid_size)).sum() / _sum.size
            if _ratio_s > ratio - threshold:
                return ((valid_size, valid_size), _ratio_s)

        return ((valid_size if p else 1, valid_size if q else 1), valid_ratio)

    def __search_sparse_nd(self, ndarray, ratio, threshold):
        init_size, valid_size = (2, 1)
        p, q = (True, True)
        valid_ratio_p, valid_ratio_q = (ratio, ratio)
        
        while True:
            is_valid_q, valid_ratio_q = self.__validate_block_nd(
                ndarray=ndarray, 
                x=q, 
                size=init_size,
                axis=1, 
                ratio=valid_ratio_q, 
                threshold=threshold,
            )

            is_valid_p, valid_ratio_p = self.__validate_block_nd(
                ndarray=ndarray, 
                x=p, 
                size=init_size,
                axis=0, 
                ratio=valid_ratio_p, 
                threshold=threshold,
            )

            if not is_valid_q and not is_valid_p:
                break

            valid_size = init_size
            init_size *= 2
            p = is_valid_p
            q = is_valid_q

        if valid_size > 1 and q and p:
            temp = ndarray.reshape(
                ndarray.shape[0] // valid_size, 
                valid_size, 
                ndarray.shape[1] // valid_size, 
                valid_size,
                *ndarray.shape[2:]
            )
            
            flag = zero_flag(temp)
            _sum = np.sum(flag, axis=(1, 3))
            ratios = (_sum == (valid_size * valid_size)).sum() / _sum.size
            
            if ratios > ratio - threshold:
                return ((valid_size, valid_size), ratios)
        
        if valid_ratio_p > valid_ratio_q:
            return (valid_size, 1), valid_ratio_p
        
        return ((1, valid_size), valid_ratio_q)

    def __search_sparse_block(self, ndarray, ratio, threshold=0.1):
        if len(ndarray.shape) == 2:  # gemm, matmul
            return self.__search_sparse_1d(ndarray, ratio, threshold)    
        elif len(ndarray.shape) == 4:  # conv2d
            return self.__search_sparse_nd(ndarray, ratio, threshold)
        return ((1, 1), ratio)

    def __perform_sparsity_search(self, threshold_size = 128, threshold_ratio = 0.4):
        self.globals.sparse_map = {}

        for key in self.globals.tensor_map.keys():
            ndarray = self.globals.tensor_map[key]
            if (construct_volume(ndarray.shape) > threshold_size):
                ratio = calculate_sparsity(ndarray)
                if ratio is not None and ratio > threshold_ratio:
                    block_size, block_ratio = self.__search_sparse_block(ndarray, ratio)
                    self.globals.sparse_map[key] = {
                        "block_size" : block_size,
                        "block_ratio" : block_ratio,
                        "ratio" : ratio,
                    }
    
    def __to_stdout(self, f: str = None, metric='MACs'):
        
        def tstr(t: tuple, splitch=','):
            s = ''
            for i, v in enumerate(t):
                s += str(v)
                if i != len(t) - 1:
                    s += splitch
            return s

        def nstr(n):
            return '{:,}'.format(n)

        assert (metric in ['MACs', 'FLOPs'])

        print_sparse_table = True
        if len(self.globals.sparse_map.keys()) == 0:
            print_sparse_table = False
        splitch = 'x'

        ptable = []

        macs = int(round(self.measurements.macs))
        params = int(self.measurements.params)
        memory = int(self.measurements.memory)

        if len(self.globals.shared_nodes.keys()):
            print(f"\n{'*' * 64}")
            print("Please note that Weight Tensors Sharing is detected:")
            for key in self.globals.shared_nodes.keys():
                print(f'Tensor:{key} ')
                print('Shared by: ')
                for node in self.globals.shared_nodes[key]:
                    print('           ', node)
                print()
            print('*' * 64)

        factor = 2 if metric == 'FLOPs' else 1

        params += 1e-18
        macs += 1e-18
        for key in self.globals.node_map.keys():
            item = self.globals.node_map[key]
            row = [key]
            if print_sparse_table:
                row.append(tstr(item['block_size'], splitch))
                row.append('{:.2%}'.format(item['block_ratio']))
                row.append('{:.2%}'.format(item['ratio']))
            row.append(nstr(int(item['macs']) * factor))
            row.append('{:.2%}'.format(item['macs'] / macs))
            row.append(nstr(int(item['memory'])))
            row.append('{:.2%}'.format(item['memory'] / memory))
            row.append(nstr(int(item['params'])))
            row.append('{:.2%}'.format(item['params'] / params))
            row.append(tstr(item['input_shape'], splitch))
            row.append(tstr(item['output_shape'], splitch))

            ptable.append(row)

        header = ['Name']
        if print_sparse_table:
            header.append('Sparse Pattern')
            header.append('Sparse Block Ratio')
            header.append('Sparse Ratio')
        header.extend([metric, 'FlOPs %', 'Memory', 'Memory %', 'Parameters', 'Parameters %', 'Input Shape',
                    'Output Shape'])

        if f is None:
            print(tabulate(ptable, headers=header, tablefmt="psql"))
        else:
            with open(f, 'w') as fp:
                headerstr = ''
                for i, item in enumerate(header):
                    headerstr += item
                    if i < len(header) - 1:
                        headerstr += ','
                headerstr += '\n'
                fp.write(headerstr)
                for row in ptable:
                    _row = ''
                    for i, element in enumerate(row):
                        _row += element
                        if i != len(row) - 1:
                            _row += ','
                    _row += '\n'
                    fp.write(_row)


if __name__ == '__main__':
    print("\n input: roberta-base-11.onnx \n")

    p = Profiler()
    macs, params = p.profile(
        'resnet50-v1-7.onnx', 
        dynamic_shapes={
            'data': np.zeros((1, 3, 224, 224), 
            np.float32)
        }, 
        hidden_operations=None, 
        stdout=True, 
        verbose=False
    )
    # macs, params = p.profile(
    #     'gpt2-10.onnx', 
    #     dynamic_shapes={
    #         'input1': construct_ndarray((1, 1, 8), dtype=np.int64),
    #     }, 
    #     hidden_operations=None, 
    #     stdout=True, 
    #     verbose=False
    # )
    # macs, params = p.profile(
    #     'roberta-base-11.onnx', 
    #     dynamic_shapes={
    #         'input_ids': construct_ndarray((1, 8), dtype=np.int64)
    #     }, 
    #     hidden_operations=None, 
    #     stdout=True, 
    #     verbose=False
    # )

    print(f"\n macs: {macs:,}")
    print(f" params: {float(params):,}")
