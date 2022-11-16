import numpy as np
import argparse

from json import dumps
from pathlib import Path

from typing import Union, List, Any
from profiler import Profiler
from hooks.common.constants import MACS

print(MACS)

class HelpFormatter(argparse.HelpFormatter):
    def __init__(self, prog: Any) -> None:
        super().__init__(prog, max_help_position=40, width=80)
        self._action_counter = 0

    def _format_action_invocation(self, action : argparse.Action) -> str:
        if not action.option_strings or action.nargs == 0:
            return super()._format_action_invocation(action)
        
        default = self._get_default_metavar_for_optional(action)
        args_string = self._format_args(action, default)
        self._action_counter += 1
        
        return ", ".join(action.option_strings) + " <" + args_string.lower().split("_")[1] + ">"
 

def _retrieve_parser() -> argparse.Namespace:
    formatter = lambda prog: HelpFormatter(prog)
    parser = argparse.ArgumentParser(description="Count the MACs / FLOPs of your ONNX model.",
        prog="onnxprofile",
        usage="%(prog)s [-h] [-v,--version] -i <model> [-d <dynamic>]",
        formatter_class=formatter,
        )
    parser.add_argument("-i", "--input", dest="_model_input_location", required=True, 
        help="specify the path to the ONNX model")
    parser.add_argument("-d", "--dynamic_inputs", nargs='+', dest="_tensor_inputs", 
        help="specify the shape of the io tensors like as follows: \"input:type[f32, i32, i64]:1x3xHxW\"")
    parser.add_argument("-c", "--console_output", action="store_true", default=False, 
        dest="_console_output", help="display a detailed report to the console")
    
    return parser.parse_args()


def _string_to_dtype(encoded_type: str) -> Union[np.float32, np.int32, np.int64]:
    if encoded_type == "f32":
        return np.float32
    elif encoded_type == "i32":
        return np.int32
    elif encoded_type == "i64":
        return np.int64
    else:
        raise TypeError(f"Unknown tensor type: {encoded_type}")


def _string_to_list(encoded_list : str, dtype: np.dtype) -> list:
    return [int(value) for value in encoded_list.split("x")]


def _string_to_ndarray(arguments: List[str]) -> np.ndarray:
    response = {}
    for argument in arguments:
        contents = argument.split(":")
        dtype = _string_to_dtype(contents[1])
        shape = _string_to_list(contents[2], dtype=np.int32)

        if len(contents) > 3:
            ndarray = np.array(_string_to_list(contents[3], 
                dtype=dtype), dtype=dtype).reshape(shape)
        else:
            ndarray = np.zeros(shape, dtype=dtype)
        response[contents[0]] = ndarray
    return response


if __name__ == "__main__":
    arguments = _retrieve_parser()
    profiler_instance = Profiler()

    if arguments._tensor_inputs is not None:
        dynamic_inputs = _string_to_ndarray(arguments._tensor_inputs)
    else:
        dynamic_inputs = None
    
    macs, params = profiler_instance.profile(model=arguments._model_input_location,
        dynamic_inputs=dynamic_inputs, stdout=arguments._console_output)
    
    output_name = f"{Path(arguments._model_input_location).stem}.txt"
    with open(output_name, "w") as stream:
        stream.write(dumps(obj={"macs": float(macs), "params": int(params)}))
    