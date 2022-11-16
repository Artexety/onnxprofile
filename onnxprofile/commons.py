from time import time
from tabulate import tabulate

from .hooks.common.constants import METRICS


class Registry(object): 
    def __init__(self):
        self.__registry = {}
    
    def __setitem__(self, key, obj):
        if key in self.__registry:
            raise KeyError(f"'{key}' was already registered!")
        
        self.__registry[key] = obj
    
    def __getitem__(self, key):
        if self.__registry.__contains__(key) is False:
            raise KeyError(f"entry '{key}' was not found!")
        
        return self.__registry[key]
    
    def __contains__(self, key):
        return key in self.__registry
    
    def __iter__(self):
        return iter(self.__registry.items())
    
    def register(self, obj = None, name = None):
        """ Registers a object under a given name.
        Can be used as decorator or method.
    
        Args:
            obj: The object to add to the registry.
            name: The name of the object.
        Raises:
            KeyError: If same name is used twice.
        """
        if obj is None:
            def dec(x):
                _name = x.__name__ if name is None else name
                self.__setitem__(_name, x)
                return x
            return dec

        _name = obj.__name__ if name is None else name
        self.__setitem__(_name, obj)
    
    def lookup(self, key):
        """ Looks up "key".
    
        Args:
            key: a string specifying the registry key for the object.
        Returns:
            Registered object if found
        Raises:
            KeyError: if "name" has not been registered.
        """
        if key in self.__registry:
            return self.__registry[key]
        else:
           raise KeyError(f"entry '{key}' was not found!")
    
    def list(self):
        """ Lists registered objects.
        
        Returns:
            A list of names of registered objects.
        """
        return self.__registry.keys()


class AttributeDict(dict):
    IMMUTABLE = "__immutable__"

    def __init__(self, *args, **kwargs):
        super(AttributeDict, self).__init__(*args, **kwargs)
        self.__dict__[AttributeDict.IMMUTABLE] = False

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if self.__dict__[AttributeDict.IMMUTABLE]:
            raise AttributeError(f"Attempted to set '{name}' to '{value}', but AttrDict is immutable")
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value

    def immutable(self, is_immutable):
        """
        Set immutability to is_immutable and recursively apply the setting
        to all nested AttributeDicts.

        Args:
            is_immutable: controls IMMUTABLE parameter.
        """
        self.__dict__[AttributeDict.IMMUTABLE] = is_immutable
        # Recursively set immutable state
        for v in self.__dict__.values():
            if isinstance(v, AttributeDict):
                v.immutable(is_immutable)
        for v in self.values():
            if isinstance(v, AttributeDict):
                v.immutable(is_immutable)

    def is_immutable(self):
        return self.__dict__[AttributeDict.IMMUTABLE]


class SimpleTimer():
    def __init__(self):
        self.start_time = time()

    def start(self):
        """ Starts timer. """
        self.start_time = time()

    def stop(self):
        """ Stops timer. """
        return time() - self.start_time


def _tuple_to_string(x: tuple, split_character: str = ",") -> str:
    response_string = ""
    for i, value in enumerate(x):
        response_string += str(value)
        if i != len(x) - 1:
            response_string += split_character
    return response_string


def _number_to_string(x: int, use_csv_format: bool = True):
    return f"{x:,}" if use_csv_format else f"{x}"


def profile_to_console(messurements: dict, maps: dict, operation_metric: int = METRICS.FLOPs) -> None:
    assert (operation_metric in [METRICS.FLOPs, METRICS.MACs]), f"unknown metric: {operation_metric}"

    has_sparse_contents = len(maps["sparse"].keys()) != 0
    factor = 2 if operation_metric == METRICS.FLOPs else 1

    table_header = ["Name"]
    if has_sparse_contents:
        table_header.extend(("Sparse Pattern", "Sparse Block Ratio", "Sparse Ratio"))
    if operation_metric == METRICS.MACs:
        table_header.extend(("MACs", "MACs %"))
    if operation_metric == METRICS.FLOPs:
        table_header.extend(("FLOPs", "FLOPs %"))
    table_header.extend(("Memory", "Memory %", "Parameters", "Parameters %", "Input Shape", "Output Shape"))

    table_body = []
    for key, item in maps["node"].items():
        row = [key]
        if has_sparse_contents:
            row.extend((_tuple_to_string(item["block_size"], split_character="x"), 
                "{:.2%}".format(item["block_ratio"]), "{:.2%}".format(item["ratio"])))

        row.extend((_number_to_string(int(item["macs"]) * factor), "{:.2%}".format(item["macs"] / messurements["macs"])))
        row.extend((_number_to_string(int(item["memory"])), "{:.2%}".format(item["memory"] / messurements["memory"])))
        row.extend((_number_to_string(int(item["params"])), "{:.2%}".format(item["params"] / messurements["params"])))
        row.append(_tuple_to_string(item["input_shape"], split_character="x"))
        row.append(_tuple_to_string(item["output_shape"], split_character="x"))
        table_body.append(row)

    table_footer = ["TOTAL"]
    if has_sparse_contents:
        table_footer.extend(("_", "_", "_"))
    table_footer.extend((
        _number_to_string(int(messurements["macs"] * factor)), "100%", 
        _number_to_string(int(messurements["memory"])), "100%", 
        _number_to_string(int(messurements["params"])), "100%", 
        "_", "_"))

    table_body.append(table_footer)
    print(tabulate(table_body, headers=table_header, tablefmt="psql"))