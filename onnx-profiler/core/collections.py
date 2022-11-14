from time import time


class Registry(object):
    """ Provides a registry for saving objects. """
    
    def __init__(self):
        """ Creates a new registry. """
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

    IMMUTABLE = '__immutable__'

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
            raise AttributeError(f'Attempted to set "{name}" to "{value}", but AttrDict is immutable')

        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value

    def immutable(self, is_immutable):
        """
        Set immutability to is_immutable and recursively apply the setting
        to all nested AttributeDicts.
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


class Timer():
    def __init__(self):
        self.start_time = time()

    def start(self):
        self.start_time = time()

    def stop(self):
        return time() - self.start_time