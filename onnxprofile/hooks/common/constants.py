
class Constants(object):
    """ Create objects with read-only (constant) attributes. """

    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getattr__(self, key):
        return self._d[key]

    def __setattr__(self, key, value):
        if (key[0] == '_'):
            super(Constants, self).__setattr__(key, value)
        else:
            raise ValueError("setattr locked", self)

ZERO_OP = 0
MACS = Constants(
    ADD = 1,
    EXP = 16,
    LOG = 16,
    SQRT = 4,
    POW = 32,
    MUL = 1,
    DIV = 2,
    CMP = 1,
    SIN = 14,
    COS = 14,
)

METRICS = Constants(
    FLOPs = 0,
    MACs = 1,
)