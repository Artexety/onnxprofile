import numpy as np

from .hooks.common.functions import construct_volume
from .hooks.common.utilities import calculate_sparsity, zero_flag

class SparsitySearch(object):
    def __init__(self) -> None:
        pass

    def search(self, tensors: dict, threshold_size: int = 128, threshold_ratio: float = 0.40) -> dict:
        """
        Searchs for the sparsity ratio for each tensor in tensors.

        Args:
            tensors (dict): A dictionary of tensors.
            threshold_size (int): The size of the threshold. Defaults to 128.
            threshold_ratio (float): Exit criterion
        Returns:
            Information about the sparsity of each tensor.
        """

        sparse_block_map = {}

        for key, ndarray in tensors.items():
            if (construct_volume(ndarray.shape) < threshold_size):
                continue
            ratio = calculate_sparsity(ndarray)
            if ratio is None or ratio < threshold_ratio:
                continue
            block_size, block_ratio = self._search_sparse_block_size(ndarray, ratio)
            sparse_block_map[key] = {
                "block_size" : block_size,
                "block_ratio" : block_ratio,
                "ratio" : ratio,
            }
        return sparse_block_map
    
    def _search_sparse_block_size(self, ndarray: np.ndarray, ratio: float, threshold: float = 0.099) -> tuple:
        if len(ndarray.shape) == 2:
            return self._search_sparse_block_size_1d(ndarray, ratio, threshold)    
        elif len(ndarray.shape) == 4:
            return self._search_sparse_block_size_nd(ndarray, ratio, threshold)
        return ((1, 1), ratio)
    
    def _validate_block_size_1d(self, ndarray: np.ndarray, x: bool, size: int, ratio: float, 
            axis: int, threshold: float = 0.099) -> tuple:
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
    
    def _validate_block_size_nd(self, ndarray: np.ndarray, x: bool, size: int, ratio: float,
            axis: int, threshold: float = 0.099):
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

    def _search_sparse_block_size_1d(self, ndarray: np.ndarray, ratio: float, threshold: float = 0.099) -> tuple:
        init_size, valid_size = 2, 1
        p, q = True, True
        valid_ratio = ratio
        while True:
            is_valid_q, valid_ratio = self._validate_block_size_1d(ndarray=ndarray, x=q, 
                size=init_size, axis=1, ratio=valid_ratio, threshold=threshold)

            is_valid_p, valid_ratio = self._validate_block_size_1d(ndarray=ndarray, x=p, 
                size=init_size, axis=0, ratio=valid_ratio, threshold=threshold)

            if not is_valid_q and not is_valid_p:
                break

            valid_size = init_size
            init_size *= 2
            p = is_valid_p
            q = is_valid_q

        if q and p:
            temp = ndarray.reshape(ndarray.shape[0] // valid_size, valid_size,
                ndarray.shape[1] // valid_size, valid_size)
            flag = zero_flag(temp)
            _sum = np.sum(flag, axis=(1, -1))
            _ratio_s = (_sum == (valid_size * valid_size)).sum() / _sum.size
            if _ratio_s > ratio - threshold:
                return ((valid_size, valid_size), _ratio_s)

        return ((valid_size if p else 1, valid_size if q else 1), valid_ratio)

    def _search_sparse_block_size_nd(self, ndarray: np.ndarray, ratio: float, threshold: float = 0.099) -> tuple:
        init_size, valid_size = (2, 1)
        p, q = (True, True)
        valid_ratio_p, valid_ratio_q = (ratio, ratio)
        
        while True:
            is_valid_q, valid_ratio_q = self._validate_block_size_nd(ndarray=ndarray, x=q, 
                size=init_size, axis=1, ratio=valid_ratio_q, threshold=threshold)

            is_valid_p, valid_ratio_p = self._validate_block_size_nd(ndarray=ndarray, x=p, 
                size=init_size, axis=0, ratio=valid_ratio_p, threshold=threshold)

            if not is_valid_q and not is_valid_p:
                break

            valid_size = init_size
            init_size *= 2
            p = is_valid_p
            q = is_valid_q

        if valid_size > 1 and q and p:
            temp = ndarray.reshape(ndarray.shape[0] // valid_size, valid_size, 
                ndarray.shape[1] // valid_size, valid_size, *ndarray.shape[2:])
            
            flag = zero_flag(temp)
            _sum = np.sum(flag, axis=(1, 3))
            ratios = (_sum == (valid_size * valid_size)).sum() / _sum.size
            
            if ratios > ratio - threshold:
                return ((valid_size, valid_size), ratios)
        
        if valid_ratio_p > valid_ratio_q:
            return (valid_size, 1), valid_ratio_p
        
        return ((1, valid_size), valid_ratio_q)