# from __future__ import annotations
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
import warnings

from typing import Any, Callable, List, Union, Dict


class O1NumHess:
    def __init__(
        self,
        x: Union[np.ndarray, np.matrix, List[float]],
        # /, # `/` only available in python >= 3.8
        grad_func: Callable[..., np.ndarray],
        **kwargs_for_grad_func, # TODO kwargs for func g can be specified both here and called
    ):

        # TODO assert grad_func is callable

        # ensure x is valid
        self.x = np.array(x)
        assert self.x.size > 0
        #   判断维度是否合法：单个常数的维度=0，能够被排除；二维要求必须是行向量，然后展开成一维的
        assert (self.x.ndim == 1) or (self.x.ndim == 2 and self.x.shape[0] == 1), f"the shape of input x: {self.x.shape} is invalid"
        if self.x.ndim == 2:
            self.x = self.x[0]

        self.grad_func = grad_func
        self.kwargs = kwargs_for_grad_func
        # self.total_cores = os.cpu_count() # total num of cpu core

    def _parallel_execute(self, x_list: List[np.ndarray], core: int = 1, total_cores: Union[int, None] = None) -> List[np.ndarray]: # type: ignore
        """
        用法：给定若干向量x构成的列表x_list，调用梯度函数g并行对每个x进行计算，返回梯度构成的列表
        """
        # ==================== check the cores
        # ensure total_cores is legal
        cpu_count = os.cpu_count()
        if not isinstance(cpu_count, int):
            assert isinstance(total_cores, int), f"Unable to obtain the maximum number of cores for the current computer, and total_cores has not been set"
            warnings.warn(f"Unable to obtain the maximum number of cores for the current computer, make sure total_cores ({total_cores}) is legal.", RuntimeWarning)
        else:
            if total_cores is None:
                warnings.warn(f"total_cores ({total_cores}) is not set or unknown, os.cpu_count() ({os.cpu_count()}) is used.", RuntimeWarning)
                total_cores:int = cpu_count
            elif not isinstance(total_cores, int):
                raise TypeError(f"total_cores must be int, {type(total_cores)} is given.")
            else: # total_cores is int
                if not 1 <= total_cores <= cpu_count:
                    raise ValueError(f"total_cores ({total_cores}) must <= os.cpu_count() ({cpu_count}) and= > 1")

        # ensure core is legal
        if not isinstance(core, int):
            raise TypeError(f"core must be int, {type(core)} is given.")
        elif core < 0:
            raise ValueError(f"The number of cores specified by the user: {core} must > 0!")
        elif core > total_cores:
            raise ValueError(f"The number of cores specified by the user: {core} exceeds the total number of available cores: {total_cores}.")
        # If the number of cores specified by the user equals 0 or cannot divide exactly the total number of cores, warning
        if core == 0:
            warnings.warn(f"The number of cores specified by the user: {core} is 0, default value 1 will be use.", RuntimeWarning)
            core = 1
        elif total_cores % core != 0:
            warnings.warn(f"The number of cores specified by the user: {core} is not a divisor of the total number of cores: {total_cores}, may lead to performance issues.", RuntimeWarning)

        # ==================== calculate
        n = len(x_list)  # 总任务数
        max_concurrent = total_cores // core  # 最大并行任务数量
        main_batch = n // max_concurrent * max_concurrent  # 前面的主要批次的任务数（阶段1）
        tail_batch = n % max_concurrent  # 尾部剩余批次的任务数（阶段2）

        result:list[np.ndarray] = [None] * n # type: ignore

        # 阶段1
        if main_batch > 0:
            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                future_to_idx = {
                    executor.submit(
                        self.grad_func,
                        x_list[i],  # x
                        i,          # index
                        core,       # core
                        **self.kwargs,
                    ): i
                    for i in range(main_batch)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    result[idx] = future.result()

        # 阶段2
        if tail_batch > 0:
            # 每个任务固定分得 total_cores // tail_batch 个核心，前面 total_cores % tail_batch 个任务多分1个核心
            core_list = [(total_cores // tail_batch) + 1 if i < (total_cores % tail_batch) else (total_cores // tail_batch) for i in range(tail_batch)]
            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                future_to_idx = {
                    executor.submit(
                        self.grad_func,
                        x_list[i + main_batch], # x
                        i + main_batch,         # index
                        core_list[i],           # core
                        **self.kwargs,
                    ): i + main_batch
                    for i in range(tail_batch)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    result[idx] = future.result()

        return result

    def singleSide(self, delta: float = 1e-6, core: int = 0, total_cores: Union[int, None] = None) -> np.ndarray:
        r"""
        $$H_{ij}\approx\frac{g_{j}(x_{1},...,x_{i}+\Delta x,...,x_{n})-g_{j}(x_{1},...,x_{i},...,x_{n})}{\Delta x}$$

        H_i = ( g(..., x_i+Δx, ...) - g(..., x_i, ...) ) / delta x
        """
        n = len(self.x)
        x_and_x_with_delta = [self.x, *(self.x + delta * np.eye(n))]
        # x_and_x_with_delta is like:  (n+1 row)
        # [(x1,    x2   , ..., xi   , ..., xn   ),
        #  (x1+Δx, x2   , ..., xi   , ..., xn   ),
        #  (x1   , x2+Δx, ..., xi   , ..., xn   ),
        #  (x1   , x2   , ..., xi+Δx, ..., xn   ),
        #  (x1   , x2   , ..., xi   , ..., xn+Δx)]
        grad_and_grad_with_delta = self._parallel_execute(x_and_x_with_delta, core=core, total_cores=total_cores)
        # each line of x_and_x_with_delta will become grad here
        hessian:np.ndarray = (np.vstack(grad_and_grad_with_delta[1:]) - grad_and_grad_with_delta[0]) / delta # type: ignore
        # each line of np.vstack(grad_and_grad_with_delta[1:]) denotes g(..., x_i-Δx, ...)
        # each line minus grad_and_grad_with_delta[0] denotes g(..., x_i+Δx, ...) - g(..., x_i, ...)

        return hessian

    def doubleSide(self, delta: float = 1e-6, core: int = 0, total_cores: Union[int, None] = None) -> np.ndarray:
        r"""
        $$H_{ij}\approx\frac{g_j(x_1,...,x_i+\Delta x,...,x_n)-g_j(x_1,...,x_i-\Delta x,...,x_n)}{2\Delta x}$$

        H_i = ( g(..., x_i+Δx, ...) - g(..., x_i-Δx, ...) ) / 2 delta x
        """
        n = len(self.x)
        all_x_with_delta = [*(self.x + delta * np.eye(n)), *(self.x - delta * np.eye(n))]
        all_grad_with_delta = self._parallel_execute(all_x_with_delta, core=core, total_cores=total_cores)
        hessian = (np.vstack(all_grad_with_delta[:n]) - np.vstack(all_grad_with_delta[n:])) / (2 * delta) # type: ignore
        # np.vstack(all_grad_with_delta[:n]) denotes g(..., x_i+Δx, ...)
        # np.vstack(all_grad_with_delta[n:]) denotes g(..., x_i-Δx, ...)
        # view the comments in singleSide() for more information

        return hessian

    """TODO 无多线程的版本，完成之前保留备用进行测试"""

    def _singleSide(self, delta: float = 1e-6) -> np.ndarray:
        r"""
        $$H_{ij}\approx\frac{g_{j}(x_{1},...,x_{i}+\Delta x,...,x_{n})-g_{j}(x_{1},...,x_{i},...,x_{n})}{\Delta x}$$

        H_i = ( g(..., x_i+Δx, ...) - g(..., x_i, ...) ) / delta x
        """
        org = self.grad_func(self.x, 0, **self.kwargs)  # g(..., x_i, ...)
        delta_g = []  # delta_g = g(..., x_i+Δx, ...) - g(..., x_i, ...)
        for i in range(len(self.x)):
            delta_forward = self.x.copy()
            delta_forward[i] += delta
            forward = self.grad_func(delta_forward, 0, **self.kwargs)
            delta_g.append(forward - org)
        hessian = np.vstack(delta_g) / delta # type: ignore

        return hessian

    def _doubleSide(self, delta: float = 1e-6) -> np.ndarray:
        r"""
        $$H_{ij}\approx\frac{g_j(x_1,...,x_i+\Delta x,...,x_n)-g_j(x_1,...,x_i-\Delta x,...,x_n)}{2\Delta x}$$

        H_i = ( g(..., x_i+Δx, ...) - g(..., x_i-Δx, ...) ) / 2 delta x
        """
        delta_g = []  # delta_g = g(..., x_i+Δx, ...) - g(..., x_i-Δx, ...)
        for i in range(len(self.x)):
            delta_forward = self.x.copy()
            delta_forward[i] += delta
            delta_backward = self.x.copy()
            delta_backward[i] -= delta
            forward = self.grad_func(delta_forward, 0, **self.kwargs)
            backward = self.grad_func(delta_backward, 0, **self.kwargs)
            delta_g.append(forward - backward)
        hessian = np.vstack(delta_g) / (2 * delta) # type: ignore

        return hessian
