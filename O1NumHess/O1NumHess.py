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
        self.total_cores = os.cpu_count() # total num of cpu core

    def _parallel_execute(self, x_list: List[np.ndarray], core: int = 1) -> List[np.ndarray]:
        """
        用法：给定若干向量x构成的列表x_list，调用梯度函数g并行对每个x进行计算，返回梯度构成的列表
        """

        # 确保用户给定的核心数量合法
        if not isinstance(core, int):
            raise TypeError(f"core must be int, {type(core)} is given.")
        elif core < 0:
            raise ValueError(f"The number of cores specified by the user: {core} must > 0!")
        elif core > self.total_cores:
            raise ValueError(f"The number of cores specified by the user: {core} exceeds the total number of available cores: {self.total_cores}.")
        # 如果用户给定的核心数量=0或不能整除总核心数量，警告
        if core == 0:
            warnings.warn(f"The number of cores specified by the user: {core} is 0, default value 1 will be use.", UserWarning)
            core = 1
        elif self.total_cores % core != 0:
            warnings.warn(f"The number of cores specified by the user: {core} is not a divisor of the total number of cores: {self.total_cores}, may lead to performance issues.", UserWarning)

        n = len(x_list)  # 总任务数
        max_concurrent = self.total_cores // core  # 最大并行任务数量
        main_batch = n // max_concurrent * max_concurrent  # 前面的主要批次的任务数（阶段1）
        tail_batch = n % max_concurrent  # 尾部剩余批次的任务数（阶段2）

        result = [None] * n

        # 阶段1
        if main_batch > 0:
            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                future_to_idx = {
                    executor.submit(
                        self.grad_func,
                        x_list[i],  # x
                        i,          # index
                        core,       #core
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
            core_list = [(self.total_cores // tail_batch) + 1 if i < (self.total_cores % tail_batch) else (self.total_cores // tail_batch) for i in range(tail_batch)]
            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                future_to_idx = {
                    executor.submit(
                        self.grad_func,
                        x_list[i + main_batch], # x
                        i + main_batch,         # index
                        core_list[i],           #core
                        **self.kwargs,
                    ): i + main_batch
                    for i in range(tail_batch)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    result[idx] = future.result()

        return result

    def singleSide(self, core: int = 0, delta: float = 1e-6) -> np.ndarray:
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
        grad_and_grad_with_delta = self._parallel_execute(x_and_x_with_delta, core=core)
        # each line of x_and_x_with_delta will become grad here
        hessian = (np.vstack(grad_and_grad_with_delta[1:]) - grad_and_grad_with_delta[0]) / delta
        # each line of np.vstack(grad_and_grad_with_delta[1:]) denotes g(..., x_i-Δx, ...)
        # each line minus grad_and_grad_with_delta[0] denotes g(..., x_i+Δx, ...) - g(..., x_i, ...)

        return hessian

    def doubleSide(self, core: int = 0, delta: float = 1e-6) -> np.ndarray:
        r"""
        $$H_{ij}\approx\frac{g_j(x_1,...,x_i+\Delta x,...,x_n)-g_j(x_1,...,x_i-\Delta x,...,x_n)}{2\Delta x}$$

        H_i = ( g(..., x_i+Δx, ...) - g(..., x_i-Δx, ...) ) / 2 delta x
        """
        n = len(self.x)
        all_x_with_delta = [*(self.x + delta * np.eye(n)), *(self.x - delta * np.eye(n))]
        all_grad_with_delta = self._parallel_execute(all_x_with_delta, core=core)
        hessian = (np.vstack(all_grad_with_delta[:n]) - np.vstack(all_grad_with_delta[n:])) / (2 * delta)
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
        hessian = np.vstack(delta_g) / delta

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
        hessian = np.vstack(delta_g) / (2 * delta)

        return hessian
