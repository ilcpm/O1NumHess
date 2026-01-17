import numpy as np
import scipy
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
from pathlib import Path
import warnings
import time
import json
import pickle
import shutil
from datetime import datetime

from typing import Any, Callable, List, Union, Dict, Sequence, Tuple


class O1NumHess:
    def __init__(
        self,
        x: Union[np.ndarray, np.matrix, List[float]],
        # /, # `/` only available in python >= 3.8
        grad_func: Callable[..., np.ndarray],
        verbosity:int = 0,
        **kwargs_for_grad_func, # TODO kwargs for func g can be specified both here and called
    ):
        self.verbosity = verbosity

        # TODO assert grad_func is callable

        # ensure x is valid
        self.x = np.array(x)
        assert self.x.size > 0
        #   判断维度是否合法：单个常数的维度=0，能够被排除；二维要求必须是行向量，然后展开成一维的
        assert (self.x.ndim == 1) or (self.x.ndim == 2 and self.x.shape[0] == 1), f"the shape of input x: {self.x.shape} is invalid"
        if self.x.ndim == 2:
            self.x = self.x[0]

        # for gradient function
        self.grad_func = grad_func
        self.kwargs = kwargs_for_grad_func

        # for calculate - 断点续算相关
        self.task_cfg_json_name = "task.json"
        self.task_result_json_name = "task_result.json"

        # 任务配置字典（既是模板也是实例，所有字段初始化为None）
        self.task_config = {
            "task_name": None,        # str, 任务名称
            "method": None,           # str, "single"/"double"/"o1numhess"
            "delta": None,            # float, 扰动步长
            "start_time": None,       # str, ISO格式时间
            "total_tasks": None,      # int, 总梯度数量
            "core": None,             # int, 每个梯度的核心数
            "total_cores": None,      # int, 总核心数
            "x_first_size": None,     # int, 第一个x的大小(用于验证)
            "status": None,           # str, "running"
        }

        # 任务结果字典
        self.task_result = {
            "task_name": None,        # str, 任务名称
            "method": None,           # str, 计算方法
            "delta": None,            # float, 扰动步长
            "start_time": None,       # str, 开始时间
            "end_time": None,         # str, 结束时间
            "total_tasks": None,      # int, 总梯度数量
            "completed_tasks": None,  # int, 已完成梯度数量
            "status": None,           # str, "completed"/"failed"
            "hessian": None,          # List[List[float]], 完成时的Hessian
            "error": None,            # Dict, 失败时的错误信息
        }

        # regularization parameters for _genODLRHessian
        self.lam = 1e-2
        self.bet = 1.5

        # width of the middle range
        self.ddmax = 5.

        # convergence parameters for the low-rank part
        self.maxiter_LR = 100
        self.thresh_LR = 1e-8

    def setVerbosity(self, verbosity: int):
        self.verbosity = verbosity

    @staticmethod
    def _getTaskdir(task_name: str) -> Path:
        return Path(f"O1NH_{task_name}")

    @staticmethod
    def _save2json(obj:Dict, path: Union[Path, str] = ".", filename: str = ""):
        """save dict to json file, used to save config and result"""
        if not isinstance(filename, str) or not filename:
            raise ValueError("filename invalid")
        os.makedirs(path, exist_ok=True)

        temp = Path(path) / (filename + ".tmp")
        result = Path(path) / filename
        temp.write_text(
            json.dumps(obj, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )
        temp.rename(result)

    def _execute_single_task(self, x, index, core, task_dir):
        """执行单个梯度计算任务并保存结果"""
        try:
            # 执行梯度计算
            grad_result = self.grad_func(x, index, core, **self.kwargs)

            # 保存结果到临时文件
            result_file = os.path.join(task_dir, f"result_{index:06d}.pkl")
            temp_file = result_file + ".tmp"

            with open(temp_file, 'wb') as f:
                pickle.dump(grad_result, f)

            # 原子性重命名，确保文件完整性
            os.rename(temp_file, result_file)

            return grad_result

        except Exception as e:
            # 清理可能的临时文件
            temp_file = os.path.join(task_dir, f"result_{index:06d}.pkl.tmp")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise e

    def _resolve_if_exists(self, task_dir: Path, if_exists: str):
        """解析if_exists参数，将"ask"模式转换为具体的执行模式

        返回值为 "error", "overwrite", 或 "continue"
        """
        if if_exists == "error":
            return "error"

        elif if_exists == "overwrite":
            return "overwrite"

        elif if_exists == "continue":
            return "continue"

        elif if_exists == "ask":
            # 检查任务状态并打印信息
            result_file = task_dir / self.task_result_json_name
            config_file = task_dir / self.task_cfg_json_name

            if result_file.exists():
                result_data = json.loads(result_file.read_text(encoding="utf-8"))
                status = result_data.get("status")
                if status == "completed":
                    print(f"Task '{task_dir.name}' already exists and is completed.")
                elif status == "failed":
                    print(f"Task '{task_dir.name}' already exists but failed.")
            elif config_file.exists():
                print(f"Task '{task_dir.name}' already exists and is incomplete.")
            else:
                print(f"Task '{task_dir.name}' already exists (no status info).")

            # 询问用户
            while True:
                choice = input(
                    "What do you want to do? (c)ontinue / (o)verwrite / (e)rror: "
                ).strip().lower()

                if choice in ("c", "continue"):
                    return "continue"
                elif choice in ("o", "overwrite"):
                    return "overwrite"
                elif choice in ("e", "error"):
                    return "error"
                else:
                    print("Invalid choice. Please enter 'c', 'o', or 'e'.")

        else:
            raise ValueError(f"Invalid if_exists value: {if_exists}")

    def _parallel_execute(
        self,
        x_list: List[np.ndarray],
        core: int = 1,
        total_cores: Union[int, None] = None, # type: ignore
        task_name: str = "hessian",
        if_exists: str = "ask",
    ) -> List[np.ndarray]:
        """
        用法：给定若干向量x构成的列表x_list，调用梯度函数g并行对每个x进行计算，返回梯度构成的列表

        Parameters:
        -----------
        x_list : List[np.ndarray]
            输入向量列表
        core : int
            每个梯度计算使用的核心数
        total_cores : Union[int, None]
            总核心数
        task_name : str
            任务名称，用于创建任务文件夹
        if_exists : str
            当任务文件夹已存在时的处理方式：
            - "ask": 询问用户
            - "continue": 继续未完成的计算
            - "overwrite": 重新开始计算
            - "error": 抛出错误
        """
        # ==================== check the cores
        # ensure total_cores is legal
        cpu_count = os.cpu_count()
        if not isinstance(cpu_count, int):
            assert isinstance(total_cores, int), f"Unable to obtain the maximum number of cores for the current computer, and total_cores has not been correctly set"
            if total_cores <= 0:
                raise ValueError(f"total_cores must > 0, {total_cores} is given")
            warnings.warn(f"Unable to obtain the maximum number of cores for the current computer, make sure total_cores ({total_cores}) is legal.", RuntimeWarning)
        else:
            if total_cores is None:
                warnings.warn(f"total_cores ({total_cores}) is not set or unknown, os.cpu_count() ({os.cpu_count()}) is used.", RuntimeWarning)
                total_cores:int = cpu_count
            elif not isinstance(total_cores, int):
                raise TypeError(f"total_cores must be int, {type(total_cores)} is given.")
            else: # total_cores is int
                if not 1 <= total_cores <= cpu_count:
                    raise ValueError(f"total_cores ({total_cores}) must <= os.cpu_count() ({cpu_count}) and >= 1")

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

        # ==================== Step 1: 处理任务文件夹的存在性，确定最终执行模式
        n = len(x_list)
        task_dir = self._getTaskdir(task_name)

        # 如果任务文件夹已存在，解析if_exists参数（ask模式会询问用户）
        if task_dir.is_dir():
            if_exists = self._resolve_if_exists(task_dir, if_exists)

            if if_exists == "error":
                raise RuntimeError(f"任务文件夹 {task_dir} 已存在")
            elif if_exists == "overwrite":
                shutil.rmtree(task_dir)
                os.makedirs(task_dir)
            elif if_exists == "continue":
                pass  # 不删除也不重建
        else:
            os.makedirs(task_dir)

        # ==================== Step 2: 处理task.json
        task_json_path = task_dir / self.task_cfg_json_name

        if if_exists == "continue" and task_json_path.exists():
            # Continue模式：读取并验证
            old_config = json.loads(task_json_path.read_text(encoding="utf-8"))

            # 验证关键参数
            if old_config.get("total_tasks") != n:
                raise ValueError(f"Task count mismatch: expected {n}, found {old_config.get('total_tasks')}")
            if old_config.get("x_first_size") != self.task_config["x_first_size"]:
                raise ValueError(f"Input size mismatch: expected {self.task_config['x_first_size']}, found {old_config.get('x_first_size')}")

            # 保留原start_time
            self.task_config["start_time"] = old_config.get("start_time")
        else:
            # 新任务或overwrite：创建新的start_time
            self.task_config["start_time"] = datetime.now().isoformat() # type: ignore

        # 更新status并写入task.json
        self.task_config["status"] = "running" # type: ignore
        self._save2json(self.task_config, task_dir, self.task_cfg_json_name)

        # ==================== Step 3: Continue模式 - 清理临时文件并读取已完成的梯度
        result:list[np.ndarray] = [None] * n # type: ignore
        finished_gradient = 0

        if if_exists == "continue":
            # 清理.tmp文件
            for tmp_file in task_dir.glob("*.tmp"):
                os.remove(tmp_file)

            # 读取已完成的梯度
            for i in range(n):
                result_file = task_dir / f"result_{i:06d}.pkl"
                if result_file.exists():
                    with open(result_file, 'rb') as f:
                        grad = pickle.load(f)

                    # 验证梯度长度
                    expected_size = x_list[i].size
                    if grad.size != expected_size:
                        raise ValueError(f"Gradient {i} size mismatch: expected {expected_size}, found {grad.size}")

                    result[i] = grad
                    finished_gradient += 1

            if self.verbosity > 0: # TODO when to print
                print(f"发现 {finished_gradient} 个已完成的任务，继续执行剩余 {n - finished_gradient} 个任务")

        # ==================== Step 4-5: 并行执行剩余的梯度计算
        # 阶段1和阶段2的并行计算逻辑（保留原有逻辑，增加first_error机制）
        max_concurrent = total_cores // core  # 最大并行任务数量
        main_batch_index = n // max_concurrent * max_concurrent  # 阶段1的任务数
        tail_batch_index = n % max_concurrent  # 阶段2的任务数

        first_error = None  # 记录第一个错误

        # 阶段1：能占满CPU的批次
        if main_batch_index > 0:
            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                future_to_idx = {
                    executor.submit(
                        self._execute_single_task,
                        x_list[i],  # x
                        i,          # index
                        core,       # core
                        task_dir,   # task_dir
                    ): i
                    for i in range(main_batch_index)
                    if result[i] is None  # 只提交未完成的任务
                }

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result[idx] = future.result()
                        finished_gradient += 1
                    except Exception as e:
                        # 记录第一个错误但继续等待其他任务
                        if first_error is None:
                            first_error = (idx, str(e))

        # 阶段2：尾部剩余任务，重新分配核心
        if tail_batch_index > 0:
            # 每个任务固定分得 total_cores // tail_batch_index 个核心，前面 total_cores % tail_batch_index 个任务多分1个核心
            core_list = [(total_cores // tail_batch_index) + 1 if i < (total_cores % tail_batch_index) else (total_cores // tail_batch_index) for i in range(tail_batch_index)]
            with ThreadPoolExecutor(max_workers=tail_batch_index) as executor:
                future_to_idx = {
                    executor.submit(
                        self._execute_single_task,
                        x_list[i + main_batch_index],   # x
                        i + main_batch_index,           # index
                        core_list[i],                   # core
                        task_dir,                       # task_dir
                    ): i + main_batch_index
                    for i in range(tail_batch_index)
                    if result[i + main_batch_index] is None  # 只提交未完成的任务
                }

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result[idx] = future.result()
                        finished_gradient += 1
                    except Exception as e:
                        # 记录第一个错误但继续等待其他任务
                        if first_error is None:
                            first_error = (idx, str(e))

        # ==================== Step 6: 如果有错误，保存失败状态并抛出异常
        if first_error is not None:
            i, error = first_error
            self.task_result.update({
                "task_name": task_name,
                "method": self.task_config["method"],
                "delta": self.task_config["delta"],
                "start_time": self.task_config["start_time"],
                "end_time": datetime.now().isoformat(),
                "total_tasks": n,
                "completed_tasks": finished_gradient,
                "status": "failed",
                "hessian": None,
                "error": {
                    "task_index": i,
                    "error": error,
                    "error_time": datetime.now().isoformat(),
                },
            }) # type: ignore
            self._save2json(self.task_result, task_dir, self.task_result_json_name)
            raise RuntimeError(f"Task {i} failed: {error}\nSee {task_dir / self.task_result_json_name} for more information")

        # ==================== Step 7: 所有梯度计算成功，返回结果
        if self.verbosity > 4:
            print(f"任务 {task_name} 并行部分完成")

        return result

    def singleSide(
        self,
        delta: float = 1e-6,
        core: int = 0,
        total_cores: Union[int, None] = None,
        task_name: str = "singleSide",
        if_exists: str = "overwrite",
    ) -> np.ndarray:
        r"""
        $$H_{ij}\approx\frac{g_{j}(x_{1},...,x_{i}+\Delta x,...,x_{n})-g_{j}(x_{1},...,x_{i},...,x_{n})}{\Delta x}$$

        H_i = ( g(..., x_i+Δx, ...) - g(..., x_i, ...) ) / delta x
        """
        # ========== Step 1: Continue模式快速返回检查
        # continue模式下，如果任务已完成，直接返回结果
        if if_exists == "continue":
            task_dir = self._getTaskdir(task_name)
            result_file = task_dir / self.task_result_json_name
            if result_file.exists():
                result_data = json.loads(result_file.read_text(encoding="utf-8"))
                if result_data.get("status") == "completed":
                    hessian_data = result_data.get("hessian")
                    if hessian_data is None:
                        raise ValueError(f"Task result corrupted: missing hessian in {result_file}")
                    if self.verbosity > 0: # TODO when to print
                        print(f"任务 {task_name} 已完成，直接返回已有结果")
                    return np.array(hessian_data)

        # ========== Step 2: 生成扰动后的坐标列表
        n = len(self.x)
        x_and_x_with_delta = [self.x, *(self.x + delta * np.eye(n))]
        # x_and_x_with_delta is like:  (n+1 row)
        # [(x1,    x2   , ..., xi   , ..., xn   ),
        #  (x1+Δx, x2   , ..., xi   , ..., xn   ),
        #  (x1   , x2+Δx, ..., xi   , ..., xn   ),
        #  (x1   , x2   , ..., xi+Δx, ..., xn   ),
        #  (x1   , x2   , ..., xi   , ..., xn+Δx)]

        # ========== Step 3: 更新任务配置（供_parallel_execute使用）
        self.task_config.update({
            "task_name": task_name,
            "method": "single",
            "delta": delta,
            "core": core,
            "total_cores": total_cores,
            "total_tasks": len(x_and_x_with_delta),
            "x_first_size": self.x.size,
            # start_time由_parallel_execute设置
        }) # type: ignore

        # ========== Step 4: 并行计算梯度
        grad_and_grad_with_delta = self._parallel_execute(
            x_and_x_with_delta,
            core=core,
            total_cores=total_cores,
            task_name=task_name,
            if_exists=if_exists,
        )

        # ========== Step 5: 计算Hessian矩阵
        # each line of x_and_x_with_delta will become grad here
        hessian:np.ndarray = (np.vstack(grad_and_grad_with_delta[1:]) - grad_and_grad_with_delta[0]) / delta # type: ignore
        # each line of np.vstack(grad_and_grad_with_delta[1:]) denotes g(..., x_i+Δx, ...)
        # each line minus grad_and_grad_with_delta[0] denotes g(..., x_i+Δx, ...) - g(..., x_i, ...)

        # ========== Step 6: 保存完整结果
        task_dir = self._getTaskdir(task_name)
        self.task_result.update({
            "task_name": task_name,
            "method": "single",
            "delta": delta,
            "start_time": self.task_config["start_time"],
            "end_time": datetime.now().isoformat(),
            "total_tasks": len(x_and_x_with_delta),
            "completed_tasks": len(x_and_x_with_delta),
            "status": "completed",
            "hessian": hessian.tolist(),
            "error": None,
        }) # type: ignore
        self._save2json(self.task_result, task_dir, self.task_result_json_name)

        # ========== Step 7: 删除任务文件夹（后期可通过注释这行来保留）
        shutil.rmtree(task_dir)

        # ========== Step 8: 返回Hessian
        return hessian

    def doubleSide(
        self,
        delta: float = 1e-6,
        core: int = 0,
        total_cores: Union[int, None] = None,
        task_name: str = "doubleSide",
        if_exists: str = "overwrite",
    ) -> np.ndarray:
        r"""
        $$H_{ij}\approx\frac{g_j(x_1,...,x_i+\Delta x,...,x_n)-g_j(x_1,...,x_i-\Delta x,...,x_n)}{2\Delta x}$$

        H_i = ( g(..., x_i+Δx, ...) - g(..., x_i-Δx, ...) ) / 2 delta x
        """
        # ========== Step 1: Continue模式快速返回检查
        # continue模式下，如果任务已完成，直接返回结果
        if if_exists == "continue":
            task_dir = self._getTaskdir(task_name)
            result_file = task_dir / self.task_result_json_name
            if result_file.exists():
                result_data = json.loads(result_file.read_text(encoding="utf-8"))
                if result_data.get("status") == "completed":
                    hessian_data = result_data.get("hessian")
                    if hessian_data is None:
                        raise ValueError(f"Task result corrupted: missing hessian in {result_file}")
                    if self.verbosity > 0: # TODO when to print
                        print(f"任务 {task_name} 已完成，直接返回已有结果")
                    return np.array(hessian_data)

        # ========== Step 2: 生成扰动后的坐标列表（双边）
        n = len(self.x)
        all_x_with_delta = [*(self.x + delta * np.eye(n)), *(self.x - delta * np.eye(n))]

        # ========== Step 3: 更新任务配置
        self.task_config.update({
            "task_name": task_name,
            "method": "double",
            "delta": delta,
            "core": core,
            "total_cores": total_cores,
            "total_tasks": len(all_x_with_delta),
            "x_first_size": self.x.size,
            # start_time由_parallel_execute设置
        }) # type: ignore

        # ========== Step 4: 并行计算梯度
        all_grad_with_delta = self._parallel_execute(
            all_x_with_delta,
            core=core,
            total_cores=total_cores,
            task_name=task_name,
            if_exists=if_exists
        )

        # ========== Step 5: 计算Hessian矩阵
        hessian = (np.vstack(all_grad_with_delta[:n]) - np.vstack(all_grad_with_delta[n:])) / (2 * delta) # type: ignore
        # np.vstack(all_grad_with_delta[:n]) denotes g(..., x_i+Δx, ...)
        # np.vstack(all_grad_with_delta[n:]) denotes g(..., x_i-Δx, ...)
        # view the comments in singleSide() for more information

        # ========== Step 6: 保存完整结果
        task_dir = self._getTaskdir(task_name)
        self.task_result.update({
            "task_name": task_name,
            "method": "double",
            "delta": delta,
            "start_time": self.task_config["start_time"],
            "end_time": datetime.now().isoformat(),
            "total_tasks": len(all_x_with_delta),
            "completed_tasks": len(all_x_with_delta),
            "status": "completed",
            "hessian": hessian.tolist(),
            "error": None,
        }) # type: ignore
        self._save2json(self.task_result, task_dir, self.task_result_json_name)

        # ========== Step 7: 删除任务文件夹（后期可通过注释这行来保留）
        shutil.rmtree(task_dir)

        # ========== Step 8: 返回Hessian
        return hessian

    def _neighborList(self,
                      distmat: np.ndarray,
                      dmax: float,
                      eps: float = 1e-8,
                      ) -> Sequence[Sequence]:
        """
        Neighbor list based on a distance matrix.
        (1) indices whose distances are < dmax are neighbors
        (2) If the above criterion gives a disconnected graph, add just enough linkages
            to make the graph connected, by following the minimum spanning tree of the graph
        Input:  x (the vector of variables)
                distmat (the distance matrix)
                dmax (distance criterion)
                eps (degeneracy criterion)
        Output: nblist (neighbor list, a list of differently sized lists;
                nblist[i] contains the list of indices that are neighbors of index i)
        """
        from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
        N = self.x.size
        nblist = []

        for i in range(N):
            nblist.append([])
            for j in range(N):
                if distmat[i,j]<dmax:
                    nblist[i].append(j)

        ncomp, labels = connected_components(distmat<dmax)
        maxdist = np.max(distmat)
        # constituent indices of each connected component
        comp_ind = [[]]*ncomp
        for i in range(ncomp):
            comp_ind[i] = np.nonzero(labels==i)[0].tolist()

        # for each pair of connected components, connect the closest pair(s) of indices
        # between this pair of connected components. Also record the closest contact distance
        distmat_comp = np.zeros([ncomp,ncomp])
        # iibest[i,j]: the list of indices of component i that are closest to component j
        iibest = {}
        for i in range(ncomp):
            for j in range(i+1,ncomp):
                d = maxdist
                iibest[i,j] = []
                for ii in comp_ind[i]:
                    for jj in comp_ind[j]:
                        if distmat[ii,jj] < d - eps:
                            d = distmat[ii,jj]
                            iibest[i,j] = [ii]
                            iibest[j,i] = [jj]
                        elif distmat[ii,jj] < d + eps: # account for distance degeneracy
                            if not ii in iibest[i,j]: iibest[i,j].append(ii)
                            if not jj in iibest[j,i]: iibest[j,i].append(jj)
                distmat_comp[i,j] = d
                distmat_comp[j,i] = d

        # generate minimum spanning tree
        tree = minimum_spanning_tree(distmat_comp).toarray()

        # connect the closest index pairs
        for i in range(ncomp):
            for j in range(i+1,ncomp):
                if tree[i,j] >= 1e-8: # components i and j are connected in the tree
                    for ii in iibest[i,j]:
                        for jj in iibest[j,i]:
                            if not jj in nblist[ii]: nblist[ii].append(jj)
                            if not ii in nblist[jj]: nblist[jj].append(ii)

        return nblist

    def _gen_displdir(self,
                      nblist: Sequence[Sequence],
                      H0: np.ndarray,
                      displdir0: np.ndarray,
                      eps: float = 1e-8,
                      eps2: float = 1e-15,
                      ) -> np.ndarray:
        '''
        Generate displacement directions for O1NumHess.
        Input:  x (the vector of variables)
                nblist (neighbor list)
                H0 (initial guess Hessian)
                displdir (the initial displacement directions provided by the user)
                eps (degeneracy threshold of patching together the local modes)
                eps2 (threshold of norm of ev)
        Output: displdir (updated displdir, including the original displdir)
        '''
        from scipy.linalg import orth
        N = displdir0.shape[0]
        Ndispl0 = displdir0.shape[1]

        # Allocate an array for all displacement directions, assuming the maximum possible
        # number of directions. Finally we'll truncate the array if necessary
        displdir = np.zeros([N,N])
        displdir[:,0:Ndispl0] = displdir0

        early_break = True
        i = 0 # necessary in case N==Ndispl0
        for i in range(N-Ndispl0):
            # Generation of seed vectors
            ev = np.zeros(N)
            coverage = np.zeros(N)
            for j in range(N):
                # If the displacement directions already span the whole space spanned by
                # the indices in the local neighborhood of index i, skip
                nnb = len(nblist[j])
                if nnb <= i+Ndispl0:
                    continue
                submat = H0[np.ix_(nblist[j],nblist[j])]

                # Local projection
                proj = orth(displdir[np.ix_(nblist[j],range(i+Ndispl0))])
                proj = np.eye(nnb) - np.matmul(proj,proj.T)
                submat = np.matmul(proj,np.matmul(submat,proj.T))
                submat = 0.5*(submat+submat.T)

                # Start an independent direction.
                # Use Hermitian diagonalization function to avoid complex eigenvaloues.
                loceig, locev = np.linalg.eigh(submat)
                locind = np.argmax(loceig)
                locev = locev[:,locind]

                # Fix sign
                ev1 = (coverage[nblist[j]]*ev[nblist[j]] + locev)/(coverage[nblist[j]]+1)
                ev2 = (coverage[nblist[j]]*ev[nblist[j]] - locev)/(coverage[nblist[j]]+1)
                norm1 = np.linalg.norm(ev1)
                norm2 = np.linalg.norm(ev2)
                if norm1>norm2+eps:
                    ev[nblist[j]] = ev1
                elif norm1<norm2-eps:
                    ev[nblist[j]] = ev2
                else:
                    # To make the result as deterministic as possible, here we arbitrarily
                    # choose to make the maximum element of locev positive
                    locind = np.argmax(abs(locev))
                    if locev[locind] > 0:
                        ev[nblist[j]] = ev1
                    else:
                        ev[nblist[j]] = ev2

                coverage[nblist[j]] += 1

            # project out previous displacement vectors
            for j in range(i+Ndispl0):
                ev -= np.dot(ev,displdir[:,j])*displdir[:,j]
            n = np.linalg.norm(ev)
            if n<eps2:
                early_break = True
                break
            else:
                early_break = False

            ev /= n
            displdir[:,i+Ndispl0] = ev

        Ndispl = i+Ndispl0
        if not early_break:
            Ndispl += 1
        displdir = displdir[:,0:Ndispl]

        return displdir

    def _vech0(self, H: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Compress the elements of H into a vector, considering:
        (1) Only those element where mask==True are kept
        (2) H is symmetrized, and then the symmetry of H is used.
        """
        H = (H + H.T)/2.
        return H[np.tril(mask)]

    def _inv_vech0(self, v: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        The inverse function of _vech0. Always returns a symmetric matrix.
        """
        N = mask.shape[0]
        H = np.zeros([N,N])
        H[np.tril(mask)] = v
        for i in range(1,N):
            H[0:i,i] = H[i,0:i]
        #mask1 = np.tril(mask)
        #k = 0
        #for i in range(N):
        #    for j in range(i):
        #        if mask1[i,j]:
        #            H[i,j] = v[k]
        #            H[j,i] = v[k]
        #            k += 1
        return H

    def _genLocalHessian(self,
                         distmat: np.ndarray,
                         displdir: np.ndarray,
                         g: np.ndarray,
                         dmax: float,
                         ) -> np.ndarray:
        """
        Given the distance matrix, a list of displacement directions, and the gradients
        along these displacement directions (divided by the step length), recover the
        Hessian, assuming that the latter is local.
        """

        if self.verbosity > 2:
            print('Enter _genLocalHessian')
            tstart = time.time()

        # Regularization term
        W2 = self.lam * np.maximum(0.,distmat-dmax)**(2.*self.bet) # type: ignore

        # Prepare the RHS vector. This also gives the dimension of the problem
        RHS = np.matmul(g,displdir.T)
        RHS = (RHS+RHS.T)/2.
        mask: np.ndarray = distmat<(dmax+self.ddmax) # type: ignore
        RHSv = self._vech0(RHS, mask)
        Ndim = RHSv.size

        # Define the MVP function as a LinearOperator
        from scipy.sparse.linalg import LinearOperator, gmres
        f1 = lambda v: np.matmul(np.matmul(self._inv_vech0(v, mask),displdir),displdir.T)
        f = lambda v: self._vech0(f1(v) + W2*self._inv_vech0(v, mask), mask)
        A = LinearOperator((Ndim,Ndim), matvec=f)

        # Call GMRES
        # TODO: implement preconditioner
        hnumv, info = gmres(A, RHSv, x0=RHSv, atol=1e-14, maxiter=1000)
        if info!=0:
            print('Warning: gmres returned with error code %d'%info)

        # Recover the desired Hessian (local part)
        hnum = self._inv_vech0(hnumv, mask)
        if self.verbosity > 2:
            tend = time.time()
            err = np.linalg.norm(g - np.matmul(hnum,displdir))
            print('Successful termination of _genLocalHessian, time = %.2f sec'%(tend-tstart)) # type: ignore
            if self.verbosity > 5:
                print('Local part of the Hessian:')
                print(hnum)
            print('Error norm of the predicted gradient: %.2e'%err)

        return hnum

    def _genODLRHessian(self,
                        distmat: np.ndarray,
                        displdir: np.ndarray,
                        g: np.ndarray,
                        dmax: float,
                        ) -> np.ndarray:
        """
        Given the distance matrix, a list of displacement directions, and the gradients
        along these displacement directions (divided by the step length), recover the
        Hessian, assuming that the latter has the Off-Diagonal Low Rank (ODLR) property.
        """

        if self.verbosity > 2:
            print('Enter _genODLRHessian')
            tstart = time.time()

        # First do the local part
        hnum = self._genLocalHessian(distmat, displdir, g, dmax)

        # The low rank part
        dampfac = 1.0
        err0 = np.inf
        for it in range(self.maxiter_LR):
            resid = g - np.matmul(hnum,displdir)
            err = np.linalg.norm(resid)
            if err < self.thresh_LR:
                break
            elif abs(err-err0) < self.thresh_LR:
                print('The gradients cannot be exactly reproduced by a symmetric Hessian.')
                print('Exit _genODLRHessian')
                break
            elif err > 1.0: # iterations diverge
                dampfac *= 0.5
                if self.verbosity > 1:
                    print('Warning: error too large, damping the correction by a factor of %.2e'%dampfac)
            if self.verbosity > 4:
                print('Iter %3d error %.2e'%(it,err))
            hcorr = np.matmul(resid, displdir.T)
            hcorr = (hcorr+hcorr.T)/2.
            hnum += dampfac*hcorr
            err0 = err

        if self.verbosity > 2:
            tend = time.time()
            print('Successful termination of _genODLRHessian, time = %.2f sec'%(tend-tstart)) # type: ignore
            print('Error norm of the predicted gradient: %.2e'%err) # type: ignore

        return hnum

    def O1NumHess(
        self,
        core: int = 0,
        delta: float = 1e-6,
        total_cores: Union[int, None] = None,
        dmax: float = 1.0,
        distmat: np.ndarray = np.zeros([0,0]),
        H0: np.ndarray = np.zeros([0,0]),
        displdir: np.ndarray = np.zeros([0,0]),
        g: np.ndarray = np.zeros([0,0]),
        g0: np.ndarray = np.zeros(0),
        doublesided: np.ndarray = np.zeros(0, dtype=bool),
        gen_new_displdir: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        The O1NumHess algorithm.
        Input:  core        (number of parallel threads per gradient calculation)
                delta       (finite difference step length)
                total_cores (total number of parallel threads)
                distmat     ("distance" between the entries. Default is: the distance between
                             entry i and entry j is |i-j|)
                dmax        (a parameter controlling the accuracy. Increasing dmax increases
                             the number of gradients, but also increases the accuracy. Sensible
                             range is 0.0~2.0)
                H0          (initial guess Hessian, used for generating the displacement
                             directions. Although a default is given, it is probably a bad choice.)
                displdir    (displacement directions, specified by the user, that need to be
                             included in the list of displacement directions, for example because
                             the gradients along these directions are known beforehand and do
                             not need to be calculated)
                g           (g[:,i] is the known gradient along the displacement direction of
                             displdir[:,i]. The second dimension of g can be smaller than that
                             of displdir. Note that g requires the knowledge of delta)
                g0          (the gradient at unperturbed x. Optional)
                doublesided (if doublesided[i]==True, the i-th displacement need to be done
                             using double sided differentiation, else with single sided
                             differentiation. This included displacement directions where the
                             gradient "g" has been given)
                gen_new_displdir (whether to generate new displacement directions based on distmat.
                                  If False, only use the user-provided displdir)
        Output: hessian     (the calculated Hessian)
                displdir    (the updated displacement directions)
                gout        (the gradient derivatives along the displacement directions.
                             It includes the user-specified gradients, i.e. g)
        """
        if self.verbosity > 0:
            print("Start O1NumHess run...")

        N = self.x.size

        # Default values
        if distmat.size==0:
            distmat = abs(np.mgrid[N,N][0]-np.mgrid[N,N][1])
        if H0.size==0:
            H0 = np.eye(N)
        if displdir.size==0:
            displdir = np.zeros([N,0])
        if g.size==0:
            g = np.zeros([N,0])
        if g0.size==0:
            # Calculate the gradient at the unperturbed x.
            # We parallelize this calculation using all available cores.
            # The calculation has index 2*N, to avoid clashing with other displacement directions
            g0 = self.grad_func(self.x,2*N,total_cores,**self.kwargs)

        # Normalize the input displdir. if there is any
        for i in range(displdir.shape[1]):
            n = np.linalg.norm(displdir[:,i])
            if n==0:
                raise ZeroDivisionError('Displacement direction %d is a zero vector'%i)
            displdir[:,i] /= n

        Ndispl0 = displdir.shape[1]
        if self.verbosity > 1:
            print("%d displacement directions given on input"%Ndispl0)
        if gen_new_displdir:
            # Get the neighbors of each entry
            nblist = self._neighborList(distmat, dmax)

            # Generate displacement directions
            displdir = self._gen_displdir(nblist, H0, displdir)
            if self.verbosity > 1:
                print("%d displacement directions generated"%(displdir.shape[1]-Ndispl0))
                print("Total number of displacement directions: %d"%displdir.shape[1])

        Ndispl = displdir.shape[1]
        Ng = g.shape[1]
        for i in range(Ndispl):
            # fix sign (to make the results reproducible)
            for j in range(N):
                # Because all displacement directions are normalized, it is guaranteed
                # that at least one j satisfies the following
                if abs(displdir[j,i])>0.5/math.sqrt(N):
                    if displdir[j,i]<0:
                        displdir[:,i] *= -1.0
                        if i<Ng:
                            g[:,i] *= -1.0
                    break

        if gen_new_displdir:
            if self.verbosity > 5:
                print("Displacement directions:")
                print(displdir)

        # Prepare the displacement vectors
        # First, count the number of displacement vectors, considering:
        # (1) Along some directions we need to do double-sided differentiation
        # (2) For some displacement directions, the corresponding gradients have already
        #     been given in g
        Ngrad = Ndispl-Ng+np.sum(doublesided[Ng:Ndispl])
        if self.verbosity > 1:
            print("%d gradients will be calculated"%Ngrad)

        x_with_delta = np.zeros([Ngrad,N])
        for i in range(Ndispl-Ng):
            x_with_delta[i,:] = self.x + delta*displdir[:,i+Ng]
        j = Ndispl-Ng
        for i in range(Ndispl-Ng):
            if doublesided[i+Ng]:
                x_with_delta[j,:] = self.x - delta*displdir[:,i+Ng]
                j += 1

        # Calculate the gradients
        if self.verbosity > 1:
            print("Start gradient calculations...")
        grad_with_delta = self._parallel_execute(x_with_delta, total_cores=total_cores, core=core, task_name="O1NumHess")
        if self.verbosity > 1:
            print("Gradient calculations finished")
            if self.verbosity > 5:
                print("Raw gradients:")
                print(grad_with_delta)

        # Divide the step length, taking special care of double-sided displacement directions
        gout = np.zeros([N,Ndispl])
        gout[:,0:Ng] = g
        j = Ndispl-Ng
        for i in range(Ng,Ndispl):
            if doublesided[i]:
                gout[:,i] = (grad_with_delta[i-Ng]-grad_with_delta[j])/(2.*delta)
                j += 1
            else:
                gout[:,i] = (grad_with_delta[i-Ng]-g0)/delta
        if self.verbosity > 5:
            print("Gradient derivatives:")
            print(gout)

        # Extract the Hessian - the essence of the O1NumHess algorithm!
        hessian = self._genODLRHessian(distmat, displdir, gout, dmax)

        return hessian, displdir, gout


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
