# from __future__ import annotations
import numpy as np
import scipy
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
import warnings
import time

from typing import Any, Callable, List, Union, Dict, Sequence, Tuple


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

        self.verbosity = 0

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

    def _parallel_execute(self, x_list: List[np.ndarray], core: int = 1, total_cores: Union[int, None] = None) -> List[np.ndarray]: # type: ignore
        """
        用法：给定若干向量x构成的列表x_list，调用梯度函数g并行对每个x进行计算，返回梯度构成的列表
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

    def _neighborList(self,
                      distmat: np.ndarray,
                      dmax: float,
                      eps: float = 1e-8,
                      ) -> Sequence[Sequence]:
        """
        Neighbor list based on a distance matrix.
        (1) indices whose distances are < dmax are neighbors
        (2) If the above criterion gives a disconnected graph, add just enough linkages
            to make the graph connected
        Input:  x (the vector of variables)
                distmat (the distance matrix)
                dmax (distance criterion)
                eps (degeneracy criterion)
        Output: nblist (neighbor list, a list of differently sized lists;
                nblist[i] contains the list of indices that are neighbors of index i)
        """
        from scipy.sparse.csgraph import connected_components
        N = self.x.size
        nblist = [[]]*N

        for i in range(N):
            nblist[i] = list(distmat[i,:]<dmax)

        ncomp, labels = connected_components(distmat<dmax)
        maxdist = np.max(distmat)
        # constituent atoms of each connected component
        comp_atoms = [[]]*ncomp
        for i in range(ncomp):
            comp_atoms[i] = np.nonzero(labels==i)
        # for each pair of connected components, connect the closest pair(s) of atoms
        # between this pair of connected components
        d = maxdist
        for i in range(ncomp):
            for j in range(i+1,ncomp):
                iibest = []
                jjbest = []
                for ii in comp_atoms[i]:
                    for jj in comp_atoms[j]:
                        if distmat[ii,jj] < d - eps:
                            d = distmat[ii,jj]
                            iibest = [ii]
                            jjbest = [jj]
                        elif distmat[ii,jj] < d + eps: # account for distance degeneracy
                            iibest.append(ii)
                            jjbest.append(jj)
                nblist[iibest] += jjbest
                nblist[jjbest] += iibest
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
                nnb = np.sum(nblist[j])
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

    def _vech0(H: np.ndarray, mask: np.ndarray) -> np.array:
        """
        Compress the elements of H into a vector, considering:
        (1) Only those element where mask==True are kept
        (2) H is symmetrized, and then the symmetry of H is used.
        """
        H = (H + H.T)/2.
        return H[np.tril(mask)]

    def _inv_vech0(v: np.array, mask: np.ndarray) -> np.ndarray:
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
        W2 = self.lam * np.maximum(0.,distmat-dmax)**(2.*self.bet)

        # Prepare the RHS vector. This also gives the dimension of the problem
        RHS = np.matmul(g,displdir.T)
        RHS = (RHS+RHS.T)/2.
        mask = distmat<(dmax+self.ddmax)
        RHSv = O1NumHess._vech0(RHS, mask)
        Ndim = RHSv.size

        # Define the MVP function as a LinearOperator
        from scipy.sparse.linalg import LinearOperator, gmres
        f1 = lambda v: np.matmul(np.matmul(O1NumHess._inv_vech0(v, mask),displdir),displdir.T)
        f = lambda v: O1NumHess._vech0(f1(v) + W2*O1NumHess._inv_vech0(v, mask), mask)
        A = LinearOperator((Ndim,Ndim), matvec=f)

        # Call GMRES
        # TODO: implement preconditioner
        hnumv, info = gmres(A, RHSv, x0=RHSv, atol=1e-14, maxiter=1000)
        if info!=0:
            print('Warning: gmres returned with error code %d'%info)

        # Recover the desired Hessian (local part)
        hnum = O1NumHess._inv_vech0(hnumv, mask)
        if self.verbosity > 2:
            tend = time.time()
            err = np.linalg.norm(g - np.matmul(hnum,displdir))
            print('Successful termination of _genLocalHessian, time = %.2f sec'%(tend-tstart))
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
            print('Successful termination of _genODLRHessian, time = %.2f sec'%(tend-tstart))
            print('Error norm of the predicted gradient: %.2e'%err)

        return hnum

    def O1NumHess(self,
                  core: int = 0,
                  delta: float = 1e-6,
                  total_cores: Union[int, None] = None,
                  dmax: float = 1.0,
                  distmat: np.ndarray = np.zeros([0,0]),
                  H0: np.ndarray = np.zeros([0,0]),
                  displdir: np.ndarray = np.zeros([0,0]),
                  g: np.array = np.zeros([0,0]),
                  g0: np.ndarray = np.zeros(0),
                  doublesided: np.array = np.zeros(0, dtype=bool),
                  gen_new_displdir: bool = True
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
        grad_with_delta = self._parallel_execute(x_with_delta, total_cores=total_cores, core=core)
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
