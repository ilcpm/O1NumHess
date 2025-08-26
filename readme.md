# O1NumHess

Calculate the seminumerical Hessian of an arbitrary N-variable function using only O(1) gradients, by assuming the Hessian has the off-diagonal low rank (ODLR) property.

Details of the O1NumHess algorithm, as well as preliminary benchmark results, can be found in our preprint paper: https://arxiv.org/abs/2508.07544

usage:

```bash
python3 setup.py install
```

Note that the above command installs the package for all users, and requires root privileges. If the user does not have access to root privileges, or if it is not desired to install the package for all users, then one should use the following command instead:

```bash
python3 setup.py install --prefix ~/.local
```

