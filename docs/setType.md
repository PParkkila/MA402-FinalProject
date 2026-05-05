# `setType`

## Description
Sets the type of Tao solver to the defined Tao method type. Once the solver type is set, solver-specific options may be set by `tao.setFromOptions()`.



## Python Signature
```python
tao.setType(tao_type: Tao.Type | str) -> None
```

## Parameters
tao_type:  Tao.Type | `str`
The Tao method type. Options are as follows:

- `"nls"`: Newton’s method with line search for unconstrained minimization.
- `"ntr"`: Newton’s method with trust region for unconstrained minimization
- `"ntl"`: Newton’s method with trust region, line search for unconstrained minimization
- `"lmvm"`: Limited memory variable metric method for unconstrained minimization
- `"cg"`: Nonlinear conjugate gradient method for unconstrained minimization
- `"nm"`: Nelder-Mead algorithm for derivate-free unconstrained minimization
- `"tron"`: Newton Trust Region method for bound constrained minimization
- `"gpcg"`: Newton Trust Region method for quadratic bound constrained minimization
- `"blmvm"`: Limited memory variable metric method for bound constrained minimization
- `"lcl"`: Linearly constrained Lagrangian method for pde-constrained minimization
- `"pounders"`: Model-based algorithm for nonlinear least squares
- `"brgn"`: Gauss-Newton algorithm with regularization. Calls C function for `TaoCreate_BRGN`.


## Returns
`None`

This function modifies the type of the solver in place.


## Notes
Calling this function will reset the convergense test to the default convergence test. If a custom convergence test has beens et, it must be set again after calling this function.

## Usage
```python
from petsc4py import PETSc

## Create the TAO solver
tao = PETSc.TAO().create(PETSc.COMM_WORLD)
tao.setType("brgn")

## Set the type to use the Gauss-Newton algorithm with regularization
tao.setType("brgn")

opts = PETSc.Options()
opts["tao_brgn_regularization_type"] = "l1dict"
tao.setFromOptions()
```


## Source Code
- **petsc4py Wrapper:** [`src/binding/petsc4py/src/petsc4py/PETSc/TAO.pyx`](https://gitlab.com/petsc/petsc/-/blob/release/src/binding/petsc4py/src/petsc4py/PETSc/TAO.pyx?ref_type=heads#L173)
- **C Header:** [`include/petsctao.h`](https://petsc.org/main/include/petsctao.h.html)
- **C Implementation:** [`src/tao/interface/taosolver.c`](https://petsc.org/release/src/tao/interface/taosolver.c.html#TaoSetType) (line 2269)
- **C Manual Page:** [`TaoSetType`](https://petsc.org/release/manualpages/Tao/TaoSetType/)
