from petsc4py import PETSc
import numpy as np
import pandas as pd

## NOTE: This is an adaptation of the code from https://petsc.org/main/src/tao/leastsquares/tutorials/cs1.c.html.

class TrendFilter:
    """
    Trend filtering using PETSc TAO with L1 dictionary regularization.

    This class formulates and solves a regularized least-squares problem
    of the form:

        minimize_x ||Ax - b||_2^2 + λ ||Dx||_1

    where:
        - A is the identity matrix (data fidelity term),
        - b is the observed data,
        - D is a discrete difference operator (first or second order),
        - λ is the regularization weight.

    The optimization is solved using PETSc's TAO solver with the
    bounded-regularized Gauss-Newton (BRGN) method.

    Parameters
    ----------
    data : array_like
        Input 1D signal to be filtered.
    order : int, optional
        Order of the difference operator:
            - 1: piecewise constant trend (total variation)
            - 2: piecewise linear trend
        Default is 1.
    reg_weight : float, optional
        Regularization weight (λ). Controls smoothness of the solution.
        Higher values produce smoother trends. Default is 1.0.

    Attributes
    ----------
    n : int
        Length of the input data.
    reg_weight : float
        Regularization parameter.
    A : PETSc.Mat
        Identity matrix representing the observation operator.
    D : PETSc.Mat
        Discrete difference matrix.
    b : PETSc.Vec
        PETSc vector containing input data.
    x : PETSc.Vec
        Solution vector (trend estimate).
    f : PETSc.Vec
        Residual vector.
    tao : PETSc.TAO
        TAO optimization solver instance.

    Notes
    -----
    - Uses L1 regularization on Dx to promote sparsity in derivatives.
    - Internally relies on PETSc's TAO solver with type "brgn".
    - Initial guess is set to the mean of the input data.
    """

    def __init__(self, data, order=1, reg_weight=1.0):
        self.n = len(data)
        self.reg_weight = reg_weight
        
        # 1. Setup Matrices
        # A is Identity (Direct observation)
        self.A = PETSc.Mat().createAIJ([self.n, self.n])
        self.A.setUp()
        for i in range(self.n):
            self.A.setValue(i, i, 1.0)
        self.A.assemblyBegin(); self.A.assemblyEnd()

        # D is the Discrete Difference Matrix
        self.D = self._create_diff_matrix(order)
        self.b = self._to_petsc_vec(data)
        
        # 2. Setup Vectors
        self.x = PETSc.Vec().createSeq(self.n)
        self.x.set(np.mean(data)) # Better initial guess
        self.f = PETSc.Vec().createSeq(self.n)
        
        # 3. Solver Config
        self.tao = PETSc.TAO().create(PETSc.COMM_WORLD)
        self.tao.setType("brgn")
        self.tao.setSolution(self.x)
        self.tao.setResidual(self._compute_residual, self.f)
        self.tao.setJacobianResidual(self._compute_jacobian, self.A, self.A)
        self.tao.setBRGNDictionaryMatrix(self.D)
        
        # 4. Set Hyperparameters
        opts = PETSc.Options()
        opts["tao_brgn_regularization_type"] = "l1dict"
        opts["tao_brgn_regularizer_weight"] = self.reg_weight
        for k, v in PETSc.Options().getAll().items():
            print(k, v)
        self.tao.setFromOptions()

    def _create_diff_matrix(self, order):
        """
        Construct a discrete difference matrix.

        Parameters
        ----------
        order : int
            Order of the difference operator:
                - 1: first-order differences
                - 2: second-order differences

        Returns
        -------
        PETSc.Mat
            Sparse matrix representing the difference operator.

        Notes
        -----
        - First-order differences compute x[i+1] - x[i].
        - Second-order differences compute x[i] - 2*x[i+1] + x[i+2].
        """
        if order == 1:
            rows = self.n - 1
            D = PETSc.Mat().createAIJ([rows, self.n])
            D.setUp()
            for i in range(rows):
                D.setValues(i, [i, i+1], [-1, 1])
        else: # Order 2: Piecewise Linear
            rows = self.n - 2
            D = PETSc.Mat().createAIJ([rows, self.n])
            D.setUp()
            for i in range(rows):
                D.setValues(i, [i, i+1, i+2], [1, -2, 1])
        
        D.assemblyBegin(); D.assemblyEnd()
        return D

    def _to_petsc_vec(self, arr):
        """
        Convert a NumPy array to a PETSc vector.

        Parameters
        ----------
        arr : array_like
            Input array.

        Returns
        -------
        PETSc.Vec
            PETSc vector containing the input data.
        """
        v = PETSc.Vec().createSeq(len(arr))
        v.setArray(arr.astype(PETSc.ScalarType))
        return v

    def _compute_residual(self, tao, x, f):
        """
        Compute the residual vector.

        Parameters
        ----------
        tao : PETSc.TAO
            TAO solver instance.
        x : PETSc.Vec
            Current solution estimate.
        f : PETSc.Vec
            Output residual vector (modified in place).

        Notes
        -----
        Residual is defined as:

            f = A x - b
        """
        self.A.mult(x, f)
        f.axpy(-1.0, self.b)

    def _compute_jacobian(self, tao, x, Amat, Pmat):
        """
        Compute the Jacobian of the residual.

        Parameters
        ----------
        tao : PETSc.TAO
            TAO solver instance.
        x : PETSc.Vec
            Current solution estimate.
        Amat : PETSc.Mat
            Jacobian matrix (to be filled).
        Pmat : PETSc.Mat
            Preconditioner matrix.

        Notes
        -----
        Currently not implemented. For identity A, the Jacobian is constant.
        """
        pass

    def solve(self):
        """
        Solve the trend filtering optimization problem.

        Returns
        -------
        numpy.ndarray
            Filtered trend estimate.

        Notes
        -----
        - Runs the TAO solver until convergence.
        - Returns a copy of the solution vector as a NumPy array.
        """
        self.tao.solve()
        return self.x.getArray().copy()
