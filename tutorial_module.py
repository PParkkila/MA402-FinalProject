from petsc4py import PETSc
import numpy as np

M, N, K = 3, 5, 4

def main():
    # 1. Initialize Matrices and Vectors
    A = PETSc.Mat().createDense([M, N])
    A.setUp()
    
    A_data = [
        [0.81, 0.91, 0.28, 0.96, 0.96],
        [0.91, 0.63, 0.55, 0.16, 0.49],
        [0.13, 0.10, 0.96, 0.97, 0.80],
    ]

    for i in range(M):
        for j in range(N):
            A.setValue(i, j, A_data[i][j])
    A.assemblyBegin(); A.assemblyEnd()

    b = PETSc.Vec().createSeq(M)
    b.setArray(np.array([0.28, 0.55, 0.96], dtype=PETSc.ScalarType))

    D = PETSc.Mat().createDense([K, N])
    D.setUp()
    for k in range(K):
        D.setValue(k, k, -1.0)
        if k + 1 < N:
            D.setValue(k, k + 1, 1.0)
    D.assemblyBegin(); D.assemblyEnd()

    x = PETSc.Vec().createSeq(N)
    x.set(0.0)
    f = PETSc.Vec().createSeq(M)

    # 2. Define the Residual and Jacobian Functions
    def compute_residual(tao, x, f):
        A.mult(x, f)
        f.axpy(-1.0, b)

    def compute_jacobian(tao, x, Amat, Pmat):
        # The Jacobian of (Ax - b) is simply A.
        # Since A is constant and already contains the values, we do nothing.
        pass

    # 3. Setup TAO Solver
    tao = PETSc.TAO().create(PETSc.COMM_WORLD)
    tao.setType("brgn")
    tao.setSolution(x)
    
    # Use setResidual for the function Ax - b
    tao.setResidual(compute_residual, f)
    
    # CRITICAL FIX: Use setJacobianResidual for least-squares solvers like BRGN
    # This maps to TaoSetResidualJacobianRoutine in the PETSc C API.
    try:
        tao.setJacobianResidual(compute_jacobian, A, A)
    except AttributeError:
        # Fallback for specific petsc4py versions
        tao.setJacobian(compute_jacobian, A, A)

    # 4. BRGN-specific Dictionary Matrix
    tao.setBRGNDictionaryMatrix(D)

    # 5. Set Options
    opts = PETSc.Options()
    opts["tao_brgn_regularization_type"] = "l1dict"
    opts["tao_brgn_regularizer_weight"] = 1e-4
    opts["tao_brgn_l1_smooth_epsilon"] = 1e-6
    opts["tao_gatol"] = 1e-8
    opts["tao_monitor"] = None  # View progress in terminal
    
    tao.setFromOptions()
    
    # 6. Solve
    print("Starting solver...")
    tao.solve()

    # 7. Output Results
    reason = tao.getConvergedReason()
    PETSc.Sys.Print(f"\n===== SOLVER TERMINATED: {reason} =====")
    if reason > 0:
        PETSc.Sys.Print("===== RESULT VECTOR =====")
        x.view()
    else:
        PETSc.Sys.Print("Solver failed to converge.")

if __name__ == "__main__":
    main()