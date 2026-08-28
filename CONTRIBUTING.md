# Contributing

DrainSinkhorn treats numerical release and performance as separate gates.

For every change:

1. add or update a test for the execution contract;
2. verify both row and column residuals with the registered upstream verifier;
3. report component comparisons with matched update, audit, verifier, tolerance,
   initialization, and input;
4. keep component speedups separate from complete-deployment speedups;
5. record CUDA device, PyTorch/CUDA/Triton/FlashSinkhorn versions, source commit,
   command, and input hash for GPU results.

CPU tests check controller semantics. CUDA tests establish numerical and backend
parity. Timing claims require a frozen GPU protocol and must not be inferred
from CPU tests or synthetic smoke runs.
