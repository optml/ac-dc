Accelerated Coordinate Descent methods for minimizing Composite functions

    a library of serial, parallel and distributed coordinate and block coordinate descent methods
    written in C++, CUDA and MATLAB
    C++ implementation consists of serial and parallel solvers
    Contains Examples for
        Truss Topology Design (TTD)
        Support Vector Machines (SVM)
        Sparse Group Lasso (SGL)
        Matrix Completion (MC) 

Read more in our Wiki.
Speedup of Parallel Implementation

The parallel implementation is written in CUDA as is intended to run on GPUs. We get speed-up of up to 120x for 3D Truss Topology Design problem as compared to the fast (C++) serial implementation.
Speedup of Serial Implementation

We implement several acceleration strategies, including

    q-shrinking 

Theoretical background

AC-DC is a collection of implementation of coordinate descent algorithms developed and analyzed in the following papers:

    Martin Takáč, Avleen Bijral, Peter Richtárik and Nathan Srebro: Mini-Batch Primal and Dual Methods for SVMs, ICML 2013 (30th International Conference on Machine Learning) 

    Martin Takáč, Jakub Mareček and Peter Richtárik: Distributed Coordinate Descent for Big Data Optimization 

    Peter Richtárik and Martin Takáč: Parallel coordinate descent methods for big data optimization, submitted for publication
    Peter Richtárik and Martin Takáč: Iteration complexity of randomized block-coordinate descent methods for minimizing a composite function, to appear in Mathematical Programming A
    Peter Richtárik and Martin Takáč: Efficient serial and parallel coordinate descent methods for huge-scale truss topology design, Operations Research Proceedings 2012
    Peter Richtárik and Martin Takáč: Efficiency of randomized coordinate descent methods on minimization problems with a composite objective function, Proceedings of the 4th Workshop on Signal Processing with Adaptive Sparse Structured Representations, June 27-30, 2011 
