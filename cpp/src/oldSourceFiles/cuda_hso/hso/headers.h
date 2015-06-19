//====================== Standard C Libs
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <sstream>
//#include "device_functions.h"
//======================= CUDA Libs
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "cublas.h"
#include "cusparse.h"

#include "cutil.h"
//======================== THRUST Libs
// http://code.google.com/p/thrust/
# include <thrust/transform_reduce.h>
# include <thrust/functional.h>
# include <cmath>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
//======================== Solvers
#include "solver/kernels.h"
#include "solver/structures.h"
#include "solver/helpers.h"
#include "solver/serial/randomL2L1Solver.h"
#include "solver/serial/greedyL2L1Solver.h"
#include "solver/parallel/randomL2L1Solver.h"
#include "solver/parallel/greedyL2L1Solver.h"

//======================== TTD Functions
#include "ttd/ttd.h"


