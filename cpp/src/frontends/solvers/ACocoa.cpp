/*
 * ClusterSolver.cpp
 *
 *  Created on: Sep 11, 2013
 *      Author: taki
 */

#include "../../solver/distributed/distributed_include.h"
#include "../../class/Context.h"
#include "../../helpers/option_console_parser.h"
#include "../../solver/settingsAndStatistics.h"
#include "../../utils/file_reader.h"
#include "../../solver/Solver.h"
#include "../../helpers/utils.h"
#include "../../solver/ClusterEngineExecutor.h"
#include "../../solver/ClusterApproxEngineExecutor.h"
#include <math.h>
#include "../../utils/distributed_instances_loader.h"
#include "../../solver/distributed/distributed_structures.h"
#include "../../helpers/option_distributed_console_parser.h"

template<typename L, typename D>
int runConsolveSolver(int argc, char *argv[]) {
	MPI_Init(&argc, &argv);
	mpi::environment env(argc, argv);
	mpi::communicator world;
	DistributedSettings distributedSettings;
	Context ctx(distributedSettings);
	consoleHelper::parseDistributedOptions(ctx, distributedSettings, argc,
			argv);

	ctx.settings.verbose = true;
	if (world.rank() != 0)
		ctx.settings.verbose = false;

	ProblemData<L, D> localInstance;
	ProblemData<L, D> globalInstance;
	int localXi = 1;
	int xi;

	{
		ProblemData<L, D> testDataset;
//		std::vector<D> A_test_csr_values;
//		std::vector<L> A_test_csr_col_idx;
//		std::vector<L> A_test_csr_row_ptr;
		if (world.rank() == 0) {
			loadDistributedSparseSVMRowData("data/rcv1_train.binary", 0, 0,
					testDataset, false);
		}

		cout << world.rank() << " going to load data" << endl;

		loadDistributedSparseSVMRowData(ctx.matrixAFile, world.rank(),
				world.size(), globalInstance, false);
		L m = globalInstance.m;
		if (world.rank() == 0) {
			globalInstance.A_test_csr_values = testDataset.A_csr_values;
			globalInstance.A_test_csr_col_idx = testDataset.A_csr_col_idx;
			globalInstance.A_test_csr_row_ptr = testDataset.A_csr_row_ptr;
			globalInstance.test_b = testDataset.b;
			if (m < testDataset.m) {
				m = testDataset.m;
			}
		}
		vall_reduce_maximum(world, &m, &globalInstance.m, 1);

		cout <<  globalInstance.m <<"XXXXXXXXXXXXXXXX"<<endl;


		std::vector<int> localNNZPerRow(globalInstance.m, 0);
		std::vector<int> reducedNNZPerRow(globalInstance.m, 0);

		for (unsigned int i = 0; i < globalInstance.A_csr_col_idx.size(); i++) {
			localNNZPerRow[globalInstance.A_csr_col_idx[i]]++;
		}
		vall_reduce(world, &localNNZPerRow[0], &reducedNNZPerRow[0],
				reducedNNZPerRow.size());
		for (int i = 0; i < globalInstance.m; i++) {
			if (localXi < reducedNNZPerRow[i]) {
				localXi = reducedNNZPerRow[i];
			}
		}
		xi = localXi;

	}

	vall_reduce(world, &globalInstance.n, &globalInstance.total_n, 1);
cout <<"total n is "<< 	globalInstance.total_n<<endl;
D lam =  1.0 / (0.0 + globalInstance.total_n);
	globalInstance.lambda =lam;

			cout <<"lam is  "<< 	globalInstance.lambda<<endl;
	localInstance.lambda = globalInstance.lambda;

	localInstance.n = globalInstance.n;
	localInstance.m = 0;

	L s = 1;
	vall_reduce_minimum(world, &globalInstance.n, &s, 1);

	TOTAL_THREADS = distributedSettings.totalThreads;
	omp_set_num_threads(TOTAL_THREADS);

	int tau = distributedSettings.iterationsPerThread
			* distributedSettings.totalThreads;

	localInstance.total_tau = tau * world.size();
	cout << "TAU is " << localInstance.total_tau << endl;

	float Xi = xi;
	localInstance.sigma = 2 * (1 + (Xi - 1) * (tau - 1) / (s - 1.0));

	localInstance.sigma = 2;

	globalInstance.sigma = localInstance.sigma;
	std::cout << "Sigma used: " << localInstance.sigma << " TOTAL THREADS USED "
			<< distributedSettings.totalThreads << std::endl;

	std::ofstream logFile;

	std::stringstream ss;
	ss << "./results/" << ctx.experimentName << "_" << world.rank() << "_"
			<< ctx.settings.lossFunction << "_" << tau << "_"
			<< distributedSettings.distributed << "_"
			<< distributedSettings.iterationsPerThread << "_"
			<< distributedSettings.iters_communicate_count << "_"
			<< localInstance.sigma << "_" << ctx.lambda << "_"
			<< distributedSettings.APPROX;

	ss << ".dat";
	logFile.open(ss.str().c_str());
	std::cout << "Logging into " << ss.str() << endl;
	distributedSettings.logFile = &logFile;
	distributedSettings.logToFile = true;

	distributedSettings.totalThreads = 1;
	distributedSettings.iterationsPerThread = 10;
	distributedSettings.iters_communicate_count = 5;
	distributedSettings.iters_bulkIterations_count = 2000;
	distributedSettings.recomputeResidualAfterEachBulkIteration = false;

	distributedSettings.distributed = AsynchronousStreamlinedOptimized;
//	distributedSettings.distributed = SynchronousReduce;

	ClusterEngineExecutor<L, D> executor(world, localInstance, globalInstance,
			&(distributedSettings));

	executor.initializeAll();

	Solver<L, D> solver(executor);

	solver.runSolver();

	logFile.close();

	MPI::Finalize();
}

int main(int argc, char *argv[]) {

	runConsolveSolver<int, double>(argc, argv);

}
