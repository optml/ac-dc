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

	if (ctx.settings.lossFunction != 2) {

		ifstream dataInfo;
		dataInfo.open(ctx.matrixAFile.c_str(), ios::in | ios::binary);
		L local_n;
		L local_m;
		L global_m;
		dataInfo >> local_n;
		dataInfo >> local_m;
		dataInfo >> global_m;
		cout << "rank " << world.rank() << " " << local_n << " " << local_m
				<< endl;
		dataInfo.close();

		std::stringstream matrixPrefix;
		std::stringstream vectorBFileName;

		if (ctx.dataANSIInput) {
			matrixPrefix << ctx.matrixAFile << "_" << world.rank()
					<< "_local_matrixA";
			vectorBFileName << ctx.vectorBFile << "_" << world.rank()
					<< "_local_vectorB.txt";

			InputOuputHelper::loadCSCData(ctx, matrixPrefix.str(),
					vectorBFileName.str(), localInstance);

			matrixPrefix.str("");
			matrixPrefix << ctx.matrixAFile << "_" << world.rank()
					<< "_global_matrixA";

			vectorBFileName.str("");
			vectorBFileName << ctx.vectorBFile << "_" << world.rank()
					<< "_global_vectorB.txt";
			InputOuputHelper::loadCSCData(ctx, matrixPrefix.str(),
					vectorBFileName.str(), globalInstance);

		} else {
			matrixPrefix << ctx.matrixAFile;
			vectorBFileName << ctx.vectorBFile << "_" << world.rank()
					<< "_local.data";
			InputOuputHelper::loadANSCIAndBinaryData(ctx, ctx.matrixAFile,
					"_local.data", vectorBFileName.str(), localInstance,
					local_n, global_m, world.rank());
			vectorBFileName.str("");
			vectorBFileName << ctx.vectorBFile << "_" << world.rank()
					<< "_global.data";
			InputOuputHelper::loadANSCIAndBinaryData(ctx, ctx.matrixAFile,
					"_global.data", vectorBFileName.str(), globalInstance,
					local_n, global_m, world.rank());

		}

		std::vector<int> localNNZPerRow(localInstance.m, 0);
		for (unsigned int i = 0; i < localInstance.A_csc_row_idx.size(); i++) {
			localNNZPerRow[localInstance.A_csc_row_idx[i]]++;
			if (localNNZPerRow[localInstance.A_csc_row_idx[i]] > localXi)
				localXi = localNNZPerRow[localInstance.A_csc_row_idx[i]];
		}

		vectorBFileName.str("");
		vectorBFileName << ctx.vectorBFile << "_" << world.rank()
				<< "_localHistogram.txt";
		ofstream histFile;
		histFile.open(vectorBFileName.str().c_str(), ios::out);
		for (unsigned int i = 0; i < localInstance.m; i++) {
			histFile << localNNZPerRow[i] << endl;
		}
		histFile.close();

		localNNZPerRow.resize(globalInstance.m, 0);
		for (int i = 0; i < globalInstance.m; i++)
			localNNZPerRow[i] = 0;

		std::vector<int> reducedNNZPerRow(globalInstance.m, 0);
		for (unsigned int i = 0; i < globalInstance.A_csc_row_idx.size(); i++) {
			localNNZPerRow[globalInstance.A_csc_row_idx[i]]++;
		}


		vall_reduce(world, &localNNZPerRow[0], &reducedNNZPerRow[0],
				reducedNNZPerRow.size());
		for (int i = 0; i < globalInstance.m; i++) {
			if (localXi < reducedNNZPerRow[i]) {
				localXi = reducedNNZPerRow[i];
			}
		}

		vectorBFileName.str("");
		vectorBFileName << ctx.vectorBFile << "_" << world.rank()
				<< "_globalHistogram.txt";
		histFile.open(vectorBFileName.str().c_str(), ios::out);
		for (unsigned int i = 0; i < globalInstance.m; i++) {
			histFile << reducedNNZPerRow[i] << endl;
		}
		histFile.close();

		vall_reduce_maximum(world, &localXi, &xi, 1);
//		if (ctx.settings.verbose) {
		if (true) {
			std::cout << world.rank() << " Local data size: " << localInstance.m
					<< " x " << localInstance.n << std::endl;
			std::cout << world.rank() << " Global data size: "
					<< globalInstance.m << " x " << globalInstance.n
					<< std::endl;
		}

		vall_reduce(world, &localInstance.n, &localInstance.total_n, 1);

	} else {

		cout << world.rank() << " going to load data" << endl;

		loadDistributedSparseSVMRowData(ctx.matrixAFile, world.rank(),
				world.size(), globalInstance, false);
		globalInstance.lambda = ctx.lambda;
		localInstance.lambda = ctx.lambda;
		localInstance.n = globalInstance.n;
		localInstance.m = 0;

		L m = globalInstance.m;

		cout << world.rank() << " data loaded  " << endl;

		vall_reduce_maximum(world, &m, &globalInstance.m, 1);
		std::vector<int> localNNZPerRow(globalInstance.m, 0);
		std::vector<int> reducedNNZPerRow(globalInstance.m, 0);

		cout << world.rank() << " reduce loaded  " << endl;

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



	vall_reduce(world, &localInstance.n, &localInstance.total_n, 1);

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

	if (ctx.settings.forcedSigma > 0) {
		localInstance.sigma = ctx.settings.forcedSigma;
	}

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

	if (distributedSettings.APPROX) {
		ClusterApproxEngineExecutor<L,D> executor(world,
				localInstance, globalInstance, &(distributedSettings));
		executor.initializeAll();
		Solver<L, D> solver(executor);
		solver.runSolver();
	} else {
		ClusterEngineExecutor<L, D> executor(world,
				localInstance, globalInstance, &(distributedSettings));

		executor.initializeAll();

		Solver<L, D> solver(executor);

		solver.runSolver();
	}

	logFile.close();

	MPI::Finalize();
}

int main(int argc, char *argv[]) {

	runConsolveSolver<int,float>(argc,argv);

}
