#include "../solver/distributed/distributed_include.h"
#include "../class/Context.h"
#include "../helpers/option_console_parser.h"
#include "../solver/settingsAndStatistics.h"
#include "../utils/file_reader.h"
#include "../solver/Solver.h"
#include "../helpers/utils.h"
#include <math.h>
#include "../utils/distributed_instances_loader.h"
#include "../utils/matrixvector.h"
#include "../solver/distributed/distributed_structures.h"
#include "../helpers/option_distributed_console_parser.h"
#include "ocsidHelper.h"
#include "readWholeData.h"
#include "../solver/distributed/distributed_essentials.h"

//#ifdef MATLAB
//
//#include "class/QuadraticLossLbfgs.h"
//#include "class/LogisticLossMatlab.h"
//
//#endif
#include  <sstream>
int main(int argc, char *argv[]) {
	MPI_Init(&argc, &argv);
	mpi::environment env(argc, argv);
	mpi::communicator world;
	DistributedSettings distributedSettings;
	Context ctx(distributedSettings);
	consoleHelper::parseDistributedOptions(ctx, distributedSettings, argc,
			argv);

	ctx.settings.verbose = true;
	if (world.rank() != 0) {
		ctx.settings.verbose = false;
	}

	ProblemData<unsigned int, double> instance;
	ProblemData<unsigned int, double> newInstance;
	//readWholeData(ctx.matrixAFile, instance, false);
	loadDistributedByFeaturesSVMRowData(ctx.matrixAFile, world.rank(), world.size(), instance, false);
	
	unsigned int finalM;
	instance.total_n = instance.n;
	//partitionByFeature(instance, newInstance, world.size(), world.rank());
	instance.theta = ctx.tmp;
	instance.lambda = ctx.lambda;
	//newInstance.total_n = instance.total_n;


	double rho = 1.0 / sqrt(instance.n);
	double mu = 0.1;

	std::vector<double> w(instance.m);
	//for (unsigned int i = 0; i < instance.m; i++)	w[i] = 0.1*rand() / (RAND_MAX + 0.0);
	std::vector<double> vk(instance.m);
	double deltak = 0.0;

	std::stringstream ss;
	ss << ctx.matrixAFile << "_ParFea_" << world.size() << ".log";
	std::ofstream logFile;
	logFile.open(ss.str().c_str());
	

	//for (unsigned int i = 0; i < K; i++){
	//	update_w(w, vk, deltak);
	if (world.rank() == 0){
		printf("Computing initial point starts!\n");
		printf("\n");
	}
	compute_initial_w(w, instance, rho, world.rank());
	distributed_PCGByD_SparseP(w, instance, mu, vk, deltak, world, world.size(), world.rank(), logFile);

	//}
	logFile.close();
	MPI::Finalize();

	return 0;

}
