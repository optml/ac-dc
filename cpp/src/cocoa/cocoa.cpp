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
#include "cocoaHelper.h"
#include "../solver/distributed/distributed_essentials.h"

#include "class/LogisticLossCD.h"
#include "class/HingeLossCD.h"
#include "class/HingeLossCDCaseC.h"

#include "class/QuadraticLossCD.h"
#include "class/SquareHingeLossCD.h"

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
	instance.theta = ctx.tmp;
	cout << "XXXXXXXx   " << instance.theta << endl;
	cout << world.rank() << " going to load data" << endl;

	loadDistributedSparseSVMRowData(ctx.matrixAFile, world.rank(), world.size(),
			instance, false);

	unsigned int finalM;

	vall_reduce_maximum(world, &instance.m, &finalM, 1);

	cout << "Local m " << instance.m << "   global m " << finalM << endl;

	instance.m = finalM;

	//	for (unsigned int i = 0; i < instance.m; i++) {
	//		for (unsigned int j = instance.A_csr_row_ptr[i];
	//				j < instance.A_csr_row_ptr[i + 1]; j++) {
	//			instance.A_csr_values[j] = instance.A_csr_values[j] * instance.b[i];
	//		}
	//		instance.b[i] = 1;
	//	}

	instance.lambda = ctx.lambda;

	std::vector<double> wBuffer(instance.m);
	std::vector<double> deltaW(instance.m);
	std::vector<double> deltaAlpha(instance.n);
	std::vector<double> w(instance.m);

	instance.x.resize(instance.n);
	cblas_set_to_zero(instance.x);
	cblas_set_to_zero(w);
	cblas_set_to_zero(deltaW);

	// compute local w
	vall_reduce(world, deltaW, w);

	cout << " Local n " << instance.n << endl;

	vall_reduce(world, &instance.n, &instance.total_n, 1);

	instance.oneOverLambdaN = 1 / (0.0 + instance.total_n * instance.lambda);

	double gamma;
	if (distributedSettings.APPROX) {
		gamma = 1;
		instance.penalty = world.size() + 0.0;
	} else {
		gamma = 1 / (world.size() + 0.0);
		instance.penalty = 1;
	}

	LossFunction<unsigned int, double> * lf;

	instance.experimentName = ctx.experimentName;

	int loss = distributedSettings.lossFunction;
	//			+ distributedSettings.APPROX * 100;

	switch (loss) {
	case 0:
		lf = new LogisticLossCD<unsigned int, double>();
		break;
	case 1:
		lf = new HingeLossCD<unsigned int, double>();
		break;

	case 2:
		lf = new SquareHingeLossCD<unsigned int, double>();
		break;
	case 3:
		lf = new QuadraticLossCD<unsigned int, double>();
		break;

		//#ifdef MATLAB
		//
		//		case 4:
		//		lf = new QuadraticLossLbfgs<unsigned int, double>();
		//		break;
		//		case 5:
		//		lf = new LogisticLossMatlab<unsigned int, double>();
		//		break;
		//
		//#endif

	default:
		break;
	}

	lf->init(instance);

	double elapsedTime = 0;

	std::stringstream ss;
	ss << ctx.matrixAFile << "_" << instance.lambda << "_"
			<< distributedSettings.lossFunction << "_"
			<< distributedSettings.iters_communicate_count << "_"
			<< distributedSettings.iterationsPerThread << "_"
			<< instance.experimentName << "_" << distributedSettings.APPROX
			<< "_" << instance.theta << "_.log";
	std::ofstream logFile;
	if (ctx.settings.verbose) {
		logFile.open(ss.str().c_str());
	}

	if (distributedSettings.iters_bulkIterations_count < 1) {
		distributedSettings.iters_bulkIterations_count = 1;
	}

	distributedSettings.iters_communicate_count =
			distributedSettings.iters_communicate_count
			/ distributedSettings.iters_bulkIterations_count;

	cout << "BULK "<<distributedSettings.iters_bulkIterations_count<<" "<< distributedSettings.iters_communicate_count<<endl;
	double start = 0;
	double finish = 0;


	double theta = 4.0 * distributedSettings.iterationsPerThread / instance.total_n;
	std::vector<double> zk(instance.n);
	std::vector<double> uk(instance.n);
	std::vector<double> Ayk(instance.m);
	std::vector<double> yk(instance.n);
	std::vector<double> deltayk(instance.n);
	std::vector<double> deltaAyk(instance.m);
	cblas_set_to_zero(uk);
	cblas_set_to_zero(yk);
	cblas_set_to_zero(Ayk);
	std::vector<double> AykBuffer(instance.m);
	double dualobj = 0;


	for (unsigned int i = 0; i < instance.n; i++)
		zk[i] = instance.x[i];
	int localsolver = distributedSettings.LocalMethods;

	switch (localsolver) {
	case 0:
		for (unsigned int t = 0; t < distributedSettings.iters_communicate_count; t++) {

			start = gettime_();

			for (int jj = 0; jj < distributedSettings.iters_bulkIterations_count; jj++) {
				cblas_set_to_zero(deltaW);
				cblas_set_to_zero(deltaAlpha);

				lf->SDCA(instance, deltaAlpha, w, deltaW, distributedSettings);
				vall_reduce(world, deltaW, wBuffer);
				cblas_sum_of_vectors(w, wBuffer, gamma);
				cblas_sum_of_vectors(instance.x, deltaAlpha, gamma);
			}
			double primalError;
			double dualError;

			//computeObjectiveValueHingeLoss(instance, world, w,dualError,primalError);
			//computeObjectiveValueQuadLoss(instance, world, w,dualError,primalError);
			//computeObjectiveValueSquaredHingeLoss(instance, world, w,dualError,primalError);
			finish = gettime_();
			elapsedTime += finish - start;

			lf->computeObjectiveValue(instance, world, w, dualError, primalError);

			if (ctx.settings.verbose) {
				cout << "Iteration " << t << " elapsed time " << elapsedTime
						<< "  error " << primalError << "    " << dualError
						<< "    " << primalError + dualError << endl;

				logFile << t << "," << elapsedTime << "," << primalError << ","
						<< dualError << "," << primalError + dualError << endl;

			}
		}
		break;
	case 1:


		for (unsigned int t = 0; t < distributedSettings.iters_communicate_count; t++) {
			start = gettime_();

			for (int jj = 0; jj < distributedSettings.iters_bulkIterations_count; jj++) {
				cblas_set_to_zero(deltaW);
				cblas_set_to_zero(deltaAlpha);
				cblas_set_to_zero(deltayk);
				cblas_set_to_zero(deltaAyk);

				lf->fast_SDCA(instance, deltaAlpha, w, deltaW, zk, uk, yk, deltayk, Ayk, deltaAyk, theta, distributedSettings);
				//for (unsigned int i = 0; i < instance.n; i++){
				//cout <<i<<"  "<< theta << "    "<<uk[i] << "    " <<zk[i] <<endl;
				//instance.x[i] = theta * theta * uk[i] + zk[i]; }
				double thetasq = theta * theta;
				theta = 0.5 * sqrt(thetasq * thetasq + 4 * thetasq) - 0.5 * thetasq;

				vall_reduce(world, deltaW, wBuffer);
				cblas_sum_of_vectors(w, wBuffer, gamma);
				//cblas_sum_of_vectors(instance.x, deltaAlpha, gamma);
				vall_reduce(world, deltaAyk, AykBuffer);
				cblas_sum_of_vectors(Ayk, AykBuffer, gamma);
				//cblas_sum_of_vectors(yk, deltayk, gamma);
			}

			double primalError;
			double dualError;

			finish = gettime_();
			elapsedTime += finish - start;

			lf->computeObjectiveValue(instance, world, w, dualError, primalError);

			if (ctx.settings.verbose) {
				cout << "Iteration " << t << " elapsed time " << elapsedTime
						<< "  error " << primalError << "    " << dualError
						<< "    " << primalError + dualError << endl;

				logFile << t << "," << elapsedTime << "," << primalError << ","
						<< dualError << "," << primalError + dualError << endl;

			}
		}
		break;
	case 2:
		for (unsigned int t = 0; t < distributedSettings.iters_communicate_count; t++) {
			start = gettime_();

			for (int jj = 0; jj <  distributedSettings.iters_bulkIterations_count; jj++) {

				cblas_set_to_zero(deltaW);
				cblas_set_to_zero(deltaAlpha);

				lf->fulldescent(instance, deltaAlpha, w, deltaW, dualobj, distributedSettings);
				//matrixvector(instance.A_csr_values, instance.A_csr_col_idx,instance.A_csr_row_ptr, instance.x, instance.m, deltaW);
				cblas_set_to_zero(w);

				vall_reduce(world, deltaW, wBuffer);
				cblas_sum_of_vectors(w, wBuffer, gamma);
				//cout << deltaW[0]<<"   "<<w[0]<<"   "<<wBuffer[0]<<endl;

				//cblas_sum_of_vectors(instance.x, deltaAlpha, gamma);

			}

			double primalError;
			double dualError;

			finish = gettime_();
			elapsedTime += finish - start;

			lf->computeObjectiveValue(instance, world, w, dualError, primalError);

			if (ctx.settings.verbose) {
				cout << "Iteration " << t << " elapsed time " << elapsedTime
						<< "  error " << primalError << "    " << dualError
						<< "    " << primalError + dualError << endl;

				logFile << t << "," << elapsedTime << "," << primalError << ","
						<< dualError << "," << primalError + dualError << endl;

			}
		}
		break;

	default:
		break;
	}

	//-------------------------------------------------------


	if (ctx.settings.verbose) {
		logFile.close();
	}

	MPI::Finalize();

	return 0;
}
