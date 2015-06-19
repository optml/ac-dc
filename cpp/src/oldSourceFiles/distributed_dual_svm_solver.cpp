#include "../solver/distributed/distributed_include.h"

ofstream myfile;
int debug_iterations = 0;
#include "../class/loss/losses.h"
#include "../solver/distributed/distributed_structures.h"
#include "../solver/distributed/distributed_solver.h"

template<typename D, typename L, typename LT>
int run_huge_exepriment(string experimentData, int argc, char * argv[]);

int main(int argc, char * argv[]) {
	MPI_Init(&argc, &argv);
	float version;
//	int rc = Zoltan_Initialize(argc, argv, &version);
//	if (rc != ZOLTAN_OK)
//		quit("Zoltan_Initialize failed. Sorry ...\n");

//	run_huge_exepriment<double, unsigned long, square_loss_traits>(
//			"/tmp/experimentNEW", argc, argv);

//	run_huge_exepriment<double, unsigned long, square_loss_traits>(
//			"/home/e248/e248/takac/work/experiment8b", argc, argv);

//	run_huge_exepriment<double, int, square_loss_traits>(
//			"./data/a1a", argc, argv);
//
//	run_huge_exepriment<double, int, loss_hinge_dual>(
//			"/home/taki/phd/otherProjects/ac-dc/cpp/data/a1a.4/a1a", argc,
//			argv);

//	myfile.open("/tmp/webspam16_v5.log");
//	run_huge_exepriment<double, int, loss_hinge_dual>(
//			"/home/taki/phd/otherProjects/ac-dc/cpp/data/test.4/a1a", argc,
//			argv);
//		myfile.close();

	myfile.open("/home/e248/e248/takac/work/webspam16_v18.log");
	run_huge_exepriment<double, int, loss_hinge_dual>(
			"/home/e248/e248/takac/work/data/webspam.16/file", argc, argv);
	myfile.close();

	MPI::Finalize();
	return 0;
}

#include "../problem_generator/distributed/generator_nesterov_to_file.h"
#include "../utils/distributed_instances_loader.h"

template<typename D, typename L, typename LT>
int run_huge_exepriment(string experimentData, int argc, char * argv[]) {
	mpi::environment env(argc, argv);
	mpi::communicator world;
	print_current_memory_consumption(world);

// Create Optimization settings
	DistributedSettings settings;
	settings.verbose = true;
	settings.bulkIterations = true;
	settings.showLastObjectiveValue = true;

	settings.partitioning = BlockedPartitioning; //Set partitioning method

// Create Optimization stats
	distributed_statistics stat("generated", &world, settings);

	gsl_rng_env_setup();
	const gsl_rng_type * T;
	gsl_rng * rr;
	T = gsl_rng_default;
	rr = gsl_rng_alloc(T);
	const int MAXIMUM_THREADS = 32;
	std::vector<gsl_rng *> rs(MAXIMUM_THREADS);
	for (int i = 0; i < MAXIMUM_THREADS; i++) {
		rs[i] = gsl_rng_alloc(T);
		unsigned long seed = i + MAXIMUM_THREADS * world.rank() * world.size();
		gsl_rng_set(rs[i], seed);
	}

	randomNumberUtil::init_omp_random_seeds(world.rank());
	randomNumberUtil::init_random_seeds(rs,
			MAXIMUM_THREADS * world.rank() * world.size());

	srand(world.rank());

// ================================LOAD DATA=======================
	ProblemData<L, D> part;

	D optimalValue = 0;

	loadDistributedSparseSVMRowData(experimentData, world.rank(), world.size(),
			part, false);

	vreduce(world, &part.n, &part.total_n, 1, 0);
	vbroadcast(world, &part.total_n, 1, 0);

	// obatining \omega
	int omega = 0;
	for (int sample = 0; sample < part.A_csr_row_ptr.size() - 1; sample++) {
		int rowOmega = part.A_csr_row_ptr[sample + 1]
				- part.A_csr_row_ptr[sample];
		if (rowOmega > omega) {
			omega = rowOmega;
		}
	}
	int totalOmega;
	vreduce(world, &omega, &totalOmega, 1, 0);
	vbroadcast(world, &totalOmega, 1, 0);

	cout << "omega : " << omega << "  " << "total Omega " << totalOmega << endl;

	part.lambda = 1 / (0.0 + part.total_n);
	part.lambda = 0.01;

	L totalFeatures;
	vreduce_max(world, &part.m, &totalFeatures, 1, 0);
	vbroadcast(world, &totalFeatures, 1, 0);
	part.m = totalFeatures;

	ProblemData<L, D> part_local;
	part_local.n = part.n;
	part_local.m = 0;
	part_local.A_csr_row_ptr.resize(part.n + 1, 0);

	// normalize ROWS!!!

//	for (L sample = 0; sample < part.n; sample++) {
//		D norm = 0;
//		for (L tmp = part.A_csr_row_ptr[sample];
//				tmp < part.A_csr_row_ptr[sample + 1]; tmp++) {
//			norm += part.A_csr_values[tmp] * part.A_csr_values[tmp];
//		}
//		if (norm > 0) {
//			norm = 1 / sqrt(norm);
//			for (L tmp = part.A_csr_row_ptr[sample];
//					tmp < part.A_csr_row_ptr[sample + 1]; tmp++) {
//				part.A_csr_values[tmp] = part.A_csr_values[tmp] * norm;
//			}
//		}
//
//	}

	stat.generated_optimal_value = optimalValue;

	long long totalN = part.total_n;

	settings.totalThreads = 8;
	settings.iterationsPerThread = 20;

//	settings.totalThreads = 2;
//	settings.iterationsPerThread = 1;

	settings.iters_communicate_count = //20
			+totalN
					/ (world.size() * settings.iterationsPerThread
							* settings.totalThreads + 0.0);
//	settings.iters_bulkIterations_count = 5;

	settings.iters_communicate_count = settings.iters_communicate_count * 5;
	settings.iters_communicate_count = 50;
	settings.iters_bulkIterations_count = 100;

//	settings.iters_communicate_count=1;
//	settings.iters_bulkIterations_count = 1;

	cout << "Solver settings  " << settings.iterationsPerThread
			<< "  communica " << settings.iters_communicate_count
			<< " recompute " << settings.iters_bulkIterations_count
			<< endl;

	settings.showInitialObjectiveValue = true;
//	settings.iters_communicate_count = 1;
//	settings.iters_bulkIterations_count = 1;

//	part.sigma = 8
//			+ ((part.n * p * world.size() / (part.m + 0.0) - 1.0)
//					* (settings.iterationsPerThread
//							* settings.totalThreads * world.size() - 1.0))
//					/ (part.n * world.size() - 1.0)
	; //FIXME compute true SIGMA!!!

//	cout << "SIGMA IS " << part.sigma << " " << part.n << " "
//			<< part.m << endl;
//	part.sigma = 1;
	world.barrier();

	data_distributor<L, D> dataDistributor;

	settings.broadcast_treshold = 2;

	settings.showIntermediateObjectiveValue = true;

	omp_set_num_threads(settings.totalThreads);
	randomNumberUtil::init_random_seeds(rs,
			settings.totalThreads * world.rank() * world.size());

	int test_case = 0;
	stat.reset();
	settings.iterationsPerThread =settings.iterationsPerThread /6;
//	part.sigma = 2*2*2*2*2*2*2*2;
	double increment = 2;
	for (int ex = 0; ex < 10; ex++) {
		settings.iterationsPerThread = settings.iterationsPerThread
				* increment;
//		increment = increment * 2;
		double tt = settings.totalThreads * settings.iterationsPerThread;


//		totalOmega=totalOmega;

		part.sigma = 1;
		if (part.n > 1 && tt > 1) {
			part.sigma += (totalOmega - 1) * (tt - 1) / (part.n - 1.0)
					+ totalOmega * (world.size() - 1) / (world.size() + 0.0)
							* (tt / (part.n + 0.0) - (tt - 1) / (part.n - 1.0));
		}
		part_local.sigma = part.sigma;
		settings.distributed = SynchronousReduce;
//		settings.distributed = AsynchronousStreamlinedOptimized;

//		switch (ex) {
//		case 0:
//			part.sigma = 1.1;
//			break;
//		case 1:
//			part.sigma = 2;
//			break;
//		case 2:
//			part.sigma = 3;
//			break;
//
//		default:
//			break;
//		}

		cout << "SIGMA IS " << part.sigma << " " << part.n << " " << part.m
				<< endl;

		distributed_solver_from_multiple_sources_structured_hybrid_barrier<D, L,
				LT>(env, world, settings, stat, part_local, part,
				dataDistributor, rs);

		if (world.rank() == 0) {
			std::cout << "TEST " << test_case++ << " (" << settings.distributed
					<< ")" << std::endl << ": Objective " << stat.last_obj_value
					<< ": error " << stat.last_obj_value - optimalValue
					<< " Runtime "
					<< boost::timer::format(stat.time_nonstop.elapsed(), 6,
							"%w") << "\n" << " " << part.sigma << std::endl;
			get_additional_times(stat);
		}

	}

//
//	distributed_solver_from_multiple_sources<D, L, LT>(env, world, settings,
//			stat, part, dataDistributor);
//	if (world.rank() == 0) {
//		std::cout << "TEST " << test_case++ << " (" << settings.distributed
//				<< ")" << std::endl << ": Objective " << stat.last_obj_value
//				<< " Runtime "
//				<< boost::timer::format(stat.time_nonstop.elapsed(), 6, "%w")
//				<< "\n" << std::endl;
//		get_additional_times(stat);
//	}
//	stat.reset();

	/*
	 part.sigma = 1;

	 create_distribution_schema_for_multiple_sources(world, inst, part,
	 dataDistributor, settings, stat);

	 if (world.rank() == 0) {
	 cout << "Hypergraph cut: " << stat.hypergraph_cut << endl;
	 }
	 world.barrier();

	 int test_case = 0;
	 //	 Solver should solve using only PART! (INST can we used for other purposes



	 settings.distributed = AsynchronousStreamlined;
	 distributed_solver_from_multiple_sources<D, L, LT>(env, world, settings,
	 stat, part, dataDistributor);
	 if (world.rank() == 0) {
	 std::cout << "TEST " << test_case++ << " (" << settings.distributed
	 << ")" << std::endl << ": Objective " << stat.last_obj_value
	 << " Runtime "
	 << boost::timer::format(stat.time_nonstop.elapsed(), 6, "%w")
	 << "\n" << std::endl;
	 get_additional_times(stat);
	 }
	 stat.reset();



	 // NOTE: on 4 computers, width 2 does not work
	 settings.torus_width = 2;
	 settings.distributed = AsynchronousTorus;
	 distributed_solver_from_multiple_sources<D, L, LT>(env, world, settings,
	 stat, part, dataDistributor);
	 if (world.rank() == 0) {
	 std::cout << "TEST " << test_case++ << " (" << settings.distributed
	 << ")" << std::endl << ": Objective " << stat.last_obj_value
	 << " Runtime "
	 << boost::timer::format(stat.time_nonstop.elapsed(), 6, "%w")
	 << "\n" << std::endl;
	 get_additional_times(stat);
	 }
	 stat.reset();

	 // NOTE: on 4 computers, width 2 does not work
	 settings.torus_width = 2;
	 settings.distributed = AsynchronousTorusOpt;
	 distributed_solver_from_multiple_sources<D, L, LT>(env, world, settings,
	 stat, part, dataDistributor);
	 if (world.rank() == 0) {
	 std::cout << "TEST " << test_case++ << " (" << settings.distributed
	 << ")" << std::endl << ": Objective " << stat.last_obj_value
	 << " Runtime "
	 << boost::timer::format(stat.time_nonstop.elapsed(), 6, "%w")
	 << "\n" << std::endl;
	 get_additional_times(stat);
	 }
	 stat.reset();

	 // NOTE: on 4 computers, width 2 does not work
	 settings.torus_width = 2;
	 settings.distributed = AsynchronousTorusOptCollectives;
	 distributed_solver_from_multiple_sources<D, L, LT>(env, world, settings,
	 stat, part, dataDistributor);
	 if (world.rank() == 0) {
	 std::cout << "TEST " << test_case++ << " (" << settings.distributed
	 << ")" << std::endl << ": Objective " << stat.last_obj_value
	 << " Runtime "
	 << boost::timer::format(stat.time_nonstop.elapsed(), 6, "%w")
	 << "\n" << std::endl;
	 get_additional_times(stat);
	 }
	 stat.reset();

	 */
	if (world.rank() == 0)
		cout << "Optimal value should be  " << optimalValue << endl;
	print_current_memory_consumption(world);
	return 0;

}
