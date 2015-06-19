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

	run_huge_exepriment<double, long, square_loss_traits>(
			"/home/d11/d11/takacn/work/data/experiment512", argc, argv);

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

	init_random_seeds(world.rank());
	init_random_seeds(rs, MAXIMUM_THREADS * world.rank() * world.size());

	srand(world.rank());

// ================================LOAD DATA=======================
	ProblemData<L, D> part_global;
	ProblemData<L, D> part_local;

	D optimalValue = loadDataFromFiles(experimentData, world.rank(),
			world.size(), part_global, part_local);
	stat.generated_optimal_value = optimalValue;
//	unsigned long n = 4000000;
//	unsigned long m = n * (world.size() * 4);
//	int p = 200;
//	double dataSize=n*p*2+m*2+n;
//	dataSize=dataSize*8/(1024*1024*1024);
//	cout << "Need "<<dataSize<<" GB per node which is "<<dataSize*world.size()<<" in total"<<endl;
//	long nnzOfSolution = 20;
//	stat.generated_optimal_value = nesterovDistributedGenerator(part, n, m, p,
//			nnzOfSolution, rs);
//cout <<"NNZ values "<< m<<" "<<n<<endl;
//	part.lambda = 1;

	long long totalN = part_global.n * world.size();

	settings.totalThreads = 8;
	settings.iterationsPerThread = 100;

	settings.iters_communicate_count = //20
			+totalN
					/ (world.size() * settings.iterationsPerThread
							* settings.totalThreads + 0.0);

//	settings.iters_bulkIterations_count = 5;
	settings.iters_communicate_count = settings.iters_communicate_count * 5;
	settings.iters_communicate_count = 1000;
	settings.iters_bulkIterations_count = 200;

//	settings.iters_bulkIterations_count = 3;
//	settings.iters_communicate_count=5;
	cout << "Solver settings  " << settings.iterationsPerThread
			<< "  communica " << settings.iters_communicate_count << endl;

	settings.showInitialObjectiveValue = true;
//	settings.iters_communicate_count = 1;
//	settings.iters_bulkIterations_count = 1;

	part_global.lambda = 1;
	part_local.lambda = part_global.lambda;

	part_global.sigma = 8
//			+ ((part.n * p * world.size() / (part.m + 0.0) - 1.0)
//					* (settings.iterationsPerThread
//							* settings.totalThreads * world.size() - 1.0))
//					/ (part.n * world.size() - 1.0)
			;//FIXME compute true SIGMA!!!
	part_global.sigma = 1.1;

	part_local.sigma = part_global.sigma;

	cout << "SIGMA IS " << part_global.sigma << " " << part_global.n << " "
			<< part_global.m << endl;
//	part.sigma = 1;
	world.barrier();

	data_distributor<L, D> dataDistributor;

	settings.broadcast_treshold = 2;

	settings.showIntermediateObjectiveValue = true;

	int test_case = 0;
	stat.reset();

////	settings.distributed = AsynchronousStreamlinedOptimized;
//	distributed_solver_from_multiple_sources_structured<D, L, LT>(env, world,
//			settings, stat, part_local, part_global, dataDistributor, rs);
//
//	if (world.rank() == 0) {
//		std::cout << "TEST " << test_case++ << " (" << settings.distributed
//				<< ")" << std::endl << ": Objective " << stat.last_obj_value
//				<< ": error " << stat.last_obj_value - optimalValue
//				<< " Runtime "
//				<< boost::timer::format(stat.time_nonstop.elapsed(), 6, "%w")
//				<< "\n" << std::endl;
//		get_additional_times(stat);
//	}
//
//	stat.reset();
//
//	distributed_solver_from_multiple_sources_structured_hybrid<D, L, LT>(env,
//			world, settings, stat, part_local, part_global, dataDistributor,
//			rs);
//
//	if (world.rank() == 0) {
//		std::cout << "TEST " << test_case++ << " (" << settings.distributed
//				<< ")" << std::endl << ": Objective " << stat.last_obj_value
//				<< ": error " << stat.last_obj_value - optimalValue
//				<< " Runtime "
//				<< boost::timer::format(stat.time_nonstop.elapsed(), 6, "%w")
//				<< "\n" << std::endl;
//		get_additional_times(stat);
//	}
//
//	stat.reset();

//	distributed_solver_from_multiple_sources_structured_hybrid_barrier<D, L, LT>(
//			env, world, settings, stat, part_local, part_global,
//			dataDistributor, rs);
//
//	if (world.rank() == 0) {
//		std::cout << "TEST " << test_case++ << " (" << settings.distributed
//				<< ")" << std::endl << ": Objective " << stat.last_obj_value
//				<< ": error " << stat.last_obj_value - optimalValue
//				<< " Runtime "
//				<< boost::timer::format(stat.time_nonstop.elapsed(), 6, "%w")
//				<< "\n" << std::endl;
//		get_additional_times(stat);
//	}

//	stat.reset();

//	distributed_solver_from_multiple_sources_structured<D, L, LT>(env, world,
//			settings, stat, part_local, part_global, dataDistributor, rs);
//
//	if (world.rank() == 0) {
//		std::cout << "TEST " << test_case++ << " (" << settings.distributed
//				<< ")" << std::endl << ": Objective " << stat.last_obj_value
//				<< ": error " << stat.last_obj_value - optimalValue
//				<< " Runtime "
//				<< boost::timer::format(stat.time_nonstop.elapsed(), 6, "%w")
//				<< "\n" << std::endl;
//		get_additional_times(stat);
//	}
//
//	stat.reset();
//
//	distributed_solver_from_multiple_sources_structured_hybrid<D, L, LT>(env,
//			world, settings, stat, part_local, part_global, dataDistributor,
//			rs);
//
//	if (world.rank() == 0) {
//		std::cout << "TEST " << test_case++ << " (" << settings.distributed
//				<< ")" << std::endl << ": Objective " << stat.last_obj_value
//				<< ": error " << stat.last_obj_value - optimalValue
//				<< " Runtime "
//				<< boost::timer::format(stat.time_nonstop.elapsed(), 6, "%w")
//				<< "\n" << std::endl;
//		get_additional_times(stat);
//	}
//
//	stat.reset();

//	for (int ex = 0; ex < 2; ex++)
	{

		settings.distributed = SynchronousReduce;
//		settings.distributed = AsynchronousStreamlinedOptimized;

//		switch (ex) {
//		case 0:
			part_global.sigma = 1.1;
//			break;
//		case 1:
//			part_global.sigma = 2;
//			break;
//		case 2:
//			part_global.sigma = 3;
//			break;
//
//		default:
//			break;
//		}

		part_local.sigma = part_global.sigma;
		cout << "SIGMA IS " << part_global.sigma << " " << part_global.n << " "
				<< part_global.m << endl;

		distributed_solver_from_multiple_sources_structured_hybrid_barrier<D, L,
				LT>(env, world, settings, stat, part_local, part_global,
				dataDistributor, rs);

		if (world.rank() == 0) {
			std::cout << "TEST " << test_case++ << " (" << settings.distributed
					<< ")" << std::endl << ": Objective " << stat.last_obj_value
					<< ": error " << stat.last_obj_value - optimalValue
					<< " Runtime "
					<< boost::timer::format(stat.time_nonstop.elapsed(), 6,
							"%w") << "\n" << " " << part_global.sigma
					<< std::endl;
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
