#include "../solver/distributed/distributed_include.h"

ofstream myfile;
int debug_iterations = 0;
#include "../class/loss/losses.h"
#include "../solver/distributed/distributed_structures.h"
#include "../solver/distributed/distributed_solver.h"

template<typename D, typename L, typename LT>
int run_distributed_test(int argc, char * argv[]);

int main(int argc, char * argv[]) {
	MPI_Init(&argc, &argv);
	float version;
	int rc = Zoltan_Initialize(argc, argv, &version);
	if (rc != ZOLTAN_OK)
		quit("Zoltan_Initialize failed. Sorry ...\n");
	// printf("MULTIPLE SOURCE CODE\n");
	run_distributed_test<float, int, square_loss_traits>(argc, argv);
	MPI::Finalize();
	return 0;
}

template<typename D, typename L, typename LT>
int run_distributed_test(int argc, char * argv[]) {
	mpi::environment env(argc, argv);
	mpi::communicator world;
	print_current_memory_consumption(world);

	// Create Optimization settings
	DistributedSettings settings;
	settings.verbose = false;
	settings.totalThreads = 1;
	settings.bulkIterations = false;

	settings.iters_communicate_count = 2000;
	settings.iters_bulkIterations_count = 1;
	settings.iterationsPerThread = 20;
	settings.showLastObjectiveValue = true;

	settings.partitioning = BlockedPartitioning; //Set partitioning method

	ProblemData<L, D> inst;
	ProblemData<L, D> part;
	// Load the instance at the root
	generate_data_with_know_optimal_value(inst, world, settings);

	part.lambda = inst.lambda;
	part.sigma = inst.sigma; //FIXME compute true SIGMA!!!
	world.barrier();

	// Create Optimization stats
	distributed_statistics stat("generated", &world, settings);

	data_distributor<L, D> dataDistributor;

	settings.broadcast_treshold = 0;
	distribute_data_from_nontrivial_sources(world, dataDistributor, inst, part,
			settings, stat);
	//	world.barrier();
	//	create_distribution_schema_for_multiple_sources(world, inst, part, dataDistributor, settings, stat);
	settings.broadcast_treshold = 2;

	part.sigma = 1;

	create_distribution_schema_for_multiple_sources(world, inst, part,
			dataDistributor, settings, stat);

	if (world.rank() == 0) {
		cout << "Hypergraph cut: " << stat.hypergraph_cut << endl;
	}
	world.barrier();

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

	int test_case = 0;
	//	 Solver should solve using only PART! (INST can we used for other purposes

	stat.reset();
	settings.distributed = SynchronousReduce;
	distributed_solver_from_multiple_sources<D, L, LT>(env, world, settings,
			stat, part, dataDistributor,rs);
	if (world.rank() == 0) {
		std::cout << "TEST " << test_case++ << " (" << settings.distributed
				<< ")" << std::endl << ": Objective " << stat.last_obj_value
				<< " Runtime "
				<< boost::timer::format(stat.time_nonstop.elapsed(), 6, "%w")
				<< "\n" << std::endl;
		get_additional_times(stat);
	}

	settings.distributed = AsynchronousStreamlined;
	distributed_solver_from_multiple_sources<D, L, LT>(env, world, settings,
			stat, part, dataDistributor,rs);
	if (world.rank() == 0) {
		std::cout << "TEST " << test_case++ << " (" << settings.distributed
				<< ")" << std::endl << ": Objective " << stat.last_obj_value
				<< " Runtime "
				<< boost::timer::format(stat.time_nonstop.elapsed(), 6, "%w")
				<< "\n" << std::endl;
		get_additional_times(stat);
	}
	stat.reset();

	settings.distributed = AsynchronousStreamlinedOptimized;
	distributed_solver_from_multiple_sources<D, L, LT>(env, world, settings,
			stat, part, dataDistributor,rs);
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
			stat, part, dataDistributor,rs);
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
			stat, part, dataDistributor,rs);
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
			stat, part, dataDistributor,rs);
	if (world.rank() == 0) {
		std::cout << "TEST " << test_case++ << " (" << settings.distributed
				<< ")" << std::endl << ": Objective " << stat.last_obj_value
				<< " Runtime "
				<< boost::timer::format(stat.time_nonstop.elapsed(), 6, "%w")
				<< "\n" << std::endl;
		get_additional_times(stat);
	}
	stat.reset();

	print_current_memory_consumption(world);
	return 0;

}
