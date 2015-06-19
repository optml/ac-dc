#include "../solver/distributed/distributed_include.h"

ofstream myfile;
int debug_iterations = 0;
#include "../class/loss/losses.h"
#include "../solver/distributed/distributed_structures.h"
#include "../solver/distributed/distributed_solver.h"

template<typename D, typename L, typename LT>
int run_huge_exepriment(int argc, char * argv[]);

#include "../problem_generator/distributed/generator_nesterov_multipart.h"

int main(int argc, char * argv[]) {
	MPI_Init(&argc, &argv);
	mpi::environment env(argc, argv);
	mpi::communicator world;
	omp_set_num_threads(32);


	int TOTAL_THREADS ;
#pragma omp parallel
	{
//		omp_set_num_threads(8);
		TOTAL_THREADS = omp_get_num_threads();
	}
	printf("Using %d threads\n",TOTAL_THREADS);


	#pragma omp parallel
	{
		cout << "My thread id is " << omp_get_thread_num() << "/"
				<< omp_get_num_threads() << " and rank is " << world.rank()
				<< endl;
	}
	MPI::Finalize();
	return 0;
}

