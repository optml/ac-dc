/*
 * This file contains an functions for experiment done in paper
 * P. Richtárik and M. Takáč
 *      Parallel Coordinate Descent Methods for Big Data Optimization
 * http://www.optimization-online.org/DB_HTML/2012/11/3688.html
 */

/*
 * abstract_loss.h
 *
 *  Created on: 19 Jan 2012
 *      Author: jmarecek and taki
 */

#ifndef DISTRIBUTED_LOSSES_H
#define DISTRIBUTED_LOSSES_H

#include "loss_abstract.h"

#include "../../solver/distributed/distributed_essentials.h"

template<typename L, typename D, typename Traits>
class DistributedLosses: public Losses<L, D, Traits> {
public:
	static inline D compute_fast_objective(const ProblemData<L, D> &part,
			const std::vector<D> &residuals, mpi::communicator &world) {
		D resids = 0;
		return resids;
	}
	static inline void bulkIterations_for_my_part_data(
			const ProblemData<L, D> &part, std::vector<D> &residuals,
			std::vector<D> & buffer_residuals, mpi::communicator &world) {
	}

};

#include "loss_square.h"

template<typename L, typename D>
class DistributedLosses<L, D, square_loss_traits> {
public:
	static inline D compute_fast_objective(const ProblemData<L, D> &part,
			const std::vector<D> &residuals, mpi::communicator &world) {
		D resids = 0;
		D sumx = 0;
		if (world.rank() == 0) {
			for (L i = 0; i < part.m; i++) {
				resids += residuals[i] * residuals[i];
			}
		}
		for (L j = 0; j < part.n; j++) {
			sumx += abs(part.x[j]);
		}
		D sumxOut = 0;
		reduce(world, sumx, sumxOut, std::plus<D>(), 0);
//				printf("%d  %f   %f   %f\n",world.rank(),sumx, sumxOut,resids);
		return 0.5 * resids + part.lambda * sumxOut;
	}

	static inline D compute_fast_objective(const ProblemData<L, D> &part,
			const ProblemData<L, D> &part_local,
			const std::vector<D> &residuals,
			const std::vector<D> &residuals_local, mpi::communicator &world) {
		D resids = 0;
		D sumx = 0;
		if (world.rank() == 0) {
			for (L i = 0; i < part.m; i++) {
				resids += residuals[i] * residuals[i];
			}
		}

		for (L i = 0; i < part_local.m; i++) {
			resids += residuals_local[i] * residuals_local[i];
		}

		for (L j = 0; j < part.n; j++) {
			sumx += abs(part.x[j]);
		}

		D sumxOut = 0;
		reduce(world, sumx, sumxOut, std::plus<D>(), 0);
		D residOut = 0;
		reduce(world, resids, residOut, std::plus<D>(), 0);

		//				printf("%d  %f   %f   %f\n",world.rank(),sumx, sumxOut,resids);
		return 0.5 * residOut + part.lambda * sumxOut;
	}

	static inline void bulkIterations_for_my_part_data(
			const ProblemData<L, D> &part, std::vector<D> &residuals,
			std::vector<D> & buffer_residuals, mpi::communicator &world) {
		if (world.rank() == 0) { //FIXME use CBLAS
			for (int i = 0; i < part.m; i++) {
				buffer_residuals[i] = -part.b[i];
			}
		} else {
			for (int i = 0; i < part.m; i++) {
				buffer_residuals[i] = 0;
			}
		}
		for (L col = 0; col < part.n; col++) {
			for (L row_tmp = part.A_csc_col_ptr[col];
					row_tmp < part.A_csc_col_ptr[col + 1]; row_tmp++) {
				buffer_residuals[part.A_csc_row_idx[row_tmp]] +=
						part.A_csc_values[row_tmp] * part.x[col];
			}
		}
		vall_reduce(world, buffer_residuals, residuals);
	}

};

#include "loss_square_hinge.h"

template<typename L, typename D>
class DistributedLosses<L, D, square_hinge_loss_traits> {
public:

	static inline void bulkIterations_for_my_part_data(
			const ProblemData<L, D> &part, std::vector<D> &residuals,
			std::vector<D> & buffer_residuals, mpi::communicator &world) {
		for (int i = 0; i < part.m; i++) {
			buffer_residuals[i] = 0; //FIXME use CBLAS
		}
		for (L col = 0; col < part.n; col++) {
			for (L row_tmp = part.A_csc_col_ptr[col];
					row_tmp < part.A_csc_col_ptr[col + 1]; row_tmp++) {
				buffer_residuals[part.A_csc_row_idx[row_tmp]] -=
						part.b[part.A_csc_row_idx[row_tmp]]
								* part.A_csc_values[row_tmp] * part.x[col];
			}
		}
		vall_reduce(world, buffer_residuals, residuals);
	}

	static inline D compute_fast_objective(const ProblemData<L, D> &part,
			const std::vector<D> &residuals, mpi::communicator &world) {
		D resids = 0;
		D sumx = 0;
		if (world.rank() == 0) {
			for (L i = 0; i < part.m; i++) {
				D tmp = (residuals[i] + 1);
				if (tmp > 0)
					resids += tmp * tmp;
			}
		}
		for (L j = 0; j < part.n; j++) {
			sumx += abs(part.x[j]);
		}
		D sumxOut = 0;
		reduce(world, sumx, sumxOut, std::plus<D>(), 0);
		return 0.5 * part.lambda * resids + sumxOut;
	}
};

#include "loss_logistic.h"

template<typename L, typename D>
class DistributedLosses<L, D, logistics_loss_traits> {
public:

	static inline void bulkIterations_for_my_part_data(
			const ProblemData<L, D> &part, std::vector<D> &residuals,
			std::vector<D> & buffer_residuals, mpi::communicator &world) {
		for (int i = 0; i < part.m; i++) {
			buffer_residuals[i] = 0; //FIXME use CBLAS
		}
		for (L col = 0; col < part.n; col++) {
			for (L row_tmp = part.A_csc_col_ptr[col];
					row_tmp < part.A_csc_col_ptr[col + 1]; row_tmp++) {
				buffer_residuals[part.A_csc_row_idx[row_tmp]] -=
						part.b[part.A_csc_row_idx[row_tmp]]
								* part.A_csc_values[row_tmp] * part.x[col];
			}
		}
		vall_reduce(world, buffer_residuals, residuals);
	}

	static inline D compute_fast_objective(const ProblemData<L, D> &part,
			const std::vector<D> &residuals, mpi::communicator &world) {
		D resids = 0;
		D sumx = 0;
		if (world.rank() == 0) {
			for (L i = 0; i < part.m; i++) {
				resids += log(1 + exp(residuals[i]));
			}
		}
		for (L j = 0; j < part.n; j++) {
			sumx += abs(part.x[j]);
		}
		//		printf("LOGISTIC OBJECTIVE  %f  %f\n",resids,0.5 * inst.lambda * resids + sumx);
		D sumxOut = 0;
		reduce(world, sumx, sumxOut, std::plus<D>(), 0);
		return 0.5 * part.lambda * resids + sumxOut;
	}

};

#include "loss_hinge_dual.h"

template<typename L, typename D>
class DistributedLosses<L, D, loss_hinge_dual> {
public:

	static inline void bulkIterations_for_my_part_data(
			ProblemData<L, D> &part, std::vector<D> &residuals,
			std::vector<D> & buffer_residuals, mpi::communicator &world) {
		for (int i = 0; i < part.m; i++) {
			buffer_residuals[i] = 0; //FIXME use CBLAS
		}
		const D delta = 1 / (part.lambda * part.total_n + 0.0);
		for (L sample = 0; sample < part.n; sample++) {
			if (part.x[sample] > 1)
				part.x[sample] = 1;
			else if (part.x[sample] < 0)
				part.x[sample] = 0;

			for (L tmp = part.A_csr_row_ptr[sample];
					tmp < part.A_csr_row_ptr[sample + 1]; tmp++) {
				buffer_residuals[part.A_csr_col_idx[tmp]] += delta
						* part.b[sample] * part.A_csr_values[tmp]
						* part.x[sample];
			}
		}
		vall_reduce(world, buffer_residuals, residuals);
	}

	static inline D compute_fast_objective(const ProblemData<L, D> &part,
			const std::vector<D> &residuals, mpi::communicator &world) {
		D resids = 0;
		D sumx = 0;
		if (world.rank() == 0) {
			for (L i = 0; i < part.m; i++) {
				resids += log(1 + exp(residuals[i]));
			}
		}
		for (L j = 0; j < part.n; j++) {
			sumx += abs(part.x[j]);
		}
		//		printf("LOGISTIC OBJECTIVE  %f  %f\n",resids,0.5 * inst.lambda * resids + sumx);
		D sumxOut = 0;
		reduce(world, sumx, sumxOut, std::plus<D>(), 0);
		return 0.5 * part.lambda * resids + sumxOut;
	}

	static inline D compute_fast_objective(ProblemData<L, D> &part,
			const ProblemData<L, D> &part_local,
			const std::vector<D> &residuals,
			const std::vector<D> &residuals_local, mpi::communicator &world) {
		D resids = 0;
		D sumLoss = 0;
		D sumX = 0;
		if (world.rank() == 0) {
			for (L i = 0; i < part.m; i++) {
				resids += residuals[i] * residuals[i];
			}
		}

//		for (L i = 0; i < part_local.m; i++) {
//			resids += residuals_local[i] * residuals_local[i];
//		}
		L good = 0;
		part.dualObjective = 0;
		for (L j = 0; j < part.n; j++) {

			D error = 0;
			for (L i = part.A_csr_row_ptr[j]; i < part.A_csr_row_ptr[j + 1];
					i++) {
				error += part.A_csr_values[i]
						* residuals[part.A_csr_col_idx[i]];
			}

			if (part.b[j] * error > 0) {
				good++;
			}

			error = 1 - part.b[j] * error;

			if (error < 0) {
				error = 0;
			}
			sumLoss += error;
			sumX += part.x[j];
		}

		D sumxLossOut = 0;
		reduce(world, sumLoss, sumxLossOut, std::plus<D>(), 0);
		D sumXOut = 0;
		reduce(world, sumX, sumXOut, std::plus<D>(), 0);
		L sumxGood = 0;
		reduce(world, good, sumxGood, std::plus<L>(), 0);

		if (world.rank() == 0) {




			part.oneZeroAccuracy = sumxGood / (0.0 + part.total_n);
			part.primalObjective = part.lambda * 0.5 * resids
					+ sumxLossOut / (0.0 + part.total_n);
			part.dualObjective = -part.lambda * 0.5 * resids
					+ sumXOut / (0.0 + part.total_n);

			cout << "XXXX "<< sumxLossOut<<"  "<< part.total_n<<"  "<<sumXOut<<"  "<<resids<<"   FINAL " << part.primalObjective - part.dualObjective  <<endl;
		}
		return part.primalObjective - part.dualObjective;
//		part.oneZeroAccuracy =
//		part.primalObjective = part.lambda * 0.5 * resids
//				+ (sumxOut + part.dualObjective) / (0.0 + part.total_n);
//		part.dualObjective = -part.lambda * 0.5 * resids
//				+ part.dualObjective / (0.0 + part.total_n);

//		D residOut = 0;
//		reduce(world, resids, residOut, std::plus<D>(), 0);
//		cout << "XXXXXXXXXX   " << 1 / (0.0 + part.total_n) * sumxOut << endl;
//		return part.lambda * resids + 1 / (0.0 + part.m) * sumxOut;
//		return part.lambda * resids + 1 / (0.0 + part.total_n) * sumxOut;
	}

};

#endif /* DISTRIBUTED_LOSSES_H */
