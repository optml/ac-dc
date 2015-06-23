/*
 * LossFunction.h
 *
 *  Created on: Dec 1, 2014
 *      Author: taki
 */

#ifndef LOSSFUNCTION_H_
#define LOSSFUNCTION_H_

template<typename L, typename D>
class LossFunction {
public:
	LossFunction(){

	}

	virtual D computeObjectiveValue(ProblemData<L, D> & instance,
			mpi::communicator & world, std::vector<D> & w, double &finalDualError,
			double &finalPrimalError){

	}


	virtual void init(ProblemData<L, D> & instance){

		}

	virtual void SDCA(ProblemData<L, D> &instance,
			std::vector<D> &deltaAlpha, std::vector<D> &w, std::vector<D> &deltaW,
			DistributedSettings & distributedSettings){

	}

	virtual void fast_SDCA(ProblemData<L, D> &instance,
			std::vector<D> &deltaAlpha, std::vector<D> &w, std::vector<D> &deltaW,
			std::vector<D> &zk, std::vector<D> &uk,
			std::vector<D> &yk, std::vector<D> &deltayk, std::vector<D> &Ayk, std::vector<D> &deltaAyk,
			D &theta, DistributedSettings & distributedSettings){

	}

	virtual void fulldescent(ProblemData<L, D> &instance,
			std::vector<D> &deltaAlpha, std::vector<D> &w, std::vector<D> &deltaW, D &dualobj,
			DistributedSettings & distributedSettings){

	}

};

#endif /* LOSSFUNCTION_H_ */
