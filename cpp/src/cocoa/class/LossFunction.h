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

	virtual void solveLocalProblem(ProblemData<L, D> &instance,
			std::vector<D> &deltaAlpha, std::vector<D> &w, std::vector<D> &deltaW,
			DistributedSettings & distributedSettings){

	}



};

#endif /* LOSSFUNCTION_H_ */
