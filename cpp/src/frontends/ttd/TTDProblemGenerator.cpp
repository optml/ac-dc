/*
 * TTDProblemGenerator.cpp
 *
 *  Created on: Sep 9, 2013
 *      Author: taki
 */

#include "../../helpers/option_console_parser.h"
#include "../../problem_generator/ttd/TTDGenerator.h"
int main(int argc, char *argv[]) {
	Context ctx;
	consoleHelper::parseConsoleOptions(ctx, argc, argv);
	TTDGenerator<float> ttd(ctx);
	if (ctx.zDim > 0) {
		ttd.generate3DFixedPoints();
		ttd.generate3DProblem();
		ttd.get3DForceVector();
	} else {
		ttd.generate2DFixedPoints();
		ttd.generate2DProblem();
		ttd.get2DForceVector();
	}

	std::cout << "Problem generated" << std::endl;
	std::cout << "Data matrix size is: " << ttd.mOut << " x " << ttd.nOut
			<< std::endl;

	ttd.storeProblemIntoFile();

	return 0;
}

