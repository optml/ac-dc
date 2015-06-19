/*
 * This is sample code which solve given data with LG and L2SVM objective with L1 regularization term
 * for different penalty parameters "C_parameters"
 */

#include "parseinputs.cpp"
#include "classifier_testing.h"
#include "../solver/svm/serial/random_L2SVM_L1Reg_Solver.h"
#include "../solver/svm/serial/random_LG_L1Reg_Solver.h"
using namespace std;

/**
 * Solve SVM Problem
 *
 * @param inputTrainFileName - file where Train data ist stored
 * @param nTrainSamples -  number of samples in Train dataset
 * @param inputTestFileName - file where Test data is stored
 * @param ntestsamples - number of samples in Test dataset
 * @param nfeatures - number of total features
 * @param nclasses - number of classes, currenlty only "2" is supported
 * @param nonzero_elements_of_input_data - maximal nonzero elements of input data
 *                                         this is due memmory allocation issue
 * @param settings - optimization settings
 * @param C_parameters - Vector of penalty parameters
 */
void solveSVMProblem(const char* inputTrainFileName, int nTrainSamples,
		const char* inputTestFileName, int ntestsamples, int nfeatures,
		int nclasses, int nonzero_elements_of_input_data,
		OptimizationSettings settings, std::vector<float> C_parameters) {
	//-------------------LOAD TRAIN DATA
	std::vector<float> h_Train_cscValA, h_Train_label;
	std::vector<int> h_Train_cscRowIndA, h_Train_cscColPtrA;
	int status = parseLibSVMdata(inputTrainFileName, &h_Train_cscValA,
			&h_Train_cscRowIndA, &h_Train_cscColPtrA, &h_Train_label,
			nTrainSamples, nfeatures, nclasses, nonzero_elements_of_input_data);
	if (status == 0) {
		return;
	}
	cout << "Train data loaded. NNZ = " << h_Train_cscValA.size() << "\n";
	//-------------------LOAD TEST DATA
	std::vector<float> h_Test_cscValA, h_Test_label;
	std::vector<int> h_Test_cscColPtrA, h_Test_cscRowIndA;
	status = parseLibSVMdata(inputTestFileName, &h_Test_cscValA,
			&h_Test_cscRowIndA, &h_Test_cscColPtrA, &h_Test_label, ntestsamples,
			nfeatures, nclasses, nonzero_elements_of_input_data);
	if (status == 0) {
		return;
	}
	cout << "Test data loaded. NNZ = " << h_Test_cscValA.size() << "\n";
	//================== Start solving ============================================
	std::vector<float> h_classifier(nfeatures, 0);
	for (int i = 0; i < C_parameters.size(); i++) {
		float C = C_parameters[i];
		//--------------Serial Random L2SVM-L1 Reg
		runSerialRandom_L2SVM_L1RegSolver(h_Train_cscValA, h_Train_cscRowIndA,
				h_Train_cscColPtrA, h_Train_label, C, settings, &h_classifier);
		float test_accuracy = testClassifierForSVM(h_classifier, h_Test_cscValA,
				h_Test_cscRowIndA, h_Test_cscColPtrA, h_Test_label);
		printf("SR-L2SVM_L1Reg:C=%f,TA=%f,ET=%f\n", C, test_accuracy,
				settings.total_execution_time);
		//--------------Serial Random LG-L1 Reg
		runSerialRandom_LG_L1Reg_Solver(h_Train_cscValA, h_Train_cscRowIndA,
				h_Train_cscColPtrA, h_Train_label, C, settings, &h_classifier);
		test_accuracy = testClassifierForSVM(h_classifier, h_Test_cscValA,
				h_Test_cscRowIndA, h_Test_cscColPtrA, h_Test_label);
		printf("SR-LG_L1Reg:C=%f,TA=%f,ET=%f\n", C, test_accuracy,
				settings.total_execution_time);

	}

}



