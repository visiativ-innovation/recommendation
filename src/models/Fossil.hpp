#pragma once

#include "model.hpp"

class Fossil : public model
{
public:
	Fossil( corpus* corp, int L, int K, double lambda, double bias_reg, int dataset_factor)
			: model(corp)
			, L(L)
			, K(K)
			, lambda(lambda)
			, bias_reg(bias_reg)
			, dataset_factor(dataset_factor) {}

	Fossil( corpus* corp, int L, int K, double lambda, double bias_reg)
			: model(corp)
			, L(L)
			, K(K)
			, lambda(lambda)
			, bias_reg(bias_reg)
			, dataset_factor(10) {}

	~Fossil(){}

	void init();
	void cleanUp();

	double prediction(int user, int prev_item, int item) { return -1; }
	double prediction(int user, vector<int>& prev_items, int item);
	void getParametersFromVector(	double*   g,
									double**  beta,
									double**  WT,
									double*** WTu,
									double*** U,
									double*** V,
									action_t  action);

	int sampleUser();
	void train(int iterations, double learn_rate);
	void oneiteration(double learn_rate);
	void updateFactors(int user_id, vector<int>& prev_items, int pos_item_id, int neg_item_id, double learn_rate);

	void sampleAUC(double* AUC_val, double* AUC_test, double* var);
	void AUC(double* AUC_val, double* AUC_test,
				  double* HIT_val, double* HIT_test,
				  double* MRR_val, double* MRR_test,
				  double* var);
	string toString();

	/* parameters */
	double* beta;
	double** U;
	double** V;
	double* WT;
	double** WTu;

	/* hyper-parameters */
	int L;
	int K;
	double lambda;
	double bias_reg;
	int dataset_factor;

	/* helper */
	vector<int>* user_matrix;
};
