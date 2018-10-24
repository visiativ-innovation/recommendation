#pragma once

#include "model.hpp"

class FossilSimple : public model
{
public:
	FossilSimple(	corpus* corp, int K, double lambda, double bias_reg)
					: model(corp)
					, K(K)
					, lambda(lambda)
					, bias_reg(bias_reg) {}

	~FossilSimple(){}

	void init();
	void cleanUp();

	double prediction(int user, int item_prev, int item);
	void getParametersFromVector(	double*   g,
									double**  alpha,
									double**  alpha_u,
									double**  beta,
									double*** U,
									double*** V,
									action_t  action);

	int sampleUser();
	void train(int iterations, double learn_rate);
	void oneiteration(double learn_rate);
	void updateFactors(int user_id, int item_id, int pos_item_id, int neg_item_id, double learn_rate);
	string toString();

	/* parameters */
	double* alpha;
	double* alpha_u;
	double* beta;
	double** U;
	double** V;

	/* hyper-parameters */
	int K;
	double lambda;
	double bias_reg;

	int test;

	/* helper */
	vector<int>* user_matrix;
};
