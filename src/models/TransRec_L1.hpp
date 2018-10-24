#pragma once

#include "model.hpp"

class TransRec_L1 : public model
{
public:
	TransRec_L1( corpus* corp, int K, double lambda, double relation_reg, double bias_reg)
				: model(corp)
				, K(K)
				, lambda(lambda)
				, relation_reg(relation_reg)
				, bias_reg(bias_reg) {}

	~TransRec_L1() {}

	void init();
	void cleanUp();

	void normalization(double** M, int ind);

	double prediction(int user, int item_prev, int item);


	void parametersToFlatVector(double*    g,
								double**   beta_item,
								double***  H,
								double***  R,
								double**   r,
								action_t   action);

	void train(int max_iter, double learn_rate);
	int sampleUser();
	void updateFactors(int user, int x, int y, int yn, double learn_rate);
	void oneIteration(double learn_rate);
	string toString();

	double** H;  // Item embeddings
	double** R;  // User embeddings
	double* r;   // Global embedding

	double*  beta_item;

	/* hyper-parameters */
	int K;
	double lambda;
	double relation_reg;
	double bias_reg;
};
