#pragma once

#include "model.hpp"

class BPRMF : public model
{
public:
	BPRMF(corpus* corp, int K, double lambda, double bias_reg)
		: model(corp)
		, K(K)
		, lambda(lambda)
		, bias_reg(bias_reg) {}

	~BPRMF(){}

	void init();
	void cleanUp();

	double prediction(int user, int item_prev, int item);
	void getParametersFromVector(	double*   g,
									double**  beta_item,
									double*** gamma_user,
									double*** gamma_item,
									action_t action);

	int sampleUser();
	void train(int iterations, double learn_rate);
	void oneiteration(double learn_rate);
	virtual void updateFactors(int user_id, int pos_item_id, int neg_item_id, double learn_rate);
	string toString();

	void normalization(double** M, int ind);

	/* auxiliary variables */
	double*  beta_item;
	double** gamma_user;
	double** gamma_item;

	/* hyper-parameters */
	int K;
	double lambda;
	double bias_reg;
};
