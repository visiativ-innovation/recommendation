#pragma once

#include "model.hpp"

class PRME : public model
{
public:
	PRME(corpus* corp, int K, int KK, double alpha, double lambda)
		: model(corp)
		, K(K)
		, KK(KK)
		, alpha(alpha)
		, lambda(lambda) {}

	~PRME(){}

	void buildPST();
	void init();
	void cleanUp();

	double prediction(int user, int item_prev, int item);
	void getParametersFromVector(	double*   g,
									double*** gamma_user,
									double*** gamma_item,
									double*** eta_item,
									action_t action);

	int sampleUser();
	void train(int iterations, double learn_rate);
	void oneiteration(double learn_rate);
	virtual void updateFactors(int user_id, int item_id, int pos_item_id, int neg_item_id, double learn_rate);
	string toString();

	/* auxiliary variables */
	double** gamma_user;
	double** gamma_item;
	double** eta_item;

	/* hyper-parameters */
	int K;
	int KK;
	double alpha;
	double lambda;
};
