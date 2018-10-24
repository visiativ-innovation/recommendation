#pragma once

#include "model.hpp"

class MC : public model
{
public:
	MC(corpus* corp, int K, double lambda)
		: model(corp)
		, K(K)
		, lambda(lambda) {}

	~MC(){}

	void init();
	void cleanUp();

	double prediction(int user, int item_prev, int item);
	void getParametersFromVector(	double*   g,
									double*** gamma_item,
									double*** eta_item,
									action_t action);

	int sampleItem();
	void train(int iterations, double learn_rate);
	void oneiteration(double learn_rate);
	virtual void updateFactors(int item_id, int pos_item_id, int neg_item_id, double learn_rate);
	string toString();

	/* variables */
	double** gamma_item;
	double** eta_item;

	vector<vector<int> > item_to_vec;
	vector<set<int> > item_to_set;

	/* hyper-parameters */
	int K;
	double lambda;

	int num_pos_trans;
};
