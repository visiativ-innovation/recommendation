#include "model.hpp"

class HRM_avg : public model
{
public:
	HRM_avg( corpus* corp, int K, double lambda)
			: model(corp)
			, K(K)
			, lambda(lambda) {}

	~HRM_avg() {}

	void init();
	void cleanUp();

	double prediction(int user, int item_prev, int item);


	void parametersToFlatVector(double*    g,
								double***  U,
								double***  V,
								action_t   action);

	void train(int max_iter, double learn_rate);

	int sampleUser();
	void updateFactors(int user, int prev_item, int item, double learn_rate);
	void oneIteration(double learn_rate);
	string toString();


	double** V;  // Item embeddings
	double** U;  // User embeddings

	/* hyper-parameters */
	int K;
	double lambda;
};
