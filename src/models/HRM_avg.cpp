#include "HRM_avg.hpp"


void HRM_avg::init()
{
	// total number of parameters
	NW = (nItems + nUsers) * K;

	// Initialize parameters and latent variables
	W = new double [NW];

	parametersToFlatVector(W, &U, &V, INIT);

	// randomized initialization
	for (int i = 0; i < NW; i++) {
		W[i] = (2.0 * rand() / RAND_MAX - 1.0) / K;
	}

	bestW = new double [NW];
}

void HRM_avg::cleanUp()
{
	parametersToFlatVector(0, &U, &V, FREE);

	delete [] W;
	delete [] bestW;
}

void HRM_avg::parametersToFlatVector(	double*    g,
										double***  U,
										double***  V,
										action_t   action)
{
	if (action == FREE) {
		delete [] *U;
		delete [] *V;
		return;
	}

	if (action == INIT) {
		*U = new double* [nUsers];
		*V = new double* [nItems];
	}

	int ind = 0;

	for (int n = 0; n < nUsers; n ++) {
		(*U)[n] = g + ind;
		ind += K;
	}

	for (int k = 0; k < nItems; k ++) {
		(*V)[k] = g + ind;
		ind += K;
	}

	if (ind != NW) {
		printf("Got incorrect index (%d != %d) at line %d of HRM_avg.cpp\n", ind, NW, __LINE__);
		exit(1);
	}
}

double HRM_avg::prediction(int user, int item_prev, int item)
{
	double pred = 0;
	for (int k = 0; k < K; k ++) {
		pred += (U[user][k] + V[item_prev][k]) * V[item][k]; // avg pooling
	}
	return pred;
}

void HRM_avg::train(int iterations, double learn_rate)
{
	printf("%s", ("\n<<< " + toString() + " >>>\n\n").c_str());

	double bestValidAUC = -1;
	int best_iter = 0;

	double timeToTrain = clock_();
	// SGD begins
	for (int iter = 1; iter <= iterations; iter ++) {

		// perform one iter of SGD
		double l_dlStart = clock_();
		oneIteration(learn_rate);
		printf("Iter: %d, took %f\n", iter, clock_() - l_dlStart);
		fflush(stdout);

		if(iter % 50 == 0) {
			double valid, test, var;
			sampleAUC(&valid, &test, &var);
			printf("[Valid AUC = %f], Test AUC = %f, Test var = %f\n", valid, test, var);
			fflush(stdout);

			if (bestValidAUC < valid) {
				bestValidAUC = valid;
				best_iter = iter;
				copyBestModel();
			} else if (iter > best_iter + 300) {
				printf("Overfitted. Exiting... \n");
				break;
			}
		}
	}

	timeToTrain = clock_() - timeToTrain;

	// copy back best parameters
	for (int w = 0; w < NW; w ++) {
		W[w] = bestW[w];
	}

	double AUC_val, AUC_test;
	double HIT_val, HIT_test;
	double MRR_val, MRR_test;
	double var;
	AUC(&AUC_val, &AUC_test, &HIT_val, &HIT_test, &MRR_val, &MRR_test, &var);

	printf("\n\n <<< %s >>> Test AUC = %f, Val AUC = %f, Test var = %f\n", toString().c_str(), AUC_test, AUC_val, var);
	printf("\n\n <<< %s >>> Test HIT50 = %f, Val HIT50 = %f\n", toString().c_str(), HIT_test, HIT_val);
	printf("\n\n <<< %s >>> Test MRR = %f, Val MRR = %f\n", toString().c_str(), MRR_test, MRR_val);

}

int HRM_avg::sampleUser()
{
	while (true) {
		int user_id = rand() % nUsers;
		if (clicked_per_user[user_id].size() < 2) {
			continue;
		}
		return user_id;
	}
}

void HRM_avg::oneIteration(double learn_rate)
{
	// working memory
	vector<int>* user_matrix = new vector<int> [nUsers];
	for (int u = 0; u < nUsers; u ++) {
		for (int i = 0; i < (int)corp->pos_per_user[u].size(); i ++) {
			user_matrix[u].push_back(corp->pos_per_user[u][i].first);
		}
	}

	// now it begins!
	// #pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < num_pos_events; i ++) {
		int user_id, item_id, pos_item_id;

		// sample user
		user_id = sampleUser();
		vector<int>& user_items = user_matrix[user_id];

		// sample positive item
		int idx = rand() % (user_items.size() - 1);
		item_id = user_items[idx];
		pos_item_id = user_items[idx + 1]; // user_items[rand() % user_items.size()];

		// now got tuple (user_id, pos_item)
		updateFactors(user_id, item_id, pos_item_id, learn_rate);
	}

	delete [] user_matrix;
}

void HRM_avg::updateFactors(int user, int prev_item, int item, double learn_rate)
{
	// sample negative item
	int neg_item;
	do {
		neg_item = rand() % nItems;
	} while (neg_item == item);

	double* V_hybrid = new double [K];
	for (int k = 0; k < K; k ++) {
		V_hybrid[k] = (U[user][k] + V[prev_item][k]);
	}

	double pred = inner(V_hybrid, V[item], K) - inner(V_hybrid, V[neg_item], K);
	double deri = 1.0 / (1.0 + exp(pred));

	for (int k = 0; k < K; k ++) {
		double deri_k = deri * (V[item][k] - V[neg_item][k]);

		V[item][k]     += learn_rate * ( deri * V_hybrid[k] - lambda * V[item][k]);
		V[neg_item][k] += learn_rate * (-deri * V_hybrid[k] - lambda * V[neg_item][k]);

		U[user][k] += learn_rate * (deri_k - lambda * U[user][k]);
		V[prev_item][k] += learn_rate * (deri_k - lambda * V[prev_item][k]);
	}

	delete [] V_hybrid;
}

string HRM_avg::toString()
{
	char str[100];
	sprintf(str, "HRM_AvgPooling_BPR_K_%d_lambda_%f", K, lambda);
	return str;
}
