#include "FPMC.hpp"

void FPMC::init()
{
	NW = K * nUsers + (K + 2 * KK) * nItems;
	W = new double [NW];
	bestW = new double [NW];

	getParametersFromVector(W, &gamma_user, &gamma_item, &kappa_item, &eta_item, INIT);

	for (int w = 0; w < NW; w ++) {
		W[w] = rand() * 1.0 / RAND_MAX;
	}
}

void FPMC::cleanUp()
{
	getParametersFromVector(W, &gamma_user, &gamma_item, &kappa_item, &eta_item, FREE);

	delete [] W;
	delete [] bestW;
}

void FPMC::getParametersFromVector(	double*   g,
									double*** gamma_user,
									double*** gamma_item,
									double*** kappa_item,
									double*** eta_item,
									action_t  action)
{
	if (action == FREE) {
		delete [] (*gamma_user);
		delete [] (*gamma_item);
		delete [] (*kappa_item);
		delete [] (*eta_item);
		return;
	}

	if (action == INIT)	{
		*gamma_user = new double* [nUsers];
		*gamma_item = new double* [nItems];
		*kappa_item = new double* [nItems];
		*eta_item = new double* [nItems];
	}

	int ind = 0;

	for (int u = 0; u < nUsers; u ++) {
		(*gamma_user)[u] = g + ind;
		ind += K;
	}
	for (int i = 0; i < nItems; i ++) {
		(*gamma_item)[i] = g + ind;
		ind += K;
	}
	for (int i = 0; i < nItems; i ++) {
		(*kappa_item)[i] = g + ind;
		ind += KK;
	}
	for (int i = 0; i < nItems; i ++) {
		(*eta_item)[i] = g + ind;
		ind += KK;
	}

	if (ind != NW) {
		printf("Got bad index (FPMC.cpp, line %d)", __LINE__);
		exit(1);
	}
}

double FPMC::prediction(int user, int item_prev, int item)
{
	return  inner(gamma_user[user], gamma_item[item], K)
			+ inner(kappa_item[item_prev], eta_item[item], KK);
}

int FPMC::sampleUser()
{
	while (true) {
		int user_id = rand() % nUsers;
		if (corp->pos_per_user[user_id].size() < 2) {
			continue;
		}
		return user_id;
	}
}

void FPMC::updateFactors(int user_id, int item_id, int pos_item_id, int neg_item_id, double learn_rate)
{
	double* gpos_minus_gneg = new double [K];
	double* epos_minus_eneg = new double [KK];

	for (int k = 0; k < K; k ++) {
		gpos_minus_gneg[k] = gamma_item[pos_item_id][k] - gamma_item[neg_item_id][k];
	}
	for (int k = 0; k < KK; k ++) {
		epos_minus_eneg[k] = eta_item[pos_item_id][k] - eta_item[neg_item_id][k];
	}

	double x_uij  = inner(gamma_user[user_id], gpos_minus_gneg, K) + inner(kappa_item[item_id], epos_minus_eneg, KK);
	double deri = 1.0 / (1.0 + exp(x_uij));

	// adjust latent factors
	for (int f = 0; f < K; f ++) {
		double deri_A = deri * gamma_user[user_id][f];

		gamma_user[user_id][f]     += learn_rate * ( deri * gpos_minus_gneg[f] - lambda * gamma_user[user_id][f]);
		gamma_item[pos_item_id][f] += learn_rate * ( deri_A - lambda * gamma_item[pos_item_id][f]);
		gamma_item[neg_item_id][f] += learn_rate * (-deri_A - lambda * gamma_item[neg_item_id][f]);
	}

	for (int f = 0; f < KK; f ++) {
		double deri_B = deri * kappa_item[item_id][f];

		kappa_item[item_id][f]   += learn_rate * ( deri * epos_minus_eneg[f] - lambda * kappa_item[item_id][f]);
		eta_item[pos_item_id][f] += learn_rate * ( deri_B - lambda * eta_item[pos_item_id][f]);
		eta_item[neg_item_id][f] += learn_rate * (-deri_B - lambda * eta_item[neg_item_id][f]);
	}

	delete [] gpos_minus_gneg;
	delete [] epos_minus_eneg;
}

void FPMC::oneiteration(double learn_rate)
{
	// working memory
	vector<pair<int,int> >* user_matrix = new vector<pair<int,int> > [nUsers];
	for (int u = 0; u < nUsers; u ++) {
		for (int i = 0; i < (int)corp->pos_per_user[u].size() - 1; i ++) {
			user_matrix[u].push_back(make_pair(corp->pos_per_user[u][i].first, corp->pos_per_user[u][i + 1].first));
		}
	}

	// now it begins!
	// #pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < num_pos_events; i ++) {
		int user_id, item_id, pos_item_id, neg_item_id;

		// sample user
		user_id = sampleUser();
		vector<pair<int,int> >& user_items = user_matrix[user_id];

		// sample positive item
		int rand_num = rand() % user_items.size();
		item_id = user_items.at(rand_num).first;
		pos_item_id = user_items.at(rand_num).second;

		// sample negative item
		do {
			neg_item_id = rand() % nItems;
		} while (neg_item_id == pos_item_id);

		// now got tuple (user_id, pos_item, neg_item)
		updateFactors(user_id, item_id, pos_item_id, neg_item_id, learn_rate);
	}

	delete [] user_matrix;
}

void FPMC::train(int iterations, double learn_rate)
{
	printf("%s", ("\n<<< " + toString() + " >>>\n\n").c_str());

	double bestValidAUC = -1;
	int best_iter = 0;

	double timeToTrain = clock_();
	// SGD begins
	for (int iter = 1; iter <= iterations; iter ++) {

		// perform one iter of SGD
		double l_dlStart = clock_();
		oneiteration(learn_rate);
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

string FPMC::toString()
{
	char str[100];
	sprintf(str, "FPMC__K_%d_KK_%d_lambda_%f", K, KK, lambda);
	return str;
}
