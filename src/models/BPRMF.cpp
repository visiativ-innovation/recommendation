#include "BPRMF.hpp"

void BPRMF::init()
{
	NW = K * (nUsers + nItems) + nItems; // bias and latent factors
	W = new double[NW];
	bestW = new double[NW];

	getParametersFromVector(W, &beta_item, &gamma_user, &gamma_item, INIT);

	for (int u = 0; u < nUsers; u ++) {
		for (int k = 0; k < K; k ++) {
			gamma_user[u][k] = rand() * 1.0 / RAND_MAX;
		}
	}
	for (int i = 0; i < nItems; i ++) {
		beta_item[i] = 0;
		for (int k = 0; k < K; k ++) {
			gamma_item[i][k] = rand() * 1.0 / RAND_MAX;
		}
	}
}

void BPRMF::cleanUp()
{
	getParametersFromVector(W, &beta_item, &gamma_user, &gamma_item, FREE);

	delete [] W;
	delete [] bestW;
}

void BPRMF::getParametersFromVector(	double*   g,
										double**  beta_item,
										double*** gamma_user,
										double*** gamma_item,
										action_t  action)
{
	if (action == FREE) {
		delete [] (*gamma_user);
		delete [] (*gamma_item);
		return;
	}

	if (action == INIT)	{
		*gamma_user = new double* [nUsers];
		*gamma_item = new double* [nItems];
	}

	int ind = 0;

	*beta_item = g + ind;
	ind += nItems;

	for (int u = 0; u < nUsers; u ++) {
		(*gamma_user)[u] = g + ind;
		ind += K;
	}
	for (int i = 0; i < nItems; i ++) {
		(*gamma_item)[i] = g + ind;
		ind += K;
	}

	if (ind != NW) {
		printf("Got bad index (BPRMF.cpp, line %d)", __LINE__);
		exit(1);
	}
}

double BPRMF::prediction(int user, int item_prev, int item)
{
	return beta_item[item] + inner(gamma_user[user], gamma_item[item], K);
}

int BPRMF::sampleUser()
{
	while (true) {
		int user_id = rand() % nUsers;
		if (clicked_per_user[user_id].size() == 0 || (int) clicked_per_user[user_id].size() == nItems) {
			continue;
		}
		return user_id;
	}
}

void BPRMF::updateFactors(int user_id, int pos_item_id, int neg_item_id, double learn_rate)
{
	double x_uij = beta_item[pos_item_id] - beta_item[neg_item_id];
	x_uij += inner(gamma_user[user_id], gamma_item[pos_item_id], K) - inner(gamma_user[user_id], gamma_item[neg_item_id], K);

	double deri = 1 / (1 + exp(x_uij));

	beta_item[pos_item_id] += learn_rate * (deri - bias_reg * beta_item[pos_item_id]);
	beta_item[neg_item_id] += learn_rate * (-deri - bias_reg * beta_item[neg_item_id]);

	// adjust latent factors
	for (int f = 0; f < K; f ++) {
		double w_uf = gamma_user[user_id][f];
		double h_if = gamma_item[pos_item_id][f];
		double h_jf = gamma_item[neg_item_id][f];

		gamma_user[user_id][f]     += learn_rate * ( deri * (h_if - h_jf) - lambda * w_uf);
		gamma_item[pos_item_id][f] += learn_rate * ( deri * w_uf - lambda * h_if);
		gamma_item[neg_item_id][f] += learn_rate * (-deri * w_uf - lambda / 10.0 * h_jf);
	}
}

void BPRMF::oneiteration(double learn_rate)
{
	// uniformally sample users in order to approximatelly optimize AUC for all users
	int user_id, pos_item_id, neg_item_id;

	// working memory
	vector<int>* user_matrix = new vector<int> [nUsers];
	for (int u = 0; u < nUsers; u ++) {
		for (vector<pair<int,int> >::iterator it = corp->pos_per_user[u].begin(); it != corp->pos_per_user[u].end(); it ++) {
			user_matrix[u].push_back(it->first);
		}
	}

	// now it begins!
	for (int i = 0; i < num_pos_events; i++) {

		// sample user
		user_id = sampleUser();
		vector<int>& user_items = user_matrix[user_id];

		// reset user if already exhausted
		if (user_items.size() == 0) {
			for (vector<pair<int,int> >::iterator it = corp->pos_per_user[user_id].begin(); it != corp->pos_per_user[user_id].end(); it ++) {
				user_items.push_back(it->first);
			}
		}

		// sample positive item
		int rand_num = rand() % user_items.size();
		pos_item_id = user_items.at(rand_num);
		user_items.at(rand_num) = user_items.back();
		user_items.pop_back();

		// sample negative item
		do {
			neg_item_id = rand() % nItems;
		} while (clicked_per_user[user_id].find(neg_item_id) != clicked_per_user[user_id].end());

		// now got tuple (user_id, pos_item, neg_item)
		updateFactors(user_id, pos_item_id, neg_item_id, learn_rate);
	}

	delete [] user_matrix;
}

void BPRMF::train(int iterations, double learn_rate)
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
			printf("[Valid AUC = %f], Test AUC = %f, Test Var = %f\n", valid, test, var);
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

string BPRMF::toString()
{
	char str[10000];
	sprintf(str, "BPR-MF__K_%d_lambda_%f_biasReg_%f", K, lambda, bias_reg);
	return str;
}
