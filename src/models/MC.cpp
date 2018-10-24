#include "MC.hpp"

void MC::init()
{
	item_to_vec = vector<vector<int> >(nItems, vector<int>());
	item_to_set = vector<set<int> >(nItems, set<int>());

	for (int u = 0; u < nUsers; u ++) {
		for (int i = 0; i < (int)corp->pos_per_user[u].size() - 1; i ++) {
			int from = corp->pos_per_user[u][i].first;
			int to = corp->pos_per_user[u][i + 1].first;

			item_to_vec[from].push_back(to);
			item_to_set[from].insert(to);
		}
	}

	// for training iterations
	num_pos_trans = 0;
	for (int i = 0; i < nItems; i ++) {
		num_pos_trans += item_to_vec[i].size();
	}

	NW = 2 * K * nItems;
	W = new double [NW];
	bestW = new double [NW];

	getParametersFromVector(W, &gamma_item, &eta_item, INIT);

	for (int w = 0; w < NW; w ++) {
		W[w] = rand() * 1.0 / RAND_MAX;
	}
}

void MC::cleanUp()
{
	getParametersFromVector(W, &gamma_item, &eta_item, FREE);

	delete [] W;
	delete [] bestW;
}

void MC::getParametersFromVector(	double*   g,
									double*** gamma_item,
									double*** eta_item,
									action_t  action)
{
	if (action == FREE) {
		delete [] (*gamma_item);
		delete [] (*eta_item);
		return;
	}

	if (action == INIT)	{
		*gamma_item = new double* [nItems];
		*eta_item = new double* [nItems];
	}

	int ind = 0;
	for (int i = 0; i < nItems; i ++) {
		(*gamma_item)[i] = g + ind;
		ind += K;
	}
	for (int i = 0; i < nItems; i ++) {
		(*eta_item)[i] = g + ind;
		ind += K;
	}

	if (ind != NW) {
		printf("Got bad index (MC.cpp, line %d)", __LINE__);
		exit(1);
	}
}

double MC::prediction(int user, int item_prev, int item)
{
	return inner(gamma_item[item_prev], eta_item[item], K);
}

int MC::sampleItem()
{
	while (true) {
		int item_id = rand() % nItems;
		if (item_to_set[item_id].size() == 0 || (int) item_to_set[item_id].size() == nItems) {
			continue;
		}
		return item_id;
	}
}

void MC::updateFactors(int item_id, int pos_item_id, int neg_item_id, double learn_rate)
{
	double* pos_minus_neg = new double [K];
	for (int k = 0; k < K; k ++) {
		pos_minus_neg[k] = eta_item[pos_item_id][k] - eta_item[neg_item_id][k];
	}

	double x_uij  = inner(gamma_item[item_id], pos_minus_neg, K);
	double deri = 1.0 / (1.0 + exp(x_uij));

	// adjust latent factors
	for (int f = 0; f < K; f ++) {
		double deri_A = deri * gamma_item[item_id][f];
		gamma_item[item_id][f]   += learn_rate * ( deri * pos_minus_neg[f] - lambda * gamma_item[item_id][f]);
		eta_item[pos_item_id][f] += learn_rate * ( deri_A - lambda * eta_item[pos_item_id][f]);
		eta_item[neg_item_id][f] += learn_rate * (-deri_A - lambda * eta_item[neg_item_id][f]);
	}

	delete [] pos_minus_neg;
}

void MC::oneiteration(double learn_rate)
{
	int item_id, pos_item_id, neg_item_id;

	// working memory
	vector<vector<int> > matrix = vector<vector<int> > (nItems, vector<int>());
	for (int i = 0; i < nItems; i ++) {
		for (vector<int>::iterator it = item_to_vec[i].begin(); it != item_to_vec[i].end(); it ++) {
			matrix[i].push_back(*it);
		}
	}

	// now it begins!
	for (int i = 0; i < num_pos_trans; i ++) {

		// sample user
		item_id = sampleItem();
		vector<int>& items = matrix[item_id];

		// reset user if already exhausted
		if (items.size() == 0) {
			for (vector<int>::iterator it = item_to_vec[item_id].begin(); it != item_to_vec[item_id].end(); it ++) {
				items.push_back(*it);
			}
		}

		// sample positive item
		int rand_num = rand() % items.size();
		pos_item_id = items.at(rand_num);
		items.at(rand_num) = items.back();
		items.pop_back();

		// sample negative item
		do {
			neg_item_id = rand() % nItems;
		} while (item_to_set[item_id].find(neg_item_id) != item_to_set[item_id].end());

		// now got tuple (user_id, pos_item, neg_item)
		updateFactors(item_id, pos_item_id, neg_item_id, learn_rate);
	}
}

void MC::train(int iterations, double learn_rate)
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
			// MultipleMetrics(50);
			double valid, test, var;
			sampleAUC(&valid, &test, &var);
			printf("[Valid AUC = %f], Test AUC = %f, Test var = %f\n", valid, test, var);
			fflush(stdout);

			if (bestValidAUC < valid) {
				bestValidAUC = valid;
				best_iter = iter;
				copyBestModel();
			} else if (iter > best_iter + 1000) {
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

	// double valid, test, var;
	// AUC(&valid, &test, &var);
	// printf("\n\n <<< %s >>> Test AUC = %f, Test var = %f\n", toString().c_str(), test, var);

	double AUC_val, AUC_test;
	double HIT_val, HIT_test;
	double MRR_val, MRR_test;
	double var;
	AUC(&AUC_val, &AUC_test, &HIT_val, &HIT_test, &MRR_val, &MRR_test, &var);

	printf("\n\n <<< %s >>> Test AUC = %f, Val AUC = %f, Test var = %f\n", toString().c_str(), AUC_test, AUC_val, var);
	printf("\n\n <<< %s >>> Test HIT50 = %f, Val HIT50 = %f\n", toString().c_str(), HIT_test, HIT_val);
	printf("\n\n <<< %s >>> Test MRR = %f, Val MRR = %f\n", toString().c_str(), MRR_test, MRR_val);

}

string MC::toString()
{
	char str[100];
	sprintf(str, "MC__K_%d_lambda_%f", K, lambda);
	return str;
}
