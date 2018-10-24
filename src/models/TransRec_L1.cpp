#include "TransRec_L1.hpp"

void TransRec_L1::init()
{
	// total number of parameters
	NW = K + nItems * K + K * nUsers + nItems;

	// Initialize parameters and latent variables
	W = new double [NW];

	parametersToFlatVector(W, &beta_item, &H, &R, &r, INIT);

	// randomized initialization
	double range = 6.0 / sqrt(K);
	for (int i = 0; i < NW; i++) {
		W[i] = range - 2 * range * rand() / RAND_MAX;
	}

	for (int i = 0; i < nItems; i ++) {
		normalization(H, i);
		beta_item[i] = 0;
	}

	for (int n = 0; n < nUsers; n ++) {
		for (int k = 0; k < K; k ++) {
			R[n][k] = 0;
		}
	}

	normalization(&r, 0);

	bestW = new double [NW];
}

void TransRec_L1::cleanUp()
{
	parametersToFlatVector(0, &beta_item, &H, &R, &r, FREE);

	delete [] W;
	delete [] bestW;
}

void TransRec_L1::parametersToFlatVector(	double*    g,
											double**   beta_item,
											double***  H,
											double***  R,
											double**   r,
											action_t   action)
{
	if (action == FREE) {
		delete [] *H;
		delete [] *R;
		return;
	}

	if (action == INIT) {
		*H = new double* [nItems];
		*R = new double* [nUsers];
	}

	int ind = 0;

	*beta_item = g + ind;
	ind += nItems;

	for (int k = 0; k < nItems; k ++) {
		(*H)[k] = g + ind;
		ind += K;
	}

	for (int n = 0; n < nUsers; n ++) {
		(*R)[n] = g + ind;
		ind += K;
	}

	*r = g + ind;
	ind += K;

	if (ind != NW) {
		printf("Got incorrect index (%d != %d) at line %d of TransRec_L1.cpp\n", ind, NW, __LINE__);
		exit(1);
	}
}

double TransRec_L1::prediction(int user, int item_prev, int item)
{
	double pred = beta_item[item];
	for (int k = 0; k < K; k ++) {
		pred += fabs(H[item_prev][k] + r[k] + R[user][k] - H[item][k]);
	}
	return -pred;
}

void TransRec_L1::train(int iterations, double learn_rate)
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

		// Observe if parameters are healthy
		if (iter % 50 == 0) {
			printf("r = ");
			for (int k = 0; k < K; k ++) {
				printf("%f  ", r[k]);
			}
			printf("\n");
			int rand_u = rand() % nUsers;
			printf("R[u] = ");
			for (int k = 0; k < K; k ++) {
				printf("%f  ", R[rand_u][k]);
			}
			printf("\n");
		}

		if(iter % 50 == 0) {
			double valid, test, var;
			sampleAUC(&valid, &test, &var);  // models are selected by AUC
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

int TransRec_L1::sampleUser()
{
	while (true) {
		int user_id = rand() % nUsers;
		if (clicked_per_user[user_id].size() < 2) {
			continue;
		}
		return user_id;
	}
}

void TransRec_L1::oneIteration(double learn_rate)
{
	// working memory
	vector<int>* user_matrix = new vector<int> [nUsers];
	for (int u = 0; u < nUsers; u ++) {
		for (int i = 0; i < (int)corp->pos_per_user[u].size(); i ++) {
			user_matrix[u].push_back(corp->pos_per_user[u][i].first);
		}
	}

	// now it begins!
	for (int i = 0; i < num_pos_events; i ++) {
		int user_id, item_id, pos_item_id, neg_item_id;

		// sample user
		user_id = sampleUser();
		vector<int>& user_items = user_matrix[user_id];

		// sample positive item
		int idx = rand() % (user_items.size() - 1);
		item_id = user_items[idx];
		pos_item_id = user_items[idx + 1]; // user_items[rand() % user_items.size()];

		// sample negative item
		do {
			neg_item_id = rand() % nItems;
		} while (clicked_per_user[user_id].find(neg_item_id) != clicked_per_user[user_id].end());

		// now got tuple (user_id, pos_item, neg_item)
		updateFactors(user_id, item_id, pos_item_id, neg_item_id, learn_rate);
	}

	delete [] user_matrix;
}

void TransRec_L1::updateFactors(int user, int x, int y, int yn, double learn_rate)
{
	double* v_x_y  = new double [K];
	double* v_x_yn = new double [K];

	for (int k = 0; k < K; k ++) {
		v_x_y[k]  = r[k] + R[user][k] + H[x][k] - H[y][k];
		v_x_yn[k] = r[k] + R[user][k] + H[x][k] - H[yn][k];
	}

	double z = -beta_item[y] + beta_item[yn];
	for (int k = 0; k < K; k ++) {
		z -= fabs(v_x_y[k]) - fabs(v_x_yn[k]);
	}

	double deri = 1 / (1 + exp(z));

	beta_item[y]   += learn_rate * (-deri - bias_reg * beta_item[y]);
	beta_item[yn]  += learn_rate * ( deri - bias_reg * beta_item[yn]);

	for (int k = 0; k < K; k ++) {
		double tmp = (v_x_y[k] > 0 ? -1 : 1) + (v_x_yn[k] > 0 ? 1 : -1);

		H[x][k]    += learn_rate * deri * tmp;
		H[y][k]    += learn_rate * deri * (v_x_y[k] > 0 ? 1 : -1);
		H[yn][k]   += learn_rate * deri * (v_x_yn[k] > 0 ? -1 : 1);
		r[k]       += learn_rate * (deri * tmp - lambda * r[k]);
		R[user][k] += learn_rate * (deri * tmp - relation_reg * R[user][k]);
	}

	// normalization
	normalization(H, x);
	normalization(H, y);
	normalization(H, yn);

	delete [] v_x_y;
	delete [] v_x_yn;
}

void TransRec_L1::normalization(double** M, int ind)
{
	double sum = 0;
	for (int k = 0; k < K; k ++) {
		sum += M[ind][k] * M[ind][k];
	}
	sum = sqrt(sum);

	if (sum > 1) {
		for (int k = 0; k < K; k ++) {
			M[ind][k] /= sum;
		}
	}
}

string TransRec_L1::toString()
{
    char str[100];
    sprintf(str, "TransRec_L1__K_%d_lambda_%f_relationReg_%f_biasReg_%f", K, lambda, relation_reg, bias_reg);
    return str;
}
