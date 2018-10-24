#include "Fossil.hpp"

void Fossil::init()
{
	NW = 2 * nItems * K + L + nItems + L * nUsers;
	W = new double [NW];
	bestW = new double [NW];

	getParametersFromVector(W, &beta, &WT, &WTu, &U, &V, INIT);

	for (int w = 0; w < NW; w ++) {
		W[w] = rand() * 0.000002 / RAND_MAX - 0.000001;
	}

	// working memory
	user_matrix = new vector<int> [nUsers];
	for (int u = 0; u < nUsers; u ++) {
		for (vector<pair<int,int> >::iterator it = corp->pos_per_user[u].begin(); it != corp->pos_per_user[u].end(); it ++) {
			user_matrix[u].push_back(it->first);
		}
	}
}

void Fossil::cleanUp()
{
	getParametersFromVector(W, &beta, &WT, &WTu, &U, &V, FREE);

	delete [] W;
	delete [] bestW;
}

void Fossil::getParametersFromVector(	double*   g,
										double**  beta,
										double**  WT,
										double*** WTu,
										double*** U,
										double*** V,
										action_t  action)
{
	if (action == FREE) {
		delete [] (*U);
		delete [] (*V);
		delete [] (*WTu);
		return;
	}

	if (action == INIT)	{
		*U = new double* [nItems];
		*V = new double* [nItems];
		*WTu = new double* [nUsers];
	}

	int ind = 0;

	*beta = g + ind;
	ind += nItems;

	*WT = g + ind;
	ind += L;

	for (int i = 0; i < nUsers; i ++) {
		(*WTu)[i] = g + ind;
		ind += L;
	}

	for (int i = 0; i < nItems; i ++) {
		(*U)[i] = g + ind;
		ind += K;
	}
	for (int i = 0; i < nItems; i ++) {
		(*V)[i] = g + ind;
		ind += K;
	}

	if (ind != NW) {
		printf("Got bad index (Fossil.cpp, line %d)", __LINE__);
		exit(1);
	}
}

double Fossil::prediction(int user, vector<int>& prev_items, int item)
{
	double* sum_pos = new double [K];
	for (int k = 0; k < K; k ++) {
		sum_pos[k] = 0;
	}

	int cnt = 0;
	for (vector<int>::iterator it = user_matrix[user].begin(); it != user_matrix[user].end(); it ++) {
		if (*it != item) {
			for (int k = 0; k < K; k ++) {
				sum_pos[k] += U[*it][k];
			}
			cnt ++;
		}
	}

	double wu = cnt > 0 ? pow(cnt, -0.2) : 0;

	for (int k = 0; k < K; k ++) {
		sum_pos[k] *= wu;
	}

	for (int ind = 0; ind < (int)prev_items.size(); ind ++) {
		int cur_prev_item = prev_items[ind];
		if (cur_prev_item != item) {
			for (int k = 0; k < K; k ++) {
				sum_pos[k] += (WT[ind] + WTu[user][ind]) * U[cur_prev_item][k];
			}
		}
	}

	double pred = beta[item] + inner(sum_pos, V[item], K);

	delete [] sum_pos;
	return pred;
}

int Fossil::sampleUser()
{
	while (true) {
		int user_id = rand() % nUsers;
		if (corp->pos_per_user[user_id].size() < 2) {
			continue;
		}
		return user_id;
	}
}

void Fossil::updateFactors(int user_id, vector<int>& prev_items, int pos_item_id, int neg_item_id, double learn_rate)
{
	double* sum_pos = new double [K];
	double* sum_neg = new double [K];

	for (int k = 0; k < K; k ++) {
		sum_pos[k] = sum_neg[k] = 0;
	}

	int cnt_pos = 0, cnt_neg = 0;
	for (vector<int>::iterator it = user_matrix[user_id].begin(); it != user_matrix[user_id].end(); it ++) {
		if (*it != pos_item_id) {
			for (int k = 0; k < K; k ++) {
				sum_pos[k] += U[*it][k];
			}
			cnt_pos ++;
		}
		if (*it != neg_item_id) {
			for (int k = 0; k < K; k ++) {
				sum_neg[k] += U[*it][k];
			}
			cnt_neg ++;
		}
	}

	double wu_pos = cnt_pos > 0 ? pow(cnt_pos, -0.2) : 0;
	double wu_neg = cnt_neg > 0 ? pow(cnt_neg, -0.2) : 0;

	for (int k = 0; k < K; k ++) {
		sum_pos[k] *= wu_pos;
		sum_neg[k] *= wu_neg;
	}

	for (int ind = 0; ind < (int)prev_items.size(); ind ++) {
		int cur_prev_item = prev_items[ind];

		if (cur_prev_item != pos_item_id) {
			for (int k = 0; k < K; k ++) {
				sum_pos[k] += (WT[ind] + WTu[user_id][ind]) * U[cur_prev_item][k];
			}
		}
		if (cur_prev_item != neg_item_id) {
			for (int k = 0; k < K; k ++) {
				sum_neg[k] += (WT[ind] + WTu[user_id][ind]) * U[cur_prev_item][k];
			}
		}
	}

	double x_uij = beta[pos_item_id] - beta[neg_item_id];
	x_uij += inner(sum_pos, V[pos_item_id], K) - inner(sum_neg, V[neg_item_id], K);
	double deri = 1 / (1 + exp(x_uij));

	beta[pos_item_id] += learn_rate * ( deri - bias_reg * beta[pos_item_id]);
	beta[neg_item_id] += learn_rate * (-deri - bias_reg * beta[neg_item_id]);

	vector<double> wts;

	for (int ind = 0; ind < (int)prev_items.size(); ind ++) {
		wts.push_back(WT[ind] + WTu[user_id][ind]);

		int cur_prev_item = prev_items[ind];

		// note the here the discount factors for learning rate may need to be tuned
		// to achieve the best performance on the specific dataset, e.g. changing it from 2 to 10
		if (cur_prev_item != pos_item_id) {
			if (cur_prev_item != neg_item_id) {
				double diff = inner(U[cur_prev_item], V[pos_item_id], K) - inner(U[cur_prev_item], V[neg_item_id], K);
				WT[ind]           += learn_rate / dataset_factor * (deri * diff - lambda / dataset_factor * WT[ind]);
				WTu[user_id][ind] += learn_rate * (deri * diff - lambda * WTu[user_id][ind]);
			} else {
				WT[ind]           += learn_rate / dataset_factor * (deri * inner(U[cur_prev_item], V[pos_item_id], K) - lambda / dataset_factor * WT[ind]);
				WTu[user_id][ind] += learn_rate * (deri * inner(U[cur_prev_item], V[pos_item_id], K) - lambda * WT[ind]);
			}
		} else {
			WT[ind]           += learn_rate / dataset_factor * (-deri * inner(U[cur_prev_item], V[neg_item_id], K) - lambda / dataset_factor * WT[ind]);
			WTu[user_id][ind] += learn_rate * (-deri * inner(U[cur_prev_item], V[neg_item_id], K) - lambda * WT[ind]);
		}
	}

	for (vector<int>::iterator it = user_matrix[user_id].begin(); it != user_matrix[user_id].end(); it ++) {
		vector<int>::iterator pos = find(prev_items.begin(), prev_items.end(), *it);

		if (pos != prev_items.end()) {
			int pt = pos - prev_items.begin();

			if (*it != pos_item_id) {
				if (*it != neg_item_id) {
					for (int k = 0; k < K; k ++) {
						U[*it][k] += learn_rate * (deri * ( (wu_pos + wts[pt]) * V[pos_item_id][k] - (wu_neg + wts[pt]) * V[neg_item_id][k]) - lambda * U[*it][k]);
					}
				} else {
					for (int k = 0; k < K; k ++) {
						U[neg_item_id][k] += learn_rate * (deri * (wu_pos + wts[pt]) * V[pos_item_id][k] - lambda * U[neg_item_id][k]);
					}
				}
			} else {
				for (int k = 0; k < K; k ++) {
					U[pos_item_id][k] += learn_rate * (-deri * (wu_neg + wts[pt]) * V[neg_item_id][k] - lambda * U[pos_item_id][k]);
				}
			}
		} else {
			if (*it != pos_item_id) {
				if (*it != neg_item_id) {
					for (int k = 0; k < K; k ++) {
						U[*it][k] += learn_rate * (deri * (wu_pos * V[pos_item_id][k] - wu_neg * V[neg_item_id][k]) - lambda * U[*it][k]);
					}
				} else {
					for (int k = 0; k < K; k ++) {
						U[neg_item_id][k] += learn_rate * (deri * wu_pos * V[pos_item_id][k] - lambda * U[neg_item_id][k]);
					}
				}
			} else {
				for (int k = 0; k < K; k ++) {
					U[pos_item_id][k] += learn_rate * (-deri * wu_neg * V[neg_item_id][k] - lambda * U[pos_item_id][k]);
				}
			}
		}
	}

	for (int k = 0; k < K; k ++) {
		V[pos_item_id][k] += learn_rate * ( deri * sum_pos[k] - lambda * V[pos_item_id][k]);
		V[neg_item_id][k] += learn_rate * (-deri * sum_neg[k] - lambda * V[neg_item_id][k]);
	}

	delete [] sum_pos;
	delete [] sum_neg;
}

void Fossil::oneiteration(double learn_rate)
{
	printf("decay_wt_0 = %f, decay_wt_1 = %f\n", WT[0], WT[1]);

	// working memory
	vector<pair<int, vector<int> > >* matrix = new vector<pair<int, vector<int> > > [nUsers];
	for (int u = 0; u < nUsers; u ++) {
		vector<pair<int,int> >& user_pos = corp->pos_per_user[u];
		for (int i = 1; i < (int)user_pos.size(); i ++) {
			vector<int> prev_items;
			for (int j = i - 1; j >= max(0, i - L); j --) {
				prev_items.push_back(user_pos[j].first);
			}
			matrix[u].push_back(make_pair(user_pos[i].first, prev_items));
		}
	}

	// now it begins!
	// #pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < num_pos_events; i ++) {
		int user_id, pos_item_id, neg_item_id;
		vector<int> prev_items;

		// sample user
		user_id = sampleUser();
		vector<pair<int, vector<int> > >& user_items = matrix[user_id];

		// sample positive item
		int rand_num = rand() % user_items.size();
		pos_item_id = user_items.at(rand_num).first;
		prev_items = user_items.at(rand_num).second;

		// sample negative item
		do {
			neg_item_id = rand() % nItems;
		} while (neg_item_id == pos_item_id || find(prev_items.begin(), prev_items.end(), neg_item_id) != prev_items.end());

		// now got tuple (user_id, pos_item, neg_item)
		updateFactors(user_id, prev_items, pos_item_id, neg_item_id, learn_rate);
	}

	delete [] matrix;
}

void Fossil::train(int iterations, double learn_rate)
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
		// cout << "Eta : ";
		// for(int i = 0; i<L; i++){
		// 	cout << WT[i] + WTu[0][i] << " - ";
		// }
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

string Fossil::toString()
{
	char str[100];
	sprintf(str, "Fossil__L_%d_K_%d_lambda_%.2f_biasReg_%.2f", L, K, lambda, bias_reg);
	return str;
}

// void Fossil::AUC(double* AUC_val, double* AUC_test, double* var)
void Fossil::AUC(double* AUC_val, double* AUC_test,
			  double* HIT_val, double* HIT_test,
			  double* MRR_val, double* MRR_test,
			  double* var)
{
	vector<double> AUC_u_val(nUsers, -1);
	vector<double> AUC_u_test(nUsers, -1);
	vector<double> val_MRR_u(nUsers, -1);
	vector<int>    val_HIT_u(nUsers, -1);
	vector<double> test_MRR_u(nUsers, -1);
	vector<int>    test_HIT_u(nUsers, -1);
	//#pragma omp parallel for schedule(dynamic)
	for (int u = 0; u < nUsers; u ++) {
		int item_test = test_per_user[u].first;
		int item_val  = val_per_user[u].first;

		if (item_test == -1) {  // invalid user
			continue;
		}

		vector<pair<int,int> >& user_pos = corp->pos_per_user[u];

		vector<int> test_prev_items;
		test_prev_items.push_back(test_per_user[u].second);
		for (int i = 1; i <= L - 1; i ++) {
			if ((int)user_pos.size() - i < 0) {
				break;
			}
			test_prev_items.push_back(user_pos[user_pos.size() - i].first);
		}

		vector<int> val_prev_items;
		for (int i = 1; i <= L; i ++) {
			if ((int)user_pos.size() - i < 0) {
				break;
			}
			val_prev_items.push_back(user_pos[user_pos.size() - i].first);
		}

		double x_u_test = prediction(u, test_prev_items, item_test);
		double x_u_val  = prediction(u, val_prev_items, item_val);

		int count_val = 0;
		int count_test = 0;
		int max = 0;
		for (int j = 0; j < nItems; j ++) {
			if (clicked_per_user[u].find(j) != clicked_per_user[u].end()
				|| j == item_test
				|| j == item_val) {
				continue;
			}
			max ++;
			double x_uj = prediction(u, val_prev_items, j);
			if (x_u_val > x_uj) {
				count_val ++;
			}

			x_uj = prediction(u, test_prev_items, j);
			if (x_u_test > x_uj) {
				count_test ++;
			}
		}
		AUC_u_val[u] = 1.0 * count_val / max;
		AUC_u_test[u] = 1.0 * count_test / max;

		int rank_pos = max - count_test + 1;
		test_MRR_u[u] = 1.0 / rank_pos;
		test_HIT_u[u] = (rank_pos <= 50);

		rank_pos = max - count_val + 1;
		val_MRR_u[u] = 1.0 / rank_pos;
		val_HIT_u[u] = (rank_pos <= 50);
	}

	// sum up AUC
	*AUC_val = 0;
	*AUC_test = 0;
	*HIT_val = 0;
	*HIT_test = 0;
	*MRR_val = 0;
	*MRR_test = 0;
	int num_user = 0;
	for (int u = 0; u < nUsers; u ++) {
		if (AUC_u_test[u] != -1) {
			*AUC_val += AUC_u_val[u];
			*AUC_test += AUC_u_test[u];
			*MRR_test += test_MRR_u[u];
			*HIT_test  += test_HIT_u[u];
			*MRR_val += val_MRR_u[u];
			*HIT_val  += val_HIT_u[u];

			num_user ++;
		}
	}
	*AUC_val /= num_user;
	*AUC_test /= num_user;

	*MRR_test /= num_user;
	*MRR_val /= num_user;

	*HIT_test /= num_user;
	*HIT_val /= num_user;

	// calculate standard deviation
	double variance = 0;
	for (int u = 0; u < nUsers; u ++) {
		if (AUC_u_test[u] != -1) {
			variance += square(AUC_u_test[u] - *AUC_test);
		}
	}
	*var = variance / num_user;
}

void Fossil::sampleAUC(double* AUC_val, double* AUC_test, double* var)
{
	vector<double> AUC_u_val(nUsers, -1);
	vector<double> AUC_u_test(nUsers, -1);

	//#pragma omp parallel for schedule(dynamic)
	for (int u = 0; u < nUsers; u ++) {
		int item_test = test_per_user[u].first;
		int item_val  = val_per_user[u].first;

		if (item_test == -1) {  // invalid user
			continue;
		}

		vector<pair<int,int> >& user_pos = corp->pos_per_user[u];

		vector<int> test_prev_items;
		test_prev_items.push_back(test_per_user[u].second);
		for (int i = 1; i <= L - 1; i ++) {
			if ((int)user_pos.size() - i < 0) {
				break;
			}
			test_prev_items.push_back(user_pos[user_pos.size() - i].first);
		}

		vector<int> val_prev_items;
		for (int i = 1; i <= L; i ++) {
			if ((int)user_pos.size() - i < 0) {
				break;
			}
			val_prev_items.push_back(user_pos[user_pos.size() - i].first);
		}

		double x_u_test = prediction(u, test_prev_items, item_test);
		double x_u_val  = prediction(u, val_prev_items, item_val);

		int count_val = 0;
		int count_test = 0;
		int max = 0;
		for (int ind = 0; ind < 2000; ind ++) {
			int j = rand() % nItems;

			if (clicked_per_user[u].find(j) != clicked_per_user[u].end()
				|| j == item_test
				|| j == item_val) {
				continue;
			}
			max ++;
			double x_uj = prediction(u, val_prev_items, j);
			if (x_u_val > x_uj) {
				count_val ++;
			}

			x_uj = prediction(u, test_prev_items, j);
			if (x_u_test > x_uj) {
				count_test ++;
			}
		}
		AUC_u_val[u] = 1.0 * count_val / max;
		AUC_u_test[u] = 1.0 * count_test / max;
	}

	// sum up AUC
	*AUC_val = 0;
	*AUC_test = 0;
	int num_user = 0;
	for (int u = 0; u < nUsers; u ++) {
		if (AUC_u_test[u] != -1) {
			*AUC_val += AUC_u_val[u];
			*AUC_test += AUC_u_test[u];
			num_user ++;
		}
	}
	*AUC_val /= num_user;
	*AUC_test /= num_user;

	// calculate standard deviation
	double variance = 0;
	for (int u = 0; u < nUsers; u ++) {
		if (AUC_u_test[u] != -1) {
			variance += square(AUC_u_test[u] - *AUC_test);
		}
	}
	*var = variance / num_user;
}
