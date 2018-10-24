#include "FossilSimple.hpp"

void FossilSimple::init()
{
	NW = 1 + nUsers + nItems + 2 * nItems * K;
	W = new double [NW];
	bestW = new double [NW];

	getParametersFromVector(W, &alpha, &alpha_u, &beta, &U, &V, INIT);

	for (int w = 0; w < NW; w ++) {
		W[w] = rand() * 0.000002 / RAND_MAX - 0.000001;
	}

	for (int i = 0; i < nItems; i ++) {
		beta[i] = 0;
	}

	// working memory
	user_matrix = new vector<int> [nUsers];
	for (int u = 0; u < nUsers; u ++) {
		for (vector<pair<int,int> >::iterator it = corp->pos_per_user[u].begin(); it != corp->pos_per_user[u].end(); it ++) {
			user_matrix[u].push_back(it->first);
		}
	}
}

void FossilSimple::cleanUp()
{
	getParametersFromVector(W, &alpha, &alpha_u, &beta, &U, &V, FREE);

	delete [] W;
	delete [] bestW;
	delete [] user_matrix;
}

void FossilSimple::getParametersFromVector(	double*   g,
											double**  alpha,
											double**  alpha_u,
											double**  beta,
											double*** U,
											double*** V,
											action_t  action)
{
	if (action == FREE) {
		delete [] (*U);
		delete [] (*V);
		return;
	}

	if (action == INIT)	{
		*U = new double* [nItems];
		*V = new double* [nItems];
	}

	int ind = 0;

	*alpha = g + ind;
	ind += 1;

	*alpha_u = g + ind;
	ind += nUsers;

	*beta = g + ind;
	ind += nItems;

	for (int i = 0; i < nItems; i ++) {
		(*U)[i] = g + ind;
		ind += K;
	}
	for (int i = 0; i < nItems; i ++) {
		(*V)[i] = g + ind;
		ind += K;
	}

	if (ind != NW) {
		printf("Got bad index (FossilSimple.cpp, line %d)", __LINE__);
		exit(1);
	}
}

double FossilSimple::prediction(int user, int item_prev, int item)
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

	double pred = beta[item];
	pred += wu * inner(sum_pos, V[item], K);
	pred += (*alpha + alpha_u[user]) * inner(U[item_prev], V[item], K);

	test = 1;
	if(test == 0) {
		string name;
		std::cout << "beta[item] : " << beta[item] << "\n";
		std::cout << "wu : " << wu << "\n";
		std::cout << "inner(sum_pos, V[item], K) : " << inner(sum_pos, V[item], K) << "\n";

		std::cout << "(*alpha) : " << (*alpha) << "\n";
		std::cout << "(alpha_u[user]) : " << (alpha_u[user]) << "\n";

		std::cout << "\n" << "U[item_prev] : " << "\n";
		for (int k = 0; k < K; k ++) {
			std::cout << U[item_prev][k] << " - "; // Long term + short term pour j sans inner join
		}
		std::cout << "\n" << "U[item_prev] : " << "\n";

		std::cout << "\n" << "V[item] : " << "\n";
		for (int k = 0; k < K; k ++) {
			std::cout << V[item][k] << " - "; // Long term + short term pour j sans inner join
		}
		std::cout << "\n" << "V[item] : " << "\n";

		std::cout << "inner(U[item_prev], V[item], K) : " << inner(U[item_prev], V[item], K)<< "\n";

		std::cout << "\n" << "pred : " << pred<< "\n";
		getline (std::cin, name);
		printf("\n");
		test = 1;
	}


	delete [] sum_pos;
	return pred;
}

int FossilSimple::sampleUser()
{
	while (true) {
		int user_id = rand() % nUsers;
		if (corp->pos_per_user[user_id].size() < 2) {
			continue;
		}
		return user_id;
	}
}

void FossilSimple::updateFactors(int user_id, int item_id, int pos_item_id, int neg_item_id, double learn_rate)
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
		sum_pos[k] = wu_pos * sum_pos[k] + (*alpha + alpha_u[user_id]) * U[item_id][k];
		sum_neg[k] = wu_neg * sum_neg[k] + (*alpha + alpha_u[user_id]) * U[item_id][k];
	}

	double x_uij = beta[pos_item_id] - beta[neg_item_id];
	x_uij += inner(sum_pos, V[pos_item_id], K) - inner(sum_neg, V[neg_item_id], K);
	double deri = 1 / (1 + exp(x_uij));

	beta[pos_item_id] += learn_rate * ( deri - bias_reg * beta[pos_item_id]);
	beta[neg_item_id] += learn_rate * (-deri - bias_reg * beta[neg_item_id]);

	double prev_alpha = *alpha + alpha_u[user_id];

	// note the here the discount factors for learning rate may need to be tuned
	// to achieve the best performance on the specific dataset
	*alpha += learn_rate / 10 * (deri * (inner(U[item_id], V[pos_item_id], K) - inner(U[item_id], V[neg_item_id], K)) - lambda / 10 * (*alpha));
	alpha_u[user_id] += learn_rate * (deri * (inner(U[item_id], V[pos_item_id], K) - inner(U[item_id], V[neg_item_id], K)) - lambda * alpha_u[user_id]);

	for (vector<int>::iterator it = user_matrix[user_id].begin(); it != user_matrix[user_id].end(); it ++) {
		if (*it == item_id) {
			// if(*it != pos_item_id) {
				for (int k = 0; k < K; k ++) {
					U[item_id][k] += learn_rate * (deri * ( (wu_pos + prev_alpha) * V[pos_item_id][k] - (wu_neg + prev_alpha) * V[neg_item_id][k]) - lambda * U[item_id][k]);
				}
			// }else {
				// for (int k = 0; k < K; k ++) {
					// U[item_id][k] += learn_rate * (deri * ( (prev_alpha) * V[pos_item_id][k] - (wu_neg + prev_alpha) * V[neg_item_id][k]) - lambda * U[item_id][k]);
				// }
			// }
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

void FossilSimple::oneiteration(double learn_rate)
{
	// working memory
	vector<pair<int,int> >* matrix = new vector<pair<int,int> > [nUsers];
	for (int u = 0; u < nUsers; u ++) {
		for (int i = 0; i < (int)corp->pos_per_user[u].size() - 1; i ++) {
			matrix[u].push_back(make_pair(corp->pos_per_user[u][i].first, corp->pos_per_user[u][i + 1].first));
		}
	}

	// now it begins!
	// #pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < num_pos_events; i ++) {
		int user_id, item_id, pos_item_id, neg_item_id;

		// sample user
		user_id = sampleUser();
		vector<pair<int,int> >& user_items = matrix[user_id];

		// sample positive item
		int rand_num = rand() % user_items.size();
		item_id = user_items.at(rand_num).first;
		pos_item_id = user_items.at(rand_num).second;

		// sample negative item
		do {
			neg_item_id = rand() % nItems;
		} while (neg_item_id == pos_item_id || neg_item_id == item_id);

		// now got tuple (user_id, pos_item, neg_item)
		updateFactors(user_id, item_id, pos_item_id, neg_item_id, learn_rate);
	}

	delete [] matrix;
}

void FossilSimple::train(int iterations, double learn_rate)
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
		printf("------------ alpha[u_x] = %f\n", alpha[rand() % nUsers]);  // In case if some parameters have blowed up
		printf("Iter: %d, took %f\n", iter, clock_() - l_dlStart);
		fflush(stdout);

		if(iter % 50 == 0) {
			test = 0;
			double valid, test, var;
			sampleAUC(&valid, &test, &var);
			printf("[Valid AUC = %f], Test AUC = %f, Test var = %f\n", valid, test, var);
			fflush(stdout);

			if (bestValidAUC < valid) {
				bestValidAUC = valid;
				best_iter = iter;
				copyBestModel();
			} else if (iter > best_iter + 500) {
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

string FossilSimple::toString()
{
	char str[100];
	sprintf(str, "FossilSimple__K_%d_lambda_%.4f_biasReg_%.4f", K, lambda, bias_reg);
	return str;
}
