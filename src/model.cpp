#include "model.hpp"

void model::AUC(double* AUC_val, double* AUC_test,
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

	// #pragma omp parallel for schedule(dynamic)
	for (int u = 0; u < nUsers; u ++) {
		int item_test 		= test_per_user[u].first;
		int item_test_prev  = test_per_user[u].second;
		int item_val  		= val_per_user[u].first;
		int item_val_prev 	= val_per_user[u].second;

		if (item_test == -1) {  // invalid user
			continue;
		}

		double x_u_test = prediction(u, item_test_prev, item_test);
		double x_u_val  = prediction(u, item_val_prev, item_val);

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
			double x_uj = prediction(u, item_val_prev, j);
			if (x_u_val > x_uj) {
				count_val ++;
			}

			x_uj = prediction(u, item_test_prev, j);
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

void model::sampleAUC(double* AUC_val, double* AUC_test, double* var)
{
    vector<double> AUC_u_val(nUsers, -1);
    vector<double> AUC_u_test(nUsers, -1);

    // #pragma omp parallel for schedule(dynamic)
    for (int u = 0; u < nUsers; u ++) {
        int item_test       = test_per_user[u].first;
        int item_test_prev  = test_per_user[u].second;
        int item_val        = val_per_user[u].first;
        int item_val_prev   = val_per_user[u].second;

        if (item_test == -1) {  // invalid user
            continue;
        }

        double x_u_test = prediction(u, item_test_prev, item_test);
        double x_u_val  = prediction(u, item_val_prev, item_val);

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
            double x_uj = prediction(u, item_val_prev, j);
            if (x_u_val > x_uj) {
                count_val ++;
            }

            x_uj = prediction(u, item_test_prev, j);
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

void model::copyBestModel()
{
	for (int w = 0; w < NW; w ++) {
		bestW[w] = W[w];
	}
}

void model::saveModel(const char* path)
{
	FILE* f = fopen_(path, "w");
	fprintf(f, "{\n");
	fprintf(f, "  \"NW\": %d,\n", NW);

	fprintf(f, "  \"W\": [");
	for (int w = 0; w < NW; w ++) {
		fprintf(f, "%f", bestW[w]);
		if (w < NW - 1) fprintf(f, ", ");
	}
	fprintf(f, "]\n");
	fprintf(f, "}\n");
	fclose(f);

	printf("\nModel saved to %s.\n", path);
}

/// model must be first initialized before calling this function
void model::loadModel(const char* path)
{
	printf("\n  loading parameters from %s.\n", path);
	ifstream in;
	in.open(path);
	if (! in.good()){
		printf("Can't read init solution from %s.\n", path);
		exit(1);
	}
	string line;
	string st;
	char ch;
	while(getline(in, line)) {
		stringstream ss(line);
		ss >> st;
		if (st == "\"NW\":") {
			int nw;
			ss >> nw;
			if (nw != NW) {
				printf("NW not match.");
				exit(1);
			}
			continue;
		}

		if (st == "\"W\":") {
			ss >> ch; // skip '['
			for (int w = 0; w < NW; w ++) {
				if (! (ss >> W[w] >> ch)) {
					printf("Read W[] error.");
					exit(1);
				}
			}
			break;
		}
	}
	in.close();
}

string model::toString()
{
	return "Empty Model!";
}

void model::MultipleMetrics(int topN, bool sample, double& valid)
{
	vector<double> val_AUC_u(nUsers, -1);
	vector<double> val_MRR_u(nUsers, -1);
	vector<int>    val_HIT_u(nUsers, -1);
	vector<double> test_AUC_u(nUsers, -1);
	vector<double> test_MRR_u(nUsers, -1);
	vector<int>    test_HIT_u(nUsers, -1);

	// #pragma omp parallel for schedule(dynamic)
	for (int u = 0; u < nUsers; u ++) {
		int item_test 		= test_per_user[u].first;
		int item_test_prev  = test_per_user[u].second;

		int item_val  		= val_per_user[u].first;
		int item_val_prev 	= val_per_user[u].second;

		if (item_test == -1) {  // invalid user
			continue;
		}

		if (sample && nUsers > 50000 && rand() % nUsers > 50000) { // if nUser is large then sample around 50000 users
			continue;
		}

		double x_u_test = prediction(u, item_test_prev, item_test);
		double x_u_val  = prediction(u, item_val_prev, item_val);

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

			double x_uj = prediction(u, item_val_prev, j);
			if (x_u_val > x_uj) {
				count_val ++;
			}

			x_uj = prediction(u, item_test_prev, j);
			if (x_u_test > x_uj) {
				count_test ++;
			}
		}

		test_AUC_u[u] = 1.0 * count_test / max;
		int rank_pos = max - count_test + 1;
		test_MRR_u[u] = 1.0 / rank_pos;
		test_HIT_u[u] = (rank_pos <= topN);

		val_AUC_u[u] = 1.0 * count_val / max;
		rank_pos = max - count_val + 1;
		val_MRR_u[u] = 1.0 / rank_pos;
		val_HIT_u[u] = (rank_pos <= topN);
	}

	// sum up AUC
	double test_auc = 0, test_mrr = 0, test_hr = 0;
	double val_auc = 0, val_mrr = 0, val_hr = 0;
	int num_user = 0;
	for (int u = 0; u < nUsers; u ++) {
		if (test_AUC_u[u] != -1) {
			test_auc += test_AUC_u[u];
			test_mrr += test_MRR_u[u];
			test_hr  += test_HIT_u[u];

			val_auc += val_AUC_u[u];
			val_mrr += val_MRR_u[u];
			val_hr  += val_HIT_u[u];

			num_user ++;
		}
	}
	test_auc /= num_user;
	test_mrr /= num_user;
	test_hr  /= num_user;

	val_auc /= num_user;
	val_mrr /= num_user;
	val_hr  /= num_user;

	printf("\n\n#Users = %d, VALID: AUC = %f, MRR = %f, HR@%d = %f\n", num_user, val_auc, val_mrr, topN, val_hr);
	printf("#Users = %d, TEST:  AUC = %f, MRR = %f, HR@%d = %f\n", num_user, test_auc, test_mrr, topN, test_hr);

	valid = val_hr; // return hit rate on the validation set
}
