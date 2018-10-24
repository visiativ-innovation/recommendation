#pragma once

#include "common.hpp"
#include "corpus.hpp"


enum action_t { COPY, INIT, FREE };

class model
{
public:
	model(corpus* corp) : corp(corp)
	{
		nUsers = corp->nUsers;
		nItems = corp->nItems;
		nClicks = corp->nClicks;

		// necessary for sampling negative items
		clicked_per_user = vector<unordered_set<int> >(nUsers, unordered_set<int>());

		// leave out the last`two' for each user
		for (int u = 0; u < nUsers; u ++) {
			if (corp->pos_per_user[u].size() < 3) {
				// printf("  Warning: user %d has only %d clicks. \n", u, (int)corp->pos_per_user[u].size());
				test_per_user.push_back(make_pair(-1, -1));
				teststamp_per_user.push_back(make_pair(-1, -1));
				val_per_user.push_back(make_pair(-1, -1));
			} else {
				int item_test = corp->pos_per_user[u].back().first;
				int test_stamp = corp->pos_per_user[u].back().second;
				corp->pos_per_user[u].pop_back();
				int item_val = corp->pos_per_user[u].back().first;
				int val_stamp = corp->pos_per_user[u].back().second;
				corp->pos_per_user[u].pop_back();
				int item_prev = corp->pos_per_user[u].back().first;

				test_per_user.push_back(make_pair(item_test, item_val));
				teststamp_per_user.push_back(make_pair(test_stamp, val_stamp));
				val_per_user.push_back(make_pair(item_val, item_prev));
			}

			for (vector<pair<int,int> >::iterator it = corp->pos_per_user[u].begin(); it != corp->pos_per_user[u].end(); it ++) {
				clicked_per_user[u].insert(it->first);
			}
		}

		// calculate num_pos_events
		num_pos_events = 0;
		for (int u = 0; u < nUsers; u ++) {
			num_pos_events += corp->pos_per_user[u].size();
		}
		std::cout << "num_pos_events :" << num_pos_events << "\n\n";
	}

	~model()
	{
	}

	/* Model parameters */
	int NW; // Total number of parameters
	double* W; // Contiguous version of all parameters
	double* bestW;

	/* Corpus related */
	corpus* corp; // dangerous
	int nUsers; // Number of users
	int nItems; // Number of items
	int nClicks; // Number of ratings

	vector<unordered_set<int> > clicked_per_user;
	vector<pair<int,int> > val_per_user;
	vector<pair<int,int> > test_per_user;
	vector<pair<int,int> > teststamp_per_user;

	int num_pos_events;

	virtual void sampleAUC(double* AUC_val, double* AUC_test, double* var);
	virtual void AUC(double* AUC_val, double* AUC_test,
				  double* HIT_val, double* HIT_test,
				  double* MRR_val, double* MRR_test,
				  double* var);

	virtual void copyBestModel();
	virtual void saveModel(const char* path);
	virtual void loadModel(const char* path);
	virtual string toString();

	void MultipleMetrics(int topN, bool sample, double& valid);

private:
	virtual double prediction(int user, int item_prev, int item) = 0;
};
