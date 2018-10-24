#pragma once

#include "model.hpp"

class REBUS : public model
{
public:
	REBUS(corpus* corp, int K, double lambda, double bias_reg, double alpha_up)
		: model(corp)
		, K(K)
		, lambda(lambda)
		, bias_reg(bias_reg)
		, type_seq("fsub")
		, find_path_stars(true)
		, nb_stars(0)
		, start_auc_test(49)
		, alpha_up(alpha_up) {}

	~REBUS(){}

	int loadPST(int minCount, int L, const char* data_path);
	void init();
	void cleanUp();
	void getParametersFromVector(	double*   g,
									double**  beta,
									double*** P,
									action_t action);

	vector<int> findPath(vector<int> prev_items);
	vector<int> findPathStars(vector<int> prev_items);

	double prediction(int user, int prev_item, int item) { return -1; }
	double prediction(int user, vector<int>& max_prev_items_list, vector<int>& prev_items_list, unordered_set<int>& prev_items_set, int item);

	int sampleUser();
	void train(int iterations, double learn_rate);
	void oneiteration(double learn_rate);
	virtual void updateFactors(int user_id, vector<int>& max_prev_items_list, vector<int>& prev_items_list, unordered_set<int>& prev_items_set, int pos_item_id, int neg_item_id, double learn_rate);
	string toString();

	void sampleAUC(double* AUC_val, double* AUC_test, double* var);
	void AUC(double* AUC_val, double* AUC_test,
				  double* HIT_val, double* HIT_test,
				  double* MRR_val, double* MRR_test,
				  double* var);

	/* auxiliary variables */
	double* beta;
	double** P;

	set<string> nodes_label;
	map<string,int> nodes_label_dict;

	/* hyper-parameters */
	int minCount;
	int K;
	int L;
	double lambda;
	double bias_reg;

	string type_seq;
	bool find_path_stars;
	int nb_stars;

	/* helper */
	vector<int>* user_matrix;
	vector<pair<int, pair<vector<int>, unordered_set<int>> > >* matrix;
	vector<pair<int, vector<int> > >* histo_user;
	int nb_items_plus_root = 0;
	int early_stopping = 300;
	int start_auc_test;
	double alpha_up;
	vector<vector<double>> eta_cumWeibullSoftmax;

};
