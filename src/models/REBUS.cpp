#include "REBUS.hpp"

int REBUS::loadPST(int minCount, int L, const char* data_path){
	printf(" -- Load Sequences Started REBUS_K_%d_minCount_%d_L_%d_lambda_%f_biasReg_%f_typeSeq_%s_findPathStar_%d_alphaUP%f -- \n\n", K, minCount, L, lambda, bias_reg, type_seq.c_str(), find_path_stars, alpha_up);

	this->minCount = minCount;
	this->L = L;
	stringstream ss(data_path);
	string item;
	vector<string> splittedStrings;
	while (getline(ss, item, '/'))
	{
		splittedStrings.push_back(item);
	}

	// File name to load
	std::stringstream sss;
	if (type_seq.compare("fsub") == 0) {
		sss << "96-FSUB/" << splittedStrings[1].substr(0, splittedStrings[1].size()-4) << "_root_fsub_minCount_" << to_string(minCount) << "_L_" << to_string(L) << ".txt";
	} else if (type_seq.compare("fseq") == 0) {
		sss << "97-FSEQ/" << splittedStrings[1].substr(0, splittedStrings[1].size()-4) << "_root_fseq_minCount_" << to_string(minCount) << "_L_" << to_string(L) << ".txt";
	} else if (type_seq.compare("fseq_bide") == 0) {
		sss << "95-FSEQ_BIDE/" << splittedStrings[1].substr(0, splittedStrings[1].size()-4) << "_root_fseq_bide_minCount_" << to_string(minCount) << "_L_" << to_string(L) << ".txt";
	} else {
		sss << "98-PST/" << splittedStrings[1].substr(0, splittedStrings[1].size()-4) << "_root_minCount_" << to_string(minCount) << "_L_" << to_string(L) << ".txt";
	}

	std::cout << "ss : " << sss.str() << "\n";
	string clickFile = sss.str();
	printf("   -Nom created -- \n\n");

	// Loading File
	int nRead = 0;
	string line;

	igzstream in;
	in.open(clickFile.c_str());
	if (! in.good()) {
		printf("Can't read clicks from %s.\n\n\n\n", clickFile.c_str());
		return(1);
	}
	string node_label;
	while (getline(in, line)) {
		stringstream ss(line);
		ss >> node_label;

		nRead ++;
		if (nRead % 100000 == 0) {
			printf(".");
			fflush(stdout);
		}

		nodes_label.insert(node_label);
	}
	in.close();

	std::cout << "  size of nodes_label" << nodes_label.size() << '\n';
	printf("   --Load file ended -- \n\n");

	// Create nodes_label_dict
	int count_i = 0 ;
	int count_not_i = 0 ;
	int idx = nItems+1;
	for (set<string>::iterator it = nodes_label.begin(); it!=nodes_label.end(); ++it){
		if ((*it).find('-') != std::string::npos)
		{
			nodes_label_dict[*it] = idx++;
			count_not_i++;
		} else {
			if((*it).compare("Root") == 0){
				nodes_label_dict[*it] = nItems;
				count_not_i++;
			} else {
				nodes_label_dict[*it] = stoi(*it);
				count_i++;
			}
		}
	}

	std::cout << "  count_not_i :" << count_not_i << '\n';
	std::cout << "  count_i :" << count_i << '\n';
	std::cout << "  nItems " << nItems << '\n';
	nb_items_plus_root = nItems + 1;
	std::cout << "  nb_items_plus_root " << nb_items_plus_root << '\n';
	printf("    --nodes_label_dict created-- \n");
	return 0;
}


void REBUS::init()
{
	printf(" --Init Started REBUS_K_%d_minCount_%d_L_%d_lambda_%f_biasReg_%f_typeSeq_%s_findPathStar_%d_alphaUP%f -- \n\n", K, minCount, L, lambda, bias_reg, type_seq.c_str(), find_path_stars, alpha_up);

	NW = K * nb_items_plus_root + nb_items_plus_root;
	W = new double [NW];
	bestW = new double [NW];

	printf(" --NW : %d -- \n\n", NW);

	getParametersFromVector(W, &beta, &P, INIT);

	for (int w = 0; w < NW; w ++) {
		W[w] = rand() * 1.0 / RAND_MAX;
	}
	printf(" --Init done -- \n\n");

	// for(int i = 0; i<L; i++){ // Rebus Etas cumWeibull
	// 	cout << "cumWeibull(" << i << ") " << cumWeibull(i) << "\n";
	// 	// string name; getline (std::cin, name);  printf("\n");
	// }

	for(int i = 0; i<=L; i++){ // Rebus Etas cumWeibullSoftmax
		cout << "i :"<< i << "\n";
		eta_cumWeibullSoftmax.push_back(vector<double>());
		for(int y = 0 ; y < i; y++){
			eta_cumWeibullSoftmax[i].push_back(cumWeibull_softmax(y, i));
			cout << "eta_cumWeibullSoftmax[" << i << "][" << y << "] :" << eta_cumWeibullSoftmax[i][y] << "\n";
		}
		// string name; getline (std::cin, name); printf("\n");
	}
	printf(" --Eta done -- \n\n");

	// working memory
	user_matrix = new vector<int> [nUsers];
	for (int u = 0; u < nUsers; u ++) {
		for (vector<pair<int,int> >::iterator it = corp->pos_per_user[u].begin(); it != corp->pos_per_user[u].end(); it ++) {
			user_matrix[u].push_back(it->first);
		}
	}
	printf(" -- user_matrix created -- \n\n");

	matrix = new vector<pair<int, pair<vector<int>, unordered_set<int>> > > [nUsers];
	histo_user = new vector<pair<int, vector<int> > > [nUsers];
	for (int u = 0; u < nUsers; u ++) {
		vector<pair<int,int> >& user_pos = corp->pos_per_user[u];
		for (int i = 1; i < (int)user_pos.size(); i ++) { // ground-truth item at timestep i
			vector<int> prev_items;
			unordered_set<int> prev_items_set;
			for (int j =  0 ; j <=  i - 1 ; j ++) {
				prev_items.push_back(user_pos[j].first);
				prev_items_set.insert(user_pos[j].first);
			}
			vector<int> max_prev_items;
			if(find_path_stars){
				max_prev_items = findPathStars(prev_items);
			} else {
				max_prev_items = findPath(prev_items);
			}

			histo_user[u].push_back(make_pair(user_pos[i].first, max_prev_items));
			matrix[u].push_back(make_pair(user_pos[i].first, make_pair(prev_items, prev_items_set)));
		}
	}
	printf(" -- matrix & histo created -- \n\n");
	printf(" --Init REBUS ended -- \n\n");
}

void REBUS::cleanUp()
{
	getParametersFromVector(W, &beta, &P, FREE);

	delete [] W;
	delete [] bestW;
	delete [] user_matrix;
	delete [] matrix;
	delete [] histo_user;
}

void REBUS::getParametersFromVector(	double*   g,
									double**  beta,
									double*** P,
									action_t  action)
{
	printf(" --getParametersFromVector Started REBUS_K_%d_minCount_%d_L_%d_lambda_%f_biasReg_%f_typeSeq_%s_findPathStar_%d_alphaUP%f -- \n\n", K, minCount, L, lambda, bias_reg, type_seq.c_str(), find_path_stars, alpha_up);

	if (action == FREE) {
		delete [] (*P);
		return;
	}

	if (action == INIT)	{
		*P = new double* [nb_items_plus_root];
	}

	int ind = 0;

	*beta = g + ind;
	ind += nb_items_plus_root;

	for (int i = 0; i < nb_items_plus_root; i ++) {
		(*P)[i] = g + ind;
		ind += K;
	}
	printf(" -- P init -- \n\n");

	if (ind != NW) {
		printf("Got bad index (REBUS.cpp, line %d)", __LINE__);
		printf("ind : %d , NW : %d\n", ind, NW );
		exit(1);
	}
	printf(" --getParametersFromVector REBUS ended -- \n\n");

}

double REBUS::prediction(int user, vector<int>& max_prev_items_list, vector<int>& prev_items_list, unordered_set<int>& prev_items_set, int item)
{
	double* p_item = new double [K];

	for (int k = 0; k < K; k ++) {
		p_item[k] = P[item][k];
	}

	// ------ User Preference ------ //
	double* sum_long = new double [K];

	for (int k = 0; k < K; k ++) {
		sum_long[k] = 0;
	}

	int cnt = 0;

	for (unordered_set<int>::iterator it = prev_items_set.begin(); it != prev_items_set.end(); it ++) { // Rebus Fism Set On Histo On
		if (*it != item) {
			for (int k = 0; k < K; k ++) {
				sum_long[k] += P[*it][k];
			}
			cnt ++;
		}
	}

	double wu = cnt > 0 ? pow(cnt, alpha_up) : 0; // Rebus dataset_factor on

	for (int k = 0; k < K; k ++) {
		sum_long[k] *= wu;
	}

	// ------ Sequential Dynamics ------ //
	double* sum_short = new double [K];

	for (int k = 0; k < K; k ++) {
		sum_short[k] = 0;
	}

	for(int ind = 0 ; ind <(int)max_prev_items_list.size(); ind ++){
		int max_prev_items = max_prev_items_list[ind] ;
		for (int k = 0; k < K; k ++) {
			// sum_short[k] += (cumWeibull(ind)) * P[max_prev_items][k]; // Rebus Etas cumWeibull
			sum_short[k] += eta_cumWeibullSoftmax[(int)max_prev_items_list.size()][ind] * P[max_prev_items][k]; // Rebus Etas cumWeibullSoftmax
		}
	}

	double* dist = new double [K];
	double* dist_squared = new double [K];

	for (int k = 0; k < K; k ++) {
		dist[k] = sum_long[k] + sum_short[k] - p_item[k];
        dist_squared[k] = (dist[k] * dist[k]);
	}

	double distance = beta[item] + sum(dist_squared,K);

	delete []  dist;
	delete []  dist_squared;
	delete []  sum_long ;
	delete []  sum_short ;
	delete []  p_item ;

	return -distance;
}

int REBUS::sampleUser()
{
	while (true) {
		int user_id = rand() % nUsers;
		if (corp->pos_per_user[user_id].size() < 2) {
			continue;
		}
		return user_id;
	}
}

void REBUS::updateFactors(int user_id, vector<int>& max_prev_items_list, vector<int>& prev_items_list, unordered_set<int>& prev_items_set, int pos_item_id, int neg_item_id, double learn_rate)
{

	double* p_pos = new double [K];
	double* p_neg = new double [K];
	double* p_pos_minus_neg = new double [K];

	for (int k = 0; k < K; k ++) {
		p_pos[k] = P[pos_item_id][k];
		p_neg[k] = P[neg_item_id][k];
		p_pos_minus_neg[k] = p_pos[k] - p_neg[k];
	}

	// ------ User Preference ------ //
	double* sum_short_pos = new double [K];
	double* sum_short_neg = new double [K];

	for (int k = 0; k < K; k ++) {
		sum_short_pos[k] = sum_short_neg[k] = 0;
	}

	int cnt_pos = 0, cnt_neg = 0;

	for (unordered_set<int>::iterator it = prev_items_set.begin(); it != prev_items_set.end(); it ++) { // Rebus Fism Set On Histo On
		if (*it != pos_item_id) {
			for (int k = 0; k < K; k ++) {
				sum_short_pos[k] += P[*it][k] ;
			}
			cnt_pos ++;
		}
		if (*it != neg_item_id) {
			for (int k = 0; k < K; k ++) {
				sum_short_neg[k] += P[*it][k] ;
			}
			cnt_neg ++;
		}
	}

	double wu_pos = cnt_pos > 0 ? pow(cnt_pos, alpha_up) : 0; // Rebus Fism dataset_factor on
	double wu_neg = cnt_neg > 0 ? pow(cnt_neg, alpha_up) : 0; // Rebus Fism dataset_factor on

	for (int k = 0; k < K; k ++) {
		sum_short_pos[k] *= wu_pos;
		sum_short_neg[k] *= wu_neg;
	}

	// ------ Sequential Dynamics ------ //
	double* sum_long = new double [K];

	for (int k = 0; k < K; k ++) {
		sum_long[k] = 0;
	}

	for(int ind = 0 ; ind <(int)max_prev_items_list.size(); ind ++){
		int max_prev_items = max_prev_items_list[ind] ;
		for (int k = 0; k < K; k ++) {
			// sum_long[k] += (cumWeibull(ind)) * P[max_prev_items][k]; // Rebus Etas cumWeibull
			sum_long[k] += eta_cumWeibullSoftmax[(int)max_prev_items_list.size()][ind] * P[max_prev_items][k]; // Rebus Etas cumWeibullSoftmax
		}
	}

	double* dist_pos = new double [K];
	double* dist_pos_squared = new double [K];

	double* dist_neg = new double [K];
	double* dist_neg_squared = new double [K];

	for (int k = 0; k < K; k ++) {
        dist_pos[k] =  sum_short_pos[k] + sum_long[k] - p_pos[k];
        dist_pos_squared[k] = (dist_pos[k] * dist_pos[k]);

        dist_neg[k] =  sum_short_neg[k] + sum_long[k] - p_neg[k];
        dist_neg_squared[k] = (dist_neg[k] * dist_neg[k]);

	}

	double x_uij =  - (beta[pos_item_id] + sum(dist_pos_squared,K)) + (beta[neg_item_id] + sum(dist_neg_squared,K));
	double deri = 1 / (1 + exp(x_uij));

	beta[pos_item_id] += learn_rate * (-deri - bias_reg * beta[pos_item_id]); // Update case [9]
	beta[neg_item_id] += learn_rate * ( deri - bias_reg * beta[neg_item_id]); // Update case [10]

	vector<double> etas;

	for(int ind = 0 ; ind <(int)max_prev_items_list.size(); ind ++){
		// etas.push_back(cumWeibull(ind)); // Rebus Etas cumWeibull
		etas.push_back(eta_cumWeibullSoftmax[(int)max_prev_items_list.size()][ind]); // Rebus Etas cumWeibullSoftmax
	}

	for (unordered_set<int>::iterator it = prev_items_set.begin(); it != prev_items_set.end(); it ++) { // Rebus Fism Histo ON
		int count_it = 1; // Rebus Fism Update Off

		vector<int>::iterator pos = find(max_prev_items_list.begin(), max_prev_items_list.end(), *it);
		if (pos != max_prev_items_list.end()) {

			double sum_eta = 0.0;
		    while (pos != max_prev_items_list.end()) // Sum of etas of pos (pos can appears more than once)
		    {
				int pt = pos - max_prev_items_list.begin();
				sum_eta += etas[pt];
		        pos = std::find(pos+1, max_prev_items_list.end(), *it);
		    }
			if (*it != pos_item_id) {
				if (*it != neg_item_id) { // Update case [1]
					for (int k = 0; k < K; k ++) {
						P[*it][k] += learn_rate * ( deri * (-(wu_pos * count_it + sum_eta) * dist_pos[k] + (wu_neg * count_it + sum_eta) * dist_neg[k]) - lambda * P[*it][k]);
					}
				}
			} else { // Update case  [4]
				for (int k = 0; k < K; k ++) {
					P[pos_item_id][k] += learn_rate * ( deri * ((1-sum_eta) * dist_pos[k] + (wu_neg * count_it + sum_eta) * dist_neg[k]) - lambda * p_pos[k]);
				}
			}
		} else {
			if (*it != pos_item_id) {
				if (*it != neg_item_id) { // Update case  [2]
					for (int k = 0; k < K; k ++) {
						P[*it][k] += learn_rate * ( deri * (-wu_pos * count_it * dist_pos[k] + wu_neg * count_it * dist_neg[k]) - lambda * P[*it][k]);
					}
				} else { // Update case  [7]
					for (int k = 0; k < K; k ++) {
						P[neg_item_id][k] += learn_rate * ( deri * (-wu_pos * count_it *  dist_pos[k] - dist_neg[k]) - lambda * p_neg[k]);
					}
				}
			} else { // Update case  [5]
				for (int k = 0; k < K; k ++) {
					P[pos_item_id][k] += learn_rate * ( deri * (dist_pos[k] + wu_neg * count_it * dist_neg[k]) - lambda * p_pos[k]);
				}
			}
		}
	}

	if(max_prev_items_list[0] == nItems){
		for (int k = 0; k < K; k ++) { // Update case  [3] (i*)
			P[nItems][k] += learn_rate * ( deri * (-dist_pos[k] + dist_neg[k]) - lambda * P[nItems][k]);
		}
	}

	unordered_set<int>::iterator neg = find(prev_items_set.begin(), prev_items_set.end(), neg_item_id); // Rebus Fism Histo On
	if (neg == prev_items_set.end()) {
		for (int k = 0; k < K; k ++) { // Update case  [8]
			P[neg_item_id][k] += learn_rate * ( deri * (-dist_neg[k]) - lambda * p_neg[k]);
		}
	}

	unordered_set<int>::iterator pos = find(prev_items_set.begin(), prev_items_set.end(), pos_item_id); // Rebus Fism Histo On
	if (pos == prev_items_set.end()) {
		for (int k = 0; k < K; k ++) { // Update case  [6]
			P[pos_item_id][k] += learn_rate * ( deri * (dist_pos[k]) - lambda * p_pos[k]);
		}
	}

	delete [] p_pos ;
	delete [] p_neg ;
	delete [] p_pos_minus_neg ;
	delete [] sum_short_pos ;
	delete [] sum_short_neg ;
	delete [] sum_long ;
	delete [] dist_pos ;
	delete [] dist_pos_squared ;
	delete [] dist_neg ;
	delete [] dist_neg_squared ;
}

// Get  prev_items : from oldest to newest
// Return  return items rank from newest to oldest
vector<int> REBUS::findPath(vector<int> prev_items)
{
	vector<int> path;
	std::string sequence;
	while(true){
		if (prev_items.size() < 1){
			break;
		}
		int item_int = prev_items.back();
		string item = std::to_string(item_int);
		prev_items.pop_back();
		if(sequence.empty()){
			sequence = item ;
		} else {
			std::stringstream ss;
			ss << item << '-' <<sequence;
			sequence = ss.str();
		}
		// std::cout << "sequence: " << sequence << "\n";

		if (nodes_label_dict.find(sequence) != nodes_label_dict.end()) {
			path.push_back(item_int);
		} else {
			break;
		}
	}

	if(path.empty()){
		path.push_back(nItems);
	}
	// std::reverse(path.begin(),path.end());
	return path;
}

// Get  prev_items : from oldest to newest
// Return  return items rank from newest to oldest
vector<int> REBUS::findPathStars(vector<int> prev_items)
{
	vector<int> path;
	std::string sequence;
	int count_start = 0;
	// int item_start =  prev_items.back();
	while(true){
		if (prev_items.size() < 1 || (nb_stars != 0 && nb_stars <= count_start)){
			break;
		}
		int item_int = prev_items.back();
		string item = std::to_string(item_int);
		prev_items.pop_back();
		if(sequence.empty()){
			if(nodes_label_dict.find(item) != nodes_label_dict.end()){
				sequence = item ;
				path.push_back(item_int);
			} else {
				count_start ++;
			}
		} else {
			std::stringstream ss;
			ss << item << '-' <<sequence;
			if(nodes_label_dict.find(ss.str()) != nodes_label_dict.end()){
				sequence = ss.str();
				path.push_back(item_int);
			} else {
				count_start ++;
			}

		}
		// std::cout << "sequence: " << sequence << "\n";
	}

	if(path.empty()){
		path.push_back(nItems);
		// path.push_back(item_start);
	}
	// std::reverse(path.begin(),path.end());
	return path;
}

void REBUS::oneiteration(double learn_rate)
{
	// printf(" -- oneiteration REBUS -- \n\n");
	for (int i = 0; i < num_pos_events; i ++) {
		int user_id, pos_item_id, neg_item_id;
		vector<int> prev_items;
		unordered_set<int> prev_items_set;
		vector<int> max_prev_items;

		// sample user
		user_id = sampleUser();
		vector<pair<int, pair<vector<int>, unordered_set<int>> > >& user_items = matrix[user_id];

		// sample positive item
		int rand_num = rand() % user_items.size();
		pos_item_id = user_items.at(rand_num).first;
		prev_items = user_items.at(rand_num).second.first; // Items rank from oldest to newest
		prev_items_set = user_items.at(rand_num).second.second;
		max_prev_items= histo_user[user_id].at(rand_num).second; //Items rank from newest to oldest
		// max_prev_items = findPath(prev_items);

		// sample negative item
		do {
			neg_item_id = rand() % nItems;
		} while (neg_item_id == pos_item_id || find(max_prev_items.begin(), max_prev_items.end(), neg_item_id) != max_prev_items.end());
		// } while (neg_item_id == pos_item_id || find(prev_items.begin(), prev_items.end(), neg_item_id) != prev_items.end());
		// } while (clicked_per_user[user_id].find(neg_item_id) != clicked_per_user[user_id].end());

		updateFactors(user_id, max_prev_items, prev_items, prev_items_set, pos_item_id, neg_item_id, learn_rate);

	}
}

void REBUS::train(int iterations, double learn_rate)
{
	printf(" --Train Strated REBUS_K_%d_minCount_%d_L_%d_lambda_%f_biasReg_%f_typeSeq_%s_findPathStar_%d_alphaUP%f -- \n\n", K, minCount, L, lambda, bias_reg, type_seq.c_str(), find_path_stars, alpha_up);
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

		// if(iter % 100 == 0 && iter > 3000) {
		if(iter % 100 == 0 && iter > start_auc_test) {
			printf("Validation ...\n");
			double valid, test, var;
			sampleAUC(&valid, &test, &var);
			printf("[Valid AUC = %f], Test AUC = %f, Test var = %f\n", valid, test, var);
			printf(" SampleAUC() --> Previous best : %f, current : %f \n",bestValidAUC, valid);
			printf(" SampleAUC() --> ite : %d, best_iter + 300 : %d \n",iter, best_iter+300);
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

string REBUS::toString()
{
	char str[250];
	sprintf(str, "REBUS_K_%d_minCount_%d_L_%d_lambda_%f_biasReg_%f_typeSeq_%s_findPathStar_%d_alphaUP%f", K, minCount, L, lambda, bias_reg, type_seq.c_str(), find_path_stars,alpha_up);
	return str;
}


void REBUS::sampleAUC(double* AUC_val, double* AUC_test, double* var)
{
	printf(" --sampleAUC Started REBUS_K_%d_minCount_%d_L_%d_lambda_%f_biasReg_%f_typeSeq_%s_findPathStar_%d_alphaUP%f -- \n\n", K, minCount, L, lambda, bias_reg, type_seq.c_str(), find_path_stars, alpha_up);


	vector<double> AUC_u_val(nUsers, -1);
	vector<double> AUC_u_test(nUsers, -1);

	// #pragma omp parallel for schedule(dynamic)
	for (int indx_u = 0; indx_u < nUsers; indx_u ++) {
		int u = indx_u ;

		int item_test = test_per_user[u].first;
		int item_val  = val_per_user[u].first;

		if(indx_u % 2000 == 0) {
			printf("indx_u : %d\n", indx_u);
		}

		if (item_test == -1) {  // invalid user
			continue;
		}

		vector<pair<int,int> >& user_pos = corp->pos_per_user[u];

		vector<int> test_prev_items;

		for (unsigned int i = 0; i < user_pos.size(); i ++) {
			test_prev_items.push_back(user_pos[i].first);
		}
		test_prev_items.push_back(test_per_user[u].second);
		unordered_set<int> test_prev_items_set(test_prev_items.begin(),test_prev_items.end());

		vector<int> val_prev_items;
		for (unsigned int i = 0; i < user_pos.size(); i ++) {
			val_prev_items.push_back(user_pos[i].first);
		}
		unordered_set<int> val_prev_items_set(val_prev_items.begin(),val_prev_items.end());

		vector<int> max_test_prev_items_list;
		vector<int> max_val_prev_items_list;
		if(find_path_stars){
			max_test_prev_items_list = findPathStars(test_prev_items);
			max_val_prev_items_list = findPathStars(val_prev_items);
		} else {
			max_test_prev_items_list = findPath(test_prev_items);
			max_val_prev_items_list = findPath(val_prev_items);
		}

		double x_u_test;
		double x_u_val;

		x_u_test = prediction(u, max_test_prev_items_list, test_prev_items, test_prev_items_set, item_test);
		x_u_val = prediction(u, max_val_prev_items_list, val_prev_items, val_prev_items_set, item_val);

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

			double x_uj = prediction(u, max_val_prev_items_list, val_prev_items, val_prev_items_set, j);
			if (x_u_val > x_uj) {
				count_val ++;
			}

			x_uj = prediction(u, max_test_prev_items_list, test_prev_items, test_prev_items_set, j);
			if (x_u_test > x_uj) {
				count_test ++;
			}
		}
		AUC_u_val[indx_u] = 1.0 * count_val / max;
		AUC_u_test[indx_u] = 1.0 * count_test / max;
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

void REBUS::AUC(double* AUC_val, double* AUC_test,
			  double* HIT_val, double* HIT_test,
			  double* MRR_val, double* MRR_test,
			  double* var)
{
	printf(" --AUC strarted REBUS_K_%d_minCount_%d_L_%d_lambda_%f_biasReg_%f_typeSeq_%s_findPathStar_%d_alphaUP%f -- \n\n", K, minCount, L, lambda, bias_reg, type_seq.c_str(), find_path_stars, alpha_up);

	vector<double> AUC_u_val(nUsers, -1);
	vector<double> AUC_u_test(nUsers, -1);
	vector<double> val_MRR_u(nUsers, -1);
	vector<int>    val_HIT_u(nUsers, -1);
	vector<double> test_MRR_u(nUsers, -1);
	vector<int>    test_HIT_u(nUsers, -1);

	// #pragma omp parallel for schedule(dynamic)
	for (int u = 0; u < nUsers; u ++) {
		int item_test = test_per_user[u].first;
		int item_val  = val_per_user[u].first;

		if(u % 2000 == 0) {
			printf("u : %d\n", u);
		}
		if (item_test == -1) {  // invalid user
			continue;
		}

		vector<pair<int,int> >& user_pos = corp->pos_per_user[u];

		vector<int> test_prev_items;

		for (unsigned int i = 0; i < user_pos.size(); i ++) {
			test_prev_items.push_back(user_pos[i].first);
		}
		test_prev_items.push_back(test_per_user[u].second);
		unordered_set<int> test_prev_items_set(test_prev_items.begin(),test_prev_items.end());

		vector<int> val_prev_items;
		for (unsigned int i = 0; i < user_pos.size(); i ++) {
			val_prev_items.push_back(user_pos[i].first);
		}
		unordered_set<int> val_prev_items_set(val_prev_items.begin(),val_prev_items.end());

		vector<int> max_test_prev_items_list;
		vector<int> max_val_prev_items_list;
		if(find_path_stars){
			max_test_prev_items_list = findPathStars(test_prev_items);
			max_val_prev_items_list = findPathStars(val_prev_items);
		} else {
			max_test_prev_items_list = findPath(test_prev_items);
			max_val_prev_items_list = findPath(val_prev_items);
		}

		double x_u_test;
		double x_u_val;

		x_u_test = prediction(u, max_test_prev_items_list, test_prev_items, test_prev_items_set, item_test);
		x_u_val = prediction(u, max_val_prev_items_list, val_prev_items, val_prev_items_set, item_val);

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

			double x_uj = prediction(u, max_val_prev_items_list, val_prev_items, val_prev_items_set, j);
			if (x_u_val > x_uj) {
				count_val ++;
			}

			x_uj = prediction(u, max_test_prev_items_list, test_prev_items, test_prev_items_set, j);
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
