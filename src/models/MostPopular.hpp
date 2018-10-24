#pragma once

#include "model.hpp"

class MostPopular : public model
{
public:
	MostPopular(corpus* corp) : model(corp) 
	{
		num_pos_per_item = vector<int>(nItems, 0);

		for (int u = 0; u < nUsers; u ++) {
			for (vector<pair<int,int> >::iterator it = corp->pos_per_user[u].begin(); it != corp->pos_per_user[u].end(); it ++) {
				num_pos_per_item[it->first] += 1;
			}
		}
	}

	~MostPopular(){}

	double prediction(int user, int item_prev, int item);

	string toString();

	vector<int> num_pos_per_item;
};
