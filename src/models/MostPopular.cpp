#include "MostPopular.hpp"


double MostPopular::prediction(int user, int item_prev, int item)
{
	return (double)num_pos_per_item[item];	// return popularity of item
}

string MostPopular::toString()
{
	return "MostPopular";
}