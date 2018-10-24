#pragma once

#include "common.hpp"

class corpus
{
public:
	corpus() {}
	~corpus() {}

	vector<vector<pair<int,int> > > pos_per_user;

	int nUsers;  // Number of users
	int nItems;  // Number of items
	int nClicks; // Number of clicks

	unordered_map<string, int> userIds; // Maps a user's string-valued ID to an integer
	unordered_map<string, int> itemIds; // Maps an item's string-valued ID to an integer

	unordered_map<int, string> rUserIds; // Inverse of the above maps
	unordered_map<int, string> rItemIds;

	void loadData(const char* clickFile, int userMin, int itemMin);

private:
	void loadClicks(const char* clickFile, int userMin, int itemMin);
};
