#include "corpus.hpp"

// rank the clicks in terms of timestamps
bool timeCompare(const pair<int,int>& firstElem, const pair<int,int>& secondElem)
{
	return firstElem.second < secondElem.second;
}

void corpus::loadData(const char* clickFile, int userMin, int itemMin)
{
	nItems = 0;
	nUsers = 0;
	nClicks = 0;

	loadClicks(clickFile, userMin, itemMin);	// determine I and U at this step

	printf("\n  \"nUsers\": %d, \"nItems\": %d, \"nClicks\": %d\n", nUsers, nItems, nClicks);
}

void corpus::loadClicks(const char* clickFile, int userMin, int itemMin)
{
	unordered_map<string, int> uCounts;
	unordered_map<string, int> iCounts;

	printf("  Loading clicks from %s, userMin = %d  itemMin = %d ", clickFile, userMin, itemMin);

	string uName; // User name
	string iName; // Item name
	float value;  // rating
	int voteTime; // Time rating was entered
	string line_comma;

	int nRead = 0; // Progress
	string line;

	igzstream in;
	in.open(clickFile);

	string clickFile_type = clickFile;
	clickFile_type = clickFile_type.substr(clickFile_type.size()-4,clickFile_type.size());
	std::cout << "\n" << "clickFile_type " << clickFile_type << "\n";

	if(clickFile_type.compare(".csv") == 0){
		if (! in.good()) {
			printf("Can't read clicks from %s.\n", clickFile);
			exit(1);
		}

		// The first pass collects statsitics
		while (getline(in, line)) {
			stringstream ss(line);
			// ss >> uName >> iName >> value >> voteTime;
			ss >> line_comma;

			stringstream sss(line_comma);
			string item;
			vector<string> splittedStrings;
			while (getline(sss, item, ','))
			{
				splittedStrings.push_back(item);
			}

			uName = splittedStrings[0];
			iName = splittedStrings[1];
			value = stof(splittedStrings[2]);
			voteTime = stoi(splittedStrings[3]);

			nRead ++;
			if (nRead % 100000 == 0) {
				printf(".");
				fflush(stdout);
			}

			if (uCounts.find(uName) == uCounts.end()) {
				uCounts[uName] = 0;
			}
			if (iCounts.find(iName) == iCounts.end()) {
			}
			uCounts[uName] ++;
			iCounts[iName] ++;
		}
	} else {
		if (! in.good()) {
			printf("Can't read clicks from %s.\n", clickFile);
			exit(1);
		}

		// The first pass collects statsitics
		while (getline(in, line)) {
			stringstream ss(line);
			ss >> uName >> iName >> value >> voteTime;

			nRead ++;
			if (nRead % 100000 == 0) {
				printf(".");
				fflush(stdout);
			}

			if (uCounts.find(uName) == uCounts.end()) {
				uCounts[uName] = 0;
			}
			if (iCounts.find(iName) == iCounts.end()) {
				iCounts[iName] = 0;
			}
			uCounts[uName] ++;
			iCounts[iName] ++;
		}
	}
	in.close();
	printf("\n  First pass: #users = %d, #items = %d, #clicks = %d\n", (int)uCounts.size(), (int)iCounts.size(), nRead);

	// The second pass is for filtering
	nUsers = 0;
	nItems = 0;
	nClicks = 0;

	igzstream in2;
	in2.open(clickFile);
	if (! in2.good()) {
		printf("Can't read clicks from %s.\n", clickFile);
		exit(1);
	}

	nRead = 0;

	if(clickFile_type.compare(".csv") == 0){
		while (getline(in2, line)) {
			stringstream ss(line);
			ss >> line_comma;

			stringstream sss(line_comma);
			string item;
			vector<string> splittedStrings;
			while (getline(sss, item, ','))
			{
				splittedStrings.push_back(item);
			}

			uName = splittedStrings[0];
			iName = splittedStrings[1];
			value = stof(splittedStrings[2]);
			voteTime = stoi(splittedStrings[3]);

			nRead ++;
			if (nRead % 100000 == 0) {
				printf(".");
				fflush(stdout);
			}

			if (uCounts[uName] < userMin) {
				continue;
			}

			if (iCounts[iName] < itemMin) {
				continue;
			}

			nClicks ++;

			// new item
			if (itemIds.find(iName) == itemIds.end()) {
				rItemIds[nItems] = iName;
				itemIds[iName] = nItems ++;
			}

			// new user
			if (userIds.find(uName) == userIds.end()) {
				rUserIds[nUsers] = uName;
				userIds[uName] = nUsers ++;
				// add a new vec
				pos_per_user.push_back(vector<pair<int,int> >());
			}
			pos_per_user[userIds[uName]].push_back(make_pair(itemIds[iName], voteTime));
		}
	} else {
		while (getline(in2, line)) {
			stringstream ss(line);
			ss >> uName >> iName >> value >> voteTime;

			nRead ++;
			if (nRead % 100000 == 0) {
				printf(".");
				fflush(stdout);
			}

			if (uCounts[uName] < userMin) {
				continue;
			}

			if (iCounts[iName] < itemMin) {
				continue;
			}

			nClicks ++;

			// new item
			if (itemIds.find(iName) == itemIds.end()) {
				rItemIds[nItems] = iName;
				itemIds[iName] = nItems ++;
			}

			// new user
			if (userIds.find(uName) == userIds.end()) {
				rUserIds[nUsers] = uName;
				userIds[uName] = nUsers ++;
				// add a new vec
				pos_per_user.push_back(vector<pair<int,int> >());
			}
			pos_per_user[userIds[uName]].push_back(make_pair(itemIds[iName], voteTime));
		}
	}
	in2.close();

	// rank clicks for each user in terms of timestamps
	printf("\n  Sorting clicks for each users ");

	#pragma omp parallel for schedule(dynamic)
	for (int u = 0; u < nUsers; u ++) {
		sort(pos_per_user[u].begin(), pos_per_user[u].end(), timeCompare);
		if (u % 10000 == 0) {
			printf(".");
			fflush(stdout);
		}
	}
	printf("\n");
}
