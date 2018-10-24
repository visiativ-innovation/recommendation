#pragma once

#include "stdio.h"
#include "stdlib.h"
#include "vector"
#include "math.h"
#include "string.h"
#include <string>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include "omp.h"
#include "map"
#include "set"
#include "vector"
#include "algorithm"
#include "sstream"
#include "gzstream.h"
#include "sys/time.h"
#include <cfloat>
#include <random>

using namespace std;

bool pairCompare(const pair<int, double>& firstElem, const pair<int, double>& secondElem);

/// Safely open a file
FILE* fopen_(const char* p, const char* m);


/// Data associated with a rating
typedef struct rating
{
	int user; // ID of the user
	int item; // ID of the item
	float val;  // Unix time of the rating

	rating(int user, int item, float val) : user(user), item(item), val(val) {}
} rating;

inline double inner(double* x, double* y, int K)
{
	double res = 0;
	for (int k = 0; k < K; k ++) {
		res += x[k] * y[k];
	}
	return res;
}

inline double inner(double* x, double* y, double* z, int K)
{
	double res = 0;
	for (int i = 0; i < K; i ++) {
		res += x[i] * y[i] * z[i];
	}
	return res;
}

inline double square(double x)
{
	return x*x;
}

inline double dsquare(double x)
{
	return 2*x;
}

inline double sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

inline double l2_sq(double* x, double* y, int K)
{
	double res = 0;
	for (int i = 0; i < K; i ++) {
		res += square(x[i] - y[i]);
	}
	return res;
}

inline double sum(double* x, int K)
{
	double res = 0;
	for (int i = 0; i < K; i ++) {
		res += x[i];
	}
	return res;
}

inline double cumWeibull(double x)
{
	double k = 2; double y = 7; // Etas cumWeibull Faible
	// double k = 1.5; double y = 5; // Etas cumWeibull Moyen
	// double k = 1; double y = 3.5; // Etas cumWeibull Fort

	return 1-(1-exp(-pow(x/y,k)));
}

inline double cumWeibull_softmax(double x, int histo_size)
{
	double res = 0.0;

	for(int i = 0 ; i < histo_size ; i ++){
		res += exp(cumWeibull(i));
	}

	return exp(cumWeibull(x)) /  res ;
}

inline double square_sum(double* x, int K)
{
	double res = 0;
	for (int i = 0; i < K; i ++) {
		res += square(x[i]);
	}
	return res;
}

inline double clock_(void)
{
	timeval tim;
	gettimeofday(&tim, NULL);
	return tim.tv_sec + (tim.tv_usec / 1000000.0);
}

static inline string &ltrim(string &s)
{
	s.erase(s.begin(), find_if(s.begin(), s.end(), not1(ptr_fun<int, int>(isspace))));
	return s;
}

// trim from end
static inline string &rtrim(string &s)
{
	s.erase(find_if(s.rbegin(), s.rend(), not1(ptr_fun<int, int>(isspace))).base(), s.end());
	return s;
}

// trim from both ends
static inline string &trim(string &s)
{
	return ltrim(rtrim(s));
}
