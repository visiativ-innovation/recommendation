#include "corpus.hpp"
#include "MostPopular.hpp"
#include "BPRMF.hpp"
#include "MC.hpp"
#include "FPMC.hpp"
#include "FossilSimple.hpp"
#include "Fossil.hpp"
#include "HRM_max.hpp"
#include "HRM_avg.hpp"
#include "PRME.hpp"
#include "TransRec.hpp"
#include "TransRec_L1.hpp"
#include "REBUS.hpp"
#include <dirent.h>
#include <stdio.h>
#include <fstream>


void go_MP(corpus* corp)
{
	MostPopular md(corp);
	double valid; md.MultipleMetrics(50, false, valid);
}

void go_BPRMF(corpus* corp, int K, double lambda, double bias_reg, int iterations, const char* model_path)
{
	BPRMF md(corp, K, lambda, bias_reg);
	md.init();
	md.train(iterations, 0.05);
	double valid; md.MultipleMetrics(50, false, valid);
	md.saveModel((string(model_path) + "__" + md.toString() + ".txt").c_str());
	md.cleanUp();
}

void go_MC(corpus* corp, int K, double lambda, int iterations, const char* model_path)
{
	MC md(corp, K, lambda);
	md.init();
	md.train(iterations, 0.05);
	double valid; md.MultipleMetrics(50, false, valid);
	md.saveModel((string(model_path) + "__" + md.toString() + ".txt").c_str());
	md.cleanUp();
}

void go_FossilSimple(corpus* corp, int K, double lambda, double bias_reg, int iterations, const char* model_path)
{
	FossilSimple md(corp, K, lambda, bias_reg);
	md.init();
	md.train(iterations, 0.05);
	md.saveModel((string(model_path) + "__" + md.toString() + ".txt").c_str());
	md.cleanUp();
}

void go_Fossil(corpus* corp, int L, int K, double lambda, double bias_reg, int iterations, const char* model_path)
{
	Fossil md(corp, L, K, lambda, bias_reg);
	md.init();
	md.train(iterations, 0.05);
	md.saveModel((string(model_path) + "__" + md.toString() + ".txt").c_str());
	md.cleanUp();
}

void go_FPMC(corpus* corp, int K, int KK, double lambda, double bias_reg, int iterations, const char* model_path)
{
	FPMC md(corp, K, KK, lambda);
	md.init();
	md.train(iterations, 0.05);
	double valid; md.MultipleMetrics(50, false, valid);
	md.saveModel((string(model_path) + "__" + md.toString() + ".txt").c_str());
	md.cleanUp();
}

void go_HRM_max(corpus* corp, int K, double lambda, double bias_reg, int iterations, const char* model_path)
{
	HRM_max md(corp, K, lambda);
	md.init();
	md.train(iterations, 0.05);
	double valid; md.MultipleMetrics(50, false, valid);
	md.saveModel((string(model_path) + "__" + md.toString() + ".txt").c_str());
	md.cleanUp();
}

void go_HRM_avg(corpus* corp, int K, double lambda, double bias_reg, int iterations, const char* model_path)
{
	HRM_avg md(corp, K, lambda);
	md.init();
	md.train(iterations, 0.05);
	double valid; md.MultipleMetrics(50, false, valid);
	md.saveModel((string(model_path) + "__" + md.toString() + ".txt").c_str());
	md.cleanUp();
}

void go_PRME(corpus* corp, int K, int KK, double alpha, double lambda, int iterations, const char* model_path)
{
	PRME md(corp, K, KK, alpha, lambda);
	md.init();
	md.train(iterations, 0.05);
	double valid; md.MultipleMetrics(50, false, valid);
	md.saveModel((string(model_path) + "__" + md.toString() + ".txt").c_str());
	md.cleanUp();
}

void go_TransRec(corpus* corp, int K, double lambda, double relation_reg, double bias_reg, int iterations, const char* model_path)
{
	TransRec md(corp, K, lambda, relation_reg, bias_reg);
	md.init();
	md.train(iterations, 0.05);
	double valid; md.MultipleMetrics(50, false, valid);
	md.saveModel((string(model_path) + "__" + md.toString() + ".txt").c_str());
	md.cleanUp();
}

void go_TransRec_L1(corpus* corp, int K, double lambda, double relation_reg, double bias_reg, int iterations, const char* model_path)
{
	TransRec_L1 md(corp, K, lambda, relation_reg, bias_reg);
	md.init();
	md.train(iterations, 0.05);
	double valid; md.MultipleMetrics(50, false, valid);
	md.saveModel((string(model_path) + "__" + md.toString() + ".txt").c_str());
	md.cleanUp();
}

void go_REBUS(corpus* corp, int K, double lambda, double bias_reg, double alpha_up, int iterations, const char* model_path, const char* data_path, int minCount, int L)
{
	string data_name = data_path;

	REBUS md(corp, K, lambda, bias_reg ,alpha_up);
	int loading = md.loadPST(minCount, L, data_path);
	if(loading == 0){
		md.init();
		md.train(iterations, 0.05);
		md.saveModel((string(model_path) + "__" + md.toString() + ".txt").c_str());
		md.cleanUp();
	} else {
		printf("No substrings found for the dataset %s with minCount : %d and L : %d\n",data_name.c_str(),minCount,L);
	}
}

// Example 1 : ./train 01-Data/visiativ-prep.csv 5 5 1 1 10 0.01 0 0 -0.6 10000 03-Models/my_model REBUS
// Example 2 : ./train 01-Data/Epinions.txt 5 5 1 12 10 0.1 0.1 0.1 -0.5 10000 03-Models/my_model TransRec

int main(int argc, char** argv)
{
	srand(0);

	if (argc != 14) {
		printf(" Parameters as following: \n");
		printf(" 1. Click triples path \n");
		printf(" 2. user min \n");
		printf(" 3. item min \n");
		printf(" 4. Min occurence for substring (MinCount) \n");
		printf(" 5. Max size for substring (L) \n");

		printf(" 6. Latent Feature Dimension (K) \n");

		printf(" 7. lambda (L2-norm regularizer) \n");
		printf(" 8. bias_reg (L2-norm regularizer for bias terms) \n");
		printf(" 9. relation_reg (L2-norm regularizer for USER translation vectors, only for TransRec \n");
		printf(" 10. alpha (for PRME and REBUS) \n");

		printf(" 11. Maximum number of iterations \n");
		printf(" 12. Model path \n");
		printf(" 13. Model name \n\n");

		exit(1);
	}

	char* data_path = argv[1];
	int user_min = atoi(argv[2]);
	int item_min = atoi(argv[3]);
	int minCount  = atoi(argv[4]);
	int L = atoi(argv[5]);

	int K  = atoi(argv[6]);

	double lambda = atof(argv[7]);
	double bias_reg = atof(argv[8]);
	double relation_reg = atof(argv[9]);
	double alpha = atof(argv[10]);

	int iter = atoi(argv[11]);
	char* model_path = argv[12];
	string type_model = argv[13];

	corpus corp;
	corp.loadData(data_path, user_min, item_min);

	if(type_model.compare("REBUS") == 0) {
		go_REBUS(&corp, K, lambda, bias_reg, alpha, iter, model_path, data_path, minCount, L);
	} else if(type_model.compare("MP") == 0) {
		go_MP(&corp);
	} else if(type_model.compare("BPRMF") == 0) {
		go_BPRMF(&corp, K, lambda, bias_reg, iter, model_path);
	} else if(type_model.compare("MC") == 0) {
		go_MC(&corp, K, lambda, iter, model_path);
	} else if(type_model.compare("FPMC") == 0) {
		go_FPMC(&corp, K, K, lambda, bias_reg, iter, model_path);
	} else if(type_model.compare("HRM_max") == 0) {
		go_HRM_max(&corp, K, lambda, bias_reg, iter, model_path);
	} else if(type_model.compare("HRM_avg") == 0) {
		go_HRM_avg(&corp, K, lambda, bias_reg, iter, model_path);
	} else if(type_model.compare("PRME") == 0) {
		go_PRME(&corp, K, K, alpha, lambda, iter, model_path);
	} else if(type_model.compare("TransRec_L1") == 0) {
		go_TransRec_L1(&corp, K, lambda, relation_reg, bias_reg, iter, model_path);
	} else if(type_model.compare("TransRec") == 0) {
		go_TransRec(&corp, K, lambda, relation_reg, bias_reg, iter, model_path);
	} else if(type_model.compare("FossilSimple") == 0) {
		go_FossilSimple(&corp, K, lambda, bias_reg, iter, model_path);
	} else if(type_model.compare("Fossil") == 0) {
		go_Fossil(&corp, L, K, lambda, bias_reg, iter, model_path);
	} else {
		printf("Unknown model !!! \n");
		printf("Possible models: \n");
		printf("  - MP \n");
		printf("  - BPRMF \n");
		printf("  - MC \n");
		printf("  - FPMC \n");
		printf("  - HRM_max \n");
		printf("  - HRM_avg \n");
		printf("  - PRME \n");
		printf("  - TransRec_L1 \n");
		printf("  - TransRec \n");
		printf("  - Fossil \n");
		printf("  - REBUS \n");

	}

	printf("}\n");
	return 0;
}
