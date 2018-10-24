# recommendation

The steps for running the code :
	1. Compile the code with the command "make"
	2. Run the program with the following parameters :
		 1 : data path
		 2 : User min (We set at 0 for MovieLens datasets and 5 for others datasets)
		 3 : Item min (We set at 0 for MovieLens datasets and 5 for others datasets)
		 4 : MinCount (Min occurence for substring)
		 4 : L (Max size for substringg)
		 4 : K (Latent Feature Dimension)
		 4 : lambda (L2-norm regularizer)
		 4 : bias_reg (L2-norm regularizer for bias terms)
		 4 : relation_reg (L2-norm regularizer for USER translation vectors, only for TransRec)
		 4 : alpha (for PRME and REBUS)
		 4 : Maximum number of iterations
		 4 : Model path
		 4 : Model name (MP, BPRMF, MC, FPMC, HRM_max, HRM_avg, PRME, TransRec_L1, TransRec, Fossil, REBUS)
		 Example 1 : ./train 01-Data/visiativ-prep.csv 5 5 1 1 10 0.01 0 0 -0.6 10000 03-Models/my_model REBUS
		 Example 2 : ./train 01-Data/Epinions.txt 5 5 1 12 10 0.1 0.1 0.1 -0.5 10000 03-Models/my_model TransRec

If you have any issues with the code, you can conctat at corentin.lonjarret@visiativ.com

All Data can be found here : https://drive.google.com/open?id=1QttRl0m24pY2ilMTg8Wo9GvluGFyb9cP

We thank R.He and J.McAuley who made available their codes and data. You can find the orignal code on their web page :
	- Ruining He : https://sites.google.com/view/ruining-he/
	- Julian McAuley : https://cseweb.ucsd.edu/~jmcauley/