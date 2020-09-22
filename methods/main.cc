#include <iostream>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstring>

#include "def.h"
#include "util.h"
#include "nns.h"

// -----------------------------------------------------------------------------
void usage() 						// display the usage of this package
{
	printf("\n"
		"-------------------------------------------------------------------\n"
		" Usage of the Package for Point-to Hyperplane NNS                  \n"
		"-------------------------------------------------------------------\n"
		"    -alg   {integer}  options of algorithms (0 - 7)\n"
		"    -n     {integer}  cardinality of the dataset\n"
		"    -qn    {integer}  number of queries\n"
		"    -d     {integer}  dimensionality of the dataset\n"
		"    -m     {integer}  #hash tables (FH and MFH)\n"
		"                      #single hasher of the compond hasher (EH, BH, MH)\n"
		"    -l     {integer}  #hash tables (EH, BH, MH)\n"
		"    -M     {integer}  #proj vecotr used for a single hasher (MH)\n"
		"    -s     {integer}  scale factor of dimension\n"
		"    -b     {float}    interval ratio\n"
		"    -cf    {string}   name of configuration\n"
		"    -dn    {string}   name of dataset\n"
		"    -ds    {string}   address of the data  set\n"
		"    -qs    {string}   address of the query set\n"
		"    -ts    {string}   address of the truth set\n"
		"    -of    {string}   output folder\n"
		"\n"
		"-------------------------------------------------------------------\n"
		" The Options of Algorithms                                         \n"
		"-------------------------------------------------------------------\n"
		"    0  - Ground-Truth\n"
		"         Param: -alg 0 -n -qn -d -ds -qs -ts\n"
		"\n"
		"    1  - Linear_Scan\n"
		"         Param: -alg 1 -n -qn -d -dn -ds -qs -ts -of\n"
		"\n"
		"    2  - Random Scan (Random Selection and Scan)\n"
		"         Param: -alg 2 -n -qn -d -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    3  - Sorted Scan (Sort and Scan)\n"
		"         Param: -alg 3 -n -qn -d -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    4  - EH (Embedding Hyperplane Hash)\n"
		"         Param: -alg 4 -n -qn -d -m -l -b -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    5  - BH (Bilinear Hyperplane Hash)\n"
		"         Param: -alg 5 -n -qn -d -m -l -b -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    6  - MH (Multilinear Hyperplane Hash)\n"
		"         Param: -alg 6 -n -qn -d -m -l -M -b -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    7  - FH (Multi-Partition Furthest Hyperpalne Hash)\n"
		"         Param: -alg 7 -n -qn -d -m -b -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    8  - FH^- (Furthest Hyperpalne Hash)\n"
		"         Param: -alg 8 -n -qn -d -m -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    9  - NH (Nearest Hyperpalne Hash with QALSH)\n"
		"         Param: -alg 9 -n -qn -d -m -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    10 - NH_LCCS (Nearest Hyperpalne Hash with LCCS-LSH)\n"
		"         Param: -alg 10 -n -qn -d -m -w -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    11 - FH Sampling\n"
		"         Param: -alg 11 -n -qn -d -m -s -b -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    12 - FH^- Sampling\n"
		"         Param: -alg 12 -n -qn -d -m -s -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    13 - NH Sampling\n"
		"         Param: -alg 13 -n -qn -d -m -s -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    14 - NH_LCCS Sampling\n"
		"         Param: -alg 14 -n -qn -d -m -w -s -cf -dn -ds -qs -ts -of\n"
		"\n"
		"-------------------------------------------------------------------\n"
		"\n\n\n");
}

// -----------------------------------------------------------------------------
int main(int nargs, char **args)
{
	srand(6); // srand((unsigned) time(NULL)); 	// 

	char   conf_name[200];			// name of configuration
	char   data_name[200];			// name of dataset
	char   data_set[200];			// address of data set
	char   query_set[200];			// address of query set
	char   truth_set[200];			// address of ground truth file
	char   output_folder[200];		// output folder

	int    alg    = -1;				// which algorithm?
	int    n      = -1;				// cardinality
	int    qn     = -1;				// query number
	int    d      = -1;				// dimensionality
	int    M      = -1;				// #proj vectors for a single hasher
	int    m      = -1;				// #hash tables (MFH,FH), #concatenated hasher (EH,BH,MH)
	int    l      = -1;				// #hash tables (EH,BH,MH)
	int    s      = -1;				// scale factor of dimension (s > 0)
	float  b      = -1.0f;			// interval ratio (0 < b < 1)
	float  w      = -1.0f;			// bucket width

	float  *data  = NULL;			// data set
	float  *query = NULL;			// query set
	Result *R     = NULL;			// k-NN ground truth
	int    cnt    = 1;
	
	while (cnt < nargs) {
		if (strcmp(args[cnt], "-alg") == 0) {
			alg = atoi(args[++cnt]);
			printf("alg       = %d\n", alg);
			assert(alg >= 0);
		}
		else if (strcmp(args[cnt], "-n") == 0) {
			n = atoi(args[++cnt]);
			printf("n         = %d\n", n);
			assert(n > 0);
		}
		else if (strcmp(args[cnt], "-qn") == 0) {
			qn = atoi(args[++cnt]);
			printf("qn        = %d\n", qn);
			assert(qn > 0);
		}
		else if (strcmp(args[cnt], "-d") == 0) {
			d = atoi(args[++cnt]) + 1; // add 1 
			printf("d         = %d\n", d);
			assert(d > 1);
		}
		else if (strcmp(args[cnt], "-m") == 0) {
			m = atoi(args[++cnt]);
			printf("m         = %d\n", m);
			assert(m > 0);
		}
		else if (strcmp(args[cnt], "-l") == 0) {
			l = atoi(args[++cnt]);
			printf("l         = %d\n", l);
			assert(l > 0);
		}
		else if (strcmp(args[cnt], "-M") == 0) {
			M = atoi(args[++cnt]);
			printf("M         = %d\n", M);
			assert(M > 2);
		}
		else if (strcmp(args[cnt], "-s") == 0) {
			s = atoi(args[++cnt]);
			printf("s         = %d\n", s);
			assert(s > 0);
		}
		else if (strcmp(args[cnt], "-b") == 0) {
			b = atof(args[++cnt]);
			printf("b         = %.2f\n", b);
			assert(b > 0.0f && b < 1.0f);
		}
		else if (strcmp(args[cnt], "-w") == 0) {
			w = atof(args[++cnt]);
			printf("w         = %.2f\n", w);
			assert(w > 0.0f);
		}
		else if (strcmp(args[cnt], "-cf") == 0) {
			strncpy(conf_name, args[++cnt], sizeof(conf_name));
			printf("conf_name = %s\n", conf_name);
		}
		else if (strcmp(args[cnt], "-dn") == 0) {
			strncpy(data_name, args[++cnt], sizeof(data_name));
			printf("data_name = %s\n", data_name);
		}
		else if (strcmp(args[cnt], "-ds") == 0) {
			strncpy(data_set, args[++cnt], sizeof(data_set));
			printf("data_set  = %s\n", data_set);
		}
		else if (strcmp(args[cnt], "-qs") == 0) {
			strncpy(query_set, args[++cnt], sizeof(query_set));
			printf("query_set = %s\n", query_set);
		}
		else if (strcmp(args[cnt], "-ts") == 0) {
			strncpy(truth_set, args[++cnt], sizeof(truth_set));
			printf("truth_set = %s\n", truth_set);
		}
		else if (strcmp(args[cnt], "-of") == 0) {
			strncpy(output_folder, args[++cnt], sizeof(output_folder));
			printf("folder    = %s\n", output_folder);

			int len = (int) strlen(output_folder);
			if (output_folder[len - 1] != '/') {
				output_folder[len] = '/';
				output_folder[len + 1] = '\0';
			}
			create_dir(output_folder);
		}
		else {
			usage();
			exit(1);
		}
		++cnt;
	}
	printf("\n");

	// -------------------------------------------------------------------------
	//  read data set and query set
	// -------------------------------------------------------------------------
	data = new float[n * d];
	if (read_bin_data(n, d, data_set, data)) exit(1);

	query = new float[qn * d];
	if (read_bin_query(qn, d, query_set, query)) exit(1);

	if (alg > 0) {
		R = new Result[qn * MAXK];
		if (read_ground_truth(qn, truth_set, R)) exit(1);
	}

	// -------------------------------------------------------------------------
	//  methods
	// -------------------------------------------------------------------------
	switch (alg) {
	case 0:
		ground_truth(n, qn, d, (const float*) data, (const float*) query, 
			truth_set);
		break;
	case 1:
		linear_scan(n, qn, d, data_name, "Linear", 
			output_folder, (const float*) data, (const float*) query, 
			(const Result*) R);
		break;
	case 2:
		random_scan(n, qn, d, conf_name, data_name, "Random_Scan", 
			output_folder, (const float*) data, (const float*) query, 
			(const Result*) R);
		break;
	case 3:
		sorted_scan(n, qn, d, conf_name, data_name, "Sorted_Scan", 
			output_folder, (const float*) data, (const float*) query, 
			(const Result*) R);
		break;
	case 4:
		embed_hash(n, qn, d, m, l, b, conf_name, data_name, "EH", 
			output_folder, (const float*) data, (const float*) query, 
			(const Result*) R);
		break;
	case 5:
		bilinear_hash(n, qn, d, m, l, b, conf_name, data_name, "BH", 
			output_folder, (const float*) data, (const float*) query, 
			(const Result*) R);
		break;
	case 6:
		multilinear_hash(n, qn, d, M, m, l, b, conf_name, data_name, "MH", 
			output_folder, (const float*) data, (const float*) query, 
			(const Result*) R);
		break;
	case 7:
		mfh_hash(n, qn, d, m, b, conf_name, data_name, "MFH", output_folder, 
			(const float*) data, (const float*) query, (const Result*) R);
		break;
	case 8:
		fh_hash(n, qn, d, m, conf_name, data_name, "FH", output_folder, 
			(const float*) data, (const float*) query, (const Result*) R);
		break;
	case 9:
		nh_hash(n, qn, d, m, conf_name, data_name, "NH", output_folder, 
			(const float*) data, (const float*) query, (const Result*) R);
		break;
	case 10:
		nh_lccs(n, qn, d, m, w, conf_name, data_name, "NH_LCCS", output_folder, 
			(const float*) data, (const float*) query, (const Result*) R);
		break;
	case 11:
		mfh_hash_sampling(n, qn, d, m, s, b, conf_name, data_name, "MFH_Sampling", 
			output_folder, (const float*) data, (const float*) query, 
			(const Result*) R);
		break;
	case 12:
		fh_hash_sampling(n, qn, d, m, s, conf_name, data_name, "FH_Sampling", 
			output_folder, (const float*) data, (const float*) query, 
			(const Result*) R);
		break;
	case 13:
		nh_hash_sampling(n, qn, d, m, s, conf_name, data_name, "NH_Sampling", 
			output_folder, (const float*) data, (const float*) query, 
			(const Result*) R);
		break;
	case 14:
		nh_lccs_sampling(n, qn, d, m, w, s, conf_name, data_name, "NH_LCCS_Sampling", 
			output_folder, (const float*) data, (const float*) query, 
			(const Result*) R);
		break;
	default:
		printf("Parameters error!\n");
		usage();
		break;
	}
	// -------------------------------------------------------------------------
	//  release space
	// -------------------------------------------------------------------------
	delete[] data;
	delete[] query; 
	if (alg > 0) delete[] R;

	return 0;
}
