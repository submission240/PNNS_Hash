#pragma once

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <sys/time.h>

#include "def.h"
#include "util.h"
#include "pri_queue.h"
#include "baseline.h"
#include "mfh.h"
#include "mfh_sampling.h"
#include "fh.h"
#include "fh_sampling.h"
#include "nh.h"
#include "nh_sampling.h"
#include "nh_lccs.h"
#include "nh_lccs_sampling.h"

// -----------------------------------------------------------------------------
int linear_scan(					// point-to-hyperplane NNS using Linear
	int   n,							// number of data  objects
	int   qn,							// number of query objects
	int   d,							// dimension of space
	const char *data_name,				// name of dataset
	const char *method_name,			// name of method
	const char *output_folder,			// output folder
	const float *data,					// data set
	const float *query,					// query set
	const Result *R);					// truth set

// -----------------------------------------------------------------------------
int random_scan(					// point-to-hyperplane NNS using Random_Scan
	int   n,							// number of data  objects
	int   qn,							// number of query objects
	int   d,							// dimension of space
	const char *conf_name,				// name of configuration
	const char *data_name,				// name of dataset
	const char *method_name,			// name of method
	const char *output_folder,			// output folder
	const float *data,					// data set
	const float *query,					// query set
	const Result *R);					// truth set

// -----------------------------------------------------------------------------
int sorted_scan(					// point-to-hyperplane NNS using Sorted_Scan
	int   n,							// number of data  objects
	int   qn,							// number of query objects
	int   d,							// dimension of space
	const char *conf_name,				// name of configuration
	const char *data_name,				// name of dataset
	const char *method_name,			// name of method
	const char *output_folder,			// output folder
	const float *data,					// data set
	const float *query,					// query set
	const Result *R);					// truth set

// -----------------------------------------------------------------------------
int embed_hash(						// point-to-hyperplane NNS using EH
	int   n,							// number of data  objects
	int   qn,							// number of query objects
	int   d,							// dimension of space
	int   m, 							// #single hasher of the compond hasher
	int   l,							// #hash tables
	float b,							// interval ratio
	const char *conf_name,				// name of configuration
	const char *data_name,				// name of dataset
	const char *method_name,			// name of method
	const char *output_folder,			// output folder
	const float *data,					// data set
	const float *query,					// query set
	const Result *R);					// truth set

// -----------------------------------------------------------------------------
int bilinear_hash(					// point-to-hyperplane NNS using BH
	int   n,							// number of data  objects
	int   qn,							// number of query objects
	int   d,							// dimension of space
	int   m, 							// #single hasher of the compond hasher
	int   l,							// #hash tables
	float b,							// interval ratio
	const char *conf_name,				// name of configuration
	const char *data_name,				// name of dataset
	const char *method_name,			// name of method
	const char *output_folder,			// output folder
	const float *data,					// data set
	const float *query,					// query set
	const Result *R);					// truth set

// -----------------------------------------------------------------------------
int multilinear_hash(				// point-to-hyperplane NNS using MH
	int   n,							// number of data  objects
	int   qn,							// number of query objects
	int   d,							// dimension of space
	int   M, 							// #proj vecotr used for a single hasher
	int   m, 							// #single hasher of the compond hasher
	int   l,							// #hash tables
	float b,							// interval ratio
	const char *conf_name,				// name of configuration
	const char *data_name,				// name of dataset
	const char *method_name,			// name of method
	const char *output_folder,			// output folder
	const float *data,					// data set
	const float *query,					// query set
	const Result *R);					// truth set

// -----------------------------------------------------------------------------
int mfh_hash(						// point-to-hyperplane NNS using MFH
	int   n,							// number of data  objects
	int   qn,							// number of query objects
	int   d,							// dimension of space
	int   m,							// #hash tables
	float b,							// interval ratio
	const char *conf_name,				// name of configuration
	const char *data_name,				// name of dataset
	const char *method_name,			// name of method
	const char *output_folder,			// output folder
	const float *data,					// data set
	const float *query,					// query set
	const Result *R);					// truth set

// -----------------------------------------------------------------------------
int fh_hash(						// point-to-hyperplane NNS using FH
	int   n,							// number of data  objects
	int   qn,							// number of query objects
	int   d,							// dimension of space
	int   m,							// #hash tables
	const char *conf_name,				// name of configuration
	const char *data_name,				// name of dataset
	const char *method_name,			// name of method
	const char *output_folder,			// output folder
	const float *data,					// data set
	const float *query,					// query set
	const Result *R);					// truth set

// -----------------------------------------------------------------------------
int nh_hash(						// point-to-hyperplane NNS using NH
	int   n,							// number of data  objects
	int   qn,							// number of query objects
	int   d,							// dimension of space
	int   m,							// #hash tables
	const char *conf_name,				// name of configuration
	const char *data_name,				// name of dataset
	const char *method_name,			// name of method
	const char *output_folder,			// output folder
	const float *data,					// data set
	const float *query,					// query set
	const Result *R);					// truth set

// -----------------------------------------------------------------------------
int nh_lccs(						// point-to-hyperplane NNS using EH
	int   n,							// number of data  objects
	int   qn,							// number of query objects
	int   d,							// dimension of space
	int   m, 							// #single hasher of the compond hasher
	float w,							// bucket width
	const char *conf_name,				// name of configuration
	const char *data_name,				// name of dataset
	const char *method_name,			// name of method
	const char *output_folder,			// output folder
	const float *data,					// data set
	const float *query,					// query set
	const Result *R);					// truth set

// -----------------------------------------------------------------------------
int mfh_hash_sampling(				// point-to-hyperplane NNS using MFH_Sampling
	int   n,							// number of data  objects
	int   qn,							// number of query objects
	int   d,							// dimension of space
	int   m,							// #hash tables
	int   s,							// scale factor of dimension
	float b,							// interval ratio
	const char *conf_name,				// name of configuration
	const char *data_name,				// name of dataset
	const char *method_name,			// name of method
	const char *output_folder,			// output folder
	const float *data,					// data set
	const float *query,					// query set
	const Result *R);					// truth set
	
// -----------------------------------------------------------------------------
int fh_hash_sampling(				// point-to-hyperplane NNS using FH_Sampling
	int   n,							// number of data  objects
	int   qn,							// number of query objects
	int   d,							// dimension of space
	int   m,							// #hash tables
	int   s,							// scale factor of dimension
	const char *conf_name,				// name of configuration
	const char *data_name,				// name of dataset
	const char *method_name,			// name of method
	const char *output_folder,			// output folder
	const float *data,					// data set
	const float *query,					// query set
	const Result *R);					// truth set

// -----------------------------------------------------------------------------
int nh_hash_sampling(				// point-to-hyperplane NNS using NH_Sampling
	int   n,							// number of data  objects
	int   qn,							// number of query objects
	int   d,							// dimension of space
	int   m,							// #hash tables
	int   s,							// scale factor of dimension
	const char *conf_name,				// name of configuration
	const char *data_name,				// name of dataset
	const char *method_name,			// name of method
	const char *output_folder,			// output folder
	const float *data,					// data set
	const float *query,					// query set
	const Result *R);					// truth set

// -----------------------------------------------------------------------------
int nh_lccs_sampling(				// point-to-hyperplane NNS using EH
	int   n,							// number of data  objects
	int   qn,							// number of query objects
	int   d,							// dimension of space
	int   m, 							// #single hasher of the compond hasher
	float w,							// bucket width
	float s, 							// scale factor of dimension
	const char *conf_name,				// name of configuration
	const char *data_name,				// name of dataset
	const char *method_name,			// name of method
	const char *output_folder,			// output folder
	const float *data,					// data set
	const float *query,					// query set
	const Result *R);					// truth set
