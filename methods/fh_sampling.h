#pragma once

#include <algorithm>
#include <cassert>
#include <vector>
#include <unordered_map>

#include "def.h"
#include "util.h"
#include "pri_queue.h"
#include "rqalsh.h"

// -----------------------------------------------------------------------------
//  FH_Hash_Sampling: Furthest Hyperplane Hash
// -----------------------------------------------------------------------------
class FH_Hash_Sampling {
public:
    FH_Hash_Sampling(				// constructor
		int   n,						// number of data objects
		int   d,						// dimension of data objects
		int   m,						// #hash tables
		int   s,						// scale factor of dimension
		const float *data);				// input data

	// -------------------------------------------------------------------------
    ~FH_Hash_Sampling();			// destructor

	// -------------------------------------------------------------------------
	void display();					// display parameters

	// -------------------------------------------------------------------------
	int nns(						// point-to-hyperplane NNS
		int   top_k,					// top-k value
		int   l,						// separation threshold
		int   cand,						// #candidates
		const float *query,				// input query
		MinK_List *list);				// top-k results (return)

	// -------------------------------------------------------------------------
	int64_t get_memory_usage()		// get memory usage
	{
		int64_t ret = 0;
		ret += sizeof(*this);
		ret += lsh_->get_memory_usage();
		return ret;
	}

protected:
    int    n_pts_;					// number of data objects
	int    dim_;					// dimension of data objects
	int    scale_;					// scale factor of dimension
	int    sample_dim_;				// sample dimension
	int    fh_dim_;					// new data dimension after transformation
	float  M_;						// max l2-norm of o' after transformation
	
	const  float *data_;			// original data objects
	RQALSH *lsh_;					// RQALSH for fh data with sampling

	// -------------------------------------------------------------------------
	void transform_data(			// data transformation
		const  float *data,				// input data
		float  *prob,					// probability vector
		bool   *checked,				// is checked?
		float  &norm,					// norm of fh_data (return)
		int    &sample_d,				// sample dimension (return)
		Result *sample_data);			// sample data (return)
	
	// -------------------------------------------------------------------------
	int sampling(					// sampling coordinate based on prob
		int   d,						// dimension
		const float *prob);				// input probability

	// -------------------------------------------------------------------------
	void transform_query(			// query transformation
		const  float *query,			// input query
		int    &sample_d,				// sample dimension (return)
		Result *sample_query);			// sample query (return)
};
