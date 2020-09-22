#pragma once

#include <algorithm>
#include <cassert>
#include <vector>
#include "../lccs_bucket/bucketAlg/lcs_int.h"

#include "def.h"
#include "util.h"
#include "pri_queue.h"

// -----------------------------------------------------------------------------
//  NH_LCCS: Embedding Hyperplane Hash
// -----------------------------------------------------------------------------
class NH_LCCS {
public:
	using LCCS = mylccs::LCCS_SORT_INT;
	using SigType = int32_t;

	NH_LCCS(						// constructor
		int   n,						// number of input data
		int   d,						// dimension of input data
		int   m,						// #hasher
		float w, 						// bucket width
		const float *data);				// input data

	// -------------------------------------------------------------------------
    virtual ~NH_LCCS();				// destructor

	// -------------------------------------------------------------------------
	void display();					// display parameters

	// -------------------------------------------------------------------------
	virtual int nns(				// point-to-hyperplane NNS
		int   top_k,					// top-k value
		int   cand,						// #candidates
        const float *query,				// input query
		MinK_List *list);				// top-k results (return)

	// -------------------------------------------------------------------------
    virtual void get_sig_data(		// get the signature of data
		const float *data, 				// input data
		float norm,
		SigType* sig) const; 			// signature (return)

	// -------------------------------------------------------------------------
    virtual void get_sig_query(		// get the signature of query
		const float *query,				// input query 
		SigType* sig) const; 			// signature (return)

	// -------------------------------------------------------------------------
	virtual int64_t get_memory_usage() // get memory usage
	{
		int64_t ret = 0;
		ret += sizeof(*this);
		ret += sizeof(float)*projv_.capacity() + sizeof(float)*projb_.capacity() 
			+ sizeof(float)*norms_.capacity();
		ret += bucketerp_.get_memory_usage();
		ret += data_sigs_.get_memory_usage(); 
		return ret;
	}

protected:
	int   n_pts_;					// number of data objects
    int   dim_;						// dimension of data objects
    int   m_;						// #single hasher of the compond hasher
	float w_;
	const float* data_;
    
    std::vector<float> projv_;		// random projection vectors
    std::vector<float> projb_;		// random projection vectors
	std::vector<float> norms_;		// norm of transformed vectors

	NDArray<2, int32_t> data_sigs_;
	float max_norm_;
	LCCS bucketerp_; 				// hash tables

	// -------------------------------------------------------------------------
	void get_norms(const float* data);
};
