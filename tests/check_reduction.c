#include <check.h>

#include <gpuarray/error.h>
#include <gpuarray/reduction.h>
#include <gpuarray/types.h>

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>


extern void *ctx;

void setup(void);
void teardown(void);


/* Defines */
#define ga_assert_ok(e) ck_assert_int_eq(e, GA_NO_ERROR)




/**
 * PRNG based on PCG XSH RR 64/32 (LCG)
 *
 * Used to generate random data for the kernel tests.
 */

/* Forward Declarations */
static       uint32_t pcgRor32 (uint32_t x, uint32_t n);
static       void     pcgSeed  (uint64_t seed);
static       uint32_t pcgRand  (void);
static       double   pcgRand01(void);
/* Definitions */
static       uint64_t pcgS =                   1;/* State */
static const uint64_t pcgM = 6364136223846793005;/* Multiplier */
static const uint64_t pcgA = 1442695040888963407;/* Addend */
static       uint32_t pcgRor32 (uint32_t x, uint32_t n){
	return (n &= 0x1F) ? x>>n | x<<(32-n) : x;
}
static       void     pcgSeed  (uint64_t seed){
	pcgS = seed;
}
static       uint32_t pcgRand  (void){
	pcgS = pcgS*pcgM + pcgA;

	/**
	 * PCG does something akin to an unbalanced Feistel round to blind the LCG
	 * state:
	 *
	 * The rightmost 59 bits are involved in an xorshift by 18.
	 * The leftmost   5 bits select a rotation of the 32 bits 58:27.
	 */

	return pcgRor32((pcgS^(pcgS>>18))>>27, pcgS>>59);
}
static       double   pcgRand01(void){
	uint64_t u = pcgRand(), l = pcgRand();
	uint64_t x = u<<32 | l;
	return x /18446744073709551616.0;
}


/**
 * Test cases.
 */

START_TEST(test_maxandargmax_reduction){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on the first and
	 * third dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,2};

	float*  pSrc    = calloc(1, sizeof(*pSrc)    * dims[0]*dims[1]*dims[2]);
	float*  pMax    = calloc(1, sizeof(*pMax)    *         dims[1]        );
	size_t* pArgmax = calloc(1, sizeof(*pArgmax) *         dims[1]        );

	ck_assert_ptr_ne(pSrc,    NULL);
	ck_assert_ptr_ne(pMax,    NULL);
	ck_assert_ptr_ne(pArgmax, NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pSrc[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaSrc;
	GpuArray gaMax;
	GpuArray gaArgmax;

	ga_assert_ok(GpuArray_empty(&gaSrc,    ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaMax,    ctx, GA_FLOAT, 1, &dims[1], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaArgmax, ctx, GA_SIZE,  1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaSrc,    pSrc, sizeof(*pSrc)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaMax,    -1));  /* 0xFFFFFFFF is a qNaN. */
	ga_assert_ok(GpuArray_memset(&gaArgmax, -1));

	ga_assert_ok(GpuArray_maxandargmax(&gaMax, &gaArgmax, &gaSrc, 2, reduxList));

	ga_assert_ok(GpuArray_read(pMax,    sizeof(*pMax)   *dims[1], &gaMax));
	ga_assert_ok(GpuArray_read(pArgmax, sizeof(*pArgmax)*dims[1], &gaArgmax));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		size_t gtArgmax = 0;
		float  gtMax    = pSrc[(0*dims[1] + j)*dims[2] + 0];

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				float v = pSrc[(i*dims[1] + j)*dims[2] + k];

				if(v > gtMax){
					gtMax    = v;
					gtArgmax = i*dims[2] + k;
				}
			}
		}

		ck_assert_msg(gtMax    == pMax[j],    "Max value mismatch!");
		ck_assert_msg(gtArgmax == pArgmax[j], "Argmax value mismatch!");
	}

	/**
	 * Deallocate.
	 */

	free(pSrc);
	free(pMax);
	free(pArgmax);
	GpuArray_clear(&gaSrc);
	GpuArray_clear(&gaMax);
	GpuArray_clear(&gaArgmax);
}END_TEST

START_TEST(test_maxandargmax_idxtranspose){
	pcgSeed(1);

	/**
	 * We test here the same reduction as test_reduction, except with a
	 * reversed reduxList {2,0} instead of {0,2}. That should lead to a
	 * transposition of the argmax "coordinates" and thus a change in its
	 * "flattened" output version.
	 */

	size_t i,j,k;
	size_t dims[3]     = {32,50,79};
	size_t prodDims    = dims[0]*dims[1]*dims[2];
	size_t rdxDims[1]  = {50};
	size_t rdxProdDims = rdxDims[0];
	const unsigned reduxList[] = {2,0};

	float*  pSrc    = calloc(1, sizeof(*pSrc)    * prodDims);
	float*  pMax    = calloc(1, sizeof(*pMax)    * rdxProdDims);
	size_t* pArgmax = calloc(1, sizeof(*pArgmax) * rdxProdDims);

	ck_assert_ptr_ne(pSrc,    NULL);
	ck_assert_ptr_ne(pMax,    NULL);
	ck_assert_ptr_ne(pArgmax, NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pSrc[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaSrc;
	GpuArray gaMax;
	GpuArray gaArgmax;

	ga_assert_ok(GpuArray_empty(&gaSrc,    ctx, GA_FLOAT, 3, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaMax,    ctx, GA_FLOAT, 1, rdxDims, GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaArgmax, ctx, GA_SIZE,  1, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaSrc,    pSrc, sizeof(*pSrc)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaMax,    -1));  /* 0xFFFFFFFF is a qNaN. */
	ga_assert_ok(GpuArray_memset(&gaArgmax, -1));

	ga_assert_ok(GpuArray_maxandargmax(&gaMax, &gaArgmax, &gaSrc, 2, reduxList));

	ga_assert_ok(GpuArray_read(pMax,    sizeof(*pMax)   *rdxProdDims, &gaMax));
	ga_assert_ok(GpuArray_read(pArgmax, sizeof(*pArgmax)*rdxProdDims, &gaArgmax));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		size_t gtArgmax = 0;
		float  gtMax    = pSrc[(0*dims[1] + j)*dims[2] + 0];

		for(k=0;k<dims[2];k++){
			for(i=0;i<dims[0];i++){
				float v = pSrc[(i*dims[1] + j)*dims[2] + k];

				if(v > gtMax){
					gtMax    = v;
					gtArgmax = k*dims[0] + i;
				}
			}
		}

		ck_assert_msg(gtMax    == pMax[j],    "Max value mismatch!");
		ck_assert_msg(gtArgmax == pArgmax[j], "Argmax value mismatch!");
	}

	/**
	 * Deallocate.
	 */

	free(pSrc);
	free(pMax);
	free(pArgmax);
	GpuArray_clear(&gaSrc);
	GpuArray_clear(&gaMax);
	GpuArray_clear(&gaArgmax);
}END_TEST

START_TEST(test_maxandargmax_veryhighrank){
	pcgSeed(1);

	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const unsigned reduxList[] = {2,4,7,5};

	float*  pSrc    = calloc(1, sizeof(*pSrc)    * prodDims);
	float*  pMax    = calloc(1, sizeof(*pMax)    * rdxProdDims);
	size_t* pArgmax = calloc(1, sizeof(*pArgmax) * rdxProdDims);

	ck_assert_ptr_ne(pSrc,    NULL);
	ck_assert_ptr_ne(pMax,    NULL);
	ck_assert_ptr_ne(pArgmax, NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pSrc[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaSrc;
	GpuArray gaMax;
	GpuArray gaArgmax;

	ga_assert_ok(GpuArray_empty(&gaSrc,    ctx, GA_FLOAT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaMax,    ctx, GA_FLOAT, 4, rdxDims, GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaArgmax, ctx, GA_SIZE,  4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaSrc,    pSrc, sizeof(*pSrc)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaMax,    -1));  /* 0xFFFFFFFF is a qNaN. */
	ga_assert_ok(GpuArray_memset(&gaArgmax, -1));

	ga_assert_ok(GpuArray_maxandargmax(&gaMax, &gaArgmax, &gaSrc, 4, reduxList));

	ga_assert_ok(GpuArray_read(pMax,    sizeof(*pMax)   *rdxProdDims, &gaMax));
	ga_assert_ok(GpuArray_read(pArgmax, sizeof(*pArgmax)*rdxProdDims, &gaArgmax));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					size_t gtArgmax = 0;
					float  gtMax    = pSrc[(((((((i)*dims[1] + j)*dims[2] + 0)*dims[3] + l)*dims[4] + 0)*dims[5] + 0)*dims[6] + o)*dims[7] + 0];

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									float v = pSrc[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];

									if(v > gtMax){
										gtMax    = v;
										gtArgmax = (((k)*dims[4] + m)*dims[7] + p)*dims[5] + n;
									}
								}
							}
						}
					}

					size_t dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					ck_assert_msg(gtMax    == pMax[dstIdx],    "Max value mismatch!");
					ck_assert_msg(gtArgmax == pArgmax[dstIdx], "Argmax value mismatch!");
				}
			}
		}
	}


	/**
	 * Deallocate.
	 */

	free(pSrc);
	free(pMax);
	free(pArgmax);
	GpuArray_clear(&gaSrc);
	GpuArray_clear(&gaMax);
	GpuArray_clear(&gaArgmax);
}END_TEST

START_TEST(test_maxandargmax_alldimsreduced){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,1,2};

	float*  pSrc    = calloc(1, sizeof(*pSrc)    * dims[0]*dims[1]*dims[2]);
	float*  pMax    = calloc(1, sizeof(*pMax)                             );
	size_t* pArgmax = calloc(1, sizeof(*pArgmax)                          );

	ck_assert_ptr_ne(pSrc,    NULL);
	ck_assert_ptr_ne(pMax,    NULL);
	ck_assert_ptr_ne(pArgmax, NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pSrc[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaSrc;
	GpuArray gaMax;
	GpuArray gaArgmax;

	ga_assert_ok(GpuArray_empty(&gaSrc,    ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaMax,    ctx, GA_FLOAT, 0, NULL,     GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaArgmax, ctx, GA_SIZE,  0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaSrc,    pSrc, sizeof(*pSrc)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaMax,    -1));  /* 0xFFFFFFFF is a qNaN. */
	ga_assert_ok(GpuArray_memset(&gaArgmax, -1));

	ga_assert_ok(GpuArray_maxandargmax(&gaMax, &gaArgmax, &gaSrc, 3, reduxList));

	ga_assert_ok(GpuArray_read(pMax,    sizeof(*pMax),    &gaMax));
	ga_assert_ok(GpuArray_read(pArgmax, sizeof(*pArgmax), &gaArgmax));


	/**
	 * Check that the destination tensors are correct.
	 */

	size_t gtArgmax = 0;
	float  gtMax    = pSrc[0];

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				float v = pSrc[(i*dims[1] + j)*dims[2] + k];

				if(v > gtMax){
					gtMax    = v;
					gtArgmax = (i*dims[1] + j)*dims[2] + k;
				}
			}
		}
	}

	ck_assert_msg(gtMax    == pMax[0],    "Max value mismatch!");
	ck_assert_msg(gtArgmax == pArgmax[0], "Argmax value mismatch!");

	/**
	 * Deallocate.
	 */

	free(pSrc);
	free(pMax);
	free(pArgmax);
	GpuArray_clear(&gaSrc);
	GpuArray_clear(&gaMax);
	GpuArray_clear(&gaArgmax);
}END_TEST

START_TEST(test_minandargmin_reduction){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on the first and
	 * third dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,2};

	float*  pSrc    = calloc(1, sizeof(*pSrc)    * dims[0]*dims[1]*dims[2]);
	float*  pMin    = calloc(1, sizeof(*pMin)    *         dims[1]        );
	size_t* pArgmin = calloc(1, sizeof(*pArgmin) *         dims[1]        );

	ck_assert_ptr_ne(pSrc,    NULL);
	ck_assert_ptr_ne(pMin,    NULL);
	ck_assert_ptr_ne(pArgmin, NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pSrc[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaSrc;
	GpuArray gaMin;
	GpuArray gaArgmin;

	ga_assert_ok(GpuArray_empty(&gaSrc,    ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaMin,    ctx, GA_FLOAT, 1, &dims[1], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaArgmin, ctx, GA_SIZE,  1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaSrc,    pSrc, sizeof(*pSrc)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaMin,    -1));  /* 0xFFFFFFFF is a qNaN. */
	ga_assert_ok(GpuArray_memset(&gaArgmin, -1));

	ga_assert_ok(GpuArray_minandargmin(&gaMin, &gaArgmin, &gaSrc, 2, reduxList));

	ga_assert_ok(GpuArray_read(pMin,    sizeof(*pMin)   *dims[1], &gaMin));
	ga_assert_ok(GpuArray_read(pArgmin, sizeof(*pArgmin)*dims[1], &gaArgmin));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		size_t gtArgmin = 0;
		float  gtMin    = pSrc[(0*dims[1] + j)*dims[2] + 0];

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				float v = pSrc[(i*dims[1] + j)*dims[2] + k];

				if(v < gtMin){
					gtMin    = v;
					gtArgmin = i*dims[2] + k;
				}
			}
		}

		ck_assert_msg(gtMin    == pMin[j],    "Min value mismatch!");
		ck_assert_msg(gtArgmin == pArgmin[j], "Argmin value mismatch!");
	}

	/**
	 * Deallocate.
	 */

	free(pSrc);
	free(pMin);
	free(pArgmin);
	GpuArray_clear(&gaSrc);
	GpuArray_clear(&gaMin);
	GpuArray_clear(&gaArgmin);
}END_TEST

START_TEST(test_minandargmin_veryhighrank){
	pcgSeed(1);

	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const unsigned reduxList[] = {2,4,7,5};

	float*  pSrc    = calloc(1, sizeof(*pSrc)    * prodDims);
	float*  pMin    = calloc(1, sizeof(*pMin)    * rdxProdDims);
	size_t* pArgmin = calloc(1, sizeof(*pArgmin) * rdxProdDims);

	ck_assert_ptr_ne(pSrc,    NULL);
	ck_assert_ptr_ne(pMin,    NULL);
	ck_assert_ptr_ne(pArgmin, NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pSrc[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaSrc;
	GpuArray gaMin;
	GpuArray gaArgmin;

	ga_assert_ok(GpuArray_empty(&gaSrc,    ctx, GA_FLOAT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaMin,    ctx, GA_FLOAT, 4, rdxDims, GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaArgmin, ctx, GA_SIZE,  4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaSrc,    pSrc, sizeof(*pSrc)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaMin,    -1));  /* 0xFFFFFFFF is a qNaN. */
	ga_assert_ok(GpuArray_memset(&gaArgmin, -1));

	ga_assert_ok(GpuArray_minandargmin(&gaMin, &gaArgmin, &gaSrc, 4, reduxList));

	ga_assert_ok(GpuArray_read(pMin,    sizeof(*pMin)   *rdxProdDims, &gaMin));
	ga_assert_ok(GpuArray_read(pArgmin, sizeof(*pArgmin)*rdxProdDims, &gaArgmin));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					size_t gtArgmin = 0;
					float  gtMin    = pSrc[(((((((i)*dims[1] + j)*dims[2] + 0)*dims[3] + l)*dims[4] + 0)*dims[5] + 0)*dims[6] + o)*dims[7] + 0];

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									float v = pSrc[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];

									if(v < gtMin){
										gtMin    = v;
										gtArgmin = (((k)*dims[4] + m)*dims[7] + p)*dims[5] + n;
									}
								}
							}
						}
					}

					size_t dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					ck_assert_msg(gtMin    == pMin[dstIdx],    "Min value mismatch!");
					ck_assert_msg(gtArgmin == pArgmin[dstIdx], "Argmin value mismatch!");
				}
			}
		}
	}


	/**
	 * Deallocate.
	 */

	free(pSrc);
	free(pMin);
	free(pArgmin);
	GpuArray_clear(&gaSrc);
	GpuArray_clear(&gaMin);
	GpuArray_clear(&gaArgmin);
}END_TEST

START_TEST(test_minandargmin_alldimsreduced){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,1,2};

	float*  pSrc    = calloc(1, sizeof(*pSrc)    * dims[0]*dims[1]*dims[2]);
	float*  pMin    = calloc(1, sizeof(*pMin)                             );
	size_t* pArgmin = calloc(1, sizeof(*pArgmin)                          );

	ck_assert_ptr_ne(pSrc,    NULL);
	ck_assert_ptr_ne(pMin,    NULL);
	ck_assert_ptr_ne(pArgmin, NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pSrc[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaSrc;
	GpuArray gaMin;
	GpuArray gaArgmin;

	ga_assert_ok(GpuArray_empty(&gaSrc,    ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaMin,    ctx, GA_FLOAT, 0, NULL,     GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaArgmin, ctx, GA_SIZE,  0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaSrc,    pSrc, sizeof(*pSrc)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaMin,    -1));  /* 0xFFFFFFFF is a qNaN. */
	ga_assert_ok(GpuArray_memset(&gaArgmin, -1));

	ga_assert_ok(GpuArray_minandargmin(&gaMin, &gaArgmin, &gaSrc, 3, reduxList));

	ga_assert_ok(GpuArray_read(pMin,    sizeof(*pMin),    &gaMin));
	ga_assert_ok(GpuArray_read(pArgmin, sizeof(*pArgmin), &gaArgmin));


	/**
	 * Check that the destination tensors are correct.
	 */

	size_t gtArgmin = 0;
	float  gtMin    = pSrc[0];

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				float v = pSrc[(i*dims[1] + j)*dims[2] + k];

				if(v < gtMin){
					gtMin    = v;
					gtArgmin = (i*dims[1] + j)*dims[2] + k;
				}
			}
		}
	}

	ck_assert_msg(gtMin    == pMin[0],    "Min value mismatch!");
	ck_assert_msg(gtArgmin == pArgmin[0], "Argmin value mismatch!");

	/**
	 * Deallocate.
	 */

	free(pSrc);
	free(pMin);
	free(pArgmin);
	GpuArray_clear(&gaSrc);
	GpuArray_clear(&gaMin);
	GpuArray_clear(&gaArgmin);
}END_TEST

START_TEST(test_argmax_reduction){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on the first and
	 * third dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,2};

	float*  pSrc    = calloc(1, sizeof(*pSrc)    * dims[0]*dims[1]*dims[2]);
	float*  pMax    = calloc(1, sizeof(*pMax)    *         dims[1]        );
	size_t* pArgmax = calloc(1, sizeof(*pArgmax) *         dims[1]        );

	ck_assert_ptr_ne(pSrc,    NULL);
	ck_assert_ptr_ne(pMax,    NULL);
	ck_assert_ptr_ne(pArgmax, NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pSrc[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaSrc;
	GpuArray gaArgmax;

	ga_assert_ok(GpuArray_empty(&gaSrc,    ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaArgmax, ctx, GA_SIZE,  1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaSrc,    pSrc, sizeof(*pSrc)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaArgmax, -1));

	ga_assert_ok(GpuArray_argmax(&gaArgmax, &gaSrc, 2, reduxList));

	ga_assert_ok(GpuArray_read(pArgmax, sizeof(*pArgmax)*dims[1], &gaArgmax));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		size_t gtArgmax = 0;
		float  gtMax    = pSrc[(0*dims[1] + j)*dims[2] + 0];

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				float v = pSrc[(i*dims[1] + j)*dims[2] + k];

				if(v > gtMax){
					gtMax    = v;
					gtArgmax = i*dims[2] + k;
				}
			}
		}

		ck_assert_msg(gtArgmax == pArgmax[j], "Argmax value mismatch!");
	}

	/**
	 * Deallocate.
	 */

	free(pSrc);
	free(pMax);
	free(pArgmax);
	GpuArray_clear(&gaSrc);
	GpuArray_clear(&gaArgmax);
}END_TEST

START_TEST(test_argmax_veryhighrank){
	pcgSeed(1);

	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const unsigned reduxList[] = {2,4,7,5};

	float*  pSrc    = calloc(1, sizeof(*pSrc)    * prodDims);
	float*  pMax    = calloc(1, sizeof(*pMax)    * rdxProdDims);
	size_t* pArgmax = calloc(1, sizeof(*pArgmax) * rdxProdDims);

	ck_assert_ptr_ne(pSrc,    NULL);
	ck_assert_ptr_ne(pArgmax, NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pSrc[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaSrc;
	GpuArray gaArgmax;

	ga_assert_ok(GpuArray_empty(&gaSrc,    ctx, GA_FLOAT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaArgmax, ctx, GA_SIZE,  4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaSrc,    pSrc, sizeof(*pSrc)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaArgmax, -1));

	ga_assert_ok(GpuArray_argmax(&gaArgmax, &gaSrc, 4, reduxList));

	ga_assert_ok(GpuArray_read(pArgmax, sizeof(*pArgmax)*rdxProdDims, &gaArgmax));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					size_t gtArgmax = 0;
					float  gtMax    = pSrc[(((((((i)*dims[1] + j)*dims[2] + 0)*dims[3] + l)*dims[4] + 0)*dims[5] + 0)*dims[6] + o)*dims[7] + 0];

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									float v = pSrc[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];

									if(v > gtMax){
										gtMax    = v;
										gtArgmax = (((k)*dims[4] + m)*dims[7] + p)*dims[5] + n;
									}
								}
							}
						}
					}

					size_t dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					ck_assert_msg(gtArgmax == pArgmax[dstIdx], "Argmax value mismatch!");
				}
			}
		}
	}


	/**
	 * Deallocate.
	 */

	free(pSrc);
	free(pMax);
	free(pArgmax);
	GpuArray_clear(&gaSrc);
	GpuArray_clear(&gaArgmax);
}END_TEST

START_TEST(test_argmax_alldimsreduced){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,1,2};

	float*  pSrc    = calloc(1, sizeof(*pSrc)    * dims[0]*dims[1]*dims[2]);
	float*  pMax    = calloc(1, sizeof(*pMax)                             );
	size_t* pArgmax = calloc(1, sizeof(*pArgmax)                          );

	ck_assert_ptr_ne(pSrc,    NULL);
	ck_assert_ptr_ne(pMax,    NULL);
	ck_assert_ptr_ne(pArgmax, NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pSrc[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaSrc;
	GpuArray gaArgmax;

	ga_assert_ok(GpuArray_empty(&gaSrc,    ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaArgmax, ctx, GA_SIZE,  0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaSrc,    pSrc, sizeof(*pSrc)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaArgmax, -1));

	ga_assert_ok(GpuArray_argmax(&gaArgmax, &gaSrc, 3, reduxList));

	ga_assert_ok(GpuArray_read(pArgmax, sizeof(*pArgmax), &gaArgmax));


	/**
	 * Check that the destination tensors are correct.
	 */

	size_t gtArgmax = 0;
	float  gtMax    = pSrc[0];

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				float v = pSrc[(i*dims[1] + j)*dims[2] + k];

				if(v > gtMax){
					gtMax    = v;
					gtArgmax = (i*dims[1] + j)*dims[2] + k;
				}
			}
		}
	}

	ck_assert_msg(gtArgmax == pArgmax[0], "Argmax value mismatch!");

	/**
	 * Deallocate.
	 */

	free(pSrc);
	free(pMax);
	free(pArgmax);
	GpuArray_clear(&gaSrc);
	GpuArray_clear(&gaArgmax);
}END_TEST

START_TEST(test_argmin_reduction){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on the first and
	 * third dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,2};

	float*  pSrc    = calloc(1, sizeof(*pSrc)    * dims[0]*dims[1]*dims[2]);
	float*  pMin    = calloc(1, sizeof(*pMin)    *         dims[1]        );
	size_t* pArgmin = calloc(1, sizeof(*pArgmin) *         dims[1]        );

	ck_assert_ptr_ne(pSrc,    NULL);
	ck_assert_ptr_ne(pMin,    NULL);
	ck_assert_ptr_ne(pArgmin, NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pSrc[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaSrc;
	GpuArray gaArgmin;

	ga_assert_ok(GpuArray_empty(&gaSrc,    ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaArgmin, ctx, GA_SIZE,  1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaSrc,    pSrc, sizeof(*pSrc)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaArgmin, -1));

	ga_assert_ok(GpuArray_argmin(&gaArgmin, &gaSrc, 2, reduxList));

	ga_assert_ok(GpuArray_read(pArgmin, sizeof(*pArgmin)*dims[1], &gaArgmin));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		size_t gtArgmin = 0;
		float  gtMin    = pSrc[(0*dims[1] + j)*dims[2] + 0];

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				float v = pSrc[(i*dims[1] + j)*dims[2] + k];

				if(v < gtMin){
					gtMin    = v;
					gtArgmin = i*dims[2] + k;
				}
			}
		}

		ck_assert_msg(gtArgmin == pArgmin[j], "Argmin value mismatch!");
	}

	/**
	 * Deallocate.
	 */

	free(pSrc);
	free(pMin);
	free(pArgmin);
	GpuArray_clear(&gaSrc);
	GpuArray_clear(&gaArgmin);
}END_TEST

START_TEST(test_argmin_veryhighrank){
	pcgSeed(1);

	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const unsigned reduxList[] = {2,4,7,5};

	float*  pSrc    = calloc(1, sizeof(*pSrc)    * prodDims);
	float*  pMin    = calloc(1, sizeof(*pMin)    * rdxProdDims);
	size_t* pArgmin = calloc(1, sizeof(*pArgmin) * rdxProdDims);

	ck_assert_ptr_ne(pSrc,    NULL);
	ck_assert_ptr_ne(pArgmin, NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pSrc[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaSrc;
	GpuArray gaArgmin;

	ga_assert_ok(GpuArray_empty(&gaSrc,    ctx, GA_FLOAT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaArgmin, ctx, GA_SIZE,  4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaSrc,    pSrc, sizeof(*pSrc)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaArgmin, -1));

	ga_assert_ok(GpuArray_argmin(&gaArgmin, &gaSrc, 4, reduxList));

	ga_assert_ok(GpuArray_read(pArgmin, sizeof(*pArgmin)*rdxProdDims, &gaArgmin));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					size_t gtArgmin = 0;
					float  gtMin    = pSrc[(((((((i)*dims[1] + j)*dims[2] + 0)*dims[3] + l)*dims[4] + 0)*dims[5] + 0)*dims[6] + o)*dims[7] + 0];

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									float v = pSrc[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];

									if(v < gtMin){
										gtMin    = v;
										gtArgmin = (((k)*dims[4] + m)*dims[7] + p)*dims[5] + n;
									}
								}
							}
						}
					}

					size_t dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					ck_assert_msg(gtArgmin == pArgmin[dstIdx], "Argmin value mismatch!");
				}
			}
		}
	}


	/**
	 * Deallocate.
	 */

	free(pSrc);
	free(pMin);
	free(pArgmin);
	GpuArray_clear(&gaSrc);
	GpuArray_clear(&gaArgmin);
}END_TEST

START_TEST(test_argmin_alldimsreduced){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,1,2};

	float*  pSrc    = calloc(1, sizeof(*pSrc)    * dims[0]*dims[1]*dims[2]);
	float*  pMin    = calloc(1, sizeof(*pMin)                             );
	size_t* pArgmin = calloc(1, sizeof(*pArgmin)                          );

	ck_assert_ptr_ne(pSrc,    NULL);
	ck_assert_ptr_ne(pMin,    NULL);
	ck_assert_ptr_ne(pArgmin, NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pSrc[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaSrc;
	GpuArray gaArgmin;

	ga_assert_ok(GpuArray_empty(&gaSrc,    ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaArgmin, ctx, GA_SIZE,  0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaSrc,    pSrc, sizeof(*pSrc)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaArgmin, -1));

	ga_assert_ok(GpuArray_argmin(&gaArgmin, &gaSrc, 3, reduxList));

	ga_assert_ok(GpuArray_read(pArgmin, sizeof(*pArgmin), &gaArgmin));


	/**
	 * Check that the destination tensors are correct.
	 */

	size_t gtArgmin = 0;
	float  gtMin    = pSrc[0];

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				float v = pSrc[(i*dims[1] + j)*dims[2] + k];

				if(v < gtMin){
					gtMin    = v;
					gtArgmin = (i*dims[1] + j)*dims[2] + k;
				}
			}
		}
	}

	ck_assert_msg(gtArgmin == pArgmin[0], "Argmin value mismatch!");

	/**
	 * Deallocate.
	 */

	free(pSrc);
	free(pMin);
	free(pArgmin);
	GpuArray_clear(&gaSrc);
	GpuArray_clear(&gaArgmin);
}END_TEST

START_TEST(test_max_reduction){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,2};

	float*  pSrc    = calloc(1, sizeof(*pSrc)    * dims[0]*dims[1]*dims[2]);
	float*  pMax    = calloc(1, sizeof(*pMax)    *         dims[1]        );

	ck_assert_ptr_ne(pSrc,    NULL);
	ck_assert_ptr_ne(pMax,    NULL);


	/**
	 * axitialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pSrc[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaSrc;
	GpuArray gaMax;

	ga_assert_ok(GpuArray_empty(&gaSrc,    ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaMax,    ctx, GA_FLOAT, 1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaSrc,    pSrc, sizeof(*pSrc)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaMax,    -1));  /* 0xFFFFFFFF is a qNaN. */

	ga_assert_ok(GpuArray_max(&gaMax, &gaSrc, 2, reduxList));

	ga_assert_ok(GpuArray_read(pMax,    sizeof(*pMax)   *dims[1], &gaMax));


	/**
	 * Check that the destaxation tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		float  gtMax    = pSrc[(0*dims[1] + j)*dims[2] + 0];

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				float v = pSrc[(i*dims[1] + j)*dims[2] + k];

				if(v > gtMax){
					gtMax    = v;
				}
			}
		}

		ck_assert_msg(gtMax    == pMax[j],    "Max value mismatch!");
	}

	/**
	 * Deallocate.
	 */

	free(pSrc);
	free(pMax);
	GpuArray_clear(&gaSrc);
	GpuArray_clear(&gaMax);
}END_TEST

START_TEST(test_max_veryhighrank){
	pcgSeed(1);

	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const unsigned reduxList[] = {2,4,7,5};

	float*  pSrc    = calloc(1, sizeof(*pSrc)    * prodDims);
	float*  pMax    = calloc(1, sizeof(*pMax)    * rdxProdDims);

	ck_assert_ptr_ne(pSrc,    NULL);
	ck_assert_ptr_ne(pMax,    NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pSrc[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaSrc;
	GpuArray gaMax;

	ga_assert_ok(GpuArray_empty(&gaSrc,    ctx, GA_FLOAT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaMax,    ctx, GA_FLOAT, 4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaSrc,    pSrc, sizeof(*pSrc)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaMax,    -1));  /* 0xFFFFFFFF is a qNaN. */

	ga_assert_ok(GpuArray_max(&gaMax, &gaSrc, 4, reduxList));

	ga_assert_ok(GpuArray_read(pMax,    sizeof(*pMax)   *rdxProdDims, &gaMax));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					float  gtMax    = pSrc[(((((((i)*dims[1] + j)*dims[2] + 0)*dims[3] + l)*dims[4] + 0)*dims[5] + 0)*dims[6] + o)*dims[7] + 0];

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									float v = pSrc[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];

									if(v > gtMax){
										gtMax    = v;
									}
								}
							}
						}
					}

					size_t dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					ck_assert_msg(gtMax    == pMax[dstIdx],    "Max value mismatch!");
				}
			}
		}
	}


	/**
	 * Deallocate.
	 */

	free(pSrc);
	free(pMax);
	GpuArray_clear(&gaSrc);
	GpuArray_clear(&gaMax);
}END_TEST

START_TEST(test_max_alldimsreduced){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,1,2};

	float*  pSrc    = calloc(1, sizeof(*pSrc)    * dims[0]*dims[1]*dims[2]);
	float*  pMax    = calloc(1, sizeof(*pMax)                             );

	ck_assert_ptr_ne(pSrc,    NULL);
	ck_assert_ptr_ne(pMax,    NULL);


	/**
	 * axitialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pSrc[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaSrc;
	GpuArray gaMax;

	ga_assert_ok(GpuArray_empty(&gaSrc,    ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaMax,    ctx, GA_FLOAT, 0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaSrc,    pSrc, sizeof(*pSrc)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaMax,    -1));  /* 0xFFFFFFFF is a qNaN. */

	ga_assert_ok(GpuArray_max(&gaMax, &gaSrc, 3, reduxList));

	ga_assert_ok(GpuArray_read(pMax,    sizeof(*pMax),    &gaMax));


	/**
	 * Check that the destaxation tensors are correct.
	 */

	float  gtMax    = pSrc[0];

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				float v = pSrc[(i*dims[1] + j)*dims[2] + k];

				if(v > gtMax){
					gtMax    = v;
				}
			}
		}
	}

	ck_assert_msg(gtMax    == pMax[0],    "Max value mismatch!");

	/**
	 * Deallocate.
	 */

	free(pSrc);
	free(pMax);
	GpuArray_clear(&gaSrc);
	GpuArray_clear(&gaMax);
}END_TEST

START_TEST(test_min_reduction){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,2};

	float*  pSrc    = calloc(1, sizeof(*pSrc)    * dims[0]*dims[1]*dims[2]);
	float*  pMin    = calloc(1, sizeof(*pMin)    *         dims[1]        );

	ck_assert_ptr_ne(pSrc,    NULL);
	ck_assert_ptr_ne(pMin,    NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pSrc[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaSrc;
	GpuArray gaMin;

	ga_assert_ok(GpuArray_empty(&gaSrc,    ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaMin,    ctx, GA_FLOAT, 1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaSrc,    pSrc, sizeof(*pSrc)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaMin,    -1));  /* 0xFFFFFFFF is a qNaN. */

	ga_assert_ok(GpuArray_min(&gaMin, &gaSrc, 2, reduxList));

	ga_assert_ok(GpuArray_read(pMin,    sizeof(*pMin)   *dims[1], &gaMin));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		float  gtMin    = pSrc[(0*dims[1] + j)*dims[2] + 0];

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				float v = pSrc[(i*dims[1] + j)*dims[2] + k];

				if(v < gtMin){
					gtMin    = v;
				}
			}
		}

		ck_assert_msg(gtMin    == pMin[j],    "Min value mismatch!");
	}

	/**
	 * Deallocate.
	 */

	free(pSrc);
	free(pMin);
	GpuArray_clear(&gaSrc);
	GpuArray_clear(&gaMin);
}END_TEST

START_TEST(test_min_veryhighrank){
	pcgSeed(1);

	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const unsigned reduxList[] = {2,4,7,5};

	float*  pSrc    = calloc(1, sizeof(*pSrc)    * prodDims);
	float*  pMin    = calloc(1, sizeof(*pMin)    * rdxProdDims);

	ck_assert_ptr_ne(pSrc,    NULL);
	ck_assert_ptr_ne(pMin,    NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pSrc[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaSrc;
	GpuArray gaMin;

	ga_assert_ok(GpuArray_empty(&gaSrc,    ctx, GA_FLOAT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaMin,    ctx, GA_FLOAT, 4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaSrc,    pSrc, sizeof(*pSrc)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaMin,    -1));  /* 0xFFFFFFFF is a qNaN. */

	ga_assert_ok(GpuArray_min(&gaMin, &gaSrc, 4, reduxList));

	ga_assert_ok(GpuArray_read(pMin,    sizeof(*pMin)   *rdxProdDims, &gaMin));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					float  gtMin    = pSrc[(((((((i)*dims[1] + j)*dims[2] + 0)*dims[3] + l)*dims[4] + 0)*dims[5] + 0)*dims[6] + o)*dims[7] + 0];

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									float v = pSrc[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];

									if(v < gtMin){
										gtMin    = v;
									}
								}
							}
						}
					}

					size_t dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					ck_assert_msg(gtMin    == pMin[dstIdx],    "Min value mismatch!");
				}
			}
		}
	}


	/**
	 * Deallocate.
	 */

	free(pSrc);
	free(pMin);
	GpuArray_clear(&gaSrc);
	GpuArray_clear(&gaMin);
}END_TEST

START_TEST(test_min_alldimsreduced){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,1,2};

	float*  pSrc    = calloc(1, sizeof(*pSrc)    * dims[0]*dims[1]*dims[2]);
	float*  pMin    = calloc(1, sizeof(*pMin)                             );

	ck_assert_ptr_ne(pSrc,    NULL);
	ck_assert_ptr_ne(pMin,    NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pSrc[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaSrc;
	GpuArray gaMin;

	ga_assert_ok(GpuArray_empty(&gaSrc,    ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaMin,    ctx, GA_FLOAT, 0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaSrc,    pSrc, sizeof(*pSrc)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaMin,    -1));  /* 0xFFFFFFFF is a qNaN. */

	ga_assert_ok(GpuArray_min(&gaMin, &gaSrc, 3, reduxList));

	ga_assert_ok(GpuArray_read(pMin,    sizeof(*pMin),    &gaMin));


	/**
	 * Check that the destination tensors are correct.
	 */

	float  gtMin    = pSrc[0];

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				float v = pSrc[(i*dims[1] + j)*dims[2] + k];

				if(v < gtMin){
					gtMin    = v;
				}
			}
		}
	}

	ck_assert_msg(gtMin    == pMin[0],    "Min value mismatch!");

	/**
	 * Deallocate.
	 */

	free(pSrc);
	free(pMin);
	GpuArray_clear(&gaSrc);
	GpuArray_clear(&gaMin);
}END_TEST

START_TEST(test_sum_reduction){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,2};
	const float TOL = 1e-5;

	float*  pS = calloc(1, sizeof(*pS)    * dims[0]*dims[1]*dims[2]);
	float*  pD = calloc(1, sizeof(*pD)    *         dims[1]        );

	ck_assert_ptr_ne(pS,    NULL);
	ck_assert_ptr_ne(pD,    NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pS[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaS;
	GpuArray gaD;

	ga_assert_ok(GpuArray_empty (&gaS, ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD, ctx, GA_FLOAT, 1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS, pS, sizeof(*pS)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD, -1));  /* 0xFFFFFFFF is a qNaN. */

	ga_assert_ok(GpuArray_sum   (&gaD, &gaS, 2, reduxList));

	ga_assert_ok(GpuArray_read  (pD,   sizeof(*pD)*dims[1], &gaD));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		float  gtD = 0;

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				float v = pS[(i*dims[1] + j)*dims[2] + k];
				gtD += v;
			}
		}

		ck_assert_double_eq_tol(gtD, pD[j], TOL);
	}

	/**
	 * Deallocate.
	 */

	free(pS);
	free(pD);
	GpuArray_clear(&gaS);
	GpuArray_clear(&gaD);
}END_TEST

START_TEST(test_sum_veryhighrank){
	pcgSeed(1);

	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const unsigned reduxList[] = {2,4,7,5};
	const float TOL    = 1e-5;

	float*  pS = calloc(1, sizeof(*pS) * prodDims);
	float*  pD = calloc(1, sizeof(*pD) * rdxProdDims);

	ck_assert_ptr_ne(pS, NULL);
	ck_assert_ptr_ne(pD, NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pS[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaS;
	GpuArray gaD;

	ga_assert_ok(GpuArray_empty (&gaS, ctx, GA_FLOAT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD, ctx, GA_FLOAT, 4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS, pS, sizeof(*pS)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD, -1));  /* 0xFFFFFFFF is a qNaN. */

	ga_assert_ok(GpuArray_sum   (&gaD, &gaS, 4, reduxList));

	ga_assert_ok(GpuArray_read  (pD,   sizeof(*pD)*rdxProdDims, &gaD));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					float gtD = 0;

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									float v = pS[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];
									gtD += v;
								}
							}
						}
					}

					size_t dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					ck_assert_double_eq_tol(gtD, pD[dstIdx], TOL);
				}
			}
		}
	}


	/**
	 * Deallocate.
	 */

	free(pS);
	free(pD);
	GpuArray_clear(&gaS);
	GpuArray_clear(&gaD);
}END_TEST

START_TEST(test_sum_alldimsreduced){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,1,2};
	const float TOL = 1e-5;

	float*  pS = calloc(1, sizeof(*pS)    * dims[0]*dims[1]*dims[2]);
	float*  pD = calloc(1, sizeof(*pD)                             );

	ck_assert_ptr_ne(pS,    NULL);
	ck_assert_ptr_ne(pD,    NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pS[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaS;
	GpuArray gaD;

	ga_assert_ok(GpuArray_empty (&gaS, ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD, ctx, GA_FLOAT, 0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS, pS, sizeof(*pS)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD, -1));  /* 0xFFFFFFFF is a qNaN. */

	ga_assert_ok(GpuArray_sum   (&gaD, &gaS, 3, reduxList));

	ga_assert_ok(GpuArray_read  (pD,   sizeof(*pD), &gaD));


	/**
	 * Check that the destination tensors are correct.
	 */

	float  gtD = 0;

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				float v = pS[(i*dims[1] + j)*dims[2] + k];
				gtD += v;
			}
		}
	}

	ck_assert_double_eq_tol(gtD, pD[0], TOL);

	/**
	 * Deallocate.
	 */

	free(pS);
	free(pD);
	GpuArray_clear(&gaS);
	GpuArray_clear(&gaD);
}END_TEST

START_TEST(test_prod_reduction){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,2};
	const float TOL = 1e-5;

	float*  pS = calloc(1, sizeof(*pS)    * dims[0]*dims[1]*dims[2]);
	float*  pD = calloc(1, sizeof(*pD)    *         dims[1]        );

	ck_assert_ptr_ne(pS,    NULL);
	ck_assert_ptr_ne(pD,    NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pS[i] = (pcgRand01()-0.5)*0.1 + 1;
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaS;
	GpuArray gaD;

	ga_assert_ok(GpuArray_empty (&gaS, ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD, ctx, GA_FLOAT, 1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS, pS, sizeof(*pS)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD, -1));  /* 0xFFFFFFFF is a qNaN. */

	ga_assert_ok(GpuArray_prod  (&gaD, &gaS, 2, reduxList));

	ga_assert_ok(GpuArray_read  (pD,   sizeof(*pD)*dims[1], &gaD));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		float  gtD = 1;

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				float v = pS[(i*dims[1] + j)*dims[2] + k];
				gtD *= v;
			}
		}

		ck_assert_double_eq_tol(gtD, pD[j], TOL);
	}

	/**
	 * Deallocate.
	 */

	free(pS);
	free(pD);
	GpuArray_clear(&gaS);
	GpuArray_clear(&gaD);
}END_TEST

START_TEST(test_prod_veryhighrank){
	pcgSeed(1);

	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const unsigned reduxList[] = {2,4,7,5};
	const float TOL    = 1e-5;

	float*  pS = calloc(1, sizeof(*pS) * prodDims);
	float*  pD = calloc(1, sizeof(*pD) * rdxProdDims);

	ck_assert_ptr_ne(pS, NULL);
	ck_assert_ptr_ne(pD, NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pS[i] = (pcgRand01()-0.5)*0.1 + 1;
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaS;
	GpuArray gaD;

	ga_assert_ok(GpuArray_empty (&gaS, ctx, GA_FLOAT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD, ctx, GA_FLOAT, 4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS, pS, sizeof(*pS)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD, -1));  /* 0xFFFFFFFF is a qNaN. */

	ga_assert_ok(GpuArray_prod  (&gaD, &gaS, 4, reduxList));

	ga_assert_ok(GpuArray_read  (pD,   sizeof(*pD)*rdxProdDims, &gaD));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					float gtD = 1;

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									float v = pS[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];
									gtD *= v;
								}
							}
						}
					}

					size_t dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					ck_assert_double_eq_tol(gtD, pD[dstIdx], TOL);
				}
			}
		}
	}


	/**
	 * Deallocate.
	 */

	free(pS);
	free(pD);
	GpuArray_clear(&gaS);
	GpuArray_clear(&gaD);
}END_TEST

START_TEST(test_prod_alldimsreduced){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,1,2};
	const float TOL = 1e-5;

	float*  pS = calloc(1, sizeof(*pS)    * dims[0]*dims[1]*dims[2]);
	float*  pD = calloc(1, sizeof(*pD)                             );

	ck_assert_ptr_ne(pS,    NULL);
	ck_assert_ptr_ne(pD,    NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pS[i] = (pcgRand01()-0.5)*0.1 + 1;
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaS;
	GpuArray gaD;

	ga_assert_ok(GpuArray_empty (&gaS, ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD, ctx, GA_FLOAT, 0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS, pS, sizeof(*pS)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD, -1));  /* 0xFFFFFFFF is a qNaN. */

	ga_assert_ok(GpuArray_prod  (&gaD, &gaS, 3, reduxList));

	ga_assert_ok(GpuArray_read  (pD,   sizeof(*pD), &gaD));


	/**
	 * Check that the destination tensors are correct.
	 */

	float  gtD = 1;

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				float v = pS[(i*dims[1] + j)*dims[2] + k];
				gtD *= v;
			}
		}
	}

	ck_assert_double_eq_tol(gtD, pD[0], TOL);

	/**
	 * Deallocate.
	 */

	free(pS);
	free(pD);
	GpuArray_clear(&gaS);
	GpuArray_clear(&gaD);
}END_TEST

START_TEST(test_prodnz_reduction){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,2};
	const float TOL = 1e-5;

	float*  pS = calloc(1, sizeof(*pS)    * dims[0]*dims[1]*dims[2]);
	float*  pD = calloc(1, sizeof(*pD)    *         dims[1]        );

	ck_assert_ptr_ne(pS,    NULL);
	ck_assert_ptr_ne(pD,    NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pS[i] = (pcgRand01()-0.5)*0.1 + 1;
		if(pcgRand01()<0.1){
			pS[i] = 0;
		}
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaS;
	GpuArray gaD;

	ga_assert_ok(GpuArray_empty (&gaS, ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD, ctx, GA_FLOAT, 1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS, pS, sizeof(*pS)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD, -1));  /* 0xFFFFFFFF is a qNaN. */

	ga_assert_ok(GpuArray_prodnz(&gaD, &gaS, 2, reduxList));

	ga_assert_ok(GpuArray_read  (pD,   sizeof(*pD)*dims[1], &gaD));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		float  gtD = 1;

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				float v = pS[(i*dims[1] + j)*dims[2] + k];
				gtD *= v==0 ? 1 : v;
			}
		}

		ck_assert_double_eq_tol(gtD, pD[j], TOL);
	}

	/**
	 * Deallocate.
	 */

	free(pS);
	free(pD);
	GpuArray_clear(&gaS);
	GpuArray_clear(&gaD);
}END_TEST

START_TEST(test_prodnz_veryhighrank){
	pcgSeed(1);

	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const unsigned reduxList[] = {2,4,7,5};
	const float TOL    = 1e-5;

	float*  pS = calloc(1, sizeof(*pS) * prodDims);
	float*  pD = calloc(1, sizeof(*pD) * rdxProdDims);

	ck_assert_ptr_ne(pS, NULL);
	ck_assert_ptr_ne(pD, NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pS[i] = (pcgRand01()-0.5)*0.1 + 1;
		if(pcgRand01()<0.1){
			pS[i] = 0;
		}
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaS;
	GpuArray gaD;

	ga_assert_ok(GpuArray_empty (&gaS, ctx, GA_FLOAT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD, ctx, GA_FLOAT, 4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS, pS, sizeof(*pS)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD, -1));  /* 0xFFFFFFFF is a qNaN. */

	ga_assert_ok(GpuArray_prodnz(&gaD, &gaS, 4, reduxList));

	ga_assert_ok(GpuArray_read  (pD,   sizeof(*pD)*rdxProdDims, &gaD));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					float gtD = 1;

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									float v = pS[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];
									gtD *= v==0 ? 1 : v;
								}
							}
						}
					}

					size_t dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					ck_assert_double_eq_tol(gtD, pD[dstIdx], TOL);
				}
			}
		}
	}


	/**
	 * Deallocate.
	 */

	free(pS);
	free(pD);
	GpuArray_clear(&gaS);
	GpuArray_clear(&gaD);
}END_TEST

START_TEST(test_prodnz_alldimsreduced){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,1,2};
	const float TOL = 1e-5;

	float*  pS = calloc(1, sizeof(*pS)    * dims[0]*dims[1]*dims[2]);
	float*  pD = calloc(1, sizeof(*pD)                             );

	ck_assert_ptr_ne(pS,    NULL);
	ck_assert_ptr_ne(pD,    NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		pS[i] = (pcgRand01()-0.5)*0.1 + 1;
		if(pcgRand01()<0.1){
			pS[i] = 0;
		}
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaS;
	GpuArray gaD;

	ga_assert_ok(GpuArray_empty (&gaS, ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD, ctx, GA_FLOAT, 0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS, pS, sizeof(*pS)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD, -1));  /* 0xFFFFFFFF is a qNaN. */

	ga_assert_ok(GpuArray_prodnz(&gaD, &gaS, 3, reduxList));

	ga_assert_ok(GpuArray_read  (pD,   sizeof(*pD), &gaD));


	/**
	 * Check that the destination tensors are correct.
	 */

	float  gtD = 1;

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				float v = pS[(i*dims[1] + j)*dims[2] + k];
				gtD *= v==0 ? 1 : v;
			}
		}
	}

	ck_assert_double_eq_tol(gtD, pD[0], TOL);

	/**
	 * Deallocate.
	 */

	free(pS);
	free(pD);
	GpuArray_clear(&gaS);
	GpuArray_clear(&gaD);
}END_TEST

START_TEST(test_and_reduction){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,2};

	uint32_t*  pS = calloc(1, sizeof(*pS)    * dims[0]*dims[1]*dims[2]);
	uint32_t*  pD = calloc(1, sizeof(*pD)    *         dims[1]        );

	ck_assert_ptr_ne(pS,    NULL);
	ck_assert_ptr_ne(pD,    NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-AND, so the bits should be 1 with high
		 * probability.
		 */

		pS[i]  = (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS[i] |= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS[i] |= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS[i] |= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS[i] |= (uint32_t)(pcgRand01() * (uint32_t)-1);
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaS;
	GpuArray gaD;

	ga_assert_ok(GpuArray_empty (&gaS, ctx, GA_UINT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD, ctx, GA_UINT, 1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS, pS, sizeof(*pS)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD, -1));

	ga_assert_ok(GpuArray_and   (&gaD, &gaS, 2, reduxList));

	ga_assert_ok(GpuArray_read  (pD,   sizeof(*pD)*dims[1], &gaD));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		uint32_t  gtD = -1;

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				uint32_t v = pS[(i*dims[1] + j)*dims[2] + k];
				gtD &= v;
			}
		}

		ck_assert_uint_eq(gtD, pD[j]);
	}

	/**
	 * Deallocate.
	 */

	free(pS);
	free(pD);
	GpuArray_clear(&gaS);
	GpuArray_clear(&gaD);
}END_TEST

START_TEST(test_and_veryhighrank){
	pcgSeed(1);

	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const unsigned reduxList[] = {2,4,7,5};

	uint32_t*  pS = calloc(1, sizeof(*pS) * prodDims);
	uint32_t*  pD = calloc(1, sizeof(*pD) * rdxProdDims);

	ck_assert_ptr_ne(pS, NULL);
	ck_assert_ptr_ne(pD, NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-AND, so the bits should be 1 with high
		 * probability.
		 */

		pS[i]  = (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS[i] |= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS[i] |= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS[i] |= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS[i] |= (uint32_t)(pcgRand01() * (uint32_t)-1);
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaS;
	GpuArray gaD;

	ga_assert_ok(GpuArray_empty (&gaS, ctx, GA_UINT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD, ctx, GA_UINT, 4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS, pS, sizeof(*pS)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD, -1));  /* 0xFFFFFFFF is a qNaN. */

	ga_assert_ok(GpuArray_and   (&gaD, &gaS, 4, reduxList));

	ga_assert_ok(GpuArray_read  (pD,   sizeof(*pD)*rdxProdDims, &gaD));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					uint32_t gtD = -1;

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									uint32_t v = pS[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];
									gtD &= v;
								}
							}
						}
					}

					size_t dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					ck_assert_uint_eq(gtD, pD[dstIdx]);
				}
			}
		}
	}


	/**
	 * Deallocate.
	 */

	free(pS);
	free(pD);
	GpuArray_clear(&gaS);
	GpuArray_clear(&gaD);
}END_TEST

START_TEST(test_and_alldimsreduced){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,1,2};

	uint32_t*  pS = calloc(1, sizeof(*pS)    * dims[0]*dims[1]*dims[2]);
	uint32_t*  pD = calloc(1, sizeof(*pD)                             );

	ck_assert_ptr_ne(pS,    NULL);
	ck_assert_ptr_ne(pD,    NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-AND, so the bits should be 1 with high
		 * probability.
		 */

		pS[i]  = (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS[i] |= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS[i] |= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS[i] |= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS[i] |= (uint32_t)(pcgRand01() * (uint32_t)-1);
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaS;
	GpuArray gaD;

	ga_assert_ok(GpuArray_empty (&gaS, ctx, GA_UINT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD, ctx, GA_UINT, 0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS, pS, sizeof(*pS)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD, -1));  /* 0xFFFFFFFF is a qNaN. */

	ga_assert_ok(GpuArray_and   (&gaD, &gaS, 3, reduxList));

	ga_assert_ok(GpuArray_read  (pD,   sizeof(*pD), &gaD));


	/**
	 * Check that the destination tensors are correct.
	 */

	uint32_t  gtD = -1;

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				uint32_t v = pS[(i*dims[1] + j)*dims[2] + k];
				gtD &= v;
			}
		}
	}

	ck_assert_uint_eq(gtD, pD[0]);

	/**
	 * Deallocate.
	 */

	free(pS);
	free(pD);
	GpuArray_clear(&gaS);
	GpuArray_clear(&gaD);
}END_TEST

START_TEST(test_or_reduction){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,2};

	uint32_t*  pS = calloc(1, sizeof(*pS)    * dims[0]*dims[1]*dims[2]);
	uint32_t*  pD = calloc(1, sizeof(*pD)    *         dims[1]        );

	ck_assert_ptr_ne(pS,    NULL);
	ck_assert_ptr_ne(pD,    NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-OR, so the bits should be 0 with high
		 * probability.
		 */

		pS[i]  = (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS[i] &= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS[i] &= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS[i] &= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS[i] &= (uint32_t)(pcgRand01() * (uint32_t)-1);
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaS;
	GpuArray gaD;

	ga_assert_ok(GpuArray_empty (&gaS, ctx, GA_UINT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD, ctx, GA_UINT, 1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS, pS, sizeof(*pS)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD, -1));

	ga_assert_ok(GpuArray_or    (&gaD, &gaS, 2, reduxList));

	ga_assert_ok(GpuArray_read  (pD,   sizeof(*pD)*dims[1], &gaD));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		uint32_t  gtD = 0;

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				uint32_t v = pS[(i*dims[1] + j)*dims[2] + k];
				gtD |= v;
			}
		}

		ck_assert_uint_eq(gtD, pD[j]);
	}

	/**
	 * Deallocate.
	 */

	free(pS);
	free(pD);
	GpuArray_clear(&gaS);
	GpuArray_clear(&gaD);
}END_TEST

START_TEST(test_or_veryhighrank){
	pcgSeed(1);

	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const unsigned reduxList[] = {2,4,7,5};

	uint32_t*  pS = calloc(1, sizeof(*pS) * prodDims);
	uint32_t*  pD = calloc(1, sizeof(*pD) * rdxProdDims);

	ck_assert_ptr_ne(pS, NULL);
	ck_assert_ptr_ne(pD, NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-OR, so the bits should be 0 with high
		 * probability.
		 */

		pS[i]  = (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS[i] &= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS[i] &= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS[i] &= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS[i] &= (uint32_t)(pcgRand01() * (uint32_t)-1);
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaS;
	GpuArray gaD;

	ga_assert_ok(GpuArray_empty (&gaS, ctx, GA_UINT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD, ctx, GA_UINT, 4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS, pS, sizeof(*pS)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD, -1));  /* 0xFFFFFFFF is a qNaN. */

	ga_assert_ok(GpuArray_or    (&gaD, &gaS, 4, reduxList));

	ga_assert_ok(GpuArray_read  (pD,   sizeof(*pD)*rdxProdDims, &gaD));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					uint32_t gtD = 0;

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									uint32_t v = pS[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];
									gtD |= v;
								}
							}
						}
					}

					size_t dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					ck_assert_uint_eq(gtD, pD[dstIdx]);
				}
			}
		}
	}


	/**
	 * Deallocate.
	 */

	free(pS);
	free(pD);
	GpuArray_clear(&gaS);
	GpuArray_clear(&gaD);
}END_TEST

START_TEST(test_or_alldimsreduced){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,1,2};

	uint32_t*  pS = calloc(1, sizeof(*pS)    * dims[0]*dims[1]*dims[2]);
	uint32_t*  pD = calloc(1, sizeof(*pD)                             );

	ck_assert_ptr_ne(pS,    NULL);
	ck_assert_ptr_ne(pD,    NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-OR, so the bits should be 0 with high
		 * probability.
		 */

		pS[i]  = (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS[i] &= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS[i] &= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS[i] &= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS[i] &= (uint32_t)(pcgRand01() * (uint32_t)-1);
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaS;
	GpuArray gaD;

	ga_assert_ok(GpuArray_empty (&gaS, ctx, GA_UINT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD, ctx, GA_UINT, 0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS, pS, sizeof(*pS)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD, -1));  /* 0xFFFFFFFF is a qNaN. */

	ga_assert_ok(GpuArray_or    (&gaD, &gaS, 3, reduxList));

	ga_assert_ok(GpuArray_read  (pD,   sizeof(*pD), &gaD));


	/**
	 * Check that the destination tensors are correct.
	 */

	uint32_t  gtD = 0;

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				uint32_t v = pS[(i*dims[1] + j)*dims[2] + k];
				gtD |= v;
			}
		}
	}

	ck_assert_uint_eq(gtD, pD[0]);

	/**
	 * Deallocate.
	 */

	free(pS);
	free(pD);
	GpuArray_clear(&gaS);
	GpuArray_clear(&gaD);
}END_TEST

START_TEST(test_xor_reduction){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,2};

	uint32_t*  pS = calloc(1, sizeof(*pS)    * dims[0]*dims[1]*dims[2]);
	uint32_t*  pD = calloc(1, sizeof(*pD)    *         dims[1]        );

	ck_assert_ptr_ne(pS,    NULL);
	ck_assert_ptr_ne(pD,    NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-XOR, so the bits should be 1 with even
		 * probability.
		 */

		pS[i]  = (uint32_t)(pcgRand01() * (uint32_t)-1);
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaS;
	GpuArray gaD;

	ga_assert_ok(GpuArray_empty (&gaS, ctx, GA_UINT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD, ctx, GA_UINT, 1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS, pS, sizeof(*pS)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD, -1));

	ga_assert_ok(GpuArray_xor   (&gaD, &gaS, 2, reduxList));

	ga_assert_ok(GpuArray_read  (pD,   sizeof(*pD)*dims[1], &gaD));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		uint32_t  gtD = 0;

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				uint32_t v = pS[(i*dims[1] + j)*dims[2] + k];
				gtD ^= v;
			}
		}

		ck_assert_uint_eq(gtD, pD[j]);
	}

	/**
	 * Deallocate.
	 */

	free(pS);
	free(pD);
	GpuArray_clear(&gaS);
	GpuArray_clear(&gaD);
}END_TEST

START_TEST(test_xor_veryhighrank){
	pcgSeed(1);

	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const unsigned reduxList[] = {2,4,7,5};

	uint32_t*  pS = calloc(1, sizeof(*pS) * prodDims);
	uint32_t*  pD = calloc(1, sizeof(*pD) * rdxProdDims);

	ck_assert_ptr_ne(pS, NULL);
	ck_assert_ptr_ne(pD, NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-XOR, so the bits should be 1 with even
		 * probability.
		 */

		pS[i]  = (uint32_t)(pcgRand01() * (uint32_t)-1);
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaS;
	GpuArray gaD;

	ga_assert_ok(GpuArray_empty (&gaS, ctx, GA_UINT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD, ctx, GA_UINT, 4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS, pS, sizeof(*pS)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD, -1));  /* 0xFFFFFFFF is a qNaN. */

	ga_assert_ok(GpuArray_xor   (&gaD, &gaS, 4, reduxList));

	ga_assert_ok(GpuArray_read  (pD,   sizeof(*pD)*rdxProdDims, &gaD));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					uint32_t gtD = 0;

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									uint32_t v = pS[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];
									gtD ^= v;
								}
							}
						}
					}

					size_t dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					ck_assert_uint_eq(gtD, pD[dstIdx]);
				}
			}
		}
	}


	/**
	 * Deallocate.
	 */

	free(pS);
	free(pD);
	GpuArray_clear(&gaS);
	GpuArray_clear(&gaD);
}END_TEST

START_TEST(test_xor_alldimsreduced){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,1,2};

	uint32_t*  pS = calloc(1, sizeof(*pS)    * dims[0]*dims[1]*dims[2]);
	uint32_t*  pD = calloc(1, sizeof(*pD)                             );

	ck_assert_ptr_ne(pS,    NULL);
	ck_assert_ptr_ne(pD,    NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-XOR, so the bits should be 1 with even
		 * probability.
		 */

		pS[i]  = (uint32_t)(pcgRand01() * (uint32_t)-1);
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaS;
	GpuArray gaD;

	ga_assert_ok(GpuArray_empty (&gaS, ctx, GA_UINT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD, ctx, GA_UINT, 0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS, pS, sizeof(*pS)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD, -1));  /* 0xFFFFFFFF is a qNaN. */

	ga_assert_ok(GpuArray_xor   (&gaD, &gaS, 3, reduxList));

	ga_assert_ok(GpuArray_read  (pD,   sizeof(*pD), &gaD));


	/**
	 * Check that the destination tensors are correct.
	 */

	uint32_t  gtD = 0;

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				uint32_t v = pS[(i*dims[1] + j)*dims[2] + k];
				gtD ^= v;
			}
		}
	}

	ck_assert_uint_eq(gtD, pD[0]);

	/**
	 * Deallocate.
	 */

	free(pS);
	free(pD);
	GpuArray_clear(&gaS);
	GpuArray_clear(&gaD);
}END_TEST

START_TEST(test_any_reduction){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,2};

	uint32_t*  pS = calloc(1, sizeof(*pS)    * dims[0]*dims[1]*dims[2]);
	uint32_t*  pD = calloc(1, sizeof(*pD)    *         dims[1]        );

	ck_assert_ptr_ne(pS,    NULL);
	ck_assert_ptr_ne(pD,    NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-ANY, so the values should be 0 with high
		 * probability.
		 */

		pS[i]  = pcgRand01() < 0.05;
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaS;
	GpuArray gaD;

	ga_assert_ok(GpuArray_empty (&gaS, ctx, GA_UINT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD, ctx, GA_UINT, 1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS, pS, sizeof(*pS)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD, -1));

	ga_assert_ok(GpuArray_any   (&gaD, &gaS, 2, reduxList));

	ga_assert_ok(GpuArray_read  (pD,   sizeof(*pD)*dims[1], &gaD));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		uint32_t  gtD = 0;

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				uint32_t v = pS[(i*dims[1] + j)*dims[2] + k];
				gtD = gtD || v;
			}
		}

		ck_assert_uint_eq(gtD, pD[j]);
	}

	/**
	 * Deallocate.
	 */

	free(pS);
	free(pD);
	GpuArray_clear(&gaS);
	GpuArray_clear(&gaD);
}END_TEST

START_TEST(test_any_veryhighrank){
	pcgSeed(1);

	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const unsigned reduxList[] = {2,4,7,5};

	uint32_t*  pS = calloc(1, sizeof(*pS) * prodDims);
	uint32_t*  pD = calloc(1, sizeof(*pD) * rdxProdDims);

	ck_assert_ptr_ne(pS, NULL);
	ck_assert_ptr_ne(pD, NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-ANY, so the values should be 0 with high
		 * probability.
		 */

		pS[i]  = pcgRand01() < 0.05;
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaS;
	GpuArray gaD;

	ga_assert_ok(GpuArray_empty (&gaS, ctx, GA_UINT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD, ctx, GA_UINT, 4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS, pS, sizeof(*pS)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD, -1));  /* 0xFFFFFFFF is a qNaN. */

	ga_assert_ok(GpuArray_any   (&gaD, &gaS, 4, reduxList));

	ga_assert_ok(GpuArray_read  (pD,   sizeof(*pD)*rdxProdDims, &gaD));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					uint32_t gtD = 0;

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									uint32_t v = pS[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];
									gtD = gtD || v;
								}
							}
						}
					}

					size_t dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					ck_assert_uint_eq(gtD, pD[dstIdx]);
				}
			}
		}
	}


	/**
	 * Deallocate.
	 */

	free(pS);
	free(pD);
	GpuArray_clear(&gaS);
	GpuArray_clear(&gaD);
}END_TEST

START_TEST(test_any_alldimsreduced){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,1,2};

	uint32_t*  pS = calloc(1, sizeof(*pS)    * dims[0]*dims[1]*dims[2]);
	uint32_t*  pD = calloc(1, sizeof(*pD)                             );

	ck_assert_ptr_ne(pS,    NULL);
	ck_assert_ptr_ne(pD,    NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-ANY, so the values should be 0 with high
		 * probability.
		 */

		pS[i]  = pcgRand01() < 0.05;
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaS;
	GpuArray gaD;

	ga_assert_ok(GpuArray_empty (&gaS, ctx, GA_UINT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD, ctx, GA_UINT, 0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS, pS, sizeof(*pS)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD, -1));  /* 0xFFFFFFFF is a qNaN. */

	ga_assert_ok(GpuArray_any   (&gaD, &gaS, 3, reduxList));

	ga_assert_ok(GpuArray_read  (pD,   sizeof(*pD), &gaD));


	/**
	 * Check that the destination tensors are correct.
	 */

	uint32_t  gtD = 0;

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				uint32_t v = pS[(i*dims[1] + j)*dims[2] + k];
				gtD = gtD || v;
			}
		}
	}

	ck_assert_uint_eq(gtD, pD[0]);

	/**
	 * Deallocate.
	 */

	free(pS);
	free(pD);
	GpuArray_clear(&gaS);
	GpuArray_clear(&gaD);
}END_TEST

START_TEST(test_all_reduction){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,2};

	uint32_t*  pS = calloc(1, sizeof(*pS)    * dims[0]*dims[1]*dims[2]);
	uint32_t*  pD = calloc(1, sizeof(*pD)    *         dims[1]        );

	ck_assert_ptr_ne(pS,    NULL);
	ck_assert_ptr_ne(pD,    NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-ALL, so the values should be non-0 with high
		 * probability.
		 */

		pS[i]  = pcgRand01() > 0.05;
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaS;
	GpuArray gaD;

	ga_assert_ok(GpuArray_empty (&gaS, ctx, GA_UINT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD, ctx, GA_UINT, 1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS, pS, sizeof(*pS)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD, -1));

	ga_assert_ok(GpuArray_all   (&gaD, &gaS, 2, reduxList));

	ga_assert_ok(GpuArray_read  (pD,   sizeof(*pD)*dims[1], &gaD));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		uint32_t  gtD = 1;

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				uint32_t v = pS[(i*dims[1] + j)*dims[2] + k];
				gtD = gtD && v;
			}
		}

		ck_assert_uint_eq(gtD, pD[j]);
	}

	/**
	 * Deallocate.
	 */

	free(pS);
	free(pD);
	GpuArray_clear(&gaS);
	GpuArray_clear(&gaD);
}END_TEST

START_TEST(test_all_veryhighrank){
	pcgSeed(1);

	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const unsigned reduxList[] = {2,4,7,5};

	uint32_t*  pS = calloc(1, sizeof(*pS) * prodDims);
	uint32_t*  pD = calloc(1, sizeof(*pD) * rdxProdDims);

	ck_assert_ptr_ne(pS, NULL);
	ck_assert_ptr_ne(pD, NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-ALL, so the values should be non-0 with high
		 * probability.
		 */

		pS[i]  = pcgRand01() > 0.05;
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaS;
	GpuArray gaD;

	ga_assert_ok(GpuArray_empty (&gaS, ctx, GA_UINT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD, ctx, GA_UINT, 4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS, pS, sizeof(*pS)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD, -1));  /* 0xFFFFFFFF is a qNaN. */

	ga_assert_ok(GpuArray_all   (&gaD, &gaS, 4, reduxList));

	ga_assert_ok(GpuArray_read  (pD,   sizeof(*pD)*rdxProdDims, &gaD));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					uint32_t gtD = 1;

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									uint32_t v = pS[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];
									gtD = gtD && v;
								}
							}
						}
					}

					size_t dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					ck_assert_uint_eq(gtD, pD[dstIdx]);
				}
			}
		}
	}


	/**
	 * Deallocate.
	 */

	free(pS);
	free(pD);
	GpuArray_clear(&gaS);
	GpuArray_clear(&gaD);
}END_TEST

START_TEST(test_all_alldimsreduced){
	pcgSeed(1);

	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,1,2};

	uint32_t*  pS = calloc(1, sizeof(*pS)    * dims[0]*dims[1]*dims[2]);
	uint32_t*  pD = calloc(1, sizeof(*pD)                             );

	ck_assert_ptr_ne(pS,    NULL);
	ck_assert_ptr_ne(pD,    NULL);


	/**
	 * Initialize source data.
	 */

	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-ALL, so the values should be non-0 with high
		 * probability.
		 */

		pS[i]  = pcgRand01() > 0.05;
	}


	/**
	 * Run the kernel.
	 */

	GpuArray gaS;
	GpuArray gaD;

	ga_assert_ok(GpuArray_empty (&gaS, ctx, GA_UINT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD, ctx, GA_UINT, 0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS, pS, sizeof(*pS)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD, -1));  /* 0xFFFFFFFF is a qNaN. */

	ga_assert_ok(GpuArray_all   (&gaD, &gaS, 3, reduxList));

	ga_assert_ok(GpuArray_read  (pD,   sizeof(*pD), &gaD));


	/**
	 * Check that the destination tensors are correct.
	 */

	uint32_t  gtD = 1;

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				uint32_t v = pS[(i*dims[1] + j)*dims[2] + k];
				gtD = gtD && v;
			}
		}
	}

	ck_assert_uint_eq(gtD, pD[0]);

	/**
	 * Deallocate.
	 */

	free(pS);
	free(pD);
	GpuArray_clear(&gaS);
	GpuArray_clear(&gaD);
}END_TEST

Suite *get_suite(void) {
	Suite *s  = suite_create("reduction");
	TCase *tc = tcase_create("basic");
	tcase_add_checked_fixture(tc, setup, teardown);
	tcase_set_timeout(tc, 120.0);

	tcase_add_test(tc, test_maxandargmax_reduction);
	tcase_add_test(tc, test_maxandargmax_idxtranspose);
	tcase_add_test(tc, test_maxandargmax_veryhighrank);
	tcase_add_test(tc, test_maxandargmax_alldimsreduced);

	tcase_add_test(tc, test_minandargmin_reduction);
	tcase_add_test(tc, test_minandargmin_veryhighrank);
	tcase_add_test(tc, test_minandargmin_alldimsreduced);

	tcase_add_test(tc, test_argmax_reduction);
	tcase_add_test(tc, test_argmax_veryhighrank);
	tcase_add_test(tc, test_argmax_alldimsreduced);

	tcase_add_test(tc, test_argmin_reduction);
	tcase_add_test(tc, test_argmin_veryhighrank);
	tcase_add_test(tc, test_argmin_alldimsreduced);

	tcase_add_test(tc, test_max_reduction);
	tcase_add_test(tc, test_max_veryhighrank);
	tcase_add_test(tc, test_max_alldimsreduced);

	tcase_add_test(tc, test_min_reduction);
	tcase_add_test(tc, test_min_veryhighrank);
	tcase_add_test(tc, test_min_alldimsreduced);

	tcase_add_test(tc, test_sum_reduction);
	tcase_add_test(tc, test_sum_veryhighrank);
	tcase_add_test(tc, test_sum_alldimsreduced);

	tcase_add_test(tc, test_prod_reduction);
	tcase_add_test(tc, test_prod_veryhighrank);
	tcase_add_test(tc, test_prod_alldimsreduced);

	tcase_add_test(tc, test_prodnz_reduction);
	tcase_add_test(tc, test_prodnz_veryhighrank);
	tcase_add_test(tc, test_prodnz_alldimsreduced);

	tcase_add_test(tc, test_and_reduction);
	tcase_add_test(tc, test_and_veryhighrank);
	tcase_add_test(tc, test_and_alldimsreduced);

	tcase_add_test(tc, test_or_reduction);
	tcase_add_test(tc, test_or_veryhighrank);
	tcase_add_test(tc, test_or_alldimsreduced);

	tcase_add_test(tc, test_xor_reduction);
	tcase_add_test(tc, test_xor_veryhighrank);
	tcase_add_test(tc, test_xor_alldimsreduced);

	tcase_add_test(tc, test_any_reduction);
	tcase_add_test(tc, test_any_veryhighrank);
	tcase_add_test(tc, test_any_alldimsreduced);

	tcase_add_test(tc, test_all_reduction);
	tcase_add_test(tc, test_all_veryhighrank);
	tcase_add_test(tc, test_all_alldimsreduced);

	suite_add_tcase(s, tc);
	return s;
}

