//Convolutional case: half the forward pass without dropout. finish the forward pass 4 times in parallel with lots of dropout, first half of backprop in parallel. Sum derivatives to finish the backward pass
//Implement dropout pretraining for ReLU units



// Ben Graham, University of Warwick, 2014
// Batch-wise and sample-wise dropout

// N.B. BatchWiseDropoutLayer applies dropout to the output layer, so output.featuresPresent.size() <= output.nFeatures
//      It therefore has to accept input hidden layers with input.featuresPresent.size() <= input.nFeatures
//      Other layer-types assume input.featuresPresent.size() == input.nFeatures
//      SimpleLayer does no dropout at all.
//      SampleWiseDropoutLayer applies dropout to the input layer (by multiplying by zero). The output layer has full size.


#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <vector>
#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/thread.hpp>
#include "cuda.h"
#include <cublas_v2.h>
using namespace std;

#include "Rng.h"
#include "CudaUtils.h"
#include "Batches.h"
#include "Layer.h"
#include "SigmoidLayer.h"
#include "BatchWiseDropoutLayer.h"
#include "SampleWiseDropoutLayer.h"
#include "SimpleLayer.h"
#include "SoftmaxClassifier.h"
#include "Dataset.h"
#include "BatchProducer.h"
#include "ANN.h"
#include "NetworkArchitectures.h"
