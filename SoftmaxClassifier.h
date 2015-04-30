////////////////////////////////////////////////////////////////////////////////////////////////
//Calculate softmaxProbability(i) - indicator(i=label)
// for i=0,1,...N-1 with N the number of character classes.
__global__ void dDerivativeOfCostWRTpreSoftmaxTopLevelWeights
(int batchSize, float* topDelta, float* topGrid,
 int* labels, int N) {
  for (int k=0;k<batchSize;k++)
    for(int i=threadIdx.x;i<N;i+=NTHREADS) {
      topDelta[k*N+i]=topGrid[k*N+i];
      if (i==labels[k])
        topDelta[k*N+i]-=1;
    }
}
__global__ void dClassifyAndSubtractOne
(float* d_probs, int* d_predictions, int batchSize, int nOut) {
  for (int i = threadIdx.x;i<batchSize;i+=NTHREADS) {
    int prediction=0;
    float maxP=d_probs[i*nOut];
    for (int k=1;k<nOut;k++) {
      if (d_probs[i*nOut+k]>maxP) {
        prediction=k;
        maxP=d_probs[i*nOut+k];
      }
    }
    d_probs[i*nOut+prediction]-=1;
    d_predictions[i]=prediction;
  }
}


//Assume no dropout in the output layer! nClasses:=input.nFeatures.
vector<vector<int> > SoftmaxClassifier(BatchInterface& input, int nTop) {
  assert(input.nFeatures==input.featuresPresent.size()); //Could bypass this requirement for training batches so long as all training labels present in the batch are a subset of elements in input.featuresPresent; useful if the number of classes is very large ?!?
  vectorCUDA<float> probs(true, input.features.size());
  cudaSafeCall(cudaMemcpy(probs.dPtr(),input.features.dPtr(), input.features.size()*sizeof(float), cudaMemcpyDeviceToDevice));
  vectorCUDA<int> pred(true, nTop*input.batchSize);
  for (int j=0;j<nTop;j++)
    dClassifyAndSubtractOne<<<1,NTHREADS>>>
      (probs.dPtr(), pred.dPtr()+j*input.batchSize,
       input.batchSize, input.nFeatures);
  cudaCheckError();

  vector<vector<int> > predictions(input.batchSize);
  vector<int> &p=pred.hVector();
  for (int i=0;i<input.batchSize;i++)
    for (int j=0;j<nTop;j++)
      predictions[i].push_back(p[j*input.batchSize+i]);
  return predictions;
}


vector<vector<int> > SoftmaxClassifier(BatchInterface& input, int nTop, vectorCUDA<int> &labels, int& mistakes) {
  vector<vector<int> > predictions=SoftmaxClassifier(input, nTop);

  mistakes+=input.batchSize;
  for (int i=0;i<input.batchSize;i++) {
    for (int j=0;j<nTop;j++) {
      if (predictions[i][j]==labels.hVector()[i]) {
        mistakes--;
      }
    }
  }
  if (input.type==TRAINBATCH) {   //Begin backprop calculation
    //top layer: d Cost / d SoftmaxInput
    dDerivativeOfCostWRTpreSoftmaxTopLevelWeights<<<1,NTHREADS>>>
      (input.batchSize, input.dfeatures.dPtr(), input.features.dPtr(),
       labels.dPtr(), input.nFeatures);
    cudaCheckError();
  }
  return predictions;
}
