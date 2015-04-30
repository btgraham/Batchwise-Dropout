class SimpleLayer : public Layer {
private:
  RNG rng;
  vectorCUDA<float> W; //Weights
  vectorCUDA<float> mW; //momentum
  vectorCUDA<float> dW; //For backprop
  vectorCUDA<float> B; //Weights
  vectorCUDA<float> mB; //momentum
  vectorCUDA<float> dB; //For backprop
  ActivationFunction fn;
public:
  int nFeaturesIn;
  int nFeaturesOut;
  SimpleLayer(int nFeaturesIn, int nFeaturesOut,
              ActivationFunction fn=NOSIGMOID) :
    nFeaturesIn(nFeaturesIn), nFeaturesOut(nFeaturesOut),
    fn(fn) {
    float scale=pow(6.0f/(nFeaturesIn+nFeaturesOut),0.5f);
    W.resize (nFeaturesIn*nFeaturesOut); W.setUniform(-scale,scale);
    mW.resize (nFeaturesIn*nFeaturesOut); mW.setZero();
    B.resize (nFeaturesOut); B.setZero();
    mB.resize (nFeaturesOut); mB.setZero();

  }
  void forwards
  (BatchInterface &input,
   BatchInterface &output) {
    assert(input.nFeatures==nFeaturesIn);
    assert(input.featuresPresent.size()==nFeaturesIn);
    output.type=input.type;
    output.batchSize=input.batchSize;
    output.nFeatures=nFeaturesOut;
    output.featuresPresent.hVector()=range(nFeaturesOut);
    output.features.resize(output.batchSize*nFeaturesOut);
    replicateArray(B.dPtr(), output.features.dPtr(), output.batchSize, nFeaturesOut);
    d_rowMajorSGEMM_alphaAB_betaC(cublasHandle,
                                  input.features.dPtr(), W.dPtr(), output.features.dPtr(),
                                  output.batchSize, nFeaturesIn, nFeaturesOut,
                                  1.0f, 1.0f,__FILE__,__LINE__);
    cudaCheckError();
    applySigmoid(output, output, fn);

    if (input.type==TRAINBATCH)
      output.dfeatures.resize(output.batchSize*output.featuresPresent.size());
  }
  void backwards(BatchInterface &input,
                 BatchInterface &output,
                 float learningRate=0.1) {
    applySigmoidBackProp(output, output, fn);

    dW.resize(nFeaturesIn*nFeaturesOut);
    dB.resize(nFeaturesOut);

    d_rowMajorSGEMM_alphaAtB_betaC(cublasHandle,
                                   input.features.dPtr(), output.dfeatures.dPtr(), dW.dPtr(),
                                   nFeaturesIn, output.batchSize, nFeaturesOut,
                                   1.0, 0.0);
    dB.setZero();
    columnSum(output.dfeatures.dPtr(), dB.dPtr(), output.batchSize, nFeaturesOut);
    cudaCheckError();

    if (input.dfeatures.size()>0)
      d_rowMajorSGEMM_alphaABt_betaC(cublasHandle,
                                     output.dfeatures.dPtr(), W.dPtr(), input.dfeatures.dPtr(),
                                     output.batchSize,nFeaturesOut,nFeaturesIn,
                                     1.0, 0.0);
    cudaCheckError();

    dGradientDescentNAG<<<nFeaturesIn,KERNELBLOCKSIZE>>>
      (dW.dPtr(), mW.dPtr(), W.dPtr(),  nFeaturesOut, learningRate);
    cudaCheckError();
    dGradientDescentNAG<<<1,KERNELBLOCKSIZE>>>
      (dB.dPtr(), mB.dPtr(), B.dPtr(), nFeaturesOut, learningRate);
    cudaCheckError();
  }
  void loadWeightsFromStream(ifstream &f) {
    f.read((char*)&W.hVector()[0],sizeof(float)*W.size());
    f.read((char*)&B.hVector()[0],sizeof(float)*B.size());
  };
  void putWeightsToStream(ofstream &f)  {
    f.write((char*)&W.hVector()[0],sizeof(float)*W.size());
    f.write((char*)&B.hVector()[0],sizeof(float)*B.size());
  };
};
