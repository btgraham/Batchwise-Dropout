class BatchWiseDropoutLayer : public Layer {
private:
  RNG rng;
  vectorCUDA<float> W; //Weights
  vectorCUDA<float> mW; //momentum
  vectorCUDA<float> w; //shrunk versions
  vectorCUDA<float> dw; //For backprop
  vectorCUDA<float> B; //Weights
  vectorCUDA<float> mB; //momentum
  vectorCUDA<float> b; //shrunk versions
  vectorCUDA<float> db; //For backprop
  ActivationFunction fn;
public:
  int nFeaturesIn;
  int nFeaturesOut;
  float dropout;
  BatchWiseDropoutLayer(int nFeaturesIn, int nFeaturesOut,
                        float dropout=0,ActivationFunction fn=NOSIGMOID) :
    nFeaturesIn(nFeaturesIn), nFeaturesOut(nFeaturesOut),
    dropout(dropout), fn(fn) {
    float scale=0;
    if (fn!=SOFTMAX)
      scale=powf(nFeaturesIn,-0.5);
    W.resize (nFeaturesIn*nFeaturesOut); W.setUniform(-scale,scale);
    mW.resize (nFeaturesIn*nFeaturesOut); mW.setZero();
    B.resize (nFeaturesOut); B.setZero();
    mB.resize (nFeaturesOut); mB.setZero();

  }
  void forwards
  (BatchInterface &input,
   BatchInterface &output) {
    output.type=input.type;
    output.batchSize=input.batchSize;
    output.nFeatures=nFeaturesOut;
    int o=nFeaturesOut*(input.type==TRAINBATCH?(1.0f-dropout):1.0f);
    output.featuresPresent.hVector()=rng.NchooseM(nFeaturesOut,o);
    assert(input.nFeatures==nFeaturesIn);
    output.features.resize(output.batchSize*output.featuresPresent.size());

    if (input.type==TRAINBATCH)
      output.dfeatures.resize(output.batchSize*output.featuresPresent.size());

    if (input.type==TRAINBATCH and nFeaturesIn+nFeaturesOut>input.featuresPresent.size()+output.featuresPresent.size()) {
      w.resize(input.featuresPresent.size()*output.featuresPresent.size());
      b.resize(output.featuresPresent.size());
      dShrinkMatrixForDropout
        <<<input.featuresPresent.size(),KERNELBLOCKSIZE>>>
        (W.dPtr(), w.dPtr(),
         input.featuresPresent.dPtr(),
         output.featuresPresent.dPtr(),
         output.nFeatures,
         output.featuresPresent.size());
      dShrinkVectorForDropout<<<1,NTHREADS>>>(B.dPtr(), b.dPtr(),
                                              output.featuresPresent.dPtr(),
                                              output.nFeatures,
                                              output.featuresPresent.size());
      cudaCheckError();
      replicateArray(b.dPtr(), output.features.dPtr(), output.batchSize, output.featuresPresent.size());
      d_rowMajorSGEMM_alphaAB_betaC(cublasHandle,
                                    input.features.dPtr(), w.dPtr(), output.features.dPtr(),
                                    output.batchSize, input.featuresPresent.size(), output.featuresPresent.size(),
                                    1.0f, 1.0f,__FILE__,__LINE__);
      cudaCheckError();

    } else {
      replicateArray(B.dPtr(), output.features.dPtr(), output.batchSize, output.featuresPresent.size());
      d_rowMajorSGEMM_alphaAB_betaC(cublasHandle,
                                    input.features.dPtr(), W.dPtr(), output.features.dPtr(),
                                    output.batchSize, input.nFeatures, output.nFeatures,
                                    1.0f-dropout, 1.0f-dropout,__FILE__,__LINE__);
      cudaCheckError();
    }
    applySigmoid(output, output, fn);
  }
  void backwards(BatchInterface &input,
                 BatchInterface &output,
                 float learningRate=0.1) {
    applySigmoidBackProp(output, output, fn);

    dw.resize(input.featuresPresent.size()*output.featuresPresent.size());
    db.resize(output.featuresPresent.size());

    d_rowMajorSGEMM_alphaAtB_betaC(cublasHandle,
                                   input.features.dPtr(), output.dfeatures.dPtr(), dw.dPtr(),
                                   input.featuresPresent.size(), output.batchSize, output.featuresPresent.size(),
                                   1.0, 0.0);
    db.setZero();
    columnSum(output.dfeatures.dPtr(), db.dPtr(), output.batchSize, output.featuresPresent.size());
    cudaCheckError();

    if (nFeaturesIn+nFeaturesOut>input.featuresPresent.size()+output.featuresPresent.size()) {
      if (input.dfeatures.size()>0) {
        d_rowMajorSGEMM_alphaABt_betaC(cublasHandle,
                                       output.dfeatures.dPtr(), w.dPtr(), input.dfeatures.dPtr(),
                                       output.batchSize,output.featuresPresent.size(),input.featuresPresent.size(),
                                       1.0, 0.0);
        cudaCheckError();
      }

      dGradientDescentMatrixNAGlite<<<input.featuresPresent.size(),KERNELBLOCKSIZE>>>
        (dw.dPtr(), mW.dPtr(), W.dPtr(),
         output.nFeatures, output.featuresPresent.size(),
         input.featuresPresent.dPtr(), output.featuresPresent.dPtr(),
         learningRate);

      dGradientDescentVectorNAGlite<<<1,NTHREADS>>>
        (db.dPtr(), mB.dPtr(), B.dPtr(),
         output.nFeatures, output.featuresPresent.size(),
         output.featuresPresent.dPtr(),
         learningRate);
    } else {
      if (input.dfeatures.size()>0) {
        d_rowMajorSGEMM_alphaABt_betaC(cublasHandle,
                                       output.dfeatures.dPtr(), W.dPtr(), input.dfeatures.dPtr(),
                                       output.batchSize,nFeaturesOut,nFeaturesIn,
                                       1.0, 0.0);
        cudaCheckError();
      }
      dGradientDescentNAG<<<nFeaturesIn,KERNELBLOCKSIZE>>>
        (dw.dPtr(), mW.dPtr(), W.dPtr(),  nFeaturesOut, learningRate);
      dGradientDescentNAG<<<1,KERNELBLOCKSIZE>>>
        (db.dPtr(), mB.dPtr(), B.dPtr(), nFeaturesOut, learningRate);
      cudaCheckError();
    }
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
