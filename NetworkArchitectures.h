class SimpleNet : public ANN {
public:
  SimpleNet(int nInputFeatures, int nFeatures, int nHiddenLayers, int nClasses, ActivationFunction fn, int cudaDevice=0, int nTop=1) : ANN(nInputFeatures, nClasses, cudaDevice, 0.0f, nTop) {
    for (int i=0;i<nHiddenLayers;i++)
      addSimpleLayer(nFeatures,fn);
    addSimpleLayer(nClasses,SOFTMAX);
  }
};


class BatchWiseDropoutNet : public ANN {
public:
  BatchWiseDropoutNet(int nInputFeatures, int nFeatures, int nHiddenLayers, int nClasses, ActivationFunction fn, float inputDropout, float dropout, int cudaDevice=0, int nTop=1) : ANN(nInputFeatures, nClasses, cudaDevice, inputDropout, nTop) {
    for (int i=0;i<nHiddenLayers;i++)
      addBatchWiseDropoutLayer(nFeatures,dropout,fn);
    addBatchWiseDropoutLayer(nClasses,0.0f,SOFTMAX);
  }
};

class SampleWiseDropoutNet : public ANN {
public:
  SampleWiseDropoutNet(int nInputFeatures, int nFeatures, int nHiddenLayers, int nClasses, ActivationFunction fn, float inputDropout, float dropout, int cudaDevice=0, int nTop=1) : ANN(nInputFeatures, nClasses, cudaDevice, 0.0f, nTop) {
    for (int i=0;i<nHiddenLayers;i++)
      addSampleWiseDropoutLayer(nFeatures,(i==0)?inputDropout:dropout,fn);
    addSampleWiseDropoutLayer(nClasses,dropout,SOFTMAX);
  }
};

