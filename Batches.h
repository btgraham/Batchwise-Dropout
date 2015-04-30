enum batchType {TRAINBATCH, TESTBATCH, UNLABELLEDBATCH};


class BatchInterface {
public:
  batchType type;
  int batchSize;                           // Number of training/test samples
  int nFeatures;                           // Features per sample
  vectorCUDA<float> features;              // For the forwards pass
  vectorCUDA<float> dfeatures;             // For the backwards/backpropagation pass
  vectorCUDA<int> featuresPresent;         // For batchwise dropout - rng.NchooseM(nFeatures,featuresPresent.size());
  //                                          Not dropped out features
  vectorCUDA<float> dropoutMask;           // For SampleWiseDropoutLayers
  void summary() {
    cout << "---------------------------------------------------\n";
    cout << "type" << type << endl;
    cout << "batchSize" << batchSize << endl;
    cout << "nFeatures" << nFeatures << endl;
    cout << "featuresPresent.size()" << featuresPresent.size() <<endl;
    cout << "features.size()" << features.size() << endl;
    cout << "dfeatures.size()" << dfeatures.size() << endl;
    cout << "---------------------------------------------------\n";
  }
};

class Batch {
public:
  BatchInterface i;
  vectorCUDA<int> labels;
  int mistakes;
  float inputDropout; //Use for batch-wise dropout during testing
  Batch(batchType type, int nFeatures, float inputDrop=0.0f) {
    i.type=type;
    i.batchSize=0;
    i.nFeatures=nFeatures;
    if (type==TRAINBATCH) {
      RNG rng;
      i.featuresPresent.hVector()=rng.NchooseM(nFeatures,nFeatures*(1-inputDrop));
      inputDropout=0;
    } else {
      i.featuresPresent.hVector()=range(nFeatures);
      inputDropout=inputDrop;
    }
    mistakes=0;
  }
};
