class ANN {
public:
  vector<Layer*> ann;
  int nInputFeatures;
  int nOutputFeatures;
  int inputSpatialSize;
  int nClasses;
  int nTop;
  float inputDropout;
  ANN (int nInputFeatures,
       int nClasses,
       int cudaDevice=0,
       float inputDropout=0.0f,
       int nTop=1) :
    nInputFeatures(nInputFeatures),
    nClasses(nClasses),
    nTop(nTop),
    inputDropout(inputDropout) {
    nOutputFeatures=nInputFeatures;
    initializeGPU(cudaDevice);
    if (inputDropout>0)
      cout << "Input Layer Dropout " << inputDropout << endl;
  }
  void addSimpleLayer(int nFeatures, ActivationFunction activationFn=RELU) {
    cout << nFeatures <<"N " << sigmoidNames[activationFn] << endl;
    ann.push_back(new SimpleLayer(nOutputFeatures, nFeatures,activationFn));
    nOutputFeatures=nFeatures;
  }
  void addBatchWiseDropoutLayer(int nFeatures, float dropout, ActivationFunction activationFn=RELU) {
    cout << nFeatures <<"N " << sigmoidNames[activationFn] << endl;
    cout << "Dropout " << dropout << endl;
    ann.push_back(new BatchWiseDropoutLayer(nOutputFeatures, nFeatures,dropout,activationFn));
    nOutputFeatures=nFeatures;
  }
  void addSampleWiseDropoutLayer(int nFeatures, float dropout, ActivationFunction activationFn=RELU) {
    cout << "Dropout " << dropout << endl;
    cout << nFeatures <<"N " << sigmoidNames[activationFn] << endl;
    ann.push_back(new SampleWiseDropoutLayer(nOutputFeatures, nFeatures, dropout, activationFn));
    nOutputFeatures=nFeatures;
  }
  void processBatch(Batch& batch, float learningRate=0.1) {
    vector<BatchInterface*> interfaces(1);
    interfaces[0]=&batch.i;
    for (int i=0;i<ann.size();i++)
      {
        interfaces.push_back(new BatchInterface);
        ann[i]->forwards(*interfaces[i],*interfaces[i+1]);
      }
    if (batch.i.type==TRAINBATCH)
      {
        SoftmaxClassifier(*interfaces.back(),nTop,batch.labels,batch.mistakes);
        for (int i=ann.size()-1;i>=0;i--) {
          ann[i]->backwards(*interfaces[i],*interfaces[i+1],learningRate);
        }
      }
    else if (batch.i.type==TESTBATCH)
      {
        SoftmaxClassifier(*interfaces.back(),nTop,batch.labels,batch.mistakes);
      }
    else if (batch.i.type==UNLABELLEDBATCH)
      {
        ofstream f;
        f.open("predictions.labels", ios::app);
        vector<vector<int> > predictions=SoftmaxClassifier(*interfaces.back(),nTop);
        for (int j=0;j<predictions.size();j++) {
          f << predictions[j][0];
          for (int k=1;k<predictions[j].size();k++)
            f << " " << predictions[j][k];
          f << endl;
        }
      }
    for (int i=0;i<ann.size();i++)
      delete interfaces[i+1];
  }
  void loadWeights(string baseName, int epoch) {
    string filename=string(baseName)+string("_epoch-")+boost::lexical_cast<string>(epoch)+string(".ann");
    ifstream f;
    f.open(filename.c_str(),ios::out | ios::binary);
    if (f) {
      cout << "Loading network parameters from " << filename << endl;
    } else {
      cout <<"Cannot find " << filename << endl;
      exit(EXIT_FAILURE);
    }
    for (int i=0;i<ann.size();i++)
      ann[i]->loadWeightsFromStream(f);
    f.close();
  }
  void saveWeights(string baseName, int epoch)  {
    string filename=string(baseName)+string("_epoch-")+boost::lexical_cast<string>(epoch)+string(".ann");
    ofstream f;
    f.open(filename.c_str(),ios::binary);
    if (f) {
      for (int i=0;i<ann.size();i++)
        ann[i]->putWeightsToStream(f);
      f.close();
    } else {
      cout <<"Cannot write " << filename << endl;
      exit(EXIT_FAILURE);
    }
  }
};



float iterate(ANN& ann, Dataset &dataset, int batchSize=128, float learningRate=0.1, bool verbose=false) {
  float errorRate=0;
  BatchProducer bp(dataset,batchSize,ann.inputDropout);
  if (dataset.type==UNLABELLEDBATCH) {
    ofstream f;
    f.open("predictions.labels");
  }
  int ctr=0;
  while(Batch* batch=bp.nextBatch()) {
    ann.processBatch(*batch,learningRate);
    ctr++;
    if (verbose and !(ctr & (ctr-1)))
      cout << ctr << " " << batch->mistakes << endl;
    errorRate+=batch->mistakes*1.0/dataset.samples.size();
    delete batch;
  }
  cout << dataset.name
       << " Mistakes: "
       << 100*errorRate
       << "%"
       << endl;
  return errorRate;
}
