class Layer {
public:
  virtual void forwards
  (BatchInterface &input, BatchInterface &output) = 0;
  virtual void backwards
  (BatchInterface &input, BatchInterface &output, float learningRate=0.1) = 0;
  virtual void loadWeightsFromStream(ifstream &f) {};
  virtual void putWeightsToStream(ofstream &f)  {};
};
