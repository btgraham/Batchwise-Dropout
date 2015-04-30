class Datum {
public:
  virtual void codifyInputData (Batch &batch)=0;
  virtual Datum* distort (RNG& rng) =0;
  int label; //-1 for unknown
  virtual ~Datum() {}
};


class Dataset {
public:
  string name;
  vector<Datum*> samples;
  int nFeatures;
  int nClasses;
  batchType type;
  void shuffle() {
    random_shuffle ( samples.begin(), samples.end());
  }
  void summary() {
    cout << "Name:           " << name << endl;
    cout << "nSamples:       " << samples.size() << endl;
    cout << "nClasses:       " << nClasses << endl;
    cout << "nFeatures:      " << nFeatures << endl;
  }
  Dataset extractValidationSet(float p=0.1) {
    Dataset val;
    val.name=name+string(" Validation set");
    name=name+string(" minus Validation set");
    val.nFeatures=nFeatures;
    val.nClasses=nClasses;
    val.type=TESTBATCH;
    shuffle();
    int size=samples.size()*p;
    for (;size>0;size--) {
      val.samples.push_back(samples.back());
      samples.pop_back();
    }
    return val;
  }
  Dataset subset(int n) {
    Dataset subset(*this);
    subset.shuffle();
    subset.samples.resize(n);
    return subset;
  }
};

class vectorDatum : public Datum {
public:
  vector<float> features;
  void codifyInputData (Batch &batch) {
    for (int i=0;i<batch.i.featuresPresent.size();i++)
      batch.i.features.hVector().push_back
        (features[batch.i.featuresPresent.hVector()[i]]*(1-batch.inputDropout));
    batch.i.batchSize++;
    batch.labels.hVector().push_back(label);
  }
  vectorDatum(int size, int label_ = -1) {
    features.resize(size);
    label=label_;
  }
  Datum* distort(RNG& rng) {
    vectorDatum* a = new vectorDatum(*this);
    return a;
  }
  ~vectorDatum() {}
};

class vectorDatum32_24 : public Datum  {  //24x24 view into the 32x32 source images
public:
  vector<float> features;
  void codifyInputData (Batch &batch) {
    for (int i=0;i<batch.i.featuresPresent.size();i++) {
      int j=batch.i.featuresPresent.hVector()[i];
      int jc=j/576;
      int jy=(j%576)/24;
      int jx=j%24;
      batch.i.features.hVector().push_back
        (features[(jx+4)+(jy+4)*28+jc*1024]*(1-batch.inputDropout));
    }
    batch.i.batchSize++;
    batch.labels.hVector().push_back(label);
  }
  vectorDatum32_24(int size, int label_ = -1) {
    features.resize(size);
    label=label_;
  }
  Datum* distort(RNG& rng) {
    vectorDatum32_24* a = new vectorDatum32_24(*this);
    int maxShift=4;
    int xshift=rng.randint(2*maxShift+1)-maxShift;
    int yshift=rng.randint(2*maxShift+1)-maxShift;
    int flip=rng.randint(2);
    for (int x=0;x<32;x++)
      for (int y=0;y<32;y++) {
        int xx=(flip?31-x:x)+xshift;
        int yy=y+yshift;
        for (int c=0;c<3;c++)
          a->features[c*1024+y*32+x]=
            (xx>=0 and xx<32 and yy>=0 and yy<32)?features[c*1024+yy*32+xx]:0;
      }
    return a;
  }
  ~vectorDatum32_24() {}
};
