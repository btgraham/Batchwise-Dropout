static int intToggleEndianness(int a) {
  int b=0;
  b+=a%256*(1<<24);a>>=8;
  b+=a%256*(1<<16);a>>=8;
  b+=a%256*(1<< 8);a>>=8;
  b+=a%256*(1<< 0);
  return b;
}

static void loadMnistC(string filename, vector<Datum*> &characters) {
  ifstream f(filename.c_str());
  if (!f) {
    cout <<"Cannot find " << filename << endl;
    exit(EXIT_FAILURE);}
  int a,n1,n2,n3;
  f.read((char*)&a,4);
  f.read((char*)&a,4);
  n1=intToggleEndianness(a);
  f.read((char*)&a,4);
  n2=intToggleEndianness(a);
  f.read((char*)&a,4);
  n3=intToggleEndianness(a);
  unsigned char *bitmap=new unsigned char[n2*n3];
  for (int i1=0;i1<n1;i1++) {
    vectorDatum* character = new vectorDatum(n2*n3,0);
    f.read((char *)bitmap,n2*n3);
    for (int j=0;j<n2*n3;j++)
      character->features[j]=bitmap[j]/255.0;
    characters.push_back(character);
  }
  delete[] bitmap;
}

static void loadMnistL(string filename, vector<Datum*> &characters) {
  ifstream f(filename.c_str());
  if (!f) {
    cout <<"Cannot find " << filename << endl;
    exit(EXIT_FAILURE);}
  int a,n;
  char l;
  f.read((char*)&a,4);
  f.read((char*)&a,4);
  n=intToggleEndianness(a);
  for (int i=0;i<n;i++) {
    f.read(&l,1);
    characters[i]->label=l;
  }
}

Dataset MnistTrainSet() {
  Dataset dataset;
  dataset.name="MNIST train set";
  dataset.type=TRAINBATCH;
  dataset.nFeatures=784;
  dataset.nClasses=10;
  string trainC("Data/MNIST/train-images-idx3-ubyte");
  string trainL("Data/MNIST/train-labels-idx1-ubyte");
  loadMnistC(trainC, dataset.samples);
  loadMnistL(trainL, dataset.samples);
  return dataset;
}
Dataset MnistTestSet() {
  Dataset dataset;
  dataset.type=TESTBATCH;
  dataset.name="MNIST test set";
  dataset.nFeatures=784;
  dataset.nClasses=10;
  string testC("Data/MNIST/t10k-images-idx3-ubyte");
  string testL("Data/MNIST/t10k-labels-idx1-ubyte");
  loadMnistC(testC, dataset.samples);
  loadMnistL(testL, dataset.samples);
  return dataset;
}
