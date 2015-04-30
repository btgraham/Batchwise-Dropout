class BatchProducer {
public:
  int batchCounter;
  boost::thread_group workers;
  Dataset& dataset;
  vector<Batch*> v;
  int batchSize;
  int nThreads;
  float inputDropout;

  Batch* nextBatch() {
    if (batchCounter<v.size()) {
      while (v[batchCounter]==NULL)
        boost::this_thread::sleep(boost::posix_time::milliseconds(10));
      return v[batchCounter++];
    } else {
      workers.join_all();
      return NULL;
    }
  }
  void batchProducerThread(int nThread) {
    RNG rng;
    for (int c=nThread;c<v.size();c+=nThreads) {
      while (c>batchCounter+5*nThreads)
        boost::this_thread::sleep(boost::posix_time::milliseconds(10));
      Batch* batch =
        new Batch(dataset.type, dataset.nFeatures, inputDropout);
      for (int i=c*batchSize;i<min((c+1)*batchSize,(int)(dataset.samples.size()));i++) {
        if (dataset.type==TRAINBATCH) {
          Datum* pic=dataset.samples[i]->distort(rng);
          pic->codifyInputData(*batch);
          delete pic;
        } else {
          dataset.samples[i]->codifyInputData(*batch);
        }
      }
      v[c]=batch;
    }
  }
  BatchProducer (Dataset &dataset, int batchSize, float inputDropout=0.0f, int nThreads=4) :
    batchCounter(0), dataset(dataset), batchSize(batchSize), inputDropout(inputDropout), nThreads(nThreads) {
    v.resize((dataset.samples.size()+batchSize-1)/batchSize,NULL);
    for (int nThread=0; nThread<nThreads; nThread++)
      workers.add_thread(new boost::thread(boost::bind(&BatchProducer::batchProducerThread,this,nThread)));
  }
};

