cifar10:
	echo "Please put .bin files from http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz in Data/CIFAR10/"
	nvcc -o DropOutNet runCifar10.cu -lrt -lcublas -lboost_thread -lboost_system -arch sm_20 -O2
	DropOutNet
mnist:
	echo "Please put http://yann.lecun.com/exdb/mnist/ ubyte files in Data/MNIST"
	nvcc -o DropOutNet runMnist.cu -lrt -lcublas -lboost_thread -lboost_system -arch sm_20 -O2
	DropOutNet
artifical:
	cd Data/Artificial/; python artificial.dataset.py
	nvcc -o DropOutNet runArtificial.cu -lrt -lcublas -lboost_thread -lboost_system -arch sm_20 -O2
	DropOutNet
