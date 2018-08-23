#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <algorithm>
#include <cstdlib>
#include <curand.h>

static const int WORK_SIZE = 10000000;

int main()
{
	// TODO: Define a vector data of doubles of size WORK_SIZE on the device.
	...  data(...);

	// Generate random number in parallel on the GPU using curand
	curandGenerator_t gen;
	// We need a "raw" pointer to the space allocated above to pass it to curand
	double* data_raw = thrust::raw_pointer_cast(data.data());
	// Initialize the random number generator and generate the sequence.
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
	curandGenerateUniformDouble(gen, data_raw, data.size());

	// TODO: Copy random numbers to data_host on the CPU
	... data_host ... ;
	// TODO: Sort random numbers in parallel on GPU
	thrust:: ... ;
	// Copy sorted sequence to CPU
	thrust::host_vector<double> data_sorted = data;
	// Sort random number on CPU
	thrust::sort(data_host.begin(), data_host.end());

	// Compare sorted sequences
	for (int i = 0; i < data_host.size(); ++i){
		if (data_sorted[i] != data_host[i])
			std::cout << "Element " << i << ": " << data_sorted[i] << "!=" << data_host[i] << "\n";
	}
	std::cout << data_host[0] << " is the smallest and " << data_host[WORK_SIZE -1] << " is the largest element.\n";

	// Clean up
	curandDestroyGenerator(gen);
	return 0;
}
