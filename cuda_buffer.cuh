#ifndef CUDA_BUFFER_CUH
#define CUDA_BUFFER_CUH
#include "abstract_buffer.hpp"
#include <vector>
#include <cstring>
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

template <typename T> __global__ void custom_memcpy(T *dst, T *src, int quantity, int from_offset, int to_offset)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
	for (int i = index; i < quantity; i += stride) {
	dst[i+to_offset] = src[i+from_offset];
  }
}

template <typename T> class cuda_buffer : public abstract_buffer<T> {
public:
	typedef abstract_buffer<T> base_buffer;
private:
	T* dev_mem;
	size_t _size;

	std::vector<T> backup() const
	{
		std::vector<T> tmp(_size);
		cudaMemcpy(tmp.data(),dev_mem,sizeof(T)*tmp.size(),cudaMemcpyDeviceToHost);
		return tmp;
	}
	void reallocateWithoutPreserve(size_t newSize)
	{
		if(newSize == _size) return;
		if(dev_mem) cudaFree(&dev_mem);
		cudaMallocManaged(&dev_mem,newSize*sizeof(T));
		_size = newSize;
	}
	void reallocateAndPreserve(size_t newSize)
	{
		if(newSize == _size) return;
		std::vector<T> tmp = backup();
		if(dev_mem) cudaFree(&dev_mem);
		cudaMalloc(&dev_mem,newSize*sizeof(T));
		cudaMemcpy(dev_mem,tmp.data(),sizeof(T)*tmp.size(),cudaMemcpyHostToDevice);
		_size = newSize;
	}

public:
	T* getDeviceMemory() { return dev_mem; }
	size_t size() const { return _size; }
	void resize(size_t newSize,bool preserve=false)
	{
		if(preserve && _size) reallocateAndPreserve(newSize);
		else reallocateWithoutPreserve(newSize);
	}
	void copy_from(const T* source, size_t quantity, bool force_resize=false, size_t to_offset=0)
	{
		if(force_resize && (_size>(quantity+to_offset))) resize(quantity+to_offset,true);
		cudaMemcpy(dev_mem+to_offset,source,sizeof(T)*quantity,cudaMemcpyHostToDevice);
	}
	void copy_from(const base_buffer& source, size_t quantity, bool force_resize=false, size_t to_offset=0, size_t from_offset=0)
	{
		if(force_resize && (_size>(quantity+to_offset))) resize(quantity+to_offset,true);
		const cuda_buffer* cudabuff = dynamic_cast<const cuda_buffer*>(&source); // Is it compatible?
		if(cudabuff) // Now we're cooking with gas!
		{
			const int blockSize = 256;
			const int numBlocks = (int(quantity) + blockSize - 1) / blockSize;
			custom_memcpy<<<numBlocks, blockSize>>>(dev_mem, cudabuff->dev_mem,
													int(quantity),
													int(from_offset),
													int(to_offset)
													);
		}
		else { // This is not CUDA :(
			std::vector<T> tmp(quantity-from_offset);
			source.copy_to(tmp.data(),quantity,from_offset);
			cudaMemcpy(dev_mem+to_offset,tmp.data(),sizeof(T)*tmp.size(),cudaMemcpyHostToDevice);
		}
	}
	virtual void copy_to(T* output, size_t quantity, size_t offset=0) const
	{
		cudaMemcpy(output,dev_mem+offset,sizeof(T)*quantity,cudaMemcpyDeviceToHost);
	}
	~cuda_buffer() {
		if(dev_mem) cudaFree(&dev_mem);
	}
	cuda_buffer()
		: dev_mem(nullptr), _size(0)
	{
		;
	}
	cuda_buffer(size_t nSize)
		: dev_mem(nullptr), _size(nSize)
	{
		cudaMallocManaged(&dev_mem,nSize*sizeof(T));
	}
	cuda_buffer(const T* source, size_t nSize)
		: dev_mem(nullptr), _size(nSize)
	{
		cudaMallocManaged(&dev_mem,nSize*sizeof(T));
		cudaMemcpy(dev_mem,source,sizeof(T)*nSize,cudaMemcpyHostToDevice);
	}
	cuda_buffer(const base_buffer& cpy)
		: dev_mem(nullptr), _size(cpy.size())
	{
		cudaMallocManaged(&dev_mem,_size*sizeof(T));
		const cuda_buffer* cudabuff = dynamic_cast<const cuda_buffer*>(&cpy); // Is it compatible?
		if(cudabuff) // Now we're cooking with gas!
		{
			const int blockSize = 256;
			const int numBlocks = (int(cpy.size()) + blockSize - 1) / blockSize;
			custom_memcpy<<<numBlocks, blockSize>>>(dev_mem, cudabuff->dev_mem,
													int(_size),
													int(0),
													int(0)
													);
		}
		else { // This is not CUDA :(
			std::vector<T> tmp(_size);
			cpy.copy_to(tmp.data(),_size);
			cudaMemcpy(dev_mem,tmp.data(),sizeof(T)*tmp.size(),cudaMemcpyHostToDevice);
		}
	}
};

#endif // CUDA_BUFFER_CUH
