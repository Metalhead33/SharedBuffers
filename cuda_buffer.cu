#ifndef CUDA_BUFFER_CUH
#define CUDA_BUFFER_CUH
#include "abstract_buffer.hpp"
#include <vector>
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

template <typename T> class cuda_buffer : public abstract_buffer<T> {
public:
	typedef abstract_buffer<T> base_buffer;
private:
	T* dev_mem;
	size_t _size;

	std::vector<T> backup() const
	{
		std::vector<T> tmp(_size);
		cudaMemcpy(tmp.data(),dev_mem,sizeof(T)*_size,cudaMemcpyDeviceToHost);
		return tmp;
	}
	void reallocateWithoutPreserve(size_t newSize)
	{
		if(newSize == _size) return;
		if(dev_mem) cudaFree(&dev_mem);
		cudaMalloc(&dev_mem,newSize*sizeof(T));
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
		if(preserve) resizeAndPreserve(newSize);
		else resizeWithoutPreserve(newSize);
	}
	void copy_from(const T* source, size_t quantity, bool force_resize=false, size_t to_offset=0)
	{
		if(force_resize && (_size>(quantity+to_offset))) resize(quantity+to_offset,true);
		cudaMemcpy(dev_mem,source,sizeof(T)*tmp.size(),cudaMemcpyHostToDevice);
	}
	virtual void copy_from(const base_buffer& source, size_t quantity, bool force_resize=false, size_t to_offset=0, size_t from_offset=0) = 0;
	virtual void copy_to(T* output, size_t quantity, size_t offset=0) const = 0;
};

#endif // CUDA_BUFFER_CUH
