#ifndef OPENCL_BUFFER_HPP
#define OPENCL_BUFFER_HPP
#include "abstract_buffer.hpp"
#include <CL/cl.hpp>
#include <memory>

struct CLCONTEXT {
	cl::Context context;
	cl::CommandQueue queue;
};
typedef std::shared_ptr<CLCONTEXT> SharedContext;

template <typename T> class opencl_buffer : public abstract_buffer<T> {
public:
	typedef abstract_buffer<T> base_buffer;
	const SharedContext context;
private:
	cl::Buffer buff;
	size_t _size;
	void resizeAndPreserve(size_t newSize)
	{
		if(newSize == _size) return;
		size_t oldsize = _size;
		std::vector<T> tmp(_size);
		context->queue.enqueueReadBuffer(buff,CL_TRUE,0,sizeof(T)*oldsize,tmp.data());
		buff = cl::Buffer(context->context,CL_MEM_READ_WRITE,sizeof(T)*newSize);
		_size = newSize;
		context->queue.enqueueWriteBuffer(buff,CL_TRUE,0,sizeof(T)*std::min(oldsize,newSize),tmp.data());
	}
	void resizeWithoutPreserve(size_t newSize)
	{
		if(newSize == _size) return;
		buff = cl::Buffer(context->context,CL_MEM_READ_WRITE,sizeof(T)*newSize);
		_size = newSize;
	}
public:
	// Necessary for basic function?
	size_t size() const { return _size; }
	void resize(size_t newSize,bool preserve=false)
	{
		if(preserve && _size) resizeAndPreserve(newSize);
		else resizeWithoutPreserve(newSize);
	}
	void copy_from(const T* source, size_t quantity, bool force_resize=false, size_t to_offset=0)
	{
		if(force_resize && (_size>(quantity+to_offset))) resize(quantity+to_offset,true);
		context->queue.enqueueWriteBuffer(buff,CL_TRUE,sizeof(T)*to_offset,sizeof(T)*std::min(quantity,_size-to_offset),source);
	}
	void copy_from(const base_buffer& source, size_t quantity, bool force_resize=false, size_t to_offset=0, size_t from_offset=0)
	{
		if(force_resize && (_size>(quantity+to_offset))) resize(quantity+to_offset,true);
		const opencl_buffer* clbuff = dynamic_cast<const opencl_buffer*>(&source);
		if(clbuff) // Now we're cooking with gas
		{
			context->queue.enqueueCopyBuffer(clbuff->buff,buff,sizeof(T)*from_offset,sizeof(T)*to_offset,sizeof(T)*quantity);
		}
		else { // Not an OpenCL buffer :(
			std::vector<T> tmp(quantity-from_offset);
			source.copy_to(tmp.data(),quantity,from_offset);
			context->queue.enqueueWriteBuffer(buff,CL_TRUE,sizeof(T)*to_offset,sizeof(T)*quantity,tmp.data());
		}
	}
	void copy_to(T* output, size_t quantity, size_t offset=0) const
	{
		context->queue.enqueueReadBuffer(buff,CL_TRUE,offset,std::min(quantity,_size-offset),output);
	}
	const cl::Buffer& getBuffer() const { return buff; }
	// Constructors
	opencl_buffer(SharedContext context)
		: context(context), buff(), _size(0)
	{
		;
	}
	opencl_buffer(SharedContext context, size_t nsize)
		: context(context), buff(context->context,CL_MEM_READ_WRITE,sizeof(T)*nsize), _size(nsize)
	{
		;
	}
	opencl_buffer(SharedContext context, const T* origin, size_t nsize)
		: context(context), buff(context->context,CL_MEM_READ_WRITE,sizeof(T)*nsize), _size(nsize)
	{
		context->queue.enqueueWriteBuffer(buff,CL_TRUE,0,sizeof(T)*nsize,origin);
	}
	opencl_buffer(SharedContext context, const base_buffer& cpy)
		: context(context), buff(context->context,CL_MEM_READ_WRITE,sizeof(T)*cpy.size()), _size(cpy.size())
	{
		std::vector<T> tmp(_size);
		cpy.copy_to(tmp.data(),_size);
		context->queue.enqueueWriteBuffer(buff,CL_TRUE,0,sizeof(T)*_size,tmp.data());
	}
	opencl_buffer(const opencl_buffer& cpy) // Copy constructor
		: context(cpy.context), buff(context->context,CL_MEM_READ_WRITE,sizeof(T)*cpy.size()), _size(cpy.size())
	{
		context->queue.enqueueCopyBuffer(cpy.buff,buff,0,0,cpy.size()*sizeof(T));
	}
};

#endif // OPENCL_BUFFER_HPP
