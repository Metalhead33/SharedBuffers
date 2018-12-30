#ifndef OPENGL_BUFFER_HPP
#define OPENGL_BUFFER_HPP
#include "abstract_buffer.hpp"
#include <vector>
#include <cstring>
// Remove the includes below, if you are using GLAD, or GLEW, or anything similar
#ifndef NO_LOAD_GL
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#endif

template <typename T> class opengl_buffer : public abstract_buffer<T>  {
public:
	typedef abstract_buffer<T> base_buffer;
private:
	const GLenum target;
	const GLenum usage;
	GLuint id;
	size_t _size;
	std::vector<T> backup() const {
		std::vector<T> bckp(_size);
		glBindBuffer(target,id);
		glGetBufferSubData(target,0,_size*sizeof(T),bckp.data());
		return bckp;
	}
	void resizeAndPreserve(size_t newSize)
	{
		if(newSize == _size) return;
		size_t old_size = _size;
		std::vector<T> backed_up = backup();
		_size = newSize;
		glBindBuffer(target,id);
		glBufferData(target,sizeof(T)*newSize,nullptr,usage);
		glBufferSubData(target,0,sizeof(T)*old_size,backed_up.data());
	}
	void resizeWithoutPreserve(size_t newSize)
	{
		if(newSize == _size) return;
		_size = newSize;
		glBindBuffer(target,id);
		glBufferData(target,sizeof(T)*newSize,nullptr,usage);
	}
public:
	GLuint getId() const { return id; }
	GLenum getTarget() const { return target; }
	GLenum getUsage() const { return usage; }
	size_t size() const { return _size; }
	void resize(size_t newSize,bool preserve=false)
	{
		if(preserve) resizeAndPreserve(newSize);
		else resizeWithoutPreserve(newSize);
	}

	void copy_from(const T* source, size_t quantity, bool force_resize=false, size_t to_offset=0)
	{
		if(force_resize && (_size>(quantity+to_offset))) resize(quantity+to_offset,true);
		glBindBuffer(target,id);
		glBufferSubData(target,sizeof(T)*to_offset,sizeof(T)*quantity,source);
	}
	void copy_from(const base_buffer& source, size_t quantity, bool force_resize=false, size_t to_offset=0, size_t from_offset=0)
	{
		if(force_resize && (_size>(quantity+to_offset))) resize(quantity+to_offset,true);
		opengl_buffer* glbuff = dynamic_cast<opengl_buffer*>(&source); // Is it compatible?
		if(glbuff) // Oh yeah, now we're cooking with gas
		{
			glBindBuffer(GL_COPY_READ_BUFFER,glbuff->id);
			glBindBuffer(GL_COPY_WRITE_BUFFER,id);
			glCopyBufferSubData(GL_COPY_READ_BUFFER,GL_COPY_WRITE_BUFFER,
								from_offset*sizeof(T),
								to_offset*sizeof(T),
								quantity*sizeof(T));
		}
		else {
			std::vector<T> tmp(quantity-from_offset);
			source.copy_to(tmp.data(),quantity,from_offset);
			glBindBuffer(target,id);
			glBufferSubData(target,sizeof(T)*to_offset,sizeof(T)*quantity,tmp.data());
		}
	}
	void copy_to(T* output, size_t quantity, size_t offset=0) const
	{
		glBindBuffer(target,id);
		glGetBufferSubData(target,offset*sizeof(T),quantity*sizeof(T),output);
	}
	~opengl_buffer() {
		glDeleteBuffers(1,&id);
	}
	opengl_buffer(GLenum target, GLenum usage)
		: target(target), usage(usage), _size(0)
	{
		glCreateBuffers(1,&id);
	}
	opengl_buffer(GLenum target, GLenum usage, size_t nSize)
		: target(target), usage(usage), _size(nSize)
	{
		glCreateBuffers(1,&id);
		glBindBuffer(target,id);
		glBufferData(target,sizeof(T)*nSize,nullptr,usage);
	}
	opengl_buffer(GLenum target, GLenum usage, const T* source, size_t nSize)
		: target(target), usage(usage), _size(nSize)
	{
		glCreateBuffers(1,&id);
		glBindBuffer(target,id);
		glBufferData(target,sizeof(T)*nSize,source,usage);
	}
	opengl_buffer(GLenum target, GLenum usage, const base_buffer& cpy)
		: target(target), usage(usage), _size(cpy.size())
	{
		std::vector<T> tmp(_size);
		cpy.copy_to(tmp.data(),_size);
		glCreateBuffers(1,&id);
		glBindBuffer(target,id);
		glBufferData(target,sizeof(T)*cpy.size(),tmp.data(),usage);
	}
	opengl_buffer(const opengl_buffer& cpy)
		: target(cpy.target), usage(cpy.usage), _size(cpy.size())
	{
		glCreateBuffers(1,&id);
		glBindBuffer(GL_COPY_READ_BUFFER,cpy.id);
		glBindBuffer(GL_COPY_WRITE_BUFFER,id);
		glCopyBufferSubData(GL_COPY_READ_BUFFER,GL_COPY_WRITE_BUFFER,
							0,
							0,
							_size*sizeof(T));
	}
};

#endif // OPENGL_BUFFER_HPP
