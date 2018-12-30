#ifndef ABSTRACT_BUFFER_HPP
#define ABSTRACT_BUFFER_HPP
#include <cstddef>

template <typename T> class abstract_buffer {
public:
	virtual ~abstract_buffer() = default;
	virtual size_t size() const = 0;
	virtual void copy_from(const T* source, size_t quantity, bool force_resize=false, size_t to_offset=0) = 0;
	virtual void copy_from(const abstract_buffer& source, size_t quantity, bool force_resize=false, size_t to_offset=0, size_t from_offset=0) = 0;
	virtual void copy_to(T* output, size_t quantity, size_t offset=0) const = 0;
	void copy_to(abstract_buffer& output, size_t quantity, size_t from_offset=0, size_t to_offset=0) const
	{
		output.copy_from(*this,quantity,false,to_offset,from_offset);
	}

	void operator=(const abstract_buffer& b) { copy_from(b,b.size(),true); }
};

#endif // ABSTRACT_BUFFER_HPP
