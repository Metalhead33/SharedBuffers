#ifndef HOST_BUFFER_HPP
#define HOST_BUFFER_HPP
#include "abstract_buffer.hpp"
#include <vector>
#include <array>
#include <cstring>
#include <initializer_list>
#include <iterator>

template <typename T> class dynamic_host_buffer : public abstract_buffer<T> {
public:
	typedef abstract_buffer<T> base_buffer;
	typedef std::initializer_list<T> initializer;
	typedef std::vector<T> buffType;

	typedef typename buffType::iterator iterator;
	typedef typename buffType::reverse_iterator reverse_iterator;
	typedef typename buffType::const_iterator const_iterator;
	typedef typename buffType::const_reverse_iterator const_reverse_iterator;
private:
	buffType buff;
public:
	// Implementations of inherited functions
	size_t size() const { return buff.size(); }
	void copy_from(const T* source, size_t quantity, bool force_resize=false, size_t to_offset=0)
	{
		if(force_resize) buff.resize(to_offset+quantity);
		memcpy(&buff[to_offset],source,sizeof(T)*std::min(buff.size()-to_offset,quantity));
	}
	void copy_from(const base_buffer& source, size_t quantity, bool force_resize=false, size_t to_offset=0, size_t from_offset=0)
	{
		if(force_resize) buff.resize(to_offset+quantity);
		source.copy_to(&buff[to_offset],std::min(buff.size()-to_offset,quantity),from_offset);
	}
	void copy_to(T* output, size_t quantity, size_t offset=0) const
	{
		memcpy(output,&buff[offset],sizeof(T)*std::min(quantity,buff.size()-offset));
	}
	// Iterators
	iterator begin() { return buff.begin(); }
	iterator end() { return buff.end(); }
	reverse_iterator rbegin() { return buff.rbegin(); }
	reverse_iterator rend() { return buff.rend(); }
	// Const iterators
	const_iterator begin() const { return buff.begin(); }
	const_iterator end() const { return buff.end(); }
	const_reverse_iterator rbegin() const { return buff.rbegin(); }
	const_reverse_iterator rend() const { return buff.rend(); }
	// Inherited for std::vector compatibility
	T& at(size_t offset) { return buff[offset]; }
	const T& at(size_t offset) const { return buff[offset]; }
	T& operator[](size_t offset) { return buff[offset]; }
	const T& operator[](size_t offset) const { return buff[offset]; }
	T* data() { return buff.data(); }
	const T* data() const { return buff.data(); }
	void resize(size_t quantity) { buff.resize(quantity); }

	// Constructors
	dynamic_host_buffer()
	{
		;
	}
	dynamic_host_buffer(size_t quantity)
		: buff(quantity)
	{
		;
	}
	dynamic_host_buffer(const base_buffer& cpy)
		: buff(cpy.size())
	{
		cpy.copy_to(buff.data(),cpy.size());
	}
	dynamic_host_buffer(const initializer& initialize)
		: buff(initialize)
	{
		;
	}
	template< class InputIt > dynamic_host_buffer(InputIt first, InputIt last)
		: buff(first,last)
	{
		;
	}
};

template <typename T, size_t Q> class static_host_buffer : public abstract_buffer<T> {
public:
	typedef abstract_buffer<T> base_buffer;
	typedef std::initializer_list<T> initializer;
	typedef std::array<T,Q> buffType;
	typedef typename buffType::iterator iterator;
	typedef typename buffType::reverse_iterator reverse_iterator;
	typedef typename buffType::const_iterator const_iterator;
	typedef typename buffType::const_reverse_iterator const_reverse_iterator;
private:
	buffType buff;
public:
	// Implementations of inherited functions
	size_t size() const { return buff.size(); }
	void copy_from(const T* source, size_t quantity, bool force_resize=false, size_t to_offset=0)
	{
		(void)(force_resize); // There is no resizing. Ever. This is static
		memcpy(&buff[to_offset],source,sizeof(T)*std::min(quantity,Q-to_offset));
	}
	void copy_from(const base_buffer& source, size_t quantity, bool force_resize=false, size_t to_offset=0, size_t from_offset=0)
	{
		(void)(force_resize); // There is no resizing. Ever. This is static
		source.copy_to(&buff[to_offset],std::min(quantity,Q-to_offset),from_offset);
	}
	void copy_to(T* output, size_t quantity, size_t offset=0) const
	{
		memcpy(output,&buff[offset],sizeof(T)*std::min(quantity,buff.size()-offset));
	}
	// Iterators
	iterator begin() { return buff.begin(); }
	iterator end() { return buff.end(); }
	reverse_iterator rbegin() { return buff.rbegin(); }
	reverse_iterator rend() { return buff.rend(); }
	// Const iterators
	const_iterator begin() const { return buff.begin(); }
	const_iterator end() const { return buff.end(); }
	const_reverse_iterator rbegin() const { return buff.rbegin(); }
	const_reverse_iterator rend() const { return buff.rend(); }
	// Inherited for std::array compatibility
	T& at(size_t offset) { return buff[offset]; }
	const T& at(size_t offset) const { return buff[offset]; }
	T& operator[](size_t offset) { return buff[offset]; }
	const T& operator[](size_t offset) const { return buff[offset]; }
	T* data() { return buff.data(); }
	const T* data() const { return buff.data(); }

	// Constructors
	static_host_buffer()
	{
		;
	}
	static_host_buffer(const base_buffer& cpy)
	{
		cpy.copy_to(buff.data(),std::min(buff.size(),cpy.size()));
	}
	static_host_buffer(const initializer& initialize)
	{
		size_t added = 0;
		for(auto it = std::begin(initialize);added < Q && it != std::end(initialize);++added)
		{
			buff[added] = *it;
		}
	}
};

#endif // HOST_BUFFER_HPP
