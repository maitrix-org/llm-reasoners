/**
 * \file array.h
 *
 * This file contains the implementation of the `array` data structure, which
 * is a resizeable ordered collection of elements with generic type `T`. In
 * addition, the file contains global helper functions for working with arrays,
 * such as resizing, linear and binary search, insertion sort, and quicksort.
 * The `pair` structure is also defined here.
 *
 * This file also contains functions that perform set operations on sorted
 * arrays and native arrays, such as union, intersection, subtraction, and
 * subset.
 *
 * <!-- Created on: Mar 3, 2012
 *          Author: asaparov -->
 */

#ifndef ARRAY_H_
#define ARRAY_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>

#include <core/core.h>

/**
 * The multiplicative factor by which array capacity is changed.
 */
#define RESIZE_FACTOR 2


namespace core {


namespace detail {
	template<typename C> static auto test_resizeable(int32_t) ->
			decltype(void(std::declval<C>().on_resize()), std::true_type{});
	template<typename C> static auto test_resizeable(int64_t) -> std::false_type;
}

/**
 * This type trait is [true_type](http://en.cppreference.com/w/cpp/types/integral_constant)
 * if and only if `T` is class with a public function `void on_resize()`.
 */
template<typename T> struct is_resizeable : decltype(core::detail::test_resizeable<T>(0)){};

/**
 * Resizes the given native array `data` with the requested capacity `new_capacity`.
 * \tparam SizeType a type that satisfies [is_integral](http://en.cppreference.com/w/cpp/types/is_integral).
 * \tparam T the generic type of the elements in `data`.
 * \param data the array to resize.
 * \param new_capacity the requested size.
 * \return `true` on success; `data` may point to a different memory address.
 * \return `false` on failure; `data` is unchanged.
 */
template<typename T, typename SizeType,
	typename std::enable_if<std::is_integral<SizeType>::value>::type* = nullptr>
inline bool resize(T*& data, const SizeType& new_capacity) {
	T* new_data = (T*) realloc(static_cast<void*>(data), new_capacity * sizeof(T));
	if (new_data == NULL) {
		fprintf(stderr, "resize ERROR: Out of memory.\n");
		return false;
	}
	data = new_data;
	return true;
}

/**
 * This function multiplies `capacity` by #RESIZE_FACTOR. It then repeats this
 * until `capacity >= new_length`.
 */
template<typename SizeType, typename std::enable_if<std::is_integral<SizeType>::value>::type* = nullptr>
inline void expand_capacity(SizeType& capacity, size_t new_length) {
	do {
		/* increase the size of the underlying array */
		capacity *= RESIZE_FACTOR;
	} while (new_length > capacity);
}

/**
 * For a given requested length `new_length`, this function calls
 * expand_capacity() to determine the new `capacity` of the native array
 * `data`. The function then attempts to resize the array with this capacity.
 * Note this function does not check whether `new_length <= capacity`.
 */
template<typename T, typename SizeType,
	typename std::enable_if<std::is_integral<SizeType>::value>::type* = nullptr>
inline bool expand(T*& data, SizeType& capacity, size_t new_length) {
	expand_capacity(capacity, new_length);
	return resize(data, capacity);
}

/**
 * For a given requested length `new_length`, this function expands the native
 * array `data` by factors of #RESIZE_FACTOR until `capacity >= new_length`.
 * If initially `new_length <= capacity`, this function does nothing.
 */
template<typename T, typename SizeType,
	typename std::enable_if<std::is_integral<SizeType>::value>::type* = nullptr>
inline bool ensure_capacity(T*& data, SizeType& capacity, size_t new_length)
{
	if (new_length <= capacity)
		return true;
	SizeType new_capacity = capacity;
	if (!expand(data, new_capacity, new_length))
		return false;
	capacity = new_capacity;
	return true;
}

/**
 * Performs a linear search through the array `data` to find the smallest index
 * `i >= start` such that `element == data[i]`.
 * \tparam Key a generic type for which operator `==` is defined for arguments of type `Key` and `T`.
 * \tparam T a generic type for which operator `==` is defined for arguments of type `Key` and `T`.
 * \return an index in `start, start + 1, ..., length - 1` if the element was found.
 * \return `length` if the element was not found.
 */
template<typename Key, typename T, typename SizeType,
	typename std::enable_if<std::is_integral<SizeType>::value>::type* = nullptr>
inline SizeType index_of(const Key& element, const T* data,
		const SizeType& length, const SizeType& start = 0)
{
	for (SizeType i = start; i < length; i++)
		if (element == data[i])
			return i;
	return length;
}

/**
 * Performs a linear search through the array `data` to find the largest index
 * `i` such that `element == data[i]`.
 * \tparam Key a generic type for which operator `==` is defined for arguments of type `Key` and `T`.
 * \tparam T a generic type for which operator `==` is defined for arguments of type `Key` and `T`.
 * \return an index in `0, 1, ..., length - 1` if the element was found.
 * \return `static_cast<unsigned int>(-1)` if the element was not found.
 */
template<typename Key, typename T, typename SizeType,
	typename std::enable_if<std::is_integral<SizeType>::value>::type* = nullptr>
inline unsigned int last_index_of(const Key& element, const T* data, const SizeType& length)
{
	unsigned int i = length;
	while (i != 0) {
		i--;
		if (element == data[i])
			return i;
	}
	return static_cast<unsigned int>(-1);
}

/**
 * A resizeable sequence of objects, stored contiguously, each with generic
 * type `T`. This structure does not automatically initialize or free its
 * elements, and so the user must appropriately free each element before the
 * array is destroyed.
 *
 * In the following example, we demonstrate a simple use-case of array. Here,
 * `a` is automatically freed by the destructor since it was initialized on the
 * stack. The expected output is `-1 -1 0 3 `.
 *
 * ```{.cpp}
 * #include <core/array.h>
 * #include <stdio.h>
 * using namespace core;
 *
 * int main() {
 * 	array<int> a = array<int>(8);
 * 	a.add(-1); a.add(-4);
 * 	a.add(3); a.add(0);
 * 	a.remove(1);
 *
 * 	printf("%d ", a[0]);
 * 	for (int element : a)
 * 		printf("%d ", element);
 * }
 * ```
 *
 *
 * However, if `a` is not allocated on the stack, the destructor will not be
 * automatically called, and so it must be freed manually using `core::free` or
 * `array::free`.
 *
 * ```{.cpp}
 * #include <core/array.h>
 * #include <stdio.h>
 * using namespace core;
 *
 * int main() {
 * 	array<int>& a = *((array<int>*) alloca(sizeof(array<int>)));
 * 	array_init(a, 8);
 * 	a.add(-1); a.add(-4);
 * 	a.add(3); a.add(0);
 * 	a.remove(1);
 *
 * 	printf("%d ", a[0]);
 * 	for (int element : a)
 * 		printf("%d ", element);
 * 	free(a);
 * }
 * ```
 *
 *
 * Also note that a number of member functions require that `T` be
 * [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable).
 * In other cases, elements should be added manually to the underlying native
 * array array::data. This structure also defines array::begin and array::end,
 * similar to Standard Template Library iterators, which enables the use of the
 * range-based for loop in the example below. In this example, the expected
 * output is `first second `.
 *
 * ```{.cpp}
 * #include <core/array.h>
 * #include <stdio.h>
 * #include <string.h>
 * using namespace core;
 *
 * struct custom_string {
 * 	char* buffer;
 *
 * 	static void free(custom_string& s) {
 * 		core::free(s.buffer);
 * 	}
 * };
 *
 * bool init(custom_string& s, const char* src) {
 * 	s.buffer = (char*) malloc(sizeof(char) * (strlen(src) + 1));
 * 	if (s.buffer == NULL)
 * 		return false;
 * 	memcpy(s.buffer, src, sizeof(char) * (strlen(src) + 1));
 * 	return true;
 * }
 *
 * int main() {
 * 	array<custom_string> a = array<custom_string>(8);
 * 	init(a[0], "first");
 * 	init(a[1], "second");
 * 	a.length = 2;
 *
 * 	for (const custom_string& s : a)
 * 		printf("%s ", s.buffer);
 * 	for (custom_string& s : a)
 * 		free(s);
 * }
 * ```
 * Note in the above example that since the array struct does not automatically
 * free its elements, they must be freed manually.
 */
template<typename T>
struct array {
	/**
	 * The underlying native array of elements.
	 */
	T* data;

	/**
	 * The length of the array.
	 */
	size_t length;

	/**
	 * The capacity of array::data.
	 */
	size_t capacity;

	/**
	 * Constructs an array with zero size and the given `initial_capacity`.
	 */
	array(size_t initial_capacity)
	{
		if (!initialize(initial_capacity))
			exit(EXIT_FAILURE);
	}

	~array() { free(); }

	/**
	 * Returns a reference to the element at the given `index`. No bounds-checking is performed.
	 */
	inline T& operator[] (size_t index) {
		return data[index];
	}

	/**
	 * Returns a const reference to the element at the given `index`. No bounds-checking is performed.
	 */
	inline const T& operator[] (size_t index) const {
		return data[index];
	}

	/**
	 * Sets the length of the array to `0`. Any elements are not freed.
	 */
	inline void clear()
	{
		length = 0;
	}

	/**
	 * Moves the last element in the array to the position given by `index` and
	 * decrements array::length by `1`. The element initially at `index` is not
	 * freed.
	 */
	void remove(size_t index)
	{
		core::move(data[length - 1], data[index]);
		length--;
	}

	/**
	 * For a given requested length `new_length`, this function expands the
	 * array by factors of #RESIZE_FACTOR until `array::capacity >= new_length`.
	 * If initially `new_length <= array::capacity`, this function does
	 * nothing. If the resize operation moved array::data to a new memory
	 * address, and `T` satisfies is_resizeable, then `x.on_resize()` is called
	 * for every element `x` in the array.
	 */
	template<typename C = T, typename std::enable_if<std::is_same<C, T>::value && is_resizeable<C>::value>::type* = nullptr>
	bool ensure_capacity(size_t new_length) {
		const T* old_data = data;
		if (!core::ensure_capacity(data, capacity, new_length)) return false;
		if (data != old_data) {
			for (unsigned int i = 0; i < length; i++)
				data[i].on_resize();
		}
		return true;
	}

	template<typename C = T, typename std::enable_if<std::is_same<C, T>::value && !is_resizeable<C>::value>::type* = nullptr>
	bool ensure_capacity(size_t new_length) {
		return core::ensure_capacity(data, capacity, new_length);
	}

	/**
	 * Adds the given native array of elements to this structure. This function
	 * uses [memcpy](http://en.cppreference.com/w/cpp/string/byte/memcpy),
	 * and so it should not be used if the elements are not
	 * [TriviallyCopyable](https://en.cppreference.com/w/cpp/named_req/TriviallyCopyable).
	 * In such a case, addition should be performed manually using the public fields.
	 * \tparam T is [TriviallyCopyable](https://en.cppreference.com/w/cpp/named_req/TriviallyCopyable).
	 */
	bool append(const T* elements, size_t size)
	{
		if (!ensure_capacity(length + size))
			return false;
		memcpy(&data[length], elements, sizeof(T) * size);
		length += size;
		return true;
	}

	/**
	 * Calls array::index_of to determine whether `element` exists in this array.
	 * \tparam Key a generic type for which operator `==` is defined for arguments of type `Key` and `T`.
	 */
	template<typename Key>
	inline bool contains(const Key& element) const {
		return index_of(element) < length;
	}

	/**
	 * Performs a linear search of the array to find the smallest index `i`
	 * such that `element == array::data[i]`.
	 * \tparam Key a generic type for which operator `==` is defined for arguments of type `Key` and `T`.
	 * \return an index in `0, 1, ..., array::length - 1` if the element was found.
	 * \return `array::length` if the element was not found.
	 */
	template<typename Key>
	inline unsigned int index_of(const Key& element) const {
		return core::index_of(element, data, (unsigned int) length);
	}

	/**
	 * Performs a linear search through the array to find the smallest index
	 * `i >= start` such that `element == array::data[i]`.
	 * \tparam Key a generic type for which operator `==` is defined for arguments of type `Key` and `T`.
	 * \return an index in `start, start + 1, ..., length - 1` if the element was found.
	 * \return `length` if the element was not found.
	 */
	template<typename Key, typename SizeType>
	inline unsigned int index_of(const Key& element, SizeType start) const {
		return core::index_of(element, data, (unsigned int) length, start);
	}

	/**
	 * Returns a reference to `array::data[0]`, ignoring any bounds.
	 */
	T& first()
	{
		return data[0];
	}

	/**
	 * Returns a const reference to `array::data[0]`, ignoring any bounds.
	 */
	const T& first() const
	{
		return data[0];
	}

	/**
	 * Returns a reference to `array::data[array::length - 1]`, ignoring any bounds.
	 */
	T& last()
	{
		return data[length - 1];
	}

	/**
	 * Returns a const reference to `array::data[array::length - 1]`, ignoring any bounds.
	 */
	const T& last() const
	{
		return data[length - 1];
	}

	/**
	 * Adds the given element to this array, increasing its capacity if
	 * necessary. The assignment operator performs the addition, and so this
	 * function should not be used if `T` is not
	 * [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable).
	 * In such a case, addition should be performed manually using the public fields.
	 * \tparam T is [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable).
	 */
	bool add(const T& element)
	{
		if (!ensure_capacity(length + 1))
			return false;
		data[length] = element;
		length++;
		return true;
	}

	/**
	 * Returns the element at `array::length - 1` and decrements array::length
	 * by `1`, ignoring any bounds.
	 */
	T pop()
	{
		length--;
		return data[length];
	}

	/**
	 * Returns an iterator to the beginning of the array.
	 */
	inline T* begin() {
		return data;
	}

	/**
	 * Returns an iterator to the end of the array.
	 */
	inline T* end() {
		return data + length;
	}

	/**
	 * Returns a const iterator to the beginning of the array.
	 */
	inline const T* begin() const {
		return data;
	}

	/**
	 * Returns a const iterator to the end of the array.
	 */
	inline const T* end() const {
		return data + length;
	}

	/**
	 * Copies the underlying fields from `src` into `dst`, effectively moving
	 * the array from `src` into `dst`.
	 */
	static inline void move(const array<T>& src, array<T>& dst) {
		dst.length = src.length;
		dst.capacity = src.capacity;
		dst.data = src.data;
	}

	template<typename Metric>
	static inline long unsigned int size_of(const array<T>& a, const Metric& metric) {
		long unsigned int sum = core::size_of(a.capacity) + core::size_of(a.length);
		for (unsigned int i = 0; i < a.length; i++)
			sum += core::size_of(a.data[i], metric);
		return sum + (a.capacity - a.length) * sizeof(T);
	}

	/**
	 * Frees array::data. This should not be used if `a` was constructed on the
	 * stack, as the destructor will automatically free array::data. The
	 * elements of `a` are not freed.
	 */
	static inline void free(array<T>& a) { a.free(); }

private:
	inline bool initialize(size_t initial_capacity)
	{
#if !defined(NDEBUG)
		if (initial_capacity == 0)
			fprintf(stderr, "array.initialize WARNING: Initial capacity is zero.\n");
#endif

		capacity = initial_capacity;
		length = 0;
		data = (T*) malloc(sizeof(T) * capacity);
		if (data == NULL) {
			fprintf(stderr, "array.initialize ERROR: Out of memory.\n");
			return false;
		}
		return true;
	}

	inline void free() {
		if (data != NULL) {
			core::free(data);
			data = NULL;
		}
	}

	template<typename K>
	friend bool array_init(array<K>& m, size_t initial_capacity);
};

/**
 * Initializes the array `m` with the given `initial_capacity`.
 */
template<typename T>
bool array_init(array<T>& m, size_t initial_capacity) {
	return m.initialize(initial_capacity);
}

/**
 * Returns array::length of `m`.
 */
template<typename T>
inline size_t size(const array<T>& m) {
	return m.length;
}

/**
 * Swaps the underlying buffers of the given arrays.
 */
template<typename T>
inline void swap(array<T>& a, array<T>& b)
{
	T* temp = a.data;
	a.data = b.data;
	b.data = temp;

	size_t swap = a.length;
	a.length = b.length;
	b.length = swap;

	swap = a.capacity;
	a.capacity = b.capacity;
	b.capacity = swap;
}

template<typename T, typename Metric>
inline long unsigned int size(const array<T>& a, const Metric& metric) {
	long unsigned int sum = size(a.capacity) + size(a.length);
	for (unsigned int i = 0; i < a.length; i++)
		sum += size(a.data[i], metric);
	return sum + (a.capacity - a.length) * sizeof(T);
}

/**
 * Compares the two given arrays and returns true if and only if `a.length == b.length`
 * and there is no index `i` such that `a.data[i] != b.data[i]`.
 */
template<typename T>
inline bool operator == (const array<T>& a, const array<T>& b) {
	if (a.length != b.length)
		return false;
	for (unsigned int i = 0; i < a.length; i++)
		if (a.data[i] != b.data[i])
			return false;
	return true;
}

/**
 * Compares the two given arrays and returns true if and only if `a.length != b.length`
 * or there is some index `i` such that `a.data[i] != b.data[i]`.
 */
template<typename T>
inline bool operator != (const array<T>& a, const array<T>& b) {
	if (a.length != b.length)
		return true;
	for (unsigned int i = 0; i < a.length; i++)
		if (a.data[i] != b.data[i])
			return true;
	return false;
}

/**
 * Performs insertion sort on the given native array `keys` with given `length`.
 * \tparam T is [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable),
 * 		[CopyConstructible](https://en.cppreference.com/w/cpp/named_req/CopyConstructible),
 * 		and [LessThanComparable](https://en.cppreference.com/w/cpp/named_req/LessThanComparable).
 */
template<typename T>
void insertion_sort(T* keys, unsigned int length)
{
	for (unsigned int i = 1; i < length; i++) {
		T item = keys[i];
		unsigned int hole = i;

		while (hole > 0 && item < keys[hole - 1]) {
			keys[hole] = keys[hole - 1];
			hole--;
		}

		keys[hole] = item;
	}
}

/**
 * Performs insertion sort on the given native array `keys` with given `length`
 * and `sorter`.
 * \tparam T satisfies is_moveable, and for which a function
 * 		`bool less_than(const T&, const T&, const Sorter&)` is defined.
 */
template<typename T, typename Sorter,
	typename std::enable_if<!std::is_integral<Sorter>::value>::type* = nullptr>
void insertion_sort(T* keys, unsigned int length, const Sorter& sorter)
{
	T& item = *((T*) malloc(sizeof(T)));
	for (unsigned int i = 1; i < length; i++) {
		move(keys[i], item);
		unsigned int hole = i;

		while (hole > 0 && less_than(item, keys[hole - 1], sorter)) {
			move(keys[hole - 1], keys[hole]);
			hole--;
		}

		move(item, keys[hole]);
	}
	free(&item);
}

/**
 * Performs insertion sort on the given native arrays `keys` and `values` with given `length`.
 * \tparam K is [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable),
 * 		[CopyConstructible](https://en.cppreference.com/w/cpp/named_req/CopyConstructible),
 * 		and [LessThanComparable](https://en.cppreference.com/w/cpp/named_req/LessThanComparable).
 * \tparam V is [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable),
 * 		[CopyConstructible](https://en.cppreference.com/w/cpp/named_req/CopyConstructible).
 */
template<typename K, typename V>
void insertion_sort(K* keys, V* values, unsigned int length)
{
	for (unsigned int i = 1; i < length; i++) {
		K item = keys[i];
		V value = values[i];
		unsigned int hole = i;

		while (hole > 0 && item < keys[hole - 1]) {
			keys[hole] = keys[hole - 1];
			values[hole] = values[hole - 1];
			hole--;
		}

		keys[hole] = item;
		values[hole] = value;
	}
}

/**
 * Performs insertion sort on the given native arrays `keys` and `values` with
 * given `length` and `sorter`.
 * \tparam K satisfies is_moveable, and for which a function
 * 		`bool less_than(const K&, const K&, const Sorter&)` is defined.
 * \tparam V satisfies is_moveable.
 */
template<typename K, typename V, typename Sorter,
	typename std::enable_if<!std::is_integral<Sorter>::value>::type* = nullptr>
void insertion_sort(K* keys, V* values, unsigned int length, const Sorter& sorter)
{
	K& item = *((K*) malloc(sizeof(K)));
	V& value = *((V*) malloc(sizeof(V)));
	for (unsigned int i = 1; i < length; i++) {
		move(keys[i], item);
		move(values[i], value);
		unsigned int hole = i;

		while (hole > 0 && less_than(item, keys[hole - 1], sorter)) {
			move(keys[hole - 1], keys[hole]);
			move(values[hole - 1], values[hole]);
			hole--;
		}

		move(item, keys[hole]);
		move(value, values[hole]);
	}
	free(&item); free(&value);
}

/**
 * Performs insertion sort on the given array `keys`.
 * \tparam T is [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable),
 * 		[CopyConstructible](https://en.cppreference.com/w/cpp/named_req/CopyConstructible),
 * 		and [LessThanComparable](https://en.cppreference.com/w/cpp/named_req/LessThanComparable).
 */
template<typename T>
inline void insertion_sort(array<T>& keys) {
	insertion_sort(keys.data, (unsigned int) keys.length);
}

/**
 * Performs insertion sort on the given array `keys` with the given `sorter`.
 * \tparam T satisfies is_moveable, and for which a function
 * 			`bool less_than(const T&, const T&, const Sorter&)` is defined.
 */
template<typename T, typename Sorter,
	typename std::enable_if<!std::is_integral<Sorter>::value>::type* = nullptr>
inline void insertion_sort(array<T>& keys, const Sorter& sorter)
{
	insertion_sort(keys.data, (unsigned int) keys.length, sorter);
}

/**
 * Performs insertion sort on the given arrays `keys` and `values`.
 * \tparam K is [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable),
 * 		[CopyConstructible](https://en.cppreference.com/w/cpp/named_req/CopyConstructible),
 * 		and [LessThanComparable](https://en.cppreference.com/w/cpp/named_req/LessThanComparable).
 * \tparam V is [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable),
 * 		[CopyConstructible](https://en.cppreference.com/w/cpp/named_req/CopyConstructible).
 */
template<typename K, typename V>
inline void insertion_sort(array<K>& keys, array<V>& values) {
	insertion_sort(keys.data, values.data, (unsigned int) keys.length);
}

/**
 * Performs insertion sort on the given arrays `keys` and `values` with the
 * given `sorter`.
 * \tparam K satisfies is_moveable, and for which a function
 * 			`bool less_than(const K&, const K&, const Sorter&)` is defined.
 * \tparam V satisfies is_moveable.
 */
template<typename K, typename V, typename Sorter,
	typename std::enable_if<!std::is_integral<Sorter>::value>::type* = nullptr>
inline void insertion_sort(array<K>& keys, array<V>& values, const Sorter& sorter)
{
	insertion_sort(keys.data, values.data, (unsigned int) keys.length, sorter);
}

/**
 * Reverses the order of the elements in the given native `array` with given `length`.
 * \tparam T satisfies is_swappable.
 */
template<typename T>
void reverse(T* array, unsigned int length) {
	for (unsigned int i = 0; i < length / 2; i++) {
		unsigned int other = length - i - 1;
		swap(array[i], array[other]);
	}
}

/**
 * Reverses the order of the elements in the given `array`.
 * \tparam T satisfies is_swappable.
 */
template<typename T>
inline void reverse(array<T>& array) {
	reverse(array.data, (unsigned int) array.length);
}

template<typename T>
inline const T& get_pivot(T* array, unsigned int start, unsigned int end) {
	return array[(end + start) / 2];
}

template<typename T>
inline void quick_sort_partition(T* array,
		unsigned int start, unsigned int end,
		unsigned int& l, unsigned int& r)
{
	const T p = get_pivot(array, start, end);
	while (true) {
		while (array[l] < p)
			l++;
		while (p < array[r])
			r--;
		if (l == r) {
			l++;
			if (r > 0) r--;
			return;
		} else if (l > r) return;
		swap(array[l++], array[r--]);
	}
}

template<typename T, typename Sorter,
	typename std::enable_if<!std::is_integral<Sorter>::value>::type* = nullptr>
inline void quick_sort_partition(
		T* array, unsigned int start, unsigned int end,
		unsigned int& l, unsigned int& r, const Sorter& sorter)
{
	T& p = *((T*) malloc(sizeof(T)));
	move(get_pivot(array, start, end), p);
	while (true) {
		while (less_than(array[l], p, sorter))
			l++;
		while (less_than(p, array[r], sorter))
			r--;
		if (l == r) {
			l++;
			if (r > 0) r--;
			break;
		} else if (l > r) break;
		swap(array[l++], array[r--]);
	}
	free(&p);
}

template<typename K, typename V>
inline void quick_sort_partition(K* keys, V* values,
		unsigned int start, unsigned int end, unsigned int& l, unsigned int& r)
{
	const K p = get_pivot(keys, start, end);
	while (true) {
		while (keys[l] < p)
			l++;
		while (p < keys[r])
			r--;
		if (l == r) {
			l++;
			if (r > 0) r--;
			return;
		} else if (l > r) return;
		swap(values[l], values[r]);
		swap(keys[l++], keys[r--]);
	}
}

template<typename K, typename V, typename Sorter,
	typename std::enable_if<!std::is_integral<Sorter>::value>::type* = nullptr>
inline void quick_sort_partition(
		K* keys, V* values, unsigned int start, unsigned int end,
		unsigned int& l, unsigned int& r, const Sorter& sorter)
{
	K& p = *((K*) malloc(sizeof(K)));
	move(get_pivot(keys, start, end), p);
	while(true) {
		while (less_than(keys[l], p, sorter))
			l++;
		while (less_than(p, keys[r], sorter))
			r--;
		if (l == r) {
			l++;
			if (r > 0) r--;
			break;
		} else if (l > r) break;
		swap(values[l], values[r]);
		swap(keys[l++], keys[r--]);
	}
	free(&p);
}

template<typename T>
void quick_sort(T* array, unsigned int start, unsigned int end)
{
	if (start >= end)
		return;
	unsigned int l = start, r = end;
	quick_sort_partition(array, start, end, l, r);
	quick_sort(array, start, r);
	quick_sort(array, l, end);
}

template<typename T, typename Sorter,
	typename std::enable_if<!std::is_integral<Sorter>::value>::type* = nullptr>
void quick_sort(T* array, unsigned int start, unsigned int end, const Sorter& sorter)
{
	if (start >= end)
		return;
	unsigned int l = start, r = end;
	quick_sort_partition(array, start, end, l, r, sorter);
	quick_sort(array, start, r, sorter);
	quick_sort(array, l, end, sorter);
}

template<typename K, typename V>
void quick_sort(K* keys, V* values, unsigned int start, unsigned int end)
{
	if (start >= end)
		return;
	unsigned int l = start, r = end;
	quick_sort_partition(keys, values, start, end, l, r);
	quick_sort(keys, values, start, r);
	quick_sort(keys, values, l, end);
}

template<typename K, typename V, typename Sorter,
	typename std::enable_if<!std::is_integral<Sorter>::value>::type* = nullptr>
void quick_sort(K* keys, V* values,
		unsigned int start, unsigned int end, const Sorter& sorter)
{
	if (start >= end)
		return;
	unsigned int l = start, r = end;
	quick_sort_partition(keys, values, start, end, l, r, sorter);
	quick_sort(keys, values, start, r, sorter);
	quick_sort(keys, values, l, end, sorter);
}

/**
 * Performs Quicksort on the given native array `keys` with given `length`.
 * This function assumes `length > 0`. If the preprocessor `NDEBUG` is not
 * defined, this function outputs a warning when `length == 0`.
 * \tparam T is [CopyConstructible](https://en.cppreference.com/w/cpp/named_req/CopyConstructible),
 * 			[LessThanComparable](https://en.cppreference.com/w/cpp/named_req/LessThanComparable),
 * 			and is_swappable.
 */
template<typename T>
inline void quick_sort(T* keys, unsigned int length) {
#if !defined(NDEBUG)
	if (length == 0) {
		fprintf(stderr, "quick_sort WARNING: Length is zero.\n");
		return;
	}
#endif
	quick_sort(keys, 0, length - 1);
}

/**
 * Performs Quicksort on the given native array `keys` with given `length` and
 * `sorter`. This function assumes `length > 0`. If the preprocessor `NDEBUG`
 * is not defined, this function outputs a warning when `length == 0`.
 * \tparam T a generic type that satisfies is_swappable, and for which a
 * 			function `bool less_than(const T&, const T&, const Sorter&)` is
 * 			defined.
 */
template<typename T, typename Sorter,
	typename std::enable_if<!std::is_integral<Sorter>::value>::type* = nullptr>
inline void quick_sort(T* keys, unsigned int length, const Sorter& sorter)
{
#if !defined(NDEBUG)
	if (length == 0) {
		fprintf(stderr, "quick_sort WARNING: Length is zero.\n");
		return;
	}
#endif
	quick_sort(keys, 0, length - 1, sorter);
}

/**
 * Performs Quicksort on the given native arrays `keys` and `values` with given
 * `length`. This function assumes `length > 0`. If the preprocessor `NDEBUG`
 * is not defined, this function outputs a warning when `length == 0`.
 * \tparam K is [CopyConstructible](https://en.cppreference.com/w/cpp/named_req/CopyConstructible),
 * 			[LessThanComparable](https://en.cppreference.com/w/cpp/named_req/LessThanComparable),
 * 			and is_swappable.
 * \tparam V satisfies is_swappable.
 */
template<typename K, typename V>
inline void quick_sort(K* keys, V* values, unsigned int length) {
#if !defined(NDEBUG)
	if (length == 0) {
		fprintf(stderr, "quick_sort WARNING: Length is zero.\n");
		return;
	}
#endif
	quick_sort(keys, values, 0, length - 1);
}

/**
 * Performs Quicksort on the given native arrays `keys` and `values` with given
 * `length` and `sorter`. This function assumes `length > 0`. If the preprocessor
 * `NDEBUG` is not defined, this function outputs a warning when `length == 0`.
 * \tparam K a generic type that satisfies is_swappable, and for which a
 * 			function `bool less_than(const K&, const K&, const Sorter&)` is
 * 			defined.
 * \tparam V satisfies is_swappable.
 */
template<typename K, typename V, typename Sorter,
	typename std::enable_if<!std::is_integral<Sorter>::value>::type* = nullptr>
inline void quick_sort(K* keys, V* values, unsigned int length, const Sorter& sorter)
{
#if !defined(NDEBUG)
	if (length == 0) {
		fprintf(stderr, "quick_sort WARNING: Length is zero.\n");
		return;
	}
#endif
	quick_sort(keys, values, 0, length - 1, sorter);
}

/**
 * Performs Quicksort on the given array `keys`. This function assumes
 * `length > 0`. If the preprocessor `NDEBUG` is not defined, this function
 * outputs a warning when `length == 0`.
 * \tparam T is [CopyConstructible](https://en.cppreference.com/w/cpp/named_req/CopyConstructible),
 * 			[LessThanComparable](https://en.cppreference.com/w/cpp/named_req/LessThanComparable),
 * 			and is_swappable.
 */
template<typename T>
inline void quick_sort(array<T>& keys) {
#if !defined(NDEBUG)
	if (keys.length == 0) {
		fprintf(stderr, "quick_sort WARNING: Length is zero.\n");
		return;
	}
#endif
	quick_sort(keys.data, 0, (unsigned int) keys.length - 1);
}

/**
 * Performs Quicksort on the given native array `keys` with the given `sorter`.
 * This function assumes `length > 0`. If the preprocessor `NDEBUG` is not
 * defined, this function outputs a warning when `length == 0`.
 * \tparam T a generic type that satisfies is_swappable, and for which a
 * 			function `bool less_than(const T&, const T&, const Sorter&)` is
 * 			defined.
 */
template<typename T, typename Sorter,
	typename std::enable_if<!std::is_integral<Sorter>::value>::type* = nullptr>
inline void quick_sort(array<T>& keys, const Sorter& sorter)
{
#if !defined(NDEBUG)
	if (keys.length == 0) {
		fprintf(stderr, "quick_sort WARNING: Length is zero.\n");
		return;
	}
#endif
	quick_sort(keys.data, 0, (unsigned int) keys.length - 1, sorter);
}

/**
 * Performs Quicksort on the given arrays `keys` and `values`. This function
 * assumes `length > 0`. If the preprocessor `NDEBUG` is not defined, this
 * function outputs a warning when `length == 0`.
 * \tparam K is [CopyConstructible](https://en.cppreference.com/w/cpp/named_req/CopyConstructible),
 * 			[LessThanComparable](https://en.cppreference.com/w/cpp/named_req/LessThanComparable),
 * 			and is_swappable.
 * \tparam V satisfies is_swappable.
 */
template<typename K, typename V>
inline void quick_sort(array<K>& keys, array<V>& values) {
#if !defined(NDEBUG)
	if (keys.length == 0) {
		fprintf(stderr, "quick_sort WARNING: Length is zero.\n");
		return;
	}
#endif
	quick_sort(keys.data, values.data, 0, (unsigned int) keys.length - 1);
}

/**
 * Performs Quicksort on the given arrays `keys` and `values` with the given
 * `sorter`. This function assumes `length > 0`. If the preprocessor `NDEBUG`
 * is not defined, this function outputs a warning when `length == 0`.
 * \tparam K a generic type that satisfies is_swappable, and for which a
 * 			function `bool less_than(const K&, const K&, const Sorter&)` is
 * 			defined.
 * \tparam V satisfies is_swappable.
 */
template<typename K, typename V, typename Sorter,
	typename std::enable_if<!std::is_integral<Sorter>::value>::type* = nullptr>
inline void quick_sort(array<K>& keys, array<V>& values, const Sorter& sorter)
{
#if !defined(NDEBUG)
	if (keys.length == 0) {
		fprintf(stderr, "quick_sort WARNING: Length is zero.\n");
		return;
	}
#endif
	quick_sort(keys.data, values.data, 0, (unsigned int) keys.length - 1, sorter);
}


/**
 * <!-- Hybrid quicksort-insertion sort. -->
 */

template<typename T>
void sort(T* array, unsigned int start, unsigned int end)
{
	if (start >= end)
		return;
	else if (start + 16 >= end) {
		insertion_sort(&array[start], end - start + 1);
		return;
	}
	unsigned int l = start, r = end;
	quick_sort_partition(array, start, end, l, r);
	sort(array, start, r);
	sort(array, l, end);
}

template<typename T, typename Sorter,
	typename std::enable_if<!std::is_integral<Sorter>::value>::type* = nullptr>
void sort(T* array, unsigned int start, unsigned int end, const Sorter& sorter)
{
	if (start >= end)
		return;
	else if (start + 16 >= end) {
		insertion_sort(&array[start], end - start + 1, sorter);
		return;
	}
	unsigned int l = start, r = end;
	quick_sort_partition(array, start, end, l, r, sorter);
	sort(array, start, r, sorter);
	sort(array, l, end, sorter);
}

template<typename K, typename V>
void sort(K* keys, V* values, unsigned int start, unsigned int end)
{
	if (start >= end)
		return;
	else if (start + 16 >= end) {
		insertion_sort(&keys[start], &values[start], end - start + 1);
		return;
	}
	unsigned int l = start, r = end;
	quick_sort_partition(keys, values, start, end, l, r);
	sort(keys, values, start, r);
	sort(keys, values, l, end);
}

template<typename K, typename V, typename Sorter,
	typename std::enable_if<!std::is_integral<Sorter>::value>::type* = nullptr>
void sort(K* keys, V* values, unsigned int start, unsigned int end, const Sorter& sorter)
{
	if (start >= end)
		return;
	else if (start + 16 >= end) {
		insertion_sort(&keys[start], &values[start], end - start + 1, sorter);
		return;
	}
	unsigned int l = start, r = end;
	quick_sort_partition(keys, values, start, end, l, r, sorter);
	sort(keys, values, start, r, sorter);
	sort(keys, values, l, end, sorter);
}

/**
 * Performs hybrid Quicksort-insertion sort on the given native array `keys`
 * with given `length`. To improve performance, the Quicksort switches to
 * insertion sort for small partitions. This function assumes `length > 0`. If
 * the preprocessor `NDEBUG` is not defined, this function outputs a warning
 * when `length == 0`.
 * \tparam T is [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable),
 * 			[CopyConstructible](https://en.cppreference.com/w/cpp/named_req/CopyConstructible)
 * 			[LessThanComparable](https://en.cppreference.com/w/cpp/named_req/LessThanComparable),
 * 			and is_swappable.
 */
template<typename T>
inline void sort(T* keys, unsigned int length) {
#if !defined(NDEBUG)
	if (length == 0) {
		fprintf(stderr, "sort WARNING: Length is zero.\n");
		return;
	}
#endif
	sort(keys, 0, length - 1);
}

/**
 * Performs hybrid Quicksort-insertion sort on the given native array `keys`
 * with given `length` and `sorter`. To improve performance, the Quicksort
 * switches to insertion sort for small partitions. This function assumes
 * `length > 0`. If the preprocessor `NDEBUG` is not defined, this function
 * outputs a warning when `length == 0`.
 * \tparam T a generic type that satisfies is_swappable and is_moveable, and
 * 			for which a function `bool less_than(const T&, const T&, const Sorter&)`
 * 			is defined.
 */
template<typename T, typename Sorter,
	typename std::enable_if<!std::is_integral<Sorter>::value>::type* = nullptr>
inline void sort(T* keys, unsigned int length, const Sorter& sorter)
{
#if !defined(NDEBUG)
	if (length == 0) {
		fprintf(stderr, "sort WARNING: Length is zero.\n");
		return;
	}
#endif
	sort(keys, 0, length - 1, sorter);
}

/**
 * Performs hybrid Quicksort-insertion sort on the given native arrays `keys`
 * and `values` with given `length`. To improve performance, the Quicksort
 * switches to insertion sort for small partitions. This function assumes
 * `length > 0`. If the preprocessor `NDEBUG` is not defined, this function
 * outputs a warning when `length == 0`.
 * \tparam K is [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable),
 * 			[CopyConstructible](https://en.cppreference.com/w/cpp/named_req/CopyConstructible)
 * 			[LessThanComparable](https://en.cppreference.com/w/cpp/named_req/LessThanComparable),
 * 			and is_swappable.
 * \tparam V satisfies is_swappable and is_moveable.
 */
template<typename K, typename V>
inline void sort(K* keys, V* values, unsigned int length) {
#if !defined(NDEBUG)
	if (length == 0) {
		fprintf(stderr, "sort WARNING: Length is zero.\n");
		return;
	}
#endif
	sort(keys, values, 0, length - 1);
}

/**
 * Performs hybrid Quicksort-insertion sort on the given native arrays `keys`
 * and `values` with given `length` and `sorter`. To improve performance, the
 * Quicksort switches to insertion sort for small partitions. This function
 * assumes `length > 0`. If the preprocessor `NDEBUG` is not defined, this
 * function outputs a warning when `length == 0`.
 * \tparam K a generic type that satisfies is_swappable and is_moveable, and
 * 			for which a function `bool less_than(const K&, const K&, const Sorter&)`
 * 			is defined.
 * \tparam V satisfies is_swappable and is_moveable.
 */
template<typename K, typename V, typename Sorter,
	typename std::enable_if<!std::is_integral<Sorter>::value>::type* = nullptr>
inline void sort(K* keys, V* values, unsigned int length, const Sorter& sorter)
{
#if !defined(NDEBUG)
	if (length == 0) {
		fprintf(stderr, "sort WARNING: Length is zero.\n");
		return;
	}
#endif
	sort(keys, values, 0, length - 1, sorter);
}

/**
 * Performs hybrid Quicksort-insertion sort on the given array `keys`. To
 * improve performance, the Quicksort switches to insertion sort for small
 * partitions. This function assumes `length > 0`. If the preprocessor `NDEBUG`
 * is not defined, this function outputs a warning when `length == 0`.
 * \tparam T is [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable),
 * 			[CopyConstructible](https://en.cppreference.com/w/cpp/named_req/CopyConstructible)
 * 			[LessThanComparable](https://en.cppreference.com/w/cpp/named_req/LessThanComparable),
 * 			and is_swappable.
 */
template<typename T>
inline void sort(array<T>& keys) {
#if !defined(NDEBUG)
	if (keys.length == 0) {
		fprintf(stderr, "sort WARNING: Length is zero.\n");
		return;
	}
#endif
	sort(keys.data, 0, (unsigned int) keys.length - 1);
}

/**
 * Performs hybrid Quicksort-insertion sort on the given array `keys` with the
 * given `sorter`. To improve performance, the Quicksort switches to insertion
 * sort for small partitions. This function assumes `length > 0`. If the
 * preprocessor `NDEBUG` is not defined, this function outputs a warning when
 * `length == 0`.
 * \tparam T a generic type that satisfies is_swappable and is_moveable, and
 * 			for which a function `bool less_than(const T&, const T&, const Sorter&)`
 * 			is defined.
 */
template<typename T, typename Sorter,
	typename std::enable_if<!std::is_integral<Sorter>::value>::type* = nullptr>
inline void sort(array<T>& keys, const Sorter& sorter)
{
#if !defined(NDEBUG)
	if (keys.length == 0) {
		fprintf(stderr, "sort WARNING: Length is zero.\n");
		return;
	}
#endif
	sort(keys.data, 0, (unsigned int) keys.length - 1, sorter);
}

/**
 * Performs hybrid Quicksort-insertion sort on the given arrays `keys` and
 * `values`. To improve performance, the Quicksort switches to insertion sort
 * for small partitions. This function assumes `length > 0`. If the
 * preprocessor `NDEBUG` is not defined, this function outputs a warning when
 * `length == 0`.
 * \tparam K is [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable),
 * 			[CopyConstructible](https://en.cppreference.com/w/cpp/named_req/CopyConstructible)
 * 			[LessThanComparable](https://en.cppreference.com/w/cpp/named_req/LessThanComparable),
 * 			and is_swappable.
 * \tparam V satisfies is_swappable and is_moveable.
 */
template<typename K, typename V>
inline void sort(array<K>& keys, array<V>& values) {
#if !defined(NDEBUG)
	if (keys.length == 0) {
		fprintf(stderr, "sort WARNING: Length is zero.\n");
		return;
	}
#endif
	sort(keys.data, values.data, 0, (unsigned int) keys.length - 1);
}

/**
 * Performs hybrid Quicksort-insertion sort on the given arrays `keys` and
 * `values` with the given `sorter`. To improve performance, the Quicksort
 * switches to insertion sort for small partitions. This function assumes
 * `length > 0`. If the preprocessor `NDEBUG` is not defined, this function
 * outputs a warning when `length == 0`.
 * \tparam K a generic type that satisfies is_swappable and is_moveable, and
 * 			for which a function `bool less_than(const K&, const K&, const Sorter&)`
 * 			is defined.
 * \tparam V satisfies is_swappable and is_moveable.
 */
template<typename K, typename V, typename Sorter,
	typename std::enable_if<!std::is_integral<Sorter>::value>::type* = nullptr>
inline void sort(array<K>& keys, array<V>& values, const Sorter& sorter)
{
#if !defined(NDEBUG)
	if (keys.length == 0) {
		fprintf(stderr, "sort WARNING: Length is zero.\n");
		return;
	}
#endif
	sort(keys.data, values.data, 0, (unsigned int) keys.length - 1, sorter);
}

/**
 * Returns `true` if and only if the given native array `items` with the given
 * `length` is sorted in non-decreasing order.
 * \tparam T a generic type for which a function `bool less_than(const T&, const T&, const Sorter&)`
 * 			is defined.
 */
template<typename T, typename Sorter>
inline bool is_sorted(const T* items, size_t length, const Sorter& sorter) {
	if (length == 0)
		return true;
	size_t first = 0, next = 0;
	while (++next != length) {
		if (less_than(items[next], items[first], sorter))
			return false;
		first = next;
	}
	return true;
}

/**
 * Returns `true` if and only if the given array `items` is sorted in
 * non-decreasing order.
 * \tparam T a generic type for which a function `bool less_than(const T&, const T&, const Sorter&)`
 * 			is defined.
 */
template<typename T, typename Sorter>
inline bool is_sorted(const array<T>& items, const Sorter& sorter) {
	return is_sorted(items.data, items.length, sorter);
}

/**
 * The default_sorter provides a default implementation of
 * `bool less_than(const T&, const T&, const default_sorter&)`
 * where the comparison is done using the operator `<`.
 */
struct default_sorter { };

template<typename T>
inline bool less_than(const T& first, const T& second, const default_sorter& sorter) {
	return (first < second);
}

/**
 * The pointer_sorter compares items using the `<` operator on the dereferenced
 * arguments.
 */
struct pointer_sorter { };

template<typename T>
inline bool less_than(const T* first, const T* second, const pointer_sorter& sorter) {
	return (*first < *second);
}

/**
 * Deletes consecutive duplicates in the given native `array` with given
 * `length` and returns the new length. Note the deleted elements are not freed.
 * \tparam T is [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable).
 */
template<typename T>
unsigned int unique(T* array, size_t length)
{
	unsigned int result = 0;
	for (unsigned int i = 1; i < length; i++) {
		if (array[result] != array[i])
			array[++result] = array[i];
	}
	return result + 1;
}

/**
 * Deletes consecutive duplicates in the given `array` with given and returns
 * the new length. Note the deleted elements are not freed.
 * \tparam T is [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable).
 */
template<typename T>
inline void unique(array<T>& a) {
	a.length = unique(a.data, a.length);
}

/* forward declarations */

unsigned int sample_uniform(unsigned int);

/**
 * Performs a Knuth shuffle on the given native `array` with given `length`.
 * \tparam T satisfies is_swappable.
 */
template<typename T>
void shuffle(T* array, unsigned int length) {
#if !defined(NDEBUG)
	if (length == 0) {
		fprintf(stderr, "shuffle WARNING: Length is zero.\n");
		return;
	}
#endif
	for (unsigned int i = length - 1; i > 0; i--) {
		unsigned int next = sample_uniform(i + 1);
		if (next != i)
			core::swap(array[next], array[i]);
	}
}

/**
 * Performs a Knuth shuffle on the given native arrays `keys` and `values` with given `length`.
 * \tparam T satisfies is_swappable.
 */
template<typename K, typename V>
void shuffle(K* keys, V* values, unsigned int length) {
#if !defined(NDEBUG)
	if (length == 0) {
		fprintf(stderr, "shuffle WARNING: Length is zero.\n");
		return;
	}
#endif
	for (unsigned int i = length - 1; i > 0; i--) {
		unsigned int next = sample_uniform(i + 1);
		if (next != i) {
			core::swap(keys[next], keys[i]);
			core::swap(values[next], values[i]);
		}
	}
}

/**
 * Performs a Knuth shuffle on the given `array`.
 * \tparam T satisfies is_swappable.
 */
template<typename T>
inline void shuffle(array<T>& items) {
	shuffle(items.data, items.length);
}

/**
 * Given sorted array `a`, this function finds the smallest index `i` such that
 * `a[i] >= b` and `i >= start` and `i < end`.
 */
template<typename T>
unsigned int linear_search(
	const T* a, const T& b,
	unsigned int start,
	unsigned int end)
{
	for (unsigned int i = start; i < end; i++)
		if (a[i] >= b) return i;
	return end;
}

/**
 * Given sorted array `a`, this function finds the smallest index `i` such that
 * `a[i] > b` and `i >= start` and `i < end`.
 */
template<typename T>
unsigned int strict_linear_search(
	const T* a, const T& b,
	unsigned int start,
	unsigned int end)
{
	for (unsigned int i = start; i < end; i++)
		if (a[i] > b) return i;
	return end;
}

/**
 * Given sorted array `a`, this function finds the smallest index `i` such that
 * `a[i] > b` and `i >= start` and `i < end`.
 */
template<typename T>
unsigned int reverse_strict_linear_search(
	const T* a, const T& b,
	unsigned int start,
	unsigned int end)
{
	for (unsigned int i = end; i > start; i--)
		if (a[i - 1] <= b) return i;
	return start;
}

/**
 * Given sorted array `a`, this function finds the smallest index `i` such that
 * `a[i] >= b` and `i >= min` and `i <= max`.
 */
/* TODO: implement a strict variant */
template<typename T>
unsigned int binary_search(
	const T* a, const T& b,
	unsigned int min,
	unsigned int max)
{
	if (a[max] < b)
		return max + 1;

	while (min < max) {
		unsigned int mid = (max + min) / 2;
		if (a[mid] < b)
			min = mid + 1;
		else max = mid;
	}

	return min;
}


/**
 * A simple pair data structure.
 */

template<typename K, typename V>
struct pair {
	/**
	 * The key object in the field.
	 */
	K key;

	/**
	 * The value object in the field.
	 */
	V value;

	pair(const K& key, const V& value) : key(key), value(value) { }

	inline bool operator == (const pair<K, V>& other) const {
		return key == other.key && value == other.value;
	}

	inline bool operator != (const pair<K, V>& other) const {
		return key != other.key || value != other.value;
	}

	static inline void move(const pair<K, V>& src, pair<K, V>& dst) {
		core::move(src.key, dst.key);
		core::move(src.value, dst.value);
	}

	static inline void swap(pair<K, V>& first, pair<K, V>& second) {
		core::swap(first.key, second.key);
		core::swap(first.value, second.value);
	}

	static inline unsigned int hash(const pair<K, V>& pair) {
		return hasher<K>::hash(pair.key) + hasher<V>::hash(pair.value);
	}

	static inline bool is_empty(const pair<K, V>& pair) {
		return hasher<K>::is_empty(pair.key);
	}

	static inline void set_empty(pair<K, V>& pair) {
		hasher<K>::set_empty(pair.key);
	}

	static inline void free(pair<K, V>& pair) {
		core::free(pair.key);
		core::free(pair.value);
	}
};

template<typename K, typename V>
inline bool operator < (const pair<K, V>& first, const pair<K, V>& second) {
	return first.key < second.key;
}

template<typename K, typename V>
inline bool operator <= (const pair<K, V>& first, const pair<K, V>& second) {
	return first.key <= second.key;
}

template<typename K, typename V>
inline bool operator > (const pair<K, V>& first, const pair<K, V>& second) {
	return first.key > second.key;
}

template<typename K, typename V>
inline bool operator >= (const pair<K, V>& first, const pair<K, V>& second) {
	return first.key >= second.key;
}

template<typename K, typename V>
constexpr pair<K, V> make_pair(const K& key, const V& value) {
	return pair<K, V>(key, value);
}

template<typename K, typename V>
inline void swap(pair<K, V>& first, pair<K, V>& second) {
	swap(first.key, second.key);
	swap(first.value, second.value);
}


/**
 * <!-- Functions for performing set operations with sorted arrays. These
 * functions assume the input arrays are sorted and their elements are
 * *distinct*.
 *
 * TODO: Extend these functions to non copy-assignable types. -->
 */

/**
 * For every index `i >= index`, this function moves each element at `i` to index `i + 1`.
 * \tparam T satisfies is_moveable.
 */
template<typename T>
void shift_right(T* list, unsigned int length, unsigned int index)
{
	for (unsigned int i = length; i > index; i--)
		move(list[i - 1], list[i]);
}

/**
 * For every index `i < index`, this function moves each element at `i + 1` to index `i`.
 * \tparam T satisfies is_moveable.
 */
template<typename T>
void shift_left(T* list, unsigned int index)
{
	for (unsigned int i = 0; i < index; i++)
		move(list[i + 1], list[i]);
}

template<bool RemoveDuplicates, typename T, typename UnionFunc>
inline void set_union_helper(UnionFunc do_union, const T& item,
		unsigned int i, unsigned int j, const T*& prev)
{
	if (RemoveDuplicates) {
		if (prev == NULL || *prev != item) {
			do_union(item, i, j);
			prev = &item;
		}
	} else {
		do_union(item, i, j);
	}
}

/**
 * Given ordered native arrays `first` and `second`, this function visits their
 * elements sequentially, and:
 *   1. For every element `x` that appears in both arrays, `union_both(x, i, j)`
 * 		is called where `first[i] == x` and `second[j] == x`.
 *   2. For every element `x` that appears in `first` but not `second`,
 * 		`union_first(x, i, j)` is called where `first[i] == x`, and `j` is the
 * 		smallest such index where `second[j] > x`.
 *   3. For every element `x` that appears in `second` but not `first`,
 * 		`union_second(x, i, j)` is called where `second[j] == x`, and `i` is
 * 		the smallest such index where `first[i] > x`.
 *
 * The visited elements are also ordered.
 *
 * \tparam T a generic type for which the operators `==`, `!=` and `<` are implemented.
 * \tparam RemoveDuplicates if `true`, this function will avoid calling the
 * 		union functions more than once for each element.
 */
template<typename T, typename UnionBoth,
	typename UnionFirst, typename UnionSecond,
	bool RemoveDuplicates = true>
void set_union(UnionBoth union_both,
	UnionFirst union_first, UnionSecond union_second,
	const T* first, unsigned int first_length,
	const T* second, unsigned int second_length)
{
	unsigned int i = 0, j = 0;
	const T* prev = NULL;
	while (i < first_length && j < second_length)
	{
		if (first[i] == second[j]) {
			set_union_helper<RemoveDuplicates>(union_both, first[i], i, j, prev);
			i++; j++;
		} else if (first[i] < second[j]) {
			set_union_helper<RemoveDuplicates>(union_first, first[i], i, j, prev);
			i++;
		} else {
			set_union_helper<RemoveDuplicates>(union_second, second[j], i, j, prev);
			j++;
		}
	}

	while (i < first_length) {
		set_union_helper<RemoveDuplicates>(union_first, first[i], i, j, prev);
		i++;
	} while (j < second_length) {
		set_union_helper<RemoveDuplicates>(union_second, second[j], i, j, prev);
		j++;
	}
}

template<bool RemoveDuplicates, typename T, typename SizeType,
	typename std::enable_if<std::is_integral<SizeType>::value>::type* = nullptr>
inline void set_union_helper(T* dst, SizeType& dst_length, const T& item) {
	if (!RemoveDuplicates || dst_length == 0 || dst[dst_length - 1] != item) {
		dst[dst_length] = item;
		dst_length++;
	}
}

/**
 * Given ordered native arrays `first` and `second`, compute their union and
 * appends the result to `dst`. The union is also ordered. This function
 * assumes `dst` has sufficient capacity to store the union.
 * \tparam T satisfies [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable)
 * 		and implements the operators `==` and `<`.
 * \tparam RemoveDuplicates if `true`, this function ignores duplicate elements.
 */
template<typename T, typename SizeType, bool RemoveDuplicates = true,
	typename std::enable_if<std::is_integral<SizeType>::value>::type* = nullptr>
void set_union(T* dst, SizeType& dst_length,
	const T* first, unsigned int first_length,
	const T* second, unsigned int second_length)
{
	unsigned int i = 0, j = 0;
	while (i < first_length && j < second_length)
	{
		if (first[i] == second[j]) {
			set_union_helper<RemoveDuplicates>(dst, dst_length, first[i]);
			i++; j++;
		} else if (first[i] < second[j]) {
			set_union_helper<RemoveDuplicates>(dst, dst_length, first[i]);
			i++;
		} else {
			set_union_helper<RemoveDuplicates>(dst, dst_length, second[j]);
			j++;
		}
	}

	while (i < first_length) {
		set_union_helper<RemoveDuplicates>(dst, dst_length, first[i]);
		i++;
	} while (j < second_length) {
		set_union_helper<RemoveDuplicates>(dst, dst_length, second[j]);
		j++;
	}
}

/**
 * Given ordered native arrays `first` and `second`, compute their union and
 * appends the result to `dst`. The union is also ordered.
 * \tparam T satisfies [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable)
 * 		and implements the operators `==` and `<`.
 * \tparam RemoveDuplicates if `true`, this function ignore duplicate elements.
 */
template<typename T, bool RemoveDuplicates = true>
inline bool set_union(array<T>& dst,
	const T* first, unsigned int first_length,
	const T* second, unsigned int second_length)
{
	if (!dst.ensure_capacity(dst.length + first_length + second_length))
		return false;
	set_union<T, size_t, RemoveDuplicates>(dst.data, dst.length, first, first_length, second, second_length);
	return true;
}

/**
 * Given ordered arrays `first` and `second`, compute their union and appends
 * the result to `dst`. The union is also ordered.
 * \tparam T satisfies [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable)
 * 		and implements the operators `==` and `<`.
 * \tparam RemoveDuplicates if `true`, this function ignore duplicate elements.
 */
template<typename T, bool RemoveDuplicates = true>
inline bool set_union(array<T>& dst, const array<T>& first, const array<T>& second) {
	return set_union<T, RemoveDuplicates>(dst,
		first.data, (unsigned int) first.length,
		second.data, (unsigned int) second.length);
}

template<typename T>
struct array_position {
	unsigned int array_id;
	unsigned int position;
	T* element;
};

/**
 * Given a collection of ordered arrays `arrays`, compute their union and
 * appends the result to `dst`. The union is also ordered.
 * NOTE: this function assumes the given arrays are all non-empty.
 * \tparam T satisfies [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable)
 * 		and implements the operators `==` and `<`.
 * \tparam ArraySetCollection a collection type where each element is accessed
 * 		using `arrays[i]` and the size of each array can be obtained using
 * 		`size(arrays[i])`. The element at index `j` for array `i` is accessed
 * 		using `arrays[i][j]`.
 */
template<typename T, typename ArraySetCollection>
bool set_union(array<T>& dst, const ArraySetCollection& arrays, unsigned int array_count)
{
	/* first ensure the destination array has enough space */
	unsigned int total_size = dst.length;
	for (unsigned int i = 0; i < array_count; i++) {
#if !defined(NDEBUG)
		if (size(arrays[i]) == 0)
			fprintf(stderr, "set_union WARNING: Input array %u is empty.\n", i);
#endif
		total_size += size(arrays[i]);
	}
	if (!dst.ensure_capacity(total_size))
		return false;

	/* TODO: we can probably use a faster heap structure */
	array_position<T>* heap = (array_position<T>*) malloc(array_count * sizeof(array_position<T>));
	if (heap == NULL) {
		fprintf(stderr, "set_union ERROR: Out of memory.\n");
		return false;
	}
	for (unsigned int i = 0; i < array_count; i++)
		heap[i] = { i, 0, &arrays[i][0] };
	std::make_heap(heap, heap + array_count);

	/* add the first item to the destination set */
	unsigned int heap_size = array_count;
	std::pop_heap(heap, heap + heap_size);
	const array_position<T>& first = heap[heap_size];
	dst.data[dst.length] = first.key;
	dst.length++;
	if (size(arrays[first.value]) > 1) {
		heap[heap_size] = { first.array_id, 1, &arrays[first.array_id][1] };
		std::push_heap(heap, heap + heap_size);
	} else { heap_size--; }

	while (heap_size > 0)
	{
		std::pop_heap(heap, heap + heap_size);
		const array_position<T>& next = heap[heap_size];
		if (next.key != dst.last()) {
			dst.data[dst.length] = next.key;
			dst.length++;
		}
		if (next.value + 1 < size(arrays[next.value])) {
			heap[heap_size] = { next.array_id, next.position + 1, arrays[next.array_id][next.position + 1] };
			std::push_heap(heap, heap + heap_size);
		} else { heap_size--; }
	}
	free(heap);
	return true;
}

/**
 * Given ordered native arrays `first` and `second`, this function visits their
 * elements sequentially, and for every element `x` that exists in both arrays,
 * `intersect(i, j)` is called where `first[i] == x` and `second[j] == x`. The
 * visited elements are also ordered.
 * \tparam T a generic type for which the operators `==` and `<` are implemented.
 * \tparam BinarySearch if `true`, binary search is used to find indices of
 * 		identical elements rather than linear search.
 */
template<typename T, typename Intersect, bool BinarySearch = false>
bool set_intersect(Intersect intersect,
	const T* first, unsigned int first_length,
	const T* second, unsigned int second_length)
{
	unsigned int i = 0, j = 0;
	while (i < first_length && j < second_length)
	{
		if (first[i] == second[j]) {
			intersect(i, j);
			i++; j++;
		} else if (first[i] < second[j]) {
			if (BinarySearch) {
				/* use binary search to find the value of i
				   such that first.data[i] >= second.data[j] */
				i = binary_search(first, second[j], i, first_length - 1);
			} else {
				i++;
			}
		} else {
			if (BinarySearch) {
				/* use binary search to find the value of j
				   such that second.data[j] >= first.data[i] */
				j = binary_search(second, first[i], j, second_length - 1);
			} else {
				j++;
			}
		}
	}
	return true;
}

/**
 * Given ordered native arrays `first` and `second`, compute the intersection
 * and append it to the native array `intersection`. The computed intersection
 * is also ordered. This function assumes `intersection` has sufficient capacity.
 * \tparam T satisfies [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable)
 * 		and implements the operators `==` and `<`.
 * \tparam BinarySearch if `true`, binary search is used to find indices of
 * 		identical elements rather than linear search.
 */
template<typename T, typename SizeType, bool BinarySearch = false,
	typename std::enable_if<std::is_integral<SizeType>::value>::type* = nullptr>
bool set_intersect(
	T* intersection, SizeType& intersection_length,
	const T* first, unsigned int first_length,
	const T* second, unsigned int second_length)
{
	unsigned int i = 0, j = 0;
	while (i < first_length && j < second_length)
	{
		if (first[i] == second[j]) {
			intersection[intersection_length] = first[i];
			intersection_length++;
			i++; j++;
		} else if (first[i] < second[j]) {
			if (BinarySearch) {
				/* use binary search to find the value of i
				   such that first.data[i] >= second.data[j] */
				i = binary_search(first, second[j], i, first_length - 1);
			} else {
				i++;
			}
		} else {
			if (BinarySearch) {
				/* use binary search to find the value of j
				   such that second.data[j] >= first.data[i] */
				j = binary_search(second, first[i], j, second_length - 1);
			} else {
				j++;
			}
		}
	}
	return true;
}

/**
 * Given ordered native arrays `first` and `second`, compute the intersection
 * and append it to the array `intersection`. The computed intersection is also
 * ordered.
 * \tparam T satisfies [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable)
 * 		and implements the operators `==` and `<`.
 * \tparam BinarySearch if `true`, binary search is used to find indices of
 * 		identical elements rather than linear search.
 */
template<typename T, bool BinarySearch = false>
inline bool set_intersect(
	array<T>& intersection,
	const T* first, unsigned int first_length,
	const T* second, unsigned int second_length)
{
	if (!intersection.ensure_capacity(intersection.length + max(first_length, second_length)))
		return false;

	return set_intersect<T, decltype(intersection.length), BinarySearch>(
		intersection.data, intersection.length,
		first, first_length, second, second_length);
}

/**
 * Given ordered arrays `first` and `second`, compute the intersection and
 * append it to the array `intersection`. The computed intersection is also
 * ordered.
 * \tparam T satisfies [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable)
 * 		and implements the operators `==` and `<`.
 * \tparam BinarySearch if `true`, binary search is used to find indices of
 * 		identical elements rather than linear search.
 */
template<typename T, bool BinarySearch = false>
inline bool set_intersect(
	array<T>& intersection,
	const array<T>& first,
	const array<T>& second)
{
	return set_intersect<T, BinarySearch>(intersection,
		first.data, (unsigned int) first.length,
		second.data, (unsigned int) second.length);
}

/**
 * Given ordered native arrays `first` and `second`, compute the intersection
 * in-place and store it in `first`. The computed intersection is also ordered.
 * \tparam T satisfies [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable)
 * 		and implements the operators `==` and `<`.
 * \tparam BinarySearch if `true`, binary search is used to find indices of
 * 		identical elements rather than linear search.
 */
template<typename T, typename SizeType, bool BinarySearch = false,
	typename std::enable_if<std::is_integral<SizeType>::value>::type* = nullptr>
void set_intersect(
	T* first, SizeType& first_length,
	const T* second, unsigned int second_length)
{
	unsigned int index = 0;
	unsigned int i = 0, j = 0;
	while (i < first_length && j < second_length)
	{
		if (first[i] == second[j]) {
			first[index] = first[i];
			index++; i++; j++;
		} else if (first[i] < second[j]) {
			if (BinarySearch) {
				/* use binary search to find the value of i
				   such that first.data[i] >= second.data[j] */
				i = binary_search(first, second[j], i, first_length - 1);
			} else {
				i++;
			}
		} else {
			if (BinarySearch) {
				/* use binary search to find the value of j
				   such that second.data[j] >= first.data[i] */
				j = binary_search(second, first[i], j, second_length - 1);
			} else {
				j++;
			}
		}
	}
	first_length = index;
}

/**
 * Given ordered array `first` and ordered native array `second`, compute the
 * intersection in-place and store it in `first`. The computed intersection is
 * also ordered.
 * \tparam T satisfies [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable)
 * 		and implements the operators `==` and `<`.
 * \tparam BinarySearch if `true`, binary search is used to find indices of
 * 		identical elements rather than linear search.
 */
template<typename T, bool BinarySearch = false>
inline void set_intersect(array<T>& first,
	const T* second, unsigned int second_length)
{
	return set_intersect<T, decltype(first.length), BinarySearch>(
			first.data, first.length, second, second_length);
}

/**
 * Given ordered arrays `first` and `second`, compute the intersection in-place
 * and store it in `first`. The computed intersection is also ordered.
 * \tparam T satisfies [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable)
 * 		and implements the operators `==` and `<`.
 * \tparam BinarySearch if `true`, binary search is used to find indices of
 * 		identical elements rather than linear search.
 */
template<typename T, bool BinarySearch = false>
inline void set_intersect(array<T>& first, const array<T>& second) {
	return set_intersect<T, BinarySearch>(first, second.data, second.length);
}

/**
 * Returns true if the intersection of `first` and `second` is non-empty,
 * where `first` and `second` are ordered native arrays.
 * \tparam T a generic type that implements the operators `==` and `<`.
 * \tparam BinarySearch if `true`, binary search is used to find indices of
 * 		identical elements rather than linear search.
 */
template<typename T, bool BinarySearch = false>
bool has_intersection(
	const T* first, unsigned int first_length,
	const T* second, unsigned int second_length)
{
	unsigned int i = 0, j = 0;
	while (i < first_length && j < second_length)
	{
		if (first[i] == second[j]) {
			return true;
			i++; j++;
		} else if (first[i] < second[j]) {
			if (BinarySearch) {
				/* use binary search to find the value of i
				   such that first.data[i] >= second.data[j] */
				i = binary_search(first, second[j], i, first_length - 1);
			} else {
				i++;
			}
		} else {
			if (BinarySearch) {
				/* use binary search to find the value of j
				   such that second.data[j] >= first.data[i] */
				j = binary_search(second, first[i], j, second_length - 1);
			} else {
				j++;
			}
		}
	}
	return false;
}

/**
 * Returns true if the intersection of `first` and `second` is non-empty,
 * where `first` and `second` are ordered arrays.
 * \tparam T a generic type that implements the operators `==` and `<`.
 * \tparam BinarySearch if `true`, binary search is used to find indices of
 * 		identical elements rather than linear search.
 */
template<typename T, bool BinarySearch = false>
inline bool has_intersection(const array<T>& first, const array<T>& second) {
	return has_intersection<T, BinarySearch>(
		first.data, (unsigned int) first.length,
		second.data, (unsigned int) second.length);
}

/**
 * Returns true if `first` is a subset of `second`, where `first` and `second`
 * are ordered native arrays.
 * \tparam T a generic type that implements the operators `==` and `<`.
 * \tparam BinarySearch if `true`, binary search is used to find indices of
 * 		identical elements rather than linear search.
 */
template<typename T, bool BinarySearch = false>
bool is_subset(
	const T* first, unsigned int first_length,
	const T* second, unsigned int second_length)
{
	unsigned int i = 0, j = 0;
	while (i < first_length && j < second_length)
	{
		if (first[i] == second[j]) {
			i++; j++;
		} else if (first[i] < second[j]) {
			return false;
		} else {
			if (BinarySearch) {
				/* use binary search to find the value of j
				   such that second.data[j] >= first.data[i] */
				j = binary_search(second, first[i], j, second_length - 1);
			} else {
				j++;
			}
		}
	}
	return (i == first_length);
}

/**
 * Given ordered native arrays `first` and `second`, this function visits their
 * elements sequentially, and for every element `x` that exists in `first` but
 * not in `second`, `emit(i)` is called where `first[i] == x`. The visited
 * elements are also ordered.
 * \tparam T a generic type that implements the operators `==` and `<`.
 * \tparam BinarySearch if `true`, binary search is used to find indices of
 * 		identical elements rather than linear search.
 */
template<typename T, typename EmitFunction, bool BinarySearch = false>
void set_subtract(EmitFunction emit,
	const T* first, unsigned int first_length,
	const T* second, unsigned int second_length)
{
	unsigned int i = 0, j = 0;
	while (i < first_length && j < second_length)
	{
		if (first[i] == second[j]) {
			i++; j++;
		} else if (first[i] < second[j]) {
			if (BinarySearch) {
				/* use binary search to find the value of i
				   such that first.data[i] >= second.data[j] */
				unsigned int next_i = binary_search(first, second[j], i, first_length - 1);
				for (; i < next_i; i++)
					emit(i);
				i = next_i;
			} else {
				emit(i);
				i++;
			}
		} else {
			if (BinarySearch) {
				/* use binary search to find the value of j
				   such that second.data[j] >= first.data[i] */
				j = binary_search(second, first[i], j, second_length - 1);
			} else {
				j++;
			}
		}
	}

	while (i < first_length) {
		emit(i);
		i++;
	}
}

/**
 * Given ordered native arrays `first` and `second`, this function computes the
 * set difference between `first` and `second` and stores the result in `dst`.
 * This function assumes `dst` has sufficient capacity. The set difference is
 * also ordered.
 * \tparam T satisfies [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable)
 * 		and implements the operators `==` and `<`.
 * \tparam BinarySearch if `true`, binary search is used to find indices of
 * 		identical elements rather than linear search.
 */
template<typename T, typename SizeType, bool BinarySearch = false,
	typename std::enable_if<std::is_integral<SizeType>::value>::type* = nullptr>
void set_subtract(T* dst, SizeType& dst_length,
	const T* first, unsigned int first_length,
	const T* second, unsigned int second_length)
{
	unsigned int i = 0, j = 0;
	while (i < first_length && j < second_length)
	{
		if (first[i] == second[j]) {
			i++; j++;
		} else if (first[i] < second[j]) {
			if (BinarySearch) {
				/* use binary search to find the value of i
				   such that first.data[i] >= second.data[j] */
				unsigned int next_i = binary_search(first, second[j], i, first_length - 1);
				for (; i < next_i; i++) {
					dst[dst_length] = first[i];
					dst_length++;
				}
				i = next_i;
			} else {
				dst[dst_length] = first[i];
				dst_length++;
				i++;
			}
		} else {
			if (BinarySearch) {
				/* use binary search to find the value of j
				   such that second.data[j] >= first.data[i] */
				j = binary_search(second, first[i], j, second_length - 1);
			} else {
				j++;
			}
		}
	}

	memcpy(dst + dst_length, first + i, (first_length - i) * sizeof(T));
	dst_length += first_length - i;
}

/**
 * Given ordered native arrays `first` and `second`, this function computes the
 * set difference between `first` and `second` and stores the result in `dst`.
 * The set difference is also ordered.
 * \tparam T satisfies [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable)
 * 		and implements the operators `==` and `<`.
 * \tparam BinarySearch if `true`, binary search is used to find indices of
 * 		identical elements rather than linear search.
 */
template<typename T, bool BinarySearch = false>
bool set_subtract(array<T>& dst,
	const T* first, unsigned int first_length,
	const T* second, unsigned int second_length)
{
	if (!dst.ensure_capacity(dst.length + max(first_length, second_length)))
		return false;
	set_subtract<T, decltype(dst.length), BinarySearch>(dst.data, dst.length, first, first_length, second, second_length);
	return true;
}

/**
 * Given ordered arrays `first` and `second`, this function computes the set
 * difference between `first` and `second` and stores the result in `dst`. The
 * set difference is also ordered.
 * \tparam T satisfies [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable)
 * 		and implements the operators `==` and `<`.
 * \tparam BinarySearch if `true`, binary search is used to find indices of
 * 		identical elements rather than linear search.
 */
template<typename T, bool BinarySearch = false>
inline bool set_subtract(array<T>& dst,
	const array<T>& first,
	const array<T>& second)
{
	return set_subtract<T, BinarySearch>(dst,
		first.data, (unsigned int) first.length,
		second.data, (unsigned int) second.length);
}

/**
 * Given ordered native arrays `first` and `second`, this function computes the
 * set difference in-place between `first` and `second` and stores the result
 * in `first`. The set difference is also ordered.
 * \tparam T satisfies [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable)
 * 		and implements the operators `==` and `<`.
 * \tparam BinarySearch if `true`, binary search is used to find indices of
 * 		identical elements rather than linear search.
 */
template<typename T, typename SizeType, bool BinarySearch = false,
	typename std::enable_if<std::is_integral<SizeType>::value>::type* = nullptr>
void set_subtract(
	T* first, SizeType& first_length,
	const T* second, unsigned int second_length)
{
	unsigned int index = 0;
	unsigned int i = 0, j = 0;
	while (i < first_length && j < second_length)
	{
		if (first[i] == second[j]) {
			i++; j++;
		} else if (first[i] < second[j]) {
			if (BinarySearch) {
				/* use binary search to find the value of i
				   such that first.data[i] >= second.data[j] */
				unsigned int next_i = binary_search(first, second[j], i, first_length - 1);
				for (; i < next_i; i++) {
					first[index] = first[i];
					index++;
				}
				i = next_i;
			} else {
				first[index] = first[i];
				index++; i++;
			}
		} else {
			if (BinarySearch) {
				/* use binary search to find the value of j
				   such that second.data[j] >= first.data[i] */
				j = binary_search(second, first[i], j, second_length - 1);
			} else {
				j++;
			}
		}
	}

	while (i < first_length) {
		first[index] = first[i];
		index++; i++;
	}
	first_length = index;
}

/**
 * Given ordered arrays `first` and `second`, this function computes the set
 * difference in-place between `first` and `second` and stores the result in
 * `first`. The set difference is also ordered.
 * \tparam T satisfies [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable)
 * 		and implements the operators `==` and `<`.
 * \tparam BinarySearch if `true`, binary search is used to find indices of
 * 		identical elements rather than linear search.
 */
template<typename T, bool BinarySearch = false>
inline void set_subtract(array<T>& first, const array<T>& second)
{
	return set_subtract<T, BinarySearch>(first.data, first.length, second.data, second.length);
}

inline void array_test(void)
{
	array<char> buf(1);
	buf.append("0123456789 ", 11);

	buf[(unsigned int) buf.length] = '\0';
	if (buf.length != 11)
		fprintf(stderr, "array test ERROR: First array length test failed.\n");
	if (buf.capacity != 16)
		fprintf(stderr, "array test ERROR: First array capacity test failed.\n");
	if (strcmp(buf.data, "0123456789 ") != 0)
		fprintf(stderr, "array test ERROR: First string comparison test failed.\n");

	buf.append("abcdefghijklmnopqrstuvwxyz ", 27);

	buf[(unsigned int) buf.length] = '\0';
	if (buf.length != 11 + 27)
		fprintf(stderr, "array test ERROR: Second array length test failed.\n");
	if (buf.capacity != 64)
		fprintf(stderr, "array test ERROR: Second array capacity test failed.\n");
	if (strcmp(buf.data, "0123456789 abcdefghijklmnopqrstuvwxyz ") != 0)
		fprintf(stderr, "array test ERROR: Second string comparison test failed.\n");

	buf.append("9876543210 ", 11);

	buf[(unsigned int) buf.length] = '\0';
	if (buf.length != 11 + 27 + 11)
		fprintf(stderr, "array test ERROR: Third array length test failed.\n");
	if (buf.capacity != 64)
		fprintf(stderr, "array test ERROR: Third array capacity test failed.\n");
	if (strcmp(buf.data, "0123456789 abcdefghijklmnopqrstuvwxyz 9876543210 ") != 0)
		fprintf(stderr, "array test ERROR: Third string comparison test failed.\n");

	/* test some of the helper functions */
	array<int> numbers = array<int>(10);
	numbers.add(4);
	numbers.add(-6);
	numbers.add(4);
	numbers.add(2);
	numbers.add(0);
	numbers.add(-6);
	numbers.add(1);
	numbers.add(4);
	numbers.add(2);

	array<int> numbers_copy = array<int>(10);
	numbers_copy.append(numbers.data, numbers.length);
	insertion_sort(numbers);
	quick_sort(numbers_copy);

	int expected[] = {-6, -6, 0, 1, 2, 2, 4, 4, 4};
	for (unsigned int i = 0; i < 9; i++) {
		if (numbers[i] != expected[i]) {
			fprintf(stderr, "array test ERROR: insertion_sort failed.\n");
			return;
		}
		if (numbers_copy[i] != expected[i]) {
			fprintf(stderr, "array test ERROR: quick_sort failed.\n");
			return;
		}
	}

	int expected_unique[] = {-6, 0, 1, 2, 4};
	unique(numbers);
	if (numbers.length != 5) {
		fprintf(stderr, "array test ERROR: unique failed.\n");
		return;
	}
	for (unsigned int i = 0; i < 5; i++) {
		if (numbers[i] != expected_unique[i]) {
			fprintf(stderr, "array test ERROR: unique failed.\n");
			return;
		}
	}

	printf("array test completed.\n");
}

} /* namespace core */

#endif /* ARRAY_H_ */
