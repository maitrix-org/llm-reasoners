/**
 * \file map.h
 * 
 * This file contains the hash_set, hash_map, and array_map data structures. It
 * also defines the default_hash function, which provides a default hash
 * algorithm (currently implemented using [xxhash](https://github.com/Cyan4973/xxHash)).
 *
 *  <!-- Created on: May 28, 2014
 *          Author: asaparov -->
 */

#ifndef MAP_H_
#define MAP_H_

#include <time.h>
#include <initializer_list>

#define XXH_INLINE_ALL
#include "array.h"
#include "xxhash.h"


namespace core {

/**
 * For hash_set and hash_map, `capacity` is the size of the underlying array
 * (i.e. the number of buckets), whereas `size` is the number of elements (i.e.
 * number of non-empty buckets). The functions hash_set::check_size and
 * hash_map::check_size compute the load factor `size / capacity` and compare
 * it to RESIZE_THRESHOLD. If the load factor is too large, the hashtable is
 * resized and the capacity is increased.
 */
#define RESIZE_THRESHOLD 1 / 2

/**
 * The multiplicative inverse of RESIZE_THRESHOLD.
 */
#define RESIZE_THRESHOLD_INVERSE 2 / 1

/**
 * The multiplicative factor by which hash_set, hash_map, and array_map capacity is changed.
 */
#define RESIZE_FACTOR 2

/**
 * A function pointer type describing a function that returns a pointer to
 * allocated memory. The first argument is the number of elements to allocate,
 * and the second argument is the number of bytes for each element.
 * [calloc](http://en.cppreference.com/w/c/memory/calloc) is an example of a
 * function with this type.
 */
typedef void*(alloc_keys_func)(size_t, size_t);


/* forward declarations */
#if !defined(DOXYGEN_IGNORE)

template<typename T>
struct hash_set;

template<typename K, typename V>
struct hash_map;

template<typename K, typename V>
struct array_map;

#endif /* DOXYGEN_IGNORE */


#if defined(__LP64__) || defined(_WIN64) || defined(__x86_64__) || defined(__ppc64__)
template<typename K, unsigned int Seed>
inline uint_fast32_t default_hash(const K& key) {
	return (uint_fast32_t) XXH64(&key, sizeof(K), Seed);
}

template<typename K, unsigned int Seed>
inline uint_fast32_t default_hash(const K* keys, unsigned int length) {
	return (uint_fast32_t) XXH64(keys, sizeof(K) * length, Seed);
}

#else
/**
 * Evaluates the hash function of the given value `key` with the given `Seed` using the default implementation.
 */
template<typename K, unsigned int Seed>
inline unsigned int default_hash(const K& key) {
	return XXH32(&key, sizeof(K), Seed);
}

/**
 * Evaluates the hash function of the given native array of values `keys` with the given `Seed` using the default implementation.
 */
template<typename K, unsigned int Seed>
inline unsigned int default_hash(const K* keys, unsigned int length) {
	return XXH32(keys, sizeof(K) * length, Seed);
}
#endif

template<typename KeyMetric, typename ValueMetric>
struct key_value_metric {
	const KeyMetric& key_metric;
	const ValueMetric& value_metric;

	constexpr key_value_metric(const KeyMetric& key_metric, const ValueMetric& value_metric) :
		key_metric(key_metric), value_metric(value_metric) { }
};

template<typename KeyMetric, typename ValueMetric>
inline constexpr key_value_metric<KeyMetric, ValueMetric> make_key_value_metric(
		const KeyMetric& key_metric, const ValueMetric& value_metric) {
	return key_value_metric<KeyMetric, ValueMetric>(key_metric, value_metric);
}

inline constexpr key_value_metric<default_metric, default_metric> make_key_value_metric() {
	return make_key_value_metric(default_metric(), default_metric());
}


/**
 * <!-- STL-style iterator implementations (useful for range-based for loops). -->
 */

/**
 * An iterator implementation, similar to those in the Standard Template
 * Library, to enable iteration of elements in a hash_set. This iterator is
 * typically initialized using hash_set::begin.
 * 
 * This definition enables the use of range-based for loops.
 */
template<typename T, bool IsConst>
struct hash_set_iterator {
	typedef typename std::conditional<IsConst, const hash_set<T>&, hash_set<T>&>::type container_type;

	/**
	 * The type of the entries returned by this iterator. If this is a const
	 * iterator, `value_type` is `const T&`. Otherwise, `value_type` is `T&`.
	 */
	typedef typename std::conditional<IsConst, const T&, T&>::type value_type;

	container_type set;
	unsigned int position;

	hash_set_iterator(container_type& set, unsigned int position) : set(set), position(position) { }

	/**
	 * Returns whether this iterator is in the same position as `other`. This
	 * function assumes the two iterators were created from the same hash_set,
	 * and that it was not modified.
	 */
	inline bool operator != (const hash_set_iterator<T, IsConst>& other) const {
		return position != other.position;
	}

	/**
	 * Returns the element in the hash_set at the current iterator position.
	 * This function assumes the hash_set was not resized and no element was
	 * removed since the last call to either the operator `++` or the
	 * constructor of this iterator, whichever came later.
	 */
	inline value_type operator * () {
		return set.keys[position];
	}

	/**
	 * Advances the position of the iterator to the next element in the hash_set.
	 */
	inline const hash_set_iterator<T, IsConst>& operator ++ () {
		do {
			++position;
		} while (position < set.capacity && hasher<T>::is_empty(set.keys[position]));
		return *this;
	}
};

/**
 * An iterator implementation, similar to those in the Standard Template
 * Library, to enable iteration of elements in a hash_map. This iterator is
 * typically initialized using hash_map::begin.
 * 
 * This definition enables the use of range-based for loops.
 */
template<typename K, typename V, bool IsConst>
struct hash_map_iterator {
	typedef typename std::conditional<IsConst, const hash_map<K, V>&, hash_map<K, V>&>::type container_type;

	/**
	 * The type of the entries returned by this iterator. If this is a const
	 * iterator, `value_type` is `core::pair<const K&, constV&>`. Otherwise,
	 * `value_type` is `core::pair<K&, V&>`.
	 */
	typedef typename std::conditional<IsConst, pair<const K&, const V&>, pair<K&, V&>>::type value_type;

	container_type map;
	unsigned int position;

	hash_map_iterator(container_type& map, unsigned int position) : map(map), position(position) { }

	/**
	 * Returns whether this iterator is in the same position as `other`. This
	 * function assumes the two iterators were created from the same hash_map,
	 * and that it was not modified.
	 */
	inline bool operator != (const hash_map_iterator<K, V, IsConst>& other) const {
		return position != other.position;
	}

	/**
	 * Returns the entry in the hash_map at the current iterator position. This
	 * function assumes the hash_map was not resized and no element was removed
	 * since the last call to either the operator `++` or the constructor of
	 * this iterator, whichever came later.
	 */
	inline value_type operator * () {
		return { map.table.keys[position], map.values[position] };
	}

	/**
	 * Advances the position of the iterator to the next entry in the hash_map.
	 */
	inline const hash_map_iterator<K, V, IsConst>& operator ++ () {
		do {
			++position;
		} while (position < map.table.capacity && hasher<K>::is_empty(map.table.keys[position]));
		return *this;
	}
};

/**
 * An iterator implementation, similar to those in the Standard Template
 * Library, to enable iteration of elements in an array_map. This iterator is
 * typically initialized using array_map::begin.
 * 
 * This definition enables the use of range-based for loops.
 */
template<typename K, typename V, bool IsConst>
struct array_map_iterator {
	typedef typename std::conditional<IsConst, const array_map<K, V>&, array_map<K, V>&>::type container_type;

	/**
	 * The type of the entries returned by this iterator. If this is a const
	 * iterator, `value_type` is `core::pair<const K&, constV&>`. Otherwise,
	 * `value_type` is `core::pair<K&, V&>`.
	 */
	typedef typename std::conditional<IsConst, pair<const K&, const V&>, pair<K&, V&>>::type value_type;

	container_type map;
	size_t position;

	array_map_iterator(container_type map, size_t position) : map(map), position(position) { }

	/**
	 * Returns whether this iterator is in the same position as `other`. This
	 * function assumes the two iterators were created from the same array_map,
	 * and that it was not modified.
	 */
	inline bool operator != (const array_map_iterator<K, V, IsConst>& other) const {
		return position != other.position;
	}

	/**
	 * Returns the entry in the array_map at the current iterator position.
	 * This function assumes the array_map was not resized and no element was
	 * removed since the last call to either the operator `++` or the
	 * constructor of this iterator, whichever came later.
	 */
	inline value_type operator * () {
		return{ map.keys[position], map.values[position] };
	}

	/**
	 * Advances the position of the iterator to the next entry in the array_map.
	 */
	inline const array_map_iterator<K, V, IsConst>& operator ++ () {
		++position;
		return *this;
	}
};

/**
 * Returns `true` only if `probe > start` and `probe <= end` where `probe`, `start`,
 * and `end` are in the additive group of integers modulo the capacity of this set.
 */
inline bool index_between(unsigned int probe, unsigned int start, unsigned int end) {
	if (end >= start) {
		return (probe > start && probe <= end);
	} else {
		return (probe <= end || probe > start);
	}
}

/**
 * An unordered associative container that contains a set of unique elements,
 * each of type `T`. The elements are stored in the native array
 * hash_set::keys, which has capacity hash_set::capacity. To compute the index
 * of an element in hash_set::keys, we first compute its hash. To do so:
 * 1. If `T` [is_pointer](http://en.cppreference.com/w/cpp/types/is_pointer),
 * 		[is_fundamental](http://en.cppreference.com/w/cpp/types/is_fundamental),
 * 		or [is_enum](http://en.cppreference.com/w/cpp/types/is_enum),
 * 		core::default_hash() provides the hash.
 * 2. For all other types, the `T::hash` function provides the hash.
 * Once the hash is computed, the index into hash_set::keys is computed using
 * `hash % hash_set::capacity` (modular division by the capacity).
 *
 * The above approach could produce the same index for distinct elements. This
 * event is known as a *collision*. We use
 * [linear probing](https://en.wikipedia.org/wiki/Linear_probing)
 * to resolve collisions: When adding an element to the hash_set, we compute
 * its `index` using the above procedure, then we inspect
 * `hash_set::keys[index]` and use core::is_empty() to determine if another
 * element already occupies that position. If so, we try the next position
 * `(index + 1) % hash_set::capacity`. We continue until we find an empty
 * index, and the element is inserted there.
 *
 * Thus, the function core::is_empty() is used to determine if a position in
 * hash_set::keys is occupied or empty. The total number of occupied positions
 * is given by hash_set::size. **WARNING:** If hash_set::keys becomes full, the
 * above linear probing mechanism could lead to an infinite loop. The function
 * hash_set::check_size should be used whenever adding new elements to avoid
 * this scenario.
 *
 * **Performance:** This data structure provides constant-time access and
 * modification, given that the load factor (`size / capacity`) is not too
 * large.
 *
 *
 * Below is an example of a simple use-case of hash_set, where the expected
 * output is `a.contains(2): false, a.contains(3): true, -4 9 3`.
 *
 * ```{.cpp}
 * #include <core/map.h>
 * #include <stdio.h>
 * using namespace core;
 *
 * int main() {
 * 	hash_set<int> a = hash_set<int>(8);
 * 	a.add(-1); a.add(-4);
 * 	a.add(3); a.add(9);
 * 	a.remove(-1);
 *
 * 	printf("a.contains(2): %s, ", a.contains(2) ? "true" : "false");
 * 	printf("a.contains(3): %s, ", a.contains(3) ? "true" : "false");
 * 	for (int element : a)
 * 		printf("%d ", element);
 * }
 * ```
 *
 *
 * However, if `a` is not allocated on the stack, the destructor will not be
 * automatically called, and so it must be freed manually using `core::free` or
 * `hash_set::free`. In the example below, the expected output is the same as
 * that of the program above: `a.contains(2): false, a.contains(3): true, -4 9 3`.
 *
 * ```{.cpp}
 * #include <core/map.h>
 * #include <stdio.h>
 * using namespace core;
 *
 * int main() {
 * 	hash_set<int>& a = *((hash_set<int>*) alloca(sizeof(hash_set<int>)));
 * 	hash_set_init(a, 8);
 * 	a.add(-1); a.add(-4);
 * 	a.add(3); a.add(9);
 * 	a.remove(-1);
 *
 * 	printf("a.contains(2): %s, ", a.contains(2) ? "true" : "false");
 * 	printf("a.contains(3): %s, ", a.contains(3) ? "true" : "false");
 * 	for (int element : a)
 * 		printf("%d ", element);
 * 	free(a);
 * }
 * ```
 *
 *
 * A number of member functions require `T` to be
 * [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable).
 * If this is not the case, those operations can be performed directly on the
 * public fields, as in the following example. In addition, when using a custom
 * struct/class with hash_set, it must implement public static functions, like
 * `hash`, `is_empty`, and `move`, as well as the operator `==`. The expected
 * output of the following example is `first second `.
 * 
 * ```{.cpp}
 * #include <core/map.h>
 * #include <stdio.h>
 * #include <string.h>
 * using namespace core;
 *
 * struct custom_string {
 * 	char* buffer;
 *
 * 	static unsigned int hash(const custom_string& s) {
 * 		return default_hash(s.buffer, strlen(s.buffer));
 * 	}
 *
 * 	static bool is_empty(const custom_string& s) {
 * 		return s.buffer == NULL;
 * 	}
 *
 * 	static void move(const custom_string& src, custom_string& dst) {
 * 		dst.buffer = src.buffer;
 * 	}
 *
 * 	static void free(custom_string& s) {
 * 		core::free(s.buffer);
 * 	}
 * };
 *
 * inline bool operator == (const custom_string& first, const custom_string& second) {
 * 	if (first.buffer == NULL)
 * 		return second.buffer == NULL;
 * 	return strcmp(first.buffer, second.buffer) == 0;
 * }
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
 * 	custom_string first, second;
 * 	init(first, "first");
 * 	init(second, "second");
 *
 * 	hash_set<custom_string> a = hash_set<custom_string>(8);
 * 	a.check_size(a.size + 2);
 *
 * 	bool contains; unsigned int index;
 * 	index = a.index_of(first, contains);
 * 	if (!contains) {
 * 		core::move(first, a.keys[index]);
 * 		a.size++;
 * 	}
 *
 * 	index = a.index_of(second, contains);
 * 	if (!contains) {
 * 		core::move(second, a.keys[index]);
 * 		a.size++;
 * 	}
 *
 * 	for (const custom_string& s : a)
 * 		printf("%s ", s.buffer);
 * 	for (custom_string& s : a)
 * 		free(s);
 * }
 * ```
 *
 * \tparam T the generic type of the elements in the set. `T` must satisfy either:
 * 		1. [is_fundamental](http://en.cppreference.com/w/cpp/types/is_fundamental),
 * 		2. [is_enum](http://en.cppreference.com/w/cpp/types/is_enum),
 * 		3. [is_pointer](http://en.cppreference.com/w/cpp/types/is_pointer),
 * 		4. implements the public static method `unsigned int hash(const T&)`,
 * 			the public static method `void is_empty(const T&)`, implements the
 * 			operators `==`, and satisfies is_moveable. Some operations also
 * 			require the operator `!=`, and public static methods
 * 			`void set_empty(T&)` and `void set_empty(T*, unsigned int)`.
 * 			**NOTE:** The first argument to the `==` and `!=` operators may be
 * 			empty.
 */
/* TODO: consider other collision resolution mechanisms */
template<typename T>
struct hash_set
{
	/**
	 * The native array of keys underlying the hashtable.
	 */
	T* keys;

	/**
	 * The capacity of hash_set::keys.
	 */
	unsigned int capacity;

	/**
	 * The number of elements in the hashtable (i.e. the number of non-empty buckets).
	 */
	unsigned int size;

	/**
	 * Constructs the hash_set with the given `initial_capacity`.
	 * \param alloc_keys a memory allocation function with prototype
	 * 		`void* alloc_keys(size_t count, size_t size)` that allocates space for
	 * 		`count` items, each with size `size`, and initializes them such that
	 * 		core::is_empty() returns `true` for each element.
	 */
	hash_set(unsigned int initial_capacity, alloc_keys_func alloc_keys = calloc) {
		if (!initialize(initial_capacity, alloc_keys)) {
			fprintf(stderr, "hash_set ERROR: Unable to allocate memory.\n");
			exit(EXIT_FAILURE);
		}
	}

	/**
	 * Constructs the hash_set and inserts the given native array of elements
	 * `set` with given `length`.
	 * \param alloc_keys a memory allocation function with prototype
	 * 		`void* alloc_keys(size_t count, size_t size)` that allocates space for
	 * 		`count` items, each with size `size`, and initializes them such that
	 * 		core::is_empty() returns `true` for each element.
	 */
	hash_set(const T* set, unsigned int length,
			alloc_keys_func alloc_keys = calloc) :
				hash_set(length * RESIZE_THRESHOLD_INVERSE + 1, alloc_keys)
	{
		for (unsigned int i = 0; i < length; i++)
			insert(set[i]);
	}

	/**
	 * Constructs the hash_set and inserts the given
	 * [initializer_list](http://en.cppreference.com/w/cpp/language/list_initialization)
	 * of elements `list`.
	 * \param alloc_keys a memory allocation function with prototype
	 * 		`void* alloc_keys(size_t count, size_t size)` that allocates space for
	 * 		`count` items, each with size `size`, and initializes them such that
	 * 		core::is_empty() returns `true` for each element.
	 */
	hash_set(const std::initializer_list<T>& list,
			alloc_keys_func alloc_keys = calloc) :
				hash_set(list.size() * RESIZE_THRESHOLD_INVERSE + 1, alloc_keys)
	{
		typename std::initializer_list<T>::iterator i;
		for (i = list.begin(); i != list.end(); i++)
			insert(*i);
	}

	~hash_set() { free(); }

	/**
	 * Forces the underlying hash_set::keys to be resized to the requested
	 * `capacity`.
	 *
	 * **WARNING:** If `new_capacity <= hash_set::size`, the hashtable could
	 * become full during the resize process, leading to an infinite loop due
	 * to the linear probing collision resolution mechanism.
	 * 
	 * \param alloc_keys a memory allocation function with prototype
	 * 		`void* alloc_keys(size_t count, size_t size)` that allocates space for
	 * 		`count` items, each with size `size`, and initializes them such that
	 * 		core::is_empty() returns `true` for each element.
	 */
	bool resize(unsigned int new_capacity,
			alloc_keys_func alloc_keys = calloc)
	{
		T* old_keys = keys;

		keys = (T*) alloc_keys(new_capacity, sizeof(T));
		if (keys == NULL) {
			/* revert changes and return error */
			keys = old_keys;
			return false;
		}

		/* iterate through keys and re-hash the elements */
		unsigned int old_capacity = capacity;
		capacity = new_capacity;
		for (unsigned int i = 0; i < old_capacity; i++) {
			if (!hasher<T>::is_empty(old_keys[i]))
				core::move(old_keys[i], keys[next_empty(old_keys[i])]);
		}
		core::free(old_keys);
		return true;
	}

	/**
	 * This function first determines whether `hash_set::size < hash_set::capacity * RESIZE_THRESHOLD`.
	 * If not, the capacity of the underlying hash_set::keys is increased by
	 * RESIZE_FACTOR until the condition is satisfied. This is useful to ensure
	 * the hashtable is sufficiently large when preparing to add new elements.
	 * \param alloc_keys a memory allocation function with prototype
	 * 		`void* alloc_keys(size_t count, size_t size)` that allocates space for
	 * 		`count` items, each with size `size`, and initializes them such that
	 * 		core::is_empty() returns `true` for each element.
	 * \returns `true` if the resize was successful, and `false` if there is insufficient memory.
	 */
	inline bool check_size(alloc_keys_func alloc_keys = calloc) {
		return check_size(size, alloc_keys);
	}

	/**
	 * For a requested number of elements `new_size`, this function first
	 * determines whether `new_size < hash_set::capacity * RESIZE_THRESHOLD`.
	 * If not, the capacity of the underlying hashtable is increased by
	 * RESIZE_FACTOR until the condition is satisfied. This is useful to ensure
	 * the hashtable is sufficiently large when preparing to add new elements.
	 * \param alloc_keys a memory allocation function with prototype
	 * 		`void* alloc_keys(size_t count, size_t size)` that allocates space for
	 * 		`count` items, each with size `size`, and initializes them such that
	 * 		core::is_empty() returns `true` for each element.
	 * \returns `true` if the resize was successful, and `false` if there is insufficient memory.
	 */
	inline bool check_size(unsigned int new_size, alloc_keys_func alloc_keys = calloc)
	{
		while (new_size >= capacity * RESIZE_THRESHOLD) {
			if (!resize(RESIZE_FACTOR * capacity, alloc_keys)) {
				fprintf(stderr, "hash_set.put ERROR: Unable to resize hashtable.\n");
				return false;
			}
		}
		return true;
	}

	/**
	 * Add the given `element` to this set. The assignment operator is used to
	 * insert each element, and so this function should not be used if `T` is not
	 * [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable).
	 * In such a case, insertion should be performed manually using
	 * hash_set::index_of to find the appropriate index and directly modifying
	 * hash_set::keys and hash_set::size. See the example in the hash_set
	 * description.
	 * \param alloc_keys a memory allocation function with prototype
	 * 		`void* alloc_keys(size_t count, size_t size)` that allocates space for
	 * 		`count` items, each with size `size`, and initializes them such that
	 * 		core::is_empty() returns `true` for each element.
	 * \tparam T is [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable).
	 */
	bool add(const T& element, alloc_keys_func alloc_keys = calloc)
	{
		if (!check_size(size, alloc_keys)) return false;
		insert(element);
		return true;
	}

	/**
	 * Adds all the elements in the hash_set `elements` to this set. The
	 * assignment operator is used to insert each element, and so this function
	 * should not be used if `T` is not
	 * [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable).
	 * In such a case, insertion should be performed manually using
	 * hash_set::index_of or hash_set::index_to_insert to find the appropriate
	 * index and directly modifying hash_set::keys and hash_set::size. See the
	 * example in the hash_set description.
	 * \param alloc_keys a memory allocation function with prototype
	 * 		`void* alloc_keys(size_t count, size_t size)` that allocates space for
	 * 		`count` items, each with size `size`, and initializes them such that
	 * 		core::is_empty() returns `true` for each element.
	 * \tparam T is [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable).
	 */
	bool add_all(const hash_set<T>& elements,
			alloc_keys_func alloc_keys = calloc)
	{
		if (!check_size(size + elements.size, alloc_keys)) return false;
		for (unsigned int i = 0; i < elements.capacity; i++)
			if (!hasher<T>::is_empty(elements.keys[i]))
				insert(elements.keys[i]);
		return true;
	}

	/**
	 * Adds all the elements in the native array `elements` with length `count`
	 * to this set. The assignment operator is used to insert each element, and
	 * so this function should not be used if `T` is not
	 * [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable).
	 * In such a case, insertion should be performed manually using
	 * hash_set::index_of or hash_set::index_to_insert and direct modification
	 * of the public fields.
	 * \param alloc_keys a memory allocation function with prototype
	 * 		`void* alloc_keys(size_t count, size_t size)` that allocates space for
	 * 		`count` items, each with size `size`, and initializes them such that
	 * 		core::is_empty() returns `true` for each element.
	 * \tparam T is [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable).
	 */
	bool add_all(const T* elements, unsigned int count,
			alloc_keys_func alloc_keys = calloc)
	{
		if (!check_size(size + count, alloc_keys)) return false;
		for (unsigned int i = 0; i < count; i++)
			insert(elements[i]);
		return true;
	}

	/**
	 * This function removes the given `element` from the set. This function
	 * does not free the removed element.
	 * \returns `true` if the element is removed, and `false` if the set does not contain `element`.
	 */
	bool remove(const T& element)
	{
#if !defined(NDEBUG)
		if (is_empty(element))
			fprintf(stderr, "hash_set.remove WARNING: Specified key is empty.\n");
#endif

		unsigned int index = hasher<T>::hash(element) % capacity;
		while (true) {
			if (keys[index] == element) {
				break;
			} else if (hasher<T>::is_empty(keys[index]))
				return false;
			index = (index + 1) % capacity;
		}

		remove_at(index);
		return true;
	}

	template<typename V>
	bool remove(const T& element, V* values)
	{
#if !defined(NDEBUG)
		if (is_empty(element))
			fprintf(stderr, "hash_set.remove WARNING: Specified key is empty.\n");
#endif

		unsigned int index = hasher<T>::hash(element) % capacity;
		while (true) {
			if (keys[index] == element) {
				break;
			} else if (hasher<T>::is_empty(keys[index]))
				return false;
			index = (index + 1) % capacity;
		}

		remove_at(values, index);
		return true;
	}

	/**
	 * This function removes the element at the bucket given by `index`. This
	 * function assumes that an element is located at the given bucket with the
	 * correct provided hash value. This function does not free the removed
	 * element.
	 */
	void remove_at(unsigned int index)
	{
		unsigned int last = index;
		unsigned int search = (index + 1) % capacity;
		while (!hasher<T>::is_empty(keys[search]))
		{
			unsigned int search_hash = hasher<T>::hash(keys[search]) % capacity;
			if (!index_between(search_hash, last, search)) {
				core::move(keys[search], keys[last]);
				last = search;
			}
			search = (search + 1) % capacity;
		}

		hasher<T>::set_empty(keys[last]);
		size--;
	}

	/**
	 * Returns `true` if `element` exists in the set, and `false` otherwise.
	 */
	bool contains(const T& element) const
	{
#if !defined(NDEBUG)
		if (is_empty(element))
			fprintf(stderr, "hashtable.contains WARNING: Specified key is empty.\n");
		if (size == capacity)
			fprintf(stderr, "hashtable.contains WARNING: Hashtable is full!\n");
#endif

		unsigned int index = hasher<T>::hash(element) % capacity;
		while (true) {
			if (keys[index] == element) {
				return true;
			} else if (hasher<T>::is_empty(keys[index])) {
				return false;
			}
			index = (index + 1) % capacity;
		}
	}

	/**
	 * If the given `element` exists in this set, this function returns the
	 * index of the bucket that contains it. If not, this function returns the
	 * index where the key would be located, if it had existed in the set.
	 */
	unsigned int index_of(const T& element) const
	{
#if !defined(NDEBUG)
		if (is_empty(element))
			fprintf(stderr, "hashtable.index_of WARNING: Specified key is empty.\n");
		if (size == capacity)
			fprintf(stderr, "hashtable.index_of WARNING: Hashtable is full!\n");
#endif

		unsigned int index = hasher<T>::hash(element) % capacity;
		while (keys[index] != element && !hasher<T>::is_empty(keys[index]))
			index = (index + 1) % capacity;
		return index;
	}

	/**
	 * If the given `element` exists in this set, this function returns the
	 * index of the bucket that contains it, and sets `contains` to `true`.
	 * If `element` is not in the set, `contains` is set to false, and this
	 * function returns the index where the key would be located, if it had
	 * existed in the set.
	 */
	inline unsigned int index_of(
			const T& element, bool& contains) const
	{
		unsigned int hash_value;
		return index_of(element, contains, hash_value);
	}

	/**
	 * If the given `element` exists in this set, this function returns the
	 * index of the bucket that contains it, and sets `contains` to `true`.
	 * If `element` is not in the set, `contains` is set to false, and this
	 * function returns the index where the key would be located, if it had
	 * existed in the set. In any case, the evaluated hash function of
	 * `element` is stored in `hash_value`.
	 */
	unsigned int index_of(const T& element,
			bool& contains, unsigned int& hash_value) const
	{
#if !defined(NDEBUG)
		if (is_empty(element))
			fprintf(stderr, "hashtable.index_of WARNING: Specified key is empty.\n");
		if (size == capacity)
			fprintf(stderr, "hashtable.index_of WARNING: Hashtable is full!\n");
#endif

		hash_value = hasher<T>::hash(element);
		unsigned int index = hash_value % capacity;
		while (true) {
			if (keys[index] == element) {
				contains = true;
				return index;
			} else if (hasher<T>::is_empty(keys[index])) {
				contains = false;
				return index;
			}
			index = (index + 1) % capacity;
		}
	}

	/**
	 * For a given `element`, this function computes and returns the index of
	 * the bucket where the element would be inserted, for example by a call to
	 * hash_set::add. `contains` is set to `true` if and only if the given
	 * `element` already exists in the set.
	 */
	inline unsigned int index_to_insert(const T& element, bool& contains)
	{
#if !defined(NDEBUG)
		if (size == capacity)
			fprintf(stderr, "hashtable.index_to_insert WARNING: Hashtable is full!\n");
#endif
		unsigned int index = hasher<T>::hash(element) % capacity;
		while (true) {
			if (hasher<T>::is_empty(keys[index])) {
				contains = false; break;
			} if (keys[index] == element) {
				contains = true; break;
			}
			index = (index + 1) % capacity;
		}
		return index;
	}

	/**
	 * For a given `element`, this function computes and returns the index of
	 * the bucket where the element would be inserted, for example by a call to
	 * hash_set::add, **assuming** the given element is not in the set.
	 */
	inline unsigned int index_to_insert(const T& element)
	{
#if !defined(NDEBUG)
		if (size == capacity)
			fprintf(stderr, "hashtable.index_to_insert WARNING: Hashtable is full!\n");
#endif
		unsigned int index = hasher<T>::hash(element) % capacity;
		while (true) {
			if (hasher<T>::is_empty(keys[index])) break;
			index = (index + 1) % capacity;
		}
		return index;
	}

	/**
	 * Removes all elements from this hash_set. Note that this function does
	 * not free each element beforehand.
	 */
	void clear() {
		hasher<T>::set_empty(keys, capacity);
		size = 0;
	}

	/**
	 * Returns `true` if this hash_set is a subset of `other`.
	 */
	bool is_subset(const hash_set<T>& other) const
	{
		for (unsigned int i = 0; i < capacity; i++)
			if (!hasher<T>::is_empty(keys[i]) && !other.contains(keys[i]))
				return false;
		return true;
	}

	bool equals(const hash_set<T>& other) const
	{
		if (size != other.size) return false;
		return is_subset(other);
	}

	/**
	 * Returns a hash_set_iterator pointing to the first element in this container.
	 * 
	 * **NOTE:** Unlike the libstdc++ [unordered_set](http://en.cppreference.com/w/cpp/container/unordered_set)
	 * and [unordered_map](http://en.cppreference.com/w/cpp/container/unordered_map)
	 * iterators, we do not keep a linked list among the elements, so if rapid
	 * iteration over elements is more critical than rapid queries, consider
	 * using an array_map.
	 */
	inline hash_set_iterator<T, false> begin() {
		return hash_set_iterator<T, false>(*this, first_empty());
	}

	/**
	 * Returns a hash_set_iterator pointing to the end of this container.
	 * 
	 * **NOTE:** Unlike the libstdc++ [unordered_set](http://en.cppreference.com/w/cpp/container/unordered_set)
	 * and [unordered_map](http://en.cppreference.com/w/cpp/container/unordered_map)
	 * iterators, we do not keep a linked list among the elements, so if rapid
	 * iteration over elements is more critical than rapid queries, consider
	 * using an array_map.
	 */
	inline hash_set_iterator<T, false> end() {
		return hash_set_iterator<T, false>(*this, capacity);
	}

	/**
	 * Returns a const hash_set_iterator pointing to the first element in this container.
	 * 
	 * **NOTE:** Unlike the libstdc++ [unordered_set](http://en.cppreference.com/w/cpp/container/unordered_set)
	 * and [unordered_map](http://en.cppreference.com/w/cpp/container/unordered_map)
	 * iterators, we do not keep a linked list among the elements, so if rapid
	 * iteration over elements is more critical than rapid queries, consider
	 * using an array_map.
	 */
	inline hash_set_iterator<T, true> begin() const {
		return hash_set_iterator<T, true>(*this, first_empty());
	}

	/**
	 * Returns a const hash_set_iterator pointing to the end of this container.
	 * 
	 * **NOTE:** Unlike the libstdc++ [unordered_set](http://en.cppreference.com/w/cpp/container/unordered_set)
	 * and [unordered_map](http://en.cppreference.com/w/cpp/container/unordered_map)
	 * iterators, we do not keep a linked list among the elements, so if rapid
	 * iteration over elements is more critical than rapid queries, consider
	 * using an array_map.
	 */
	inline hash_set_iterator<T, true> end() const {
		return hash_set_iterator<T, true>(*this, capacity);
	}

	/**
	 * Swaps the contents of the hash_set `first` with that of `second`.
	 */
	static void swap(hash_set<T>& first, hash_set<T>& second) {
		core::swap(first.keys, second.keys);
		core::swap(first.capacity, second.capacity);
		core::swap(first.size, second.size);
	}

	/**
	 * Moves the contents of the hash_set `src` into `dst`. Note this function
	 * does not copy the contents of the underlying hash_set::keys, it merely
	 * copies the pointer.
	 */
	static inline void move(const hash_set<T>& src, hash_set<T>& dst) {
		dst.keys = src.keys;
		dst.capacity = src.capacity;
		dst.size = src.size;
	}

	/**
	 * Copies the contents of the hash_set `src` into `dst`.
	 * \param alloc_keys a memory allocation function with prototype
	 * 		`void* alloc_keys(size_t count, size_t size)` that allocates space for
	 * 		`count` items, each with size `size`, and initializes them such that
	 * 		core::is_empty() returns `true` for each element.
	 */
	static inline bool copy(const hash_set<T>& src, hash_set<T>& dst, alloc_keys_func alloc_keys = calloc) {
		dst.capacity = src.capacity;
		dst.size = src.size;
		dst.keys = (T*) alloc_keys(src.capacity, sizeof(T));
		if (dst.keys == NULL) return false;

		for (unsigned int i = 0; i < src.capacity; i++) {
			if (is_empty(src.keys[i])) continue;
			if (!copy(src.keys[i], dst.keys[i])) {
				free(dst); return false;
			}
		}
		return true;
	}

	template<typename Metric>
	static inline long unsigned int size_of(const hash_set<T>& set, const Metric& metric)
	{
		long unsigned int sum = core::size_of(set.capacity) + core::size_of(set.size);
		for (unsigned int i = 0; i < set.capacity; i++) {
			if (is_empty(set.keys[i]))
				sum += sizeof(T);
			else sum += core::size_of(set.keys[i], metric);
		}
		return sum;
	}

	/**
	 * Frees hash_set::keys. This should not be used if `set` was constructed
	 * on the stack, as the destructor will automatically free hash_set::data.
	 * The elements of `set` are not freed.
	 */
	static inline void free(hash_set<T>& set) { set.free(); }

private:
	bool initialize(
			unsigned int initial_capacity,
			alloc_keys_func alloc_keys)
	{
#if !defined(NDEBUG)
		if (initial_capacity == 0)
			fprintf(stderr, "hashtable.initialize WARNING: Initial capacity is zero.\n");
#endif

		size = 0;
		capacity = initial_capacity;

		keys = (T*) alloc_keys(capacity, sizeof(T));
		return (keys != NULL);
	}

	inline void free() {
		core::free(keys);
	}

	inline void place(
			const T& element, unsigned int index)
	{
#if !defined(NDEBUG)
		if (is_empty(element))
			fprintf(stderr, "hashtable.place WARNING: Specified key is empty.\n");
#endif

		keys[index] = element;
	}

	inline void insert_unique(const T& element)
	{
		place(element, next_empty(element));
	}

	inline unsigned int first_empty() const {
		unsigned int index = 0;
		while (index < capacity && hasher<T>::is_empty(keys[index]))
			index++;
		return index;
	}

	inline unsigned int next_empty(const T& element)
	{
#if !defined(NDEBUG)
		if (size == capacity)
			fprintf(stderr, "hashtable.next_empty WARNING: Hashtable is full!\n");
#endif
		unsigned int index = hasher<T>::hash(element) % capacity;
		while (!hasher<T>::is_empty(keys[index]))
			index = (index + 1) % capacity;
		return index;
	}

	inline void insert(const T& element)
	{
		bool contains;
		unsigned int index = index_to_insert(element, contains);
		if (!contains) {
			place(element, index);
			size++;
		}
	}

	template<typename V>
	void remove_at(V* values, unsigned int index)
	{
		unsigned int last = index;
		unsigned int search = (index + 1) % capacity;
		if (!hasher<T>::is_empty(keys[search]))
		{
			do {
				unsigned int search_hash = hasher<T>::hash(keys[search]) % capacity;
				if (!index_between(search_hash, last, search)) {
					core::move(keys[search], keys[last]);
					core::move(values[search], values[last]);
					last = search;
				}
				search = (search + 1) % capacity;
			} while (!hasher<T>::is_empty(keys[search]));
		}

		hasher<T>::set_empty(keys[last]);
		size--;
	}

	template<typename K, typename V>
	friend struct hash_map;

	template<typename K>
	friend bool hash_set_init(hash_set<K>& set,
			unsigned int capacity, alloc_keys_func alloc_keys);

	template<typename K>
	friend bool hash_set_init(hash_set<K>& set,
			const std::initializer_list<T>& list,
			alloc_keys_func alloc_keys);
};

/**
 * Initializes the hash_set `set` with the given initial `capacity`.
 * \param alloc_keys a memory allocation function with prototype
 * 		`void* alloc_keys(size_t count, size_t size)` that allocates space for
 * 		`count` items, each with size `size`, and initializes them such that
 * 		core::is_empty() returns `true` for each element.
 */
template<typename T>
bool hash_set_init(
		hash_set<T>& set, unsigned int capacity,
		alloc_keys_func alloc_keys = calloc)
{
	if (!set.initialize(capacity, alloc_keys)) {
		fprintf(stderr, "hash_set_init ERROR: Unable to allocate memory.\n");
		return false;
	}
	return true;
}

/**
 * Swaps the underlying buffers of the given hash_sets `first` and `second`.
 */
template<typename T>
void swap(hash_set<T>& first, hash_set<T>& second) {
	T* keys_swap = first.keys;
	first.keys = second.keys;
	second.keys = keys_swap;

	unsigned int ui_swap = first.size;
	first.size = second.size;
	second.size = ui_swap;

	ui_swap = first.capacity;
	first.capacity = second.capacity;
	second.capacity = ui_swap;
}

/**
 * An unordered associative container that contains a set of key-value pairs,
 * where the keys are unique and have type `K` and the values have type `V`.
 * The keys are stored in a hash_set structure, and the values are stored in
 * hash_map::values, which is a native array parallel to the key array in
 * hash_set::keys. To compute the index of a key-value pair, we compute the
 * index of the key using the algorithm described in hash_set. The value has
 * the same index in hash_map::values. This structure uses the same linear
 * probing approach to resolve collisions (i.e. when two distinct keys are
 * computed to have the same index).
 * 
 * **WARNING:** As with hash_set, if the map becomes full, the linear probing
 * mechanism could lead to an infinite loop in many hash_map operations. The
 * function hash_map::check_size should be used whenever adding new elements
 * to avoid this scenario.
 *
 * **Performance:** This data structure provides constant-time access and
 * modification, given that the load factor (`size / capacity`) is not too
 * large.
 * 
 * Below is a simple example using a hash_map, where the expected output is
 * `a.get(3): c, a.get(4): d,  1:a 4:d 3:c`.
 *
 * ```{.cpp}
 * #include <core/map.h>
 * #include <stdio.h>
 * using namespace core;
 *
 * int main() {
 * 	hash_map<int, char> a = hash_map<int, char>(8);
 * 	a.put(1, 'a'); a.put(2, 'b');
 * 	a.put(3, 'c'); a.put(4, 'd');
 * 	a.remove(2);
 *
 * 	printf("a.get(3): %c, ", a.get(3));
 * 	printf("a.get(4): %c,  ", a.get(4));
 * 	for (const auto& entry : a)
 * 		printf("%d:%c ", entry.key, entry.value);
 * }
 * ```
 * 
 * 
 * However, if `a` is not allocated on the stack, the destructor will not be
 * automatically called, and so it must be freed manually using `core::free`
 * or `hash_map::free`. In the example below, the expected output is the same
 * as that of the program above: `a.get(3): c, a.get(4): d,  1:a 4:d 3:c`.
 *
 * ```{.cpp}
 * #include <core/map.h>
 * #include <stdio.h>
 * using namespace core;
 *
 * int main() {
 * 	hash_map<int, char>& a = *((hash_map<int, char>*) malloc(sizeof(hash_map<int, char>)));
 * 	hash_map_init(a, 8);
 * 	a.put(1, 'a'); a.put(2, 'b');
 * 	a.put(3, 'c'); a.put(4, 'd');
 * 	a.remove(2);
 *
 * 	printf("a.get(3): %c, ", a.get(3));
 * 	printf("a.get(4): %c,  ", a.get(4));
 * 	for (const auto& entry : a)
 * 		printf("%d:%c ", entry.key, entry.value);
 * 	free(a);
 * }
 * ```
 *
 *
 * A number of member functions require `K` and `V` to be
 * [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable).
 * If this is not the case, those operations can be performed directly on the
 * public fields, as in the following example. In addition, when using a custom
 * struct/class as the key type in hash_map, it must implement public static
 * functions, like `hash`, `is_empty`, and `move`, as well as the operator
 * `==`. The expected output of the following example is `first:1 second:2 `.
 *
 * ```{.cpp}
 * #include <core/map.h>
 * #include <stdio.h>
 * #include <string.h>
 * using namespace core;
 *
 * struct custom_string {
 * 	char* buffer;
 *
 * 	static unsigned int hash(const custom_string& s) {
 * 		return default_hash(s.buffer, strlen(s.buffer));
 * 	}
 *
 * 	static bool is_empty(const custom_string& s) {
 * 		return s.buffer == NULL;
 * 	}
 *
 * 	static void move(const custom_string& src, custom_string& dst) {
 * 		dst.buffer = src.buffer;
 * 	}
 *
 * 	static void free(custom_string& s) {
 * 		core::free(s.buffer);
 * 	}
 * };
 *
 * inline bool operator == (const custom_string& first, const custom_string& second) {
 * 	if (first.buffer == NULL)
 * 		return second.buffer == NULL;
 * 	return strcmp(first.buffer, second.buffer) == 0;
 * }
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
 * 	custom_string key_one, key_two;
 * 	int value_one = 1, value_two = 2;
 * 	init(key_one, "first");
 * 	init(key_two, "second");
 *
 * 	hash_map<custom_string, int> a = hash_map<custom_string, int>(8);
 * 	a.check_size(a.table.size + 2);
 *
 * 	bool contains; unsigned int index;
 * 	a.get(key_one, contains, index);
 * 	if (!contains) {
 * 		core::move(key_one, a.table.keys[index]);
 * 		a.values[index] = value_one;
 * 		a.table.size++;
 * 	}
 *
 * 	a.get(key_two, contains, index);
 * 	if (!contains) {
 * 		core::move(key_two, a.table.keys[index]);
 * 		a.values[index] = value_two;
 * 		a.table.size++;
 * 	}
 *
 * 	for (const auto& entry : a)
 * 		printf("%s:%d ", entry.key.buffer, entry.value);
 * 	for (auto entry : a)
 * 		free(entry.key);
 * }
 * ```
 *
 * \tparam K the generic type of the keys in the map. `K` must satisfy either:
 * 		1. [is_fundamental](http://en.cppreference.com/w/cpp/types/is_fundamental),
 * 		2. [is_enum](http://en.cppreference.com/w/cpp/types/is_enum),
 * 		3. [is_pointer](http://en.cppreference.com/w/cpp/types/is_pointer),
 * 		4. implements the public static method `unsigned int hash(const T&)`,
 * 			the public static method `void is_empty(const T&)`, implements the
 * 			operators `==`, and satisfies is_moveable. Some operations also
 * 			require the operator `!=`, and public static methods
 * 			`void set_empty(T&)` and `void set_empty(T*, unsigned int)`.
 * 			**NOTE:** The first argument to the `==` and `!=` operators may be
 * 			empty.
 * \tparam V the generic type of the values in the map. `V` must satisfy is_moveable.
 */
/* TODO: consider other collision resolution mechanisms */
template<typename K, typename V>
struct hash_map
{
	/**
	 * The type of the keys in this hash_map.
	 */
	typedef K key_type;

	/**
	 * The type of the values in this hash_map.
	 */
	typedef V value_type;

	/**
	 * The underlying hashtable containing the keys.
	 */
	hash_set<K> table;

	/**
	 * An array parallel to hash_set::keys in hash_map::table,
	 * containing a value at every non-empty bucket index.
	 */
	V* values;

	/**
	 * Constructs the hash_map with the given initial `capacity`.
	 * \param alloc_keys a memory allocation function with prototype
	 * 		`void* alloc_keys(size_t count, size_t size)` that allocates space for
	 * 		`count` items, each with size `size`, and initializes them such that
	 * 		core::is_empty() returns `true` for each key.
	 */
	hash_map(unsigned int capacity, alloc_keys_func alloc_keys = calloc) :
		table(capacity, alloc_keys)
	{
		if (!initialize_values()) {
			fprintf(stderr, "hash_map ERROR: Unable to allocate memory.\n");
			exit(EXIT_FAILURE);
		}
	}

	/**
	 * Constructs the hash_map and inserts the array of keys `map`, where each
	 * element in `map` is interpreted as a key, and its corresponding value is
	 * the index of the element.
	 * \param alloc_keys a memory allocation function with prototype
	 * 		`void* alloc_keys(size_t count, size_t size)` that allocates space for
	 * 		`count` items, each with size `size`, and initializes them such that
	 * 		core::is_empty() returns `true` for each key.
	 * \tparam V satisfies [is_integral](http://en.cppreference.com/w/cpp/types/is_integral).
	 */
	hash_map(const K* map, unsigned int length,
			alloc_keys_func alloc_keys = calloc) :
				table(length * RESIZE_THRESHOLD_INVERSE + 1, alloc_keys)
	{
		if (!initialize_values()) {
			fprintf(stderr, "hash_map ERROR: Unable to allocate memory.\n");
			exit(EXIT_FAILURE);
		}
		for (unsigned int i = 0; i < length; i++)
			insert(map[i], i);
	}

	/**
	 * Constructs the hash_map and inserts the list of key-value pairs given by
	 * the native arrays `keys` and `values`.
	 * \param alloc_keys a memory allocation function with prototype
	 * 		`void* alloc_keys(size_t count, size_t size)` that allocates space for
	 * 		`count` items, each with size `size`, and initializes them such that
	 * 		core::is_empty() returns `true` for each key.
	 */
	hash_map(const K* keys, const V* values,
			unsigned int length, alloc_keys_func alloc_keys = calloc) :
				table(length * RESIZE_THRESHOLD_INVERSE + 1, alloc_keys)
	{
		if (!initialize_values()) {
			fprintf(stderr, "hash_map ERROR: Unable to allocate memory.\n");
			exit(EXIT_FAILURE);
		}
		for (unsigned int i = 0; i < length; i++)
			insert(keys[i], values[i]);
	}

	/**
	 * Constructs the hash_map and inserts the given
	 * [initializer_list](http://en.cppreference.com/w/cpp/language/list_initialization)
	 * of core::pair entries.
	 * \param alloc_keys a memory allocation function with prototype
	 * 		`void* alloc_keys(size_t count, size_t size)` that allocates space for
	 * 		`count` items, each with size `size`, and initializes them such that
	 * 		core::is_empty() returns `true` for each key.
	 */
	hash_map(const std::initializer_list<pair<K, V>>& list,
			alloc_keys_func alloc_keys = calloc) :
		table(list.size() * RESIZE_THRESHOLD_INVERSE + 1, alloc_keys)
	{
		if (!initialize_values()) {
			fprintf(stderr, "hash_map ERROR: Unable to allocate memory.\n");
			exit(EXIT_FAILURE);
		}
		typename std::initializer_list<pair<K, V>>::iterator i;
		for (i = list.begin(); i != list.end(); i++)
			insert(i->key, i->value);
	}

	~hash_map() {
		::free(values);
	}

	/**
	 * Forces the underlying hash_map::table.keys and hash_map::values to be
	 * resized to the requested `capacity`.
	 *
	 * **WARNING:** If `new_capacity <= hash_map::table.size`, the hashtable
	 * could become full during the resize process, leading to an infinite loop
	 * due to the linear probing collision resolution mechanism.
	 * 
	 * \param alloc_keys a memory allocation function with prototype
	 * 		`void* alloc_keys(size_t count, size_t size)` that allocates space for
	 * 		`count` items, each with size `size`, and initializes them such that
	 * 		core::is_empty() returns `true` for each key.
	 */
	bool resize(unsigned int new_capacity,
			alloc_keys_func alloc_keys = calloc)
	{
		K* old_keys = table.keys;
		V* old_values = values;

		table.keys = (K*) alloc_keys(new_capacity, sizeof(K));
		if (table.keys == NULL) {
			/* revert changes and return error */
			table.keys = old_keys;
			return false;
		}

		values = (V*) malloc(sizeof(V) * new_capacity);
		if (values == NULL) {
			/* revert changes and return error */
			::free(table.keys);
			table.keys = old_keys;
			values = old_values;
			return false;
		}

		/* iterate through keys and re-hash the elements */
		unsigned int old_capacity = table.capacity;
		table.capacity = new_capacity;
		for (unsigned int i = 0; i < old_capacity; i++) {
			if (!hasher<K>::is_empty(old_keys[i])) {
				unsigned int new_bucket = table.next_empty(old_keys[i]);
				core::move(old_keys[i], table.keys[new_bucket]);
				core::move(old_values[i], values[new_bucket]);
			}
		}
		::free(old_keys);
		::free(old_values);
		return true;
	}

	/**
	 * This function first determines whether `hash_map::table.size < hash_map::table.capacity * RESIZE_THRESHOLD`.
	 * If not, the capacities of the underlying hash_map::table and
	 * hash_map::values is increased by RESIZE_FACTOR until the condition is
	 * satisfied. This is useful to ensure the hashtable is sufficiently large
	 * when preparing to add new elements.
	 * \param alloc_keys a memory allocation function with prototype
	 * 		`void* alloc_keys(size_t count, size_t size)` that allocates space for
	 * 		`count` items, each with size `size`, and initializes them such that
	 * 		core::is_empty() returns `true` for each key.
	 * \returns `true` if the resize was successful, and `false` if there is insufficient memory.
	 */
	inline bool check_size(alloc_keys_func alloc_keys = calloc) {
		return check_size(table.size, alloc_keys);
	}

	/**
	 * For a requested number of elements `new_size`, this function first
	 * determines whether `new_size < hash_map::table.capacity * RESIZE_THRESHOLD`.
	 * If not, the capacities of the underlying hash_map::table and
	 * hash_map::values is increased by RESIZE_FACTOR until the condition is
	 * satisfied. This is useful to ensure the hashtable is sufficiently large
	 * when preparing to add new elements.
	 * \param alloc_keys a memory allocation function with prototype
	 * 		`void* alloc_keys(size_t count, size_t size)` that allocates space for
	 * 		`count` items, each with size `size`, and initializes them such that
	 * 		core::is_empty() returns `true` for each key.
	 * \returns `true` if the resize was successful, and `false` if there is insufficient memory.
	 */
	inline bool check_size(unsigned int new_size, alloc_keys_func alloc_keys = calloc)
	{
		while (new_size >= table.capacity * RESIZE_THRESHOLD) {
			if (!resize(RESIZE_FACTOR * table.capacity, alloc_keys)) {
				fprintf(stderr, "hash_map.put ERROR: Unable to resize hashtable.\n");
				return false;
			}
		}
		return true;
	}

	/**
	 * Adds the given key-value pair to this map. The assignment operator is
	 * used insert the entry, and so this function should not be used if `K`
	 * and `V` are not [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable).
	 * In such a case, insertion should be performed manually by using
	 * hash_map::get or hash_set::index_to_insert to find the appropriate index
	 * and directly modifying hash_map::table.keys, hash_map::table.size, and
	 * hash_map::values. See the example in the hash_map description.
	 * \param alloc_keys a memory allocation function with prototype
	 * 		`void* alloc_keys(size_t count, size_t size)` that allocates space for
	 * 		`count` items, each with size `size`, and initializes them such that
	 * 		core::is_empty() returns `true` for each key.
	 * \tparam K is [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable).
	 * \tparam V is [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable).
	 */
	bool put(const K& key, const V& value,
			alloc_keys_func alloc_keys = calloc)
	{
		if (!check_size(table.size, alloc_keys)) return false;
		insert(key, value);
		return true;
	}

	/**
	 * Adds the given key-value pairs in `elements` to this map. The assignment
	 * operator is used insert each entry, and so this function should not be
	 * used if `K` and `V` are not [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable).
	 * In such a case, insertion should be performed manually by using
	 * hash_map::get or hash_map::index_to_insert to find the appropriate index
	 * and directly modifying hash_map::table.keys, hash_map::table.size, and
	 * hash_map::values. See the example in the hash_map description.
	 * \param alloc_keys a memory allocation function with prototype
	 * 		`void* alloc_keys(size_t count, size_t size)` that allocates space for
	 * 		`count` items, each with size `size`, and initializes them such that
	 * 		core::is_empty() returns `true` for each key.
	 * \tparam K is [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable).
	 * \tparam V is [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable).
	 */
	bool put_all(const hash_map<K, V>& elements,
			alloc_keys_func alloc_keys = calloc)
	{
		if (!check_size(table.size + elements.table.size, alloc_keys))
			return false;
		for (unsigned int i = 0; i < elements.table.capacity; i++)
			if (!hasher<K>::is_empty(elements.table.keys[i]))
				insert(elements.table.keys[i], elements.values[i]);
		return true;
	}

	/**
	 * This function removes `key` and associated value from the map. This
	 * function does not free the removed element.
	 * \returns `true` if the element is removed, and `false` if the set does not contain `element`.
	 */
	inline bool remove(const K& key)
	{
		return table.remove(key, values);
	}

	/**
	 * This function removes the entry at the bucket given by `index`. This
	 * function assumes that an entry is located at the given bucket with the
	 * correct provided hash value. This function does not free the removed key or
	 * value.
	 */
	inline void remove_at(unsigned int index)
	{
		table.remove_at(values, index);
	}

	/**
	 * Retrieves the value associated with the given `key`. This function
	 * assumes the given key exists in the map.
	 */
	V& get(const K& key) const
	{
		return values[table.index_of(key)];
	}

	/**
	 * If the given `key` exists in this map, this function returns the value
	 * associated with the key, and sets `contains` to `true`. If `key` is not
	 * in the map, `contains` is set to `false`. 
	 */
	inline V& get(const K& key, bool& contains) const
	{
		unsigned int index;
		return get(key, contains, index);
	}

	/**
	 * If the given `key` exists in this map, this function returns the value
	 * associated with the key, sets `contains` to `true`, and sets `index` to
	 * the bucket containing the key. If `key` is not in the map, `contains` is
	 * set to `false`, and `index` is set to the index where the key would be
	 * located, if it had existed in the map.
	 */
	V& get(const K& key, bool& contains, unsigned int& index) const
	{
		index = table.index_of(key, contains);
		return values[index];
	}

	/**
	 * If the given `key` exists in this map, this function returns the value
	 * associated with the key, sets `contains` to `true`, and sets `index` to
	 * the bucket containing the key. If `key` is not in the map, `contains` is
	 * set to `false`, and `index` is set to the index where the key would be
	 * located, if it had existed in the map. In any case, the evaluated hash
	 * function of `key` is stored in `hash_value`.
	 */
	V& get(const K& key, bool& contains,
			unsigned int& index,
			unsigned int& hash_value) const
	{
		index = table.index_of(key, contains, hash_value);
		return values[index];
	}

	/**
	 * Removes all entries from this hash_map. Note that this function does
	 * not free each entry beforehand.
	 */
	inline void clear() {
		table.clear();
	}

	/**
	 * Returns a hash_map_iterator pointing to the first entry in this container.
	 * 
	 * **NOTE:** Unlike the libstdc++ [unordered_set](http://en.cppreference.com/w/cpp/container/unordered_set)
	 * and [unordered_map](http://en.cppreference.com/w/cpp/container/unordered_map)
	 * iterators, we do not keep a linked list among the elements, so if rapid
	 * iteration over elements is more critical than rapid queries, consider
	 * using an array_map.
	 */
	inline hash_map_iterator<K, V, false> begin() {
		return hash_map_iterator<K, V, false>(*this, table.first_empty());
	}

	/**
	 * Returns a hash_map_iterator pointing to the end of this container.
	 * 
	 * **NOTE:** Unlike the libstdc++ [unordered_set](http://en.cppreference.com/w/cpp/container/unordered_set)
	 * and [unordered_map](http://en.cppreference.com/w/cpp/container/unordered_map)
	 * iterators, we do not keep a linked list among the elements, so if rapid
	 * iteration over elements is more critical than rapid queries, consider
	 * using an array_map.
	 */
	inline hash_map_iterator<K, V, false> end() {
		return hash_map_iterator<K, V, false>(*this, table.capacity);
	}

	/**
	 * Returns a const hash_map_iterator pointing to the first entry in this container.
	 * 
	 * **NOTE:** Unlike the libstdc++ [unordered_set](http://en.cppreference.com/w/cpp/container/unordered_set)
	 * and [unordered_map](http://en.cppreference.com/w/cpp/container/unordered_map)
	 * iterators, we do not keep a linked list among the elements, so if rapid
	 * iteration over elements is more critical than rapid queries, consider
	 * using an array_map.
	 */
	inline hash_map_iterator<K, V, true> begin() const {
		return hash_map_iterator<K, V, true>(*this, table.first_empty());
	}

	/**
	 * Returns a const hash_map_iterator pointing to the end of this container.
	 * 
	 * **NOTE:** Unlike the libstdc++ [unordered_set](http://en.cppreference.com/w/cpp/container/unordered_set)
	 * and [unordered_map](http://en.cppreference.com/w/cpp/container/unordered_map)
	 * iterators, we do not keep a linked list among the elements, so if rapid
	 * iteration over elements is more critical than rapid queries, consider
	 * using an array_map.
	 */
	inline hash_map_iterator<K, V, true> end() const {
		return hash_map_iterator<K, V, true>(*this, table.capacity);
	}

	/**
	 * Swaps the contents of the hash_map `first` with that of `second`.
	 */
	static inline void swap(hash_map<K, V>& first, hash_map<K, V>& second) {
		hash_set<K>::swap(first.table, second.table);
		core::swap(first.values, second.values);
	}

	/**
	 * Moves the contents of the hash_map `src` into `dst`. Note this function
	 * does not copy the contents of the underlying hash_map::table or
	 * hash_map::values, it merely copies the pointers.
	 */
	static inline void move(const hash_map<K, V>& src, hash_map<K, V>& dst) {
		hash_set<K>::move(src.table, dst.table);
		dst.values = src.values;
	}

	/**
	 * Copies the contents of the hash_map `src` into `dst`.
	 * \param alloc_keys a memory allocation function with prototype
	 * 		`void* alloc_keys(size_t count, size_t size)` that allocates space for
	 * 		`count` items, each with size `size`, and initializes them such that
	 * 		core::is_empty() returns `true` for each key.
	 */
	static inline bool copy(const hash_map<K, V>& src, hash_map<K, V>& dst, alloc_keys_func alloc_keys = calloc) {
		dst.table.capacity = src.table.capacity;
		dst.table.size = src.table.size;
		dst.table.keys = (K*) alloc_keys(src.table.capacity, sizeof(K));
		if (dst.table.keys == NULL) return false;
		dst.values = (V*) malloc(sizeof(V) * src.table.capacity);
		if (dst.values == NULL) {
			core::free(dst.table.keys);
			return false;
		}

		for (unsigned int i = 0; i < src.table.capacity; i++) {
			if (is_empty(src.table.keys[i])) continue;
			if (!core::copy(src.table.keys[i], dst.table.keys[i])) {
				free(dst); return false;
			} else if (!core::copy(src.values[i], dst.values[i])) {
				core::free(dst.table.keys[i]);
				set_empty(dst.table.keys[i]);
				free(dst); return false;
			}
		}
		return true;
	}

	template<typename KeyMetric, typename ValueMetric>
	static inline long unsigned int size_of(const hash_map<K, V>& map,
			const key_value_metric<KeyMetric, ValueMetric>& metric)
	{
		long unsigned int sum = core::size_of(map.table.capacity) + core::size_of(map.table.size);
		for (unsigned int i = 0; i < map.table.capacity; i++) {
			if (is_empty(map.table.keys[i]))
				sum += sizeof(K) + sizeof(V);
			else sum += core::size_of(map.table.keys[i], metric.key_metric) + core::size_of(map.values[i], metric.value_metric);
		}
		return sum;
	}

	/**
	 * Frees hash_map::table and hash_map::values. This should not be used if
	 * `map` was constructed on the stack, as the destructor will automatically
	 * free hash_map::table and hash_map::values. The existing entries of `map`
	 * are not freed.
	 */
	static inline void free(hash_map<K, V>& map) {
		core::free(map.table);
		core::free(map.values);
	}

private:
	/* NOTE: this function assumes table is initialized */
	bool initialize_values() {
		values = (V*) malloc(sizeof(V) * table.capacity);
		if (values == NULL) {
			core::free(table);
			return false;
		}
		return true;
	}

	bool initialize(
			unsigned int initial_capacity,
			alloc_keys_func alloc_keys)
	{
		if (!table.initialize(initial_capacity, alloc_keys))
			return false;
		return initialize_values();
	}

	inline void place(const K& key, const V& value, unsigned int index)
	{
		table.place(key, index);
		values[index] = value;
	}

	inline void insert_unique(
			const K& key, const V& value)
	{
		place(key, value, table.next_empty(key));
	}

	inline void insert(const K& key, const V& value)
	{
		bool contains;
		unsigned int index = table.index_to_insert(key, contains);
		if (!contains) {
			place(key, value, index);
			table.size++;
		}
	}

	template<typename T, typename U>
	friend bool hash_map_init(hash_map<T, U>& map,
			unsigned int capacity, alloc_keys_func alloc_keys);
};

/**
 * Initializes the hash_map `map` with the given initial `capacity`.
 * \param alloc_keys a memory allocation function with prototype
 * 		`void* alloc_keys(size_t count, size_t size)` that allocates space for
 * 		`count` items, each with size `size`, and initializes them such that
 * 		core::is_empty() returns `true` for each key.
 */
template<typename K, typename V>
bool hash_map_init(
		hash_map<K, V>& map,
		unsigned int capacity,
		alloc_keys_func alloc_keys = calloc)
{
	if (!map.initialize(capacity, alloc_keys)) {
		fprintf(stderr, "hash_map_init ERROR: Unable to allocate memory.\n");
		return false;
	}
	return true;
}

/**
 * Swaps the underlying buffers of the given hash_maps `first` and `second`.
 */
template<typename K, typename V>
void swap(hash_map<K, V>& first, hash_map<K, V>& second) {
	K* keys_swap = first.table.keys;
	first.table.keys = second.table.keys;
	second.table.keys = keys_swap;

	V* values_swap = first.values;
	first.values = second.values;
	second.values = values_swap;

	unsigned int ui_swap = first.table.size;
	first.table.size = second.table.size;
	second.table.size = ui_swap;

	ui_swap = first.table.capacity;
	first.table.capacity = second.table.capacity;
	second.table.capacity = ui_swap;
}

/**
 * A sequentially-ordered associative container that contains a set of
 * key-value pairs, where the keys are unique and have type `K` and the values
 * have type `V`. The keys are stored sequentially in a native array
 * array_map::keys, and the values are stored sequentially in a parallel native
 * array array_map::values.
 *
 * When inserting a key-value pair into an array_map, a linear search is
 * performed to determine if the key already exists in the map. If so, its
 * corresponding value is replaced by the new value. If not, the new key and
 * value are inserted at the ends of their respective arrays, and
 * array_map::size is incremented.
 *
 * **Performance:** This data structure provides linear-time access and
 * modification. However, due to locality of reference, this data structure
 * may perform operations more quickly than hash_map if the size of the map
 * is small.
 *
 * A simple example of the use of array_map is given below, where the expected
 * output is `a.get(3): c, a.get(4): d,  1:a 4:d 3:c `.
 *
 * ```{.cpp}
 * #include <core/map.h>
 * #include <stdio.h>
 * using namespace core;
 *
 * int main() {
 * 	array_map<int, char> a = array_map<int, char>(8);
 * 	a.put(1, 'a'); a.put(2, 'b');
 * 	a.put(3, 'c'); a.put(4, 'd');
 * 	a.remove(2);
 *
 * 	printf("a.get(3): %c, ", a.get(3));
 * 	printf("a.get(4): %c,  ", a.get(4));
 * 	for (const auto& entry : a)
 * 		printf("%d:%c ", entry.key, entry.value);
 * }
 * ```
 *
 *
 * However, if `a` is not allocated on the stack, the destructor will not be
 * automatically called, and so it must be freed using `core::free` or
 * `hash_map::free`. In the example below, the expected output is the same as
 * that of the program above: `a.get(3): c, a.get(4): d,  1:a 4:d 3:c `.
 *
 * ```{.cpp}
 * #include <core/map.h>
 * #include <stdio.h>
 * using namespace core;
 *
 * int main() {
 * 	array_map<int, char>& a = *((array_map<int, char>*) alloca(sizeof(array_map<int, char>)));
 * 	array_map_init(a, 8);
 * 	a.put(1, 'a'); a.put(2, 'b');
 * 	a.put(3, 'c'); a.put(4, 'd');
 * 	a.remove(2);
 *
 * 	printf("a.get(3): %c, ", a.get(3));
 * 	printf("a.get(4): %c,  ", a.get(4));
 * 	for (const auto& entry : a)
 * 		printf("%d:%c ", entry.key, entry.value);
 * 	free(a);
 * }
 * ```
 *
 *
 * A number of member functions require `K` and `V` to be
 * [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable).
 * If this is not the case, those operations can be performed directly on the
 * public fields, as in the following example. The expected output is
 * `first:1 second:2 `.
 *
 * ```{.cpp}
 * #include <core/map.h>
 * #include <stdio.h>
 * #include <string.h>
 * using namespace core;
 *
 * struct custom_string {
 * 	char* buffer;
 *
 * 	static void move(const custom_string& src, custom_string& dst) {
 * 		dst.buffer = src.buffer;
 * 	}
 *
 * 	static void free(custom_string& s) {
 * 		core::free(s.buffer);
 * 	}
 * };
 *
 * inline bool operator == (const custom_string& first, const custom_string& second) {
 * 	if (first.buffer == NULL)
 * 		return second.buffer == NULL;
 * 	return strcmp(first.buffer, second.buffer) == 0;
 * }
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
 * 	custom_string key_one, key_two;
 * 	int value_one = 1, value_two = 2;
 * 	init(key_one, "first");
 * 	init(key_two, "second");
 *
 * 	array_map<custom_string, int> a = array_map<custom_string, int>(8);
 * 	a.ensure_capacity(2);
 *
 * 	unsigned int index;
 * 	a.get(key_one, index);
 * 	if (index == a.size) {
 * 		core::move(key_one, a.keys[index]);
 * 		a.values[index] = value_one;
 * 		a.size++;
 * 	}
 *
 * 	a.get(key_two, index);
 * 	if (index == a.size) {
 * 		core::move(key_two, a.keys[index]);
 * 		a.values[index] = value_two;
 * 		a.size++;
 * 	}
 *
 * 	for (const auto& entry : a)
 * 		printf("%s:%d ", entry.key.buffer, entry.value);
 * 	for (auto entry : a)
 * 		free(entry.key);
 * }
 * ```
 */
template<typename K, typename V>
struct array_map {
	/**
	 * The type of the keys in this array_map.
	 */
	typedef K key_type;
	
	/**
	 * The type of the values in this array_map.
	 */
	typedef V value_type;

	/**
	 * The native array of keys.
	 */
	K* keys;

	/**
	 * The native array of values parallel to array_map::keys.
	 */
	V* values;

	/**
	 * The capacity of array_map::keys and array_map::values.
	 */
	size_t capacity;

	/**
	 * The number of entries in the array_map.
	 */
	size_t size;

	/**
	 * Constructs the hash_map with the given `initial_capacity`.
	 */
	array_map(size_t initial_capacity) : size(0) {
		if (!initialize(initial_capacity)) {
			fprintf(stderr, "array_map ERROR: Error during initialization.\n");
			exit(EXIT_FAILURE);
		}
	}

	~array_map() { free(); }

	/**
	 * Given the requested number of elements `new_length`, this function
	 * determines whether array_map::capacity is sufficient. If not, it
	 * attempts to increase its capacity by factors of RESIZE_FACTOR.
	 * \returns `true` if the resize was successful, and `false` if there is insufficient memory.
	 */
	bool ensure_capacity(size_t new_length) {
		if (new_length <= capacity)
			return true;

		size_t new_capacity = capacity;
		if (!expand(keys, new_capacity, new_length))
			return false;
		if (!resize(values, new_capacity))
			return false;
		capacity = new_capacity;
		return true;
	}

	/**
	 * Adds the given key-value pair to this map. The assignment operator is
	 * used insert the entry, and so this function should not be used if `K`
	 * and `V` are not [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable).
	 * In such a case, insertion should be performed manually by using
	 * array_map::get to compute the appropriate index and directly modifying
	 * the the public fields. See the example in the description of array_map.
	 * \tparam K is [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable).
	 * \tparam V is [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable).
	 */
	bool put(const K& key, const V& value) {
		size_t index = index_of(key);
		if (index < size) {
			values[index] = value;
			return true;
		}

		if (!ensure_capacity(size + 1))
			return false;
		keys[size] = key;
		values[size] = value;
		size++;
		return true;
	}

	/**
	 * Performs a linear search to find the index of the given `key`. If the
	 * `key` is not in this map, array_map::size is returned.
	 */
	template<typename Key>
	inline size_t index_of(const Key& key) const {
		return core::index_of(key, keys, size);
	}

	/**
	 * Performs a linear search to find the index of the given `key`, with the
	 * search beginning at the index `start`. If the `key` is not in this map,
	 * array_map::size is returned.
	 */
	template<typename Key>
	inline unsigned int index_of(const Key& key, unsigned int start) const {
		for (unsigned int i = start; i < size; i++)
			if (keys[i] == key)
				return i;
		return (unsigned int) size;
	}

	/**
	 * Performs a reverse linear search to find the index of the given `key`.
	 * If the `key` is not in this map, `static_cast<unsigned int>(-1)` is
	 * returned.
	 */
	template<typename Key>
	inline unsigned int last_index_of(const Key& key) const {
		return core::last_index_of(key, keys, size);
	}

	/**
	 * Returns `true` if the given `key` exists in the map, and `false` otherwise.
	 */
	inline bool contains(const K& key) const {
		return index_of(key) < size;
	}

	/**
	 * Retrieves the value associated with the given `key`. This function
	 * assumes the given key exists in the map.
	 */
	template<typename Key>
	inline V& get(const Key& key) {
		return values[index_of(key)];
	}

	/**
	 * Retrieves the const value associated with the given `key`. This function
	 * assumes the given key exists in the map.
	 */
	template<typename Key>
	inline const V& get(const Key& key) const {
		return values[index_of(key)];
	}

	/**
	 * If the given `key` exists in the map, this function returns the value
	 * associated with the key, and sets `index` to the index of
	 * array_map::keys where the key is located. If `key` does not exist in the
	 * map, `index` is set to array_map::size.
	 */
	template<typename Key>
	inline V& get(const Key& key, unsigned int& index) {
		index = index_of(key);
		return values[index];
	}

	/**
	 * If the given `key` exists in the map, this function returns the value
	 * associated with the key, and sets `contains` to `true`. If `key` does
	 * not exist in the map, `contains` is set to `false`.
	 */
	template<typename Key>
	inline V& get(const Key& key, bool& contains) {
		size_t index = index_of(key);
		contains = (index != size);
		return values[index];
	}

	/**
	 * If the given `key` exists in the map, this function returns the const
	 * value associated with the key, and sets `contains` to `true`. If `key`
	 * does not exist in the map, `contains` is set to `false`.
	 */
	template<typename Key>
	inline const V& get(const Key& key, bool& contains) const {
		unsigned int index = index_of(key);
		contains = (index != size);
		return values[index];
	}

	/**
	 * This function removes `key` and associated value from the map. This
	 * function does not free the removed element.
	 * \returns `true` if the element is removed, and `false` if the set does not contain `element`.
	 */
	bool remove(const K& key) {
		size_t index = index_of(key);
		if (index == size)
			return false;
		remove_at(index);
		return true;
	}

	/**
	 * This function removes the key-value pair located at the given `index` in
	 * the map. This function does not free the removed element. This function
	 * assumes `0 <= index < array_map::size`.
	 * \tparam K satisfies is_moveable.
	 * \tparam V satisfies is_moveable.
	 */
	inline void remove_at(size_t index) {
		size--;
		if (index == size)
			return;

		/* move the last item into the position of the removed item */
		core::move(keys[size], keys[index]);
		core::move(values[size], values[index]);
	}

	/**
	 * Removes all entries from this array_map. Note that this function does
	 * not free each entry beforehand.
	 */
	inline void clear() {
		size = 0;
	}

	/**
	 * Returns an array_map_iterator pointing to the first entry in this container.
	 */
	inline array_map_iterator<K, V, false> begin() {
		return array_map_iterator<K, V, false>(*this, 0);
	}

	/**
	 * Returns an array_map_iterator pointing to the end of this container.
	 */
	inline array_map_iterator<K, V, false> end() {
		return array_map_iterator<K, V, false>(*this, size);
	}

	/**
	 * Returns a const array_map_iterator pointing to the first entry in this container.
	 */
	inline array_map_iterator<K, V, true> begin() const {
		return array_map_iterator<K, V, true>(*this, 0);
	}

	/**
	 * Returns a const array_map_iterator pointing to the end of this container.
	 */
	inline array_map_iterator<K, V, true> end() const {
		return array_map_iterator<K, V, true>(*this, size);
	}

	/**
	 * Swaps the contents of the array_map `first` with that of `second`.
	 */
	static inline void swap(array_map<K, V>& first, array_map<K, V>& second) {
		core::swap(first.keys, second.keys);
		core::swap(first.values, second.values);
		core::swap(first.capacity, second.capacity);
		core::swap(first.size, second.size);
	}

	/**
	 * Moves the contents of the array_map `src` into `dst`. Note this function
	 * does not copy the contents of the underlying array_map::keys or
	 * array_map::values, it merely copies the pointers.
	 */
	static inline void move(const array_map<K, V>& src, array_map<K, V>& dst) {
		dst.keys = src.keys;
		dst.values = src.values;
		dst.capacity = src.capacity;
		dst.size = src.size;
	}

	template<typename KeyMetric, typename ValueMetric>
	static inline long unsigned int size_of(const array_map<K, V>& map,
			const key_value_metric<KeyMetric, ValueMetric>& metric)
	{
		long unsigned int sum = core::size_of(map.capacity) + core::size_of(map.size);
		for (unsigned int i = 0; i < map.size; i++)
			sum += core::size_of(map.keys[i], metric.key_metric) + core::size_of(map.values[i], metric.value_metric);
		return sum + (map.capacity - map.size) * (sizeof(K) + sizeof(V));
	}

	static inline long unsigned int size_of(const array_map<K, V>& map, const default_metric& metric) {
		return size_of(map, make_key_value_metric(default_metric(), default_metric()));
	}

	/**
	 * Frees array_map::keys and array_map::values. This should not be used if
	 * `map` was constructed on the stack, as the destructor will automatically
	 * free array_map::keys and array_map::values. The existing entries of
	 * `map` are not freed.
	 */
	static inline void free(array_map<K, V>& map) { map.free(); }

private:
	inline bool initialize(size_t initial_capacity) {
		capacity = initial_capacity;
		keys = (K*) malloc(sizeof(K) * capacity);
		if (keys == NULL) {
			fprintf(stderr, "array_map.initialize ERROR: Out of memory.\n");
			return false;
		}
		values = (V*) malloc(sizeof(V) * capacity);
		if (values == NULL) {
			core::free(keys);
			fprintf(stderr, "array_map.initialize ERROR: Out of memory.\n");
			return false;
		}
		return true;
	}

	inline void free() {
		core::free(keys);
		core::free(values);
	}

	template<typename A, typename B>
	friend bool array_map_init(array_map<A, B>&, size_t);
};

/**
 * Initializes the array_map `map` with the given `initial_capacity`.
 */
template<typename K, typename V>
bool array_map_init(array_map<K, V>& map, size_t initial_capacity) {
	map.size = 0;
	return map.initialize(initial_capacity);
}

/**
 * Returns the number of elements in the given hash_set `set`.
 */
template<typename T>
inline unsigned int size(const hash_set<T>& set) {
	return set.size;
}

/**
 * Returns the number of elements in the given hash_map `map`.
 */
template<typename K, typename V>
inline unsigned int size(const hash_map<K, V>& map) {
	return map.table.size;
}

/**
 * Returns the number of elements in the given array_map `map`.
 */
template<typename K, typename V>
inline unsigned int size(const array_map<K, V>& map) {
	return map.size;
}

/**
 * Assuming the given `map` has value type that satisfies
 * [is_integral](http://en.cppreference.com/w/cpp/types/is_integral),
 * and the values in the map are unique and no larger than `size(map)`,
 * this function returns an array of pointers to the keys in `map`, where the
 * pointer at index `i` references the key in `map` that corresponds to the
 * value `i`. The caller of this function is responsible for the memory of the
 * returned native array, and must call free to release it.
 * \tparam MapType a map type that allows range-based for iteration, and for
 * 		which the function `unsigned int size(const MapType&)` is defined.
 */
template<typename MapType>
inline const typename MapType::key_type** invert(const MapType& map) {
	const typename MapType::key_type** inverse =
			(const typename MapType::key_type**) calloc(size(map) + 1, sizeof(typename MapType::key_type*));
	if (inverse == NULL) {
		fprintf(stderr, "invert ERROR: Unable to invert map. Out of memory.\n");
		return NULL;
	}
	for (const auto& entry : map)
		inverse[entry.value] = &entry.key;
	return inverse;
}


inline bool hash_map_test(void)
{
	hash_map<int, const char*> map = hash_map<int, const char*>(4);
	bool contains_four = true, contains_minus_seven = true;
	map.get(4, contains_four);
	map.get(-7, contains_minus_seven);
	if (contains_four || contains_minus_seven) {
		fprintf(stderr, "hash_map_test ERROR: Map with no inserted elements should be empty.\n");
		return false;
	}

	/* test insertion and retrieval operations */
	map.put(-7, "negative seven");
	map.put(4, "four");

	if (map.table.size != 2) {
		fprintf(stderr, "hash_map_test ERROR: Map size is %d after adding two elements.\n", map.table.size);
		return false;
	}
	if (strcmp(map.get(-7), "negative seven")
	 || strcmp(map.get(4), "four"))
	{
		fprintf(stderr, "hash_map_test ERROR: Simple hashtable insertion failed.\n");
		return false;
	}

	map.put(4, "new four");
	map.put(5, "five");
	map.put(12, "twelve");
	map.put(7, "seven");
	map.put(13, "thirteen");

	/* test automatic capacity resizing */
	if (map.table.size != 6) {
		fprintf(stderr, "hash_map_test ERROR: Map size is %d after adding six elements.\n", map.table.size);
		return false;
	}
	if (map.table.capacity != 4 * RESIZE_FACTOR * RESIZE_FACTOR) {
		fprintf(stderr, "hash_map_test ERROR: Unexpected hashtable capacity.\n");
		return false;
	}
	if (strcmp(map.get(4), "new four")
	 || strcmp(map.get(12), "twelve")
	 || strcmp(map.get(7), "seven")
	 || strcmp(map.get(-7), "negative seven"))
	{
		fprintf(stderr, "hash_map_test ERROR: Additional hashtable insertion failed.\n");
		return false;
	}

	/* test removal operation */
	if (!map.remove(4)) {
		fprintf(stderr, "hash_map_test ERROR: Removal of key '4' failed.\n");
		return false;
	}
	if (strcmp(map.get(12), "twelve")
	 || strcmp(map.get(13), "thirteen")
	 || strcmp(map.get(5), "five")
	 || strcmp(map.get(-7), "negative seven")
	 || strcmp(map.get(7), "seven")) {
		fprintf(stderr, "hash_map_test ERROR: Hashtable lookup failed after removal.\n");
		return false;
	}

	bool contains = true;
	map.get(4, contains);
	if (contains) {
		fprintf(stderr, "hash_map_test ERROR: Retrieval of removed key failed.\n");
		return false;
	}

	return true;
}

inline bool array_map_test(void)
{
	array_map<int, const char*> map = array_map<int, const char*>(4);
	bool contains_four = true, contains_minus_seven = true;
	map.get(4, contains_four);
	map.get(-7, contains_minus_seven);
	if (contains_four || contains_minus_seven) {
		fprintf(stderr, "array_map_test ERROR: Map with no inserted elements should be empty.\n");
		return false;
	}

	/* test insertion and retrieval operations */
	map.put(-7, "negative seven");
	map.put(4, "four");

	if (map.size != 2) {
		fprintf(stderr, "array_map_test ERROR: Map size is %zu after adding two elements.\n", map.size);
		return false;
	}
	if (strcmp(map.get(-7), "negative seven")
	 || strcmp(map.get(4), "four"))
	{
		fprintf(stderr, "array_map_test ERROR: Simple hashtable insertion failed.\n");
		return false;
	}

	map.put(4, "new four");
	map.put(5, "five");
	map.put(12, "twelve");
	map.put(7, "seven");
	map.put(13, "thirteen");

	if (map.size != 6) {
		fprintf(stderr, "array_map_test ERROR: Map size is %zu after adding six elements.\n", map.size);
		return false;
	}
	if (strcmp(map.get(4), "new four")
	 || strcmp(map.get(12), "twelve")
	 || strcmp(map.get(7), "seven")
	 || strcmp(map.get(-7), "negative seven"))
	{
		fprintf(stderr, "array_map_test ERROR: Additional hashtable insertion failed.\n");
		return false;
	}

	/* test removal operation */
	if (!map.remove(4)) {
		fprintf(stderr, "array_map_test ERROR: Removal of key '4' failed.\n");
		return false;
	}
	if (strcmp(map.get(12), "twelve")
	 || strcmp(map.get(13), "thirteen")
	 || strcmp(map.get(5), "five")
	 || strcmp(map.get(-7), "negative seven")
	 || strcmp(map.get(7), "seven")) {
		fprintf(stderr, "array_map_test ERROR: Hashtable lookup failed after removal.\n");
		return false;
	}

	bool contains = true;
	map.get(4, contains);
	if (contains) {
		fprintf(stderr, "array_map_test ERROR: Retrieval of removed key failed.\n");
		return false;
	}

	return true;
}

} /* namespace core */

#endif /* MAP_H_ */
