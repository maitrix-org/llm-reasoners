/**
 * \file utility.h
 *
 * This file contains a number of useful miscellaneous definitions and
 * implementations, such as a core::string structure, a binary logarithm
 * function, and a function to access the directory structure in the
 * filesystem.
 *
 * <!-- Created on: Jan 8, 2014
 *          Author: asaparov -->
 */

#ifndef UTILITY_H_
#define UTILITY_H_

#include <stdio.h>
#include "io.h"

#if defined(_WIN32)
#if !defined(UNICODE)
#error "Unicode support required. Please compile with UNICODE preprocessor definition."
#endif
#define _WINSOCKAPI_
#include <windows.h>
#include <intrin.h>
#undef max
#undef min
#else
#include <dirent.h>
#include <sys/stat.h>
#endif


namespace core {

/* a useful type trait for detecting whether the function 'print_special_string' is defined */
namespace detail {
	template<typename Stream, typename... Printer> static auto test_print_special_string(int32_t) ->
			decltype(bool(print_special_string(0u, std::declval<Stream&>(), std::declval<Printer&&>()...)), std::true_type{});
	template<typename Stream, typename... Printer> static auto test_print_special_string(int64_t) -> std::false_type;
}

template<typename Stream, typename... Printer> struct has_print_special_string : decltype(core::detail::test_print_special_string<Stream, Printer...>(0)){};


/**
 * A basic string structure, containing a native array of `char` elements and an `unsigned int length`.
 */
struct string {
	/**
	 * The length of the string in characters.
	 */
	unsigned int length;

	/**
	 * The native `char` array containing the string data.
	 */
	char* data;

	/**
	 * A constructor that does not initialize any fields. **WARNING:** The
	 * destructor will free string::data, which this constructor does not
	 * initialize. The user must initialize string::data using `init` or
	 * manually before the string is destroyed.
	 */
	string() { }

	/**
	 * Constructs the string by copying from the given null-terminated C string `src`.
	 */
	string(const char* src) {
		if (!initialize(src, (unsigned int) strlen(src)))
			exit(EXIT_FAILURE);
	}

	/**
	 * Constructs the string by copying from the given native char array `src`
	 * with given `length`.
	 */
	string(const char* src, unsigned int length) {
		if (!initialize(src, length))
			exit(EXIT_FAILURE);
	}

	/**
	 * Constructs the string by initializing string::data with size `length`
	 * but without settings its contents.
	 */
	explicit string(unsigned int length) {
		if (!initialize(length))
			exit(EXIT_FAILURE);
	}

	~string() {
		core::free(data);
	}

	/**
	 * Accesses the character at the given `index`.
	 */
	inline char& operator [] (unsigned int index) {
		return data[index];
	}

	/**
	 * Accesses the character at the given `index`.
	 */
	inline const char& operator [] (unsigned int index) const {
		return data[index];
	}

	/**
	 * Initializes this string by copying from `s`. Note that if this string
	 * was previously initialized, it is not freed.
	 */
	inline void operator = (const string& s) {
		initialize(s.data, s.length);
	}

	/**
	 * Appends the given null-terminated C string to this string.
	 */
	inline void operator += (const char* src) {
		unsigned int src_length = (unsigned int) strlen(src);
		char* new_data = (char*) realloc(data, sizeof(char) * (length + src_length));
		if (new_data == NULL) {
			fprintf(stderr, "string.operator += ERROR: Unable to expand string.\n");
			exit(EXIT_FAILURE);
		}

		data = new_data;
		memcpy(data + length, src, sizeof(char) * src_length);
		length += src_length;
	}

	/**
	 * Appends the given string to this string.
	 */
	inline void operator += (const string& src) {
		unsigned int src_length = src.length;
		char* new_data = (char*) realloc(data, sizeof(char) * (length + src_length));
		if (new_data == NULL) {
			fprintf(stderr, "string.operator += ERROR: Unable to expand string.\n");
			exit(EXIT_FAILURE);
		}

		data = new_data;
		memcpy(data + length, src.data, sizeof(char) * src_length);
		length += src_length;
	}

	/**
	 * Returns whether the current string precedes `other` in lexicographical order.
	 */
	inline bool operator < (const string& other) const {
		for (unsigned int i = 0; ; i++) {
			if (i == length) {
				if (i == other.length) return false;
				else return true;
			} else if (i == other.length) {
				return false;
			}

			if (data[i] > other.data[i])
				return false;
			else if (data[i] < other.data[i])
				return true;
		}
	}

	/**
	 * Returns the smallest index `i` such that `string::data[i] == c`.
	 */
	inline unsigned int index_of(char c) const {
		for (unsigned int i = 0; i < length; i++) {
			if (data[i] == c)
				return i;
		}
		return length;
	}

	/**
	 * Returns whether string::data is NULL. This enables strings to be used as keys in hashtables.
	 */
	static inline bool is_empty(const string& key) {
		return key.data == NULL;
	}

	/**
	 * Sets string::data to NULL. This enables strings to be used as keys in hashtables.
	 */
	static inline void set_empty(string& key) {
		key.data = NULL;
	}

	/**
	 * Sets string::data to NULL for every element in `keys`. This enables
	 * strings to be used as keys in hashtables.
	 */
	static inline void set_empty(string* keys, unsigned int length) {
		memset(static_cast<void*>(keys), 0, sizeof(string) * length);
	}

	/**
	 * Returns the hash of the given `key`.
	 */
	static inline unsigned int hash(const string& key) {
		return default_hash(key.data, key.length);
	}

	/**
	 * Copies the string::length and string::data pointer from `src` to `dst`.
	 * Note this function does not create a new pointer and copy the character
	 * contents, it simply copies the `char*` pointer.
	 */
	static inline void move(const string& src, string& dst) {
		dst.length = src.length;
		dst.data = src.data;
	}

	/**
	 * Copies the string in `src` to `dst`. This function initializes a new
	 * string::data array in `dst` and copies the contents from `src.data`.
	 */
	static inline bool copy(const string& src, string& dst) {
		return dst.initialize(src.data, src.length);
	}

	/**
	 * Swaps the contents and lengths of `first` and `second`.
	 */
	static inline void swap(string& first, string& second) {
		core::swap(first.length, second.length);
		core::swap(first.data, second.data);
	}

	template<typename Metric>
	static inline long unsigned int size_of(const string& str, const Metric& metric) {
		return core::size_of(str.length) + sizeof(char) * str.length;
	}

	/**
	 * Frees the underlying `char` array in `str`.
	 */
	static inline void free(string& str) {
		core::free(str.data);
	}

private:
	bool initialize(unsigned int src_length) {
		length = src_length;
		data = (char*) malloc(sizeof(char) * (length == 0 ? 1 : length));
		if (data == NULL) {
			fprintf(stderr, "string.initialize ERROR: Unable to initialize string.\n");
			return false;
		}
		return true;
	}

	bool initialize(const char* src, unsigned int src_length) {
		if (!initialize(src_length))
			return false;
		memcpy(data, src, sizeof(char) * length);
		return true;
	}

	friend bool init(string&, const char*, unsigned int);
	friend bool init(string&, unsigned int);
};

/**
 * Initializes the string `dst` with the given native `char` array `src` and the given `length`.
 */
inline bool init(string& dst, const char* src, unsigned int length) {
	return dst.initialize(src, length);
}

/**
 * Initializes the string `dst` with the given string `src`.
 */
inline bool init(string& dst, const string& src) {
	return init(dst, src.data, src.length);
}

/**
 * Initializes the string `dst` by allocating string::data with size `length`,
 * but this function does not set its contents.
 */
inline bool init(string& dst, unsigned int length) {
	return dst.initialize(length);
}

/**
 * Reads a string `s` from `in`.
 * \param s an uninitialized string structure. This function initializes `s`,
 * 		and the caller is responsible for its memory and must call free to
 * 		release its memory resources.
 */
template<typename Stream>
inline bool read(string& s, Stream& in) {
	if (!read(s.length, in)) return false;
	s.data = (char*) malloc(sizeof(char) * s.length);
	if (s.data == NULL)
		return false;
	return read(s.data, in, s.length);
}

/**
 * Writes the string `s` to `out`.
 */
template<typename Stream>
inline bool write(const string& s, Stream& out) {
	if (!write(s.length, out)) return false;
	return write(s.data, out, s.length);
}

/**
 * Prints the string `s` to `stream`.
 */
template<typename Stream>
inline bool print(const string& s, Stream&& stream) {
	return fwrite(s.data, sizeof(char), s.length, stream) == s.length;
}

/**
 * Compares the string `first` to the null-terminated C string `second` and
 * returns `true` if they are equivalent, and `false` otherwise.
 */
inline bool operator == (const string& first, const char* second) {
	for (unsigned int i = 0; i < first.length; i++) {
		if (first[i] != second[i])
			return false;
	}
	if (second[first.length] != '\0')
		return false;
	return true;
}

/**
 * Compares the null-terminated C string `first` to the string `second` and
 * returns `true` if they are equivalent, and `false` otherwise.
 */
inline bool operator == (const char* first, const string& second) {
	return (second == first);
}

/**
 * Compares the string `first` to the string `second` and returns `true` if
 * they are equivalent, and `false` otherwise.
 */
inline bool operator == (const string& first, const string& second) {
	if (first.length != second.length) return false;
	/* we are guaranteed that only the first may be uninitialized */
	if (first.data == NULL)
		return false;
	return memcmp(first.data, second.data, first.length * sizeof(char)) == 0;
}

/**
 * Compares the string `first` to the null-terminated C string `second` and
 * returns `false` if they are equivalent, and `true` otherwise.
 */
inline bool operator != (const string& first, const char* second) {
	return !(first == second);
}

/**
 * Compares the null-terminated C string `first` to the string `second` and
 * returns `false` if they are equivalent, and `true` otherwise.
 */
inline bool operator != (const char* first, const string& second) {
	return !(second == first);
}

/**
 * Compares the string `first` to the string `second` and returns `false` if
 * they are equivalent, and `true` otherwise.
 */
inline bool operator != (const string& first, const string& second) {
	if (first.length != second.length) return true;
	if (first.data == NULL) {
		if (second.data == NULL) return true;
		else return false;
	} else if (second.data == NULL)
		return false;
	return memcmp(first.data, second.data, first.length * sizeof(char)) != 0;
}

struct sequence {
	unsigned int* tokens;
	unsigned int length;

	sequence(unsigned int* src, unsigned int length) : tokens(src), length(length) { }

	inline bool operator = (const sequence& src) {
		return initialize(src);
	}

	inline unsigned int& operator [] (unsigned int index) {
		return tokens[index];
	}

	inline unsigned int operator [] (unsigned int index) const {
		return tokens[index];
	}

	static inline unsigned int hash(const sequence& key) {
		return default_hash(key.tokens, key.length);
	}

	static inline bool is_empty(const sequence& key) {
		return key.tokens == NULL;
	}

	static inline void set_empty(sequence& key) {
		key.tokens = NULL;
	}

	static inline void move(const sequence& src, sequence& dst) {
		dst.tokens = src.tokens;
		dst.length = src.length;
	}

	static inline bool copy(const sequence& src, sequence& dst) {
		return dst.initialize(src);
	}

	static inline void swap(sequence& first, sequence& second) {
		core::swap(first.tokens, second.tokens);
		core::swap(first.length, second.length);
	}

	static inline void free(sequence& seq) {
		core::free(seq.tokens);
	}

private:
	inline bool initialize(const sequence& src) {
		length = src.length;
		tokens = (unsigned int*) malloc(sizeof(unsigned int) * length);
		if (tokens == NULL) {
			fprintf(stderr, "sequence.initialize ERROR: Out of memory.\n");
			return false;
		}
		memcpy(tokens, src.tokens, sizeof(unsigned int) * length);
		return true;
	}
};

inline bool init(sequence& seq, unsigned int length) {
	seq.length = length;
	seq.tokens = (unsigned int*) malloc(sizeof(unsigned int) * length);
	if (seq.tokens == NULL) {
		fprintf(stderr, "init ERROR: Insufficient memory for token array in sequence.\n");
		return false;
	}
	return true;
}

inline bool init(sequence& seq, const sequence& src) {
	return sequence::copy(src, seq);
}

inline bool operator == (const sequence& first, const sequence& second) {
	/* only the first argument can be uninitialized */
	if (first.tokens == NULL) return false;
	if (first.length != second.length) return false;
	for (unsigned int i = 0; i < first.length; i++)
		if (first.tokens[i] != second.tokens[i]) return false;
	return true;
}

inline bool operator != (const sequence& first, const sequence& second) {
	return !(first == second);
}

inline bool operator < (const sequence& first, const sequence& second) {
	if (first.length < second.length) return true;
	else if (first.length > second.length) return false;
	for (unsigned int i = 0; i < first.length; i++) {
		if (first[i] < second[i]) return true;
		else if (first[i] > second[i]) return false;
	}
	return false;
}

template<typename Stream>
inline bool read(sequence& item, Stream& in) {
	if (!read(item.length, in)) return false;
	item.tokens = (unsigned int*) malloc(sizeof(unsigned int) * item.length);
	if (item.tokens == NULL) return false;
	if (!read(item.tokens, in, item.length)) {
		free(item.tokens);
		return false;
	}
	return true;
}

template<typename Stream>
inline bool write(const sequence& item, Stream& out) {
	return write(item.length, out) && write(item.tokens, out, item.length);
}

template<typename Stream, typename... Printer>
inline bool print(const sequence& item,
	Stream&& out, Printer&&... printer)
{
	if (item.length == 0) return true;
	bool success = print(item[0], out, std::forward<Printer>(printer)...);
	for (unsigned int i = 1; i < item.length; i++)
		success &= print(' ', out) && print(item[i], out, std::forward<Printer>(printer)...);
	return success;
}

/**
 * A scribe that maps unsigned integer indices to core::string pointers.
 * 
 * ```{.cpp}
 * #include <core/utility.h>
 * using namespace core;
 * 
 * int main() {
 * 	string first = "first";
 * 	string second = "second";
 * 
 * 	string_map_scribe scribe;
 * 	scribe.map = (const string**) malloc(sizeof(const string*) * 2);
 * 	scribe.map[0] = &first; scribe.map[1] = &second;
 * 	scribe.length = 2;
 * 
 * 	print(0, stdout, scribe); print(' ', stdout);
 * 	print(1, stdout, scribe);
 * 
 * 	free(scribe.map);
 * }
 * ```
 * The expected output of this program is `first second`.
 * 
 * Another way to construct this structure is to convert a
 * hash_map<string, unsigned int> into a core::string** array using the
 * core::invert() function.
 * 
 * ```{.cpp}
 * #include <core/utility.h>
 * using namespace core;
 * 
 * int main() {
 * 	hash_map<string, unsigned int> token_map(16);
 * 	token_map.put("first", 0);
 * 	token_map.put("second", 1);
 * 
 * 	string_map_scribe scribe;
 * 	scribe.map = invert(token_map);
 * 	scribe.length = 2;
 * 
 * 	print(0, stdout, scribe); print(' ', stdout);
 * 	print(1, stdout, scribe);
 * 
 * 	free(scribe.map);
 * 	for (auto entry : token_map)
 * 		free(entry.key);
 * }
 * ```
 * The expected output of this program is `first second`. Notice that since
 * hash_map does not automatically free its elements, we do it manually using a
 * range-based for loop.
 */
struct string_map_scribe {
	/**
	 * The native array of `const string*` elements that represents the
	 * effective map from unsigned integers to strings.
	 */
	const string** map;

	/**
	 * The length of the native array string_map_scribe::map.
	 */
	unsigned int length;
};

template<typename Stream, typename... Printer, typename std::enable_if<has_print_special_string<Stream, Printer...>::value>::type* = nullptr>
inline bool print_special_string_helper(unsigned int item, Stream& out, Printer&&... printer) {
	return print_special_string(item, out, std::forward<Printer>(printer)...);
}

template<typename Stream, typename... Printer, typename std::enable_if<!has_print_special_string<Stream, Printer...>::value>::type* = nullptr>
inline bool print_special_string_helper(unsigned int item, Stream& out, Printer&&... printer) {
	fprintf(stderr, "print ERROR: The unsigned int %u exceeds the bounds of the "
			"string_map_scribe. Did you forget to implement print_special_string?\n", item);
	return true;
}

/**
 * Prints `item` to `out` using the string_map_scribe `printer`. If
 * `item < printer.length`, the string at index `item` is accessed from
 * string_map_scribe::map and printed to `out`. Otherwise, there are two cases:
 * 	1. If the function `bool print_special_string(unsigned int, Stream&)` is
 * 		defined, this function calls it with arguments `item` and `out`.
 * 	2. If such a function is not defined, an error message is printed to
 * 		[stderr](http://en.cppreference.com/w/cpp/io/c) and the function
 * 		returns `true`.
 *
 * \param string_printer a scribe for which the function
 * 		`bool print(const string&, Stream&, Printer&&...)` is defined, which is
 * 		used to print the string itself. Note that since this is a variadic
 * 		argument, it may be empty.
 */
template<typename Stream, typename... Printer>
inline bool print(unsigned int item, Stream&& out, const string_map_scribe& printer, Printer&&... string_printer)
{
	if (item < printer.length)
		return print(*printer.map[item], out, std::forward<Printer>(string_printer)...);
	else return print_special_string_helper(item, out, std::forward<Printer>(string_printer)...);
}

/**
 * Looks for the token `identifier` in the given hash_map `map`. If such a key
 * exists in the map, `id` is set to its corresponding value and `true` is
 * returned. If not, a new entry is added to the map with `identifier` as the
 * key and `map.table.size + 1` as its value, and `id` is set to this new value.
 * \returns `true` upon success, or `false` if the hash_map `map` could not be
 * 		resized to accommodate the new entry.
 */
bool get_token(const string& identifier, unsigned int& id, hash_map<string, unsigned int>& map) {
	if (!map.check_size()) {
		fprintf(stderr, "get_token ERROR: Unable to expand token map.\n");
		return false;
	}

	bool contains;
	unsigned int bucket;
	unsigned int& value = map.get(identifier, contains, bucket);
	if (!contains) {
		map.table.keys[bucket] = identifier;
		map.table.size++;
		value = map.table.size;
	}
	id = value;
	return true;
}

/**
 * Opens the file with the given `filename` with the given `mode` and returns
 * either a [FILE](http://en.cppreference.com/w/c/io) pointer on success, or
 * `NULL` on failure. [fclose](http://en.cppreference.com/w/c/io/fclose) should
 * be used to close the stream once reading is complete.
 * 
 * On Windows, this function uses
 * [fopen_s](https://msdn.microsoft.com/en-us/library/z5hh6ee9.aspx) to open
 * the stream.
 */
inline FILE* open_file(const char* filename, const char* mode) {
#if defined(_WIN32)
	FILE* file;
	if (fopen_s(&file, filename, mode) != 0)
		return NULL;
	return file;
#else
	return fopen(filename, mode);
#endif
}

/**
 * Reads the contents of the file whose path is given by `filename`. If the
 * file cannot be opened for reading, or if there is insufficient memory to
 * allocate a buffer, `NULL` is returned. `bytes_read` is set to the number of
 * bytes read, and the file contents are returned. The caller is responsible
 * for the memory of the returned buffer and must call free to release its
 * memory resources.
 * \tparam AppendNull if `true`, a null terminating character is appended.
 */
template<bool AppendNull>
inline char* read_file(const char* filename, size_t& bytes_read)
{
	FILE* fin = open_file(filename, "rb");
	if (fin == NULL || fseek(fin, 0, SEEK_END) != 0) {
		if (fin != NULL) {
#if defined(_WIN32)
			errno = (int) GetLastError();
#endif
			perror("read_file ERROR");
			fclose(fin);
		}
		return NULL;
	}

	long int filesize = ftell(fin);
	if (filesize == -1L) {
		fprintf(stderr, "read_file ERROR: `ftell` returned error.\n");
		fclose(fin);
		return NULL;
	}

	if (fseek(fin, 0, SEEK_SET) != 0) {
		fprintf(stderr, "read_file ERROR: `seek` returned error.\n");
		fclose(fin);
		return NULL;
	}

	char* data = (char*) malloc(sizeof(char) * (AppendNull ? (filesize + 1) : filesize));
	if (data == NULL) {
		fprintf(stderr, "read_file ERROR: Out of memory.\n");
		fclose(fin);
		return NULL;
	}
	bytes_read = fread(data, sizeof(char), filesize, fin);
	fclose(fin);
	if (AppendNull)
		data[filesize] = '\0';
	return data;
}

/**
 * This function inspects the directory given by the path `directory` and adds
 * the list of filenames in that directory to the string array `out`.
 * \see http://stackoverflow.com/questions/306533/how-do-i-get-a-list-of-files-in-a-directory-in-c
 */
inline bool get_files_in_directory(array<string>& out, const char* directory)
{
#if defined(_WIN32)
	HANDLE dir;
	WIN32_FIND_DATA file_data;

	size_t required;
	if (mbstowcs_s(&required, NULL, 0, directory, 0) != 0)
		return false;
	required--; /* ignore the null terminator */
	wchar_t* dir_prefix = (wchar_t*) malloc(sizeof(wchar_t) * (required + 3));
	if (dir_prefix == NULL) {
		fprintf(stderr, "get_files_in_directory ERROR: Out of memory.\n");
		return false;
	}
	if (mbstowcs_s(&required, dir_prefix, required + 3, directory, required) != 0) {
		free(dir_prefix);
		return false;
	}
	required--; /* ignore the null terminator */
	dir_prefix[required] = '/';
	dir_prefix[required + 1] = '*';
	dir_prefix[required + 2] = '\0';
	if ((dir = FindFirstFile(dir_prefix, &file_data)) == INVALID_HANDLE_VALUE) {
		free(dir_prefix);
		return false;
	}
	free(dir_prefix);

	do {
		const bool is_directory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
		if (file_data.cFileName[0] == '.' || is_directory)
			continue;

		if (!out.ensure_capacity(out.length + 1))
			return false;

		if (wcstombs_s(&required, NULL, 0, file_data.cFileName, 0) != 0
		 || !init(out[(unsigned int) out.length], (unsigned int) required))
			return false;
		out.length++;
		if (wcstombs_s(&required, out.last().data, required, file_data.cFileName, wcslen(file_data.cFileName)) != 0)
			return false;
		out.last().length--; /* remove the null terminator */
	} while (FindNextFile(dir, &file_data));

    FindClose(dir);
    return true;
#else
    DIR* dir;
    dirent* ent;
    struct stat st;

    dir = opendir(directory);
    if (dir == NULL)
    	return false;
    while ((ent = readdir(dir)) != NULL) {
        string full_file_name = string(directory);
        full_file_name += "/";
        full_file_name += ent->d_name;
        full_file_name += " ";
        full_file_name[full_file_name.length - 1] = '\0';
        if (ent->d_name[0] == '.' || stat(full_file_name.data, &st) == -1)
            continue;

        const bool is_directory = (st.st_mode & S_IFDIR) != 0;
        if (is_directory)
            continue;

        if (!out.ensure_capacity(out.length + 1))
        	return false;
        init(out[out.length], ent->d_name);
        out.length++;
    }
    closedir(dir);
    return true;
#endif
}

} /* namespace core */

#endif /* UTILITY_H_ */
