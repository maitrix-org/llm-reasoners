/**
 * \file io.h
 *
 * This file defines serialization and deserialization functions `read` and
 * `write`, as well as the `print` function for all fundamental types and core
 * data structures.
 *
 * The file also contains the definition and implementation of the
 * `core::memory_stream` class, which may be used to read/write to an in-memory
 * buffer.
 *
 * Scribes
 * -------
 *
 * The `read`, `write`, and `print` functions in this library follow a very
 * regular argument structure: The first argument is the object to
 * read/write/print. The second argument is the stream to read from/write
 * to/print to. Most functions also require a third (optional) argument called
 * the `scribe`, which controls how the subobjects are read/written/printed.
 * The scribe is typically passed to the read/write/print functions when
 * operating on subobjects of the given object.
 *
 * Calling read/write/print with core::default_scribe will call the same
 * function without the third argument. This largely corresponds to the
 * "default" behavior of those functions.
 *
 * For example, `write(const core::array<T>& a, Stream& out, Writer&&... writer)`
 * will call the function `write(a[i], out, writer)` for every element `a[i]`
 * in `a`. This enables users to define their own scribes, and define new ways
 * to read/write/print objects without having to re-implement the
 * read/write/print functions for container structures such as core::array. The
 * following example demonstrates how the behavior of the `print` function for
 * an array of integers can be altered using a custom scribe.
 *
 * ```{.cpp}
 * #include <core/io.h>
 * using namespace core;
 *
 * struct my_string_scribe {
 * 	const char* strings[3];
 * };
 *
 * template<typename Stream>
 * bool print(int i, Stream& out, const my_string_scribe& printer) {
 *     return print(printer.strings[i], out);
 * }
 *
 * int main() {
 * 	array<int> a = array<int>(8);
 * 	a.add(1); a.add(2); a.add(0);
 *
 * 	print(a, stdout); print(' ', stdout);
 *
 * 	default_scribe def;
 * 	print(a, stdout, def); print(' ', stdout);
 *
 * 	my_string_scribe printer;
 * 	printer.strings[0] = "vici";
 * 	printer.strings[1] = "veni";
 * 	printer.strings[2] = "vidi";
 *
 * 	print(a, stdout, printer);
 * }
 * ```
 *
 * This example has expected output `[1, 2, 0] [1, 2, 0] [veni, vidi, vici]`.
 *
 * <!-- Created on: Aug 29, 2014
 *          Author: asaparov -->
 */

#ifndef IO_H_
#define IO_H_

#include "array.h"
#include "map.h"

#include <stdarg.h>
#include <cstdint>
#include <errno.h>

#if defined(__APPLE__)
#include <cwchar>

inline size_t mbrtoc32(char32_t* wc, const char* s, size_t n, mbstate_t* ps) {
	wchar_t c;
	size_t result = mbrtowc(&c, s, n, ps);
	if (result < (size_t) -2 && wc != nullptr)
		*wc = c;
	return result;
}

#else /* __APPLE__ */
#include <cuchar>
#endif /* __APPLE__ */

namespace core {

/**
 * Reads `sizeof(T)` bytes from `in` and writes them to the memory referenced
 * by `value`. This function does not perform endianness transformations.
 * \param in the stream given by a [FILE](https://en.cppreference.com/w/c/io) pointer.
 * \tparam T satisfies [is_fundamental](https://en.cppreference.com/w/cpp/types/is_fundamental).
 */
template<typename T, typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr>
inline bool read(T& value, FILE* in) {
	return (fread(&value, sizeof(T), 1, in) == 1);
}

/**
 * Reads `length` elements from `in` and writes them to the native array
 * `values`. This function does not perform endianness transformations.
 * \param in the stream given by a [FILE](https://en.cppreference.com/w/c/io) pointer.
 * \tparam T satisfies [is_fundamental](https://en.cppreference.com/w/cpp/types/is_fundamental).
 */
template<typename T, typename SizeType, typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr>
inline bool read(T* values, FILE* in, SizeType length) {
	return (fread(values, sizeof(T), length, in) == length);
}

/**
 * Writes `sizeof(T)` bytes to `out` from the memory referenced by `value`.
 * This function does not perform endianness transformations.
 * \param out the stream given by a [FILE](https://en.cppreference.com/w/c/io) pointer.
 * \tparam T satisfies [is_fundamental](https://en.cppreference.com/w/cpp/types/is_fundamental).
 */
template<typename T, typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr>
inline bool write(const T& value, FILE* out) {
	return (fwrite(&value, sizeof(T), 1, out) == 1);
}

/**
 * Writes `length` elements to `out` from the native array `values`. This
 * function does not perform endianness transformations.
 * \param out the stream given by a [FILE](https://en.cppreference.com/w/c/io) pointer.
 * \tparam T satisfies [is_fundamental](https://en.cppreference.com/w/cpp/types/is_fundamental).
 */
template<typename T, typename SizeType, typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr>
inline bool write(const T* values, FILE* out, SizeType length) {
	return (fwrite(values, sizeof(T), length, out) == length);
}

/**
 * Prints the character `value` to the stream given by the
 * [FILE](https://en.cppreference.com/w/c/io) pointer `out`.
 */
inline bool print(const char& value, FILE* out) {
	return (fputc(value, out) != EOF);
}

/**
 * Prints the int `value` to the stream given by the
 * [FILE](https://en.cppreference.com/w/c/io) pointer `out`.
 */
inline bool print(const int& value, FILE* out) {
	return (fprintf(out, "%d", value) > 0);
}

/**
 * Prints the long `value` to the stream given by the
 * [FILE](https://en.cppreference.com/w/c/io) pointer `out`.
 */
inline bool print(const long& value, FILE* out) {
	return (fprintf(out, "%ld", value) > 0);
}

/**
 * Prints the long long `value` to the stream given by the
 * [FILE](https://en.cppreference.com/w/c/io) pointer `out`.
 */
inline bool print(const long long& value, FILE* out) {
	return (fprintf(out, "%lld", value) > 0);
}

/**
 * Prints the unsigned int `value` to the stream given by the
 * [FILE](https://en.cppreference.com/w/c/io) pointer `out`.
 */
inline bool print(const unsigned int& value, FILE* out) {
	return (fprintf(out, "%u", value) > 0);
}

/**
 * Prints the unsigned long `value` to the stream given by the
 * [FILE](https://en.cppreference.com/w/c/io) pointer `out`.
 */
inline bool print(const unsigned long& value, FILE* out) {
	return (fprintf(out, "%lu", value) > 0);
}

/**
 * Prints the unsigned long long `value` to the stream given by the
 * [FILE](https://en.cppreference.com/w/c/io) pointer `out`.
 */
inline bool print(const unsigned long long& value, FILE* out) {
	return (fprintf(out, "%llu", value) > 0);
}

/**
 * Prints the float `value` to the stream given by the
 * [FILE](https://en.cppreference.com/w/c/io) pointer `out`.
 */
inline bool print(const float& value, FILE* out) {
	return (fprintf(out, "%f", (double) value) > 0);
}

/**
 * Prints the double `value` to the stream given by the
 * [FILE](https://en.cppreference.com/w/c/io) pointer `out`.
 */
inline bool print(const double& value, FILE* out) {
	return (fprintf(out, "%lf", value) > 0);
}

/**
 * Prints the float `value` to the stream given by the
 * [FILE](https://en.cppreference.com/w/c/io) pointer `out`
 * with the given `precision`.
 */
inline bool print(const float& value, FILE* out, unsigned int precision) {
	return (fprintf(out, "%.*f", precision, (double) value) > 0);
}

/**
 * Prints the double `value` to the stream given by the
 * [FILE](https://en.cppreference.com/w/c/io) pointer `out`
 * with the given `precision`.
 */
inline bool print(const double& value, FILE* out, unsigned int precision) {
	return (fprintf(out, "%.*lf", precision, value) > 0);
}

/**
 * Prints the null-terminated C string `value` to the stream given by the
 * [FILE](https://en.cppreference.com/w/c/io) pointer `out`.
 */
inline bool print(const char* values, FILE* out) {
	return (fprintf(out, "%s", values) >= 0);
}

namespace detail {
	template<typename A, typename C> static auto test_readable(int32_t) ->
			decltype(bool(read(std::declval<A&>(), std::declval<C&>())), std::true_type{});
	template<typename A, typename C> static auto test_readable(int64_t) -> std::false_type;

	template<typename A, typename C> static auto test_writeable(int32_t) ->
			decltype(bool(write(std::declval<const A&>(), std::declval<C&>())), std::true_type{});
	template<typename A, typename C> static auto test_writeable(int64_t) -> std::false_type;

	template<typename A, typename C> static auto test_printable(int32_t) ->
			decltype(bool(print(std::declval<const A&>(), std::declval<C&>())), std::true_type{});
	template<typename A, typename C> static auto test_printable(int64_t) -> std::false_type;
}

/**
 * This type trait is [true_type](https://en.cppreference.com/w/cpp/types/integral_constant)
 * if and only if the function `bool read(integral&, T&)` is defined where
 * `integral` is any integral type.
 */
template<typename T> struct is_readable : and_type<
	decltype(core::detail::test_readable<bool, T>(0)),
	decltype(core::detail::test_readable<char, T>(0)),
	decltype(core::detail::test_readable<char16_t, T>(0)),
	decltype(core::detail::test_readable<char32_t, T>(0)),
	decltype(core::detail::test_readable<wchar_t, T>(0)),
	decltype(core::detail::test_readable<short, T>(0)),
	decltype(core::detail::test_readable<int, T>(0)),
	decltype(core::detail::test_readable<long, T>(0)),
	decltype(core::detail::test_readable<long long, T>(0)),
	decltype(core::detail::test_readable<unsigned char, T>(0)),
	decltype(core::detail::test_readable<unsigned short, T>(0)),
	decltype(core::detail::test_readable<unsigned int, T>(0)),
	decltype(core::detail::test_readable<unsigned long, T>(0)),
	decltype(core::detail::test_readable<unsigned long long, T>(0))>::type {};

/**
 * This type trait is [true_type](https://en.cppreference.com/w/cpp/types/integral_constant)
 * if and only if the function `bool fwrite(const integral&, T&)` is defined where
 * `integral` is any integral type.
 */
template<typename T> struct is_writeable : and_type<
	decltype(core::detail::test_writeable<bool, T>(0)),
	decltype(core::detail::test_writeable<char, T>(0)),
	decltype(core::detail::test_writeable<char16_t, T>(0)),
	decltype(core::detail::test_writeable<char32_t, T>(0)),
	decltype(core::detail::test_writeable<wchar_t, T>(0)),
	decltype(core::detail::test_writeable<short, T>(0)),
	decltype(core::detail::test_writeable<int, T>(0)),
	decltype(core::detail::test_writeable<long, T>(0)),
	decltype(core::detail::test_writeable<long long, T>(0)),
	decltype(core::detail::test_writeable<unsigned char, T>(0)),
	decltype(core::detail::test_writeable<unsigned short, T>(0)),
	decltype(core::detail::test_writeable<unsigned int, T>(0)),
	decltype(core::detail::test_writeable<unsigned long, T>(0)),
	decltype(core::detail::test_writeable<unsigned long long, T>(0))>::type {};

/**
 * This type trait is [true_type](https://en.cppreference.com/w/cpp/types/integral_constant)
 * if and only if the function `bool print(value, T&)` is defined.
 */
template<typename T> struct is_printable : and_type<
	decltype(core::detail::test_printable<char, T>(0)),
	decltype(core::detail::test_printable<int, T>(0)),
	decltype(core::detail::test_printable<unsigned int, T>(0)),
	decltype(core::detail::test_printable<unsigned long, T>(0)),
	decltype(core::detail::test_printable<unsigned long long, T>(0)),
	decltype(core::detail::test_printable<float, T>(0)),
	decltype(core::detail::test_printable<double, T>(0))>::type {};

/**
 * Represents a stream to read/write from an in-memory buffer.
 */
struct memory_stream {
	/**
	 * The size of the stream.
	 */
	unsigned int length;

	/**
	 * The current position of the stream in the buffer.
	 */
	unsigned int position;

	/**
	 * The underlying buffer.
	 */
	char* buffer;

	/**
	 * The default constructor does not initialize any fields.
	 */
	memory_stream() { }

	/**
	 * Initializes the stream with memory_stream::length given by
	 * `initial_capacity` and memory_stream::position set to `0`.
	 * memory_stream::buffer is allocated but not initialized to any value.
	 */
	memory_stream(unsigned int initial_capacity) : length(initial_capacity), position(0) {
		buffer = (char*) malloc(sizeof(char) * length);
		if (buffer == NULL) {
			fprintf(stderr, "memory_stream ERROR: Unable to initialize buffer.\n");
			exit(EXIT_FAILURE);
		}
	}

	/**
	 * Initializes the stream with the memory_stream::buffer given by `buf`,
	 * memory_stream::length given by `length`, and memory_stream::position set
	 * to `0`.
	 */
	memory_stream(const char* buf, unsigned int length) : length(length), position(0) {
		buffer = (char*) malloc(sizeof(char) * length);
		if (buffer == NULL) {
			fprintf(stderr, "memory_stream ERROR: Unable to initialize buffer.\n");
			exit(EXIT_FAILURE);
		}
		memcpy(buffer, buf, sizeof(char) * length);
	}

	~memory_stream() {
		free(buffer);
	}

	/**
	 * Reads a number of bytes given by `bytes` from the memory_stream and
	 * writes them to `dst`. This function assumes `dst` has sufficient capacity.
	 */
	inline bool read(void* dst, unsigned int bytes) {
		if (position + bytes >= length)
			return false;
		memcpy(dst, buffer + position, bytes);
		position += bytes;
		return true;
	}

	/**
	 * Checks whether the stream has sufficient size for an additional number
	 * of bytes given by `bytes` at its current memory_stream::position. If
	 * not, this function attempts to expand the buffer to a new size computed
	 * as `memory_stream::position + bytes`.
	 */
	inline bool ensure_capacity(unsigned int bytes) {
		if (position + bytes <= length)
			return true;

		unsigned int new_length = length;
		if (!core::expand(buffer, new_length, position + bytes))
			return false;
		length = new_length;
		return true;
	}

	/**
	 * Writes a number of bytes given by `bytes` from the given native array
	 * `src` to the current position in this stream. memory_stream::ensure_capacity
	 * is called to ensure the underlying buffer has sufficient size.
	 */
	inline bool write(const void* src, unsigned int bytes) {
		if (!ensure_capacity(bytes))
			return false;
		memcpy(buffer + position, src, bytes);
		position += bytes;
		return true;
	}
};

/**
 * Reads `sizeof(T)` bytes from `in` and writes them to the memory referenced
 * by `value`. This function does not perform endianness transformations.
 * \param in a memory_stream.
 * \tparam T satisfies [is_fundamental](https://en.cppreference.com/w/cpp/types/is_fundamental).
 */
template<typename T, typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr>
inline bool read(T& value, memory_stream& in) {
	return in.read(&value, sizeof(T));
}

/**
 * Reads `length` elements from `in` and writes them to the native array
 * `values`. This function does not perform endianness transformations.
 * \param in a memory_stream.
 * \tparam T satisfies [is_fundamental](https://en.cppreference.com/w/cpp/types/is_fundamental).
 */
template<typename T, typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr>
inline bool read(T* values, memory_stream& in, unsigned int length) {
	return in.read(values, (unsigned int) sizeof(T) * length);
}

/**
 * Writes `sizeof(T)` bytes to `out` from the memory referenced by `value`.
 * This function does not perform endianness transformations.
 * \param out a memory_stream.
 * \tparam T satisfies [is_fundamental](https://en.cppreference.com/w/cpp/types/is_fundamental).
 */
template<typename T, typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr>
inline bool write(const T& value, memory_stream& out) {
	return out.write(&value, sizeof(T));
}

/**
 * Writes `length` elements to `out` from the native array `values`. This
 * function does not perform endianness transformations.
 * \param out a memory_stream.
 * \tparam T satisfies [is_fundamental](https://en.cppreference.com/w/cpp/types/is_fundamental).
 */
template<typename T, typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr>
inline bool write(const T* values, memory_stream& out, unsigned int length) {
	return out.write(values, (unsigned int) sizeof(T) * length);
}

/**
 * Reads an array of `n` elements, each with a size of `size` bytes, from the
 * memory_stream `in`, to the memory address referenced by `dst`.
 * \see This function mirrors the equivalent [fread](http://en.cppreference.com/w/cpp/io/c/fread)
 * 			for [FILE](https://en.cppreference.com/w/c/io) pointer streams.
 * \returns the number of elements read.
 */
inline size_t fread(void* dst, size_t size, size_t count, memory_stream& in) {
	size_t num_bytes = size * count;
	if (in.position + num_bytes > in.length) {
		count = (in.length - in.position) / size;
		num_bytes = size * count;
	}

	memcpy(dst, in.buffer + in.position, num_bytes);
	in.position += num_bytes;
	return count;
}

/**
 * Writes the array of `n` elements, each with a size of `size` bytes, from the
 * memory address referenced by `src` to the memory_stream `out`.
 * \see This function mirrors the equivalent [fwrite](http://en.cppreference.com/w/cpp/io/c/fwrite)
 * 			for [FILE](https://en.cppreference.com/w/c/io) pointer streams.
 * \returns either `n` if the write is successful, or `0` upon failure.
 */
inline size_t fwrite(const void* src, size_t size, size_t n, memory_stream& out) {
	if (out.write(src, (unsigned int) (size * n)))
		return n;
	else return 0;
}

/**
 * Retrieves the current position in the given memory_stream.
 * \see This function mirrors the equivalent [fgetpos](https://en.cppreference.com/w/c/io/fgetpos)
 * 		for [FILE](https://en.cppreference.com/w/c/io) pointer streams.
 * \returns 0 on success; nonzero value otherwise.
 */
inline int fgetpos(const memory_stream& stream, fpos_t* pos) {
#if defined(_WIN32) || defined(__APPLE__) /* on Windows or Mac */
	*pos = (fpos_t) stream.position;
#else /* on Windows or Linux */
	pos->__pos = stream.position;
#endif
	return 0;
}

/**
 * Sets the current position in the given memory_stream.
 * \see This function mirrors the equivalent [fsetpos](https://en.cppreference.com/w/c/io/fsetpos)
 * 		for [FILE](https://en.cppreference.com/w/c/io) pointer streams.
 * \returns 0 on success; nonzero value otherwise.
 */
inline int fsetpos(memory_stream& stream, const fpos_t* pos) {
#if defined(_WIN32) || defined(__APPLE__) /* on Windows or Mac */
	stream.position = (unsigned int) *pos;
#else /* on Windows or Linux */
	stream.position = pos->__pos;
#endif
	return 0;
}

/**
 * Writes the given character `c` to the memory_stream `out`.
 * \see This function mirrors the equivalent [fputc](http://en.cppreference.com/w/cpp/io/c/fputc)
 * 		for [FILE](https://en.cppreference.com/w/c/io) pointer streams.
 */
inline int fputc(int c, memory_stream& out) {
	char ch = (char) c;
	if (out.write(&ch, sizeof(char)))
		return c;
	else return EOF;
}

/**
 * Writes the given null-terminated C string `s` to the memory_stream `out`.
 * \see This function mirrors the equivalent [fputs](http://en.cppreference.com/w/cpp/io/c/fputs)
 * 		for [FILE](https://en.cppreference.com/w/c/io) pointer streams.
 */
inline int fputs(const char* s, memory_stream& out) {
	if (out.write(s, (unsigned int) strlen(s)))
		return 1;
	else return EOF;
}

/**
 * Writes the given arguments according to the format string `format` to the
 * memory_stream `out`.
 * \see This function mirrors the equivalent [fprintf](http://en.cppreference.com/w/cpp/io/c/fprintf)
 * 		for [FILE](https://en.cppreference.com/w/c/io) pointer streams.
 * \returns the number of bytes written to the stream, or `-1` upon error.
 */
inline int fprintf(memory_stream& out, const char* format, ...) {
	va_list argptr;
	va_start(argptr, format);

#if defined(_WIN32)
	int required = _vscprintf(format, argptr);
	if (!out.ensure_capacity((unsigned int) required + 1)) {
		fprintf(stderr, "fprintf ERROR: Unable to expand memory stream buffer.\n");
		va_end(argptr);
		return -1;
	}

	int written = vsnprintf_s(out.buffer + out.position, out.length - out.position, (size_t) required, format, argptr);
#else
	int written = vsnprintf(out.buffer + out.position, out.length - out.position, format, argptr);
	if (written < 0) {
		va_end(argptr);
		return -1;
	} else if ((unsigned) written < out.length - out.position) {
		va_end(argptr);
		out.position += written;
		return written;
	}

	if (!out.ensure_capacity(written + 1)) {
		fprintf(stderr, "fprintf ERROR: Unable to expand memory stream buffer.\n");
		va_end(argptr);
		return -1;
	}
	written = vsnprintf(out.buffer + out.position, out.length - out.position, format, argptr);
#endif

	va_end(argptr);
	if (written < 0) return -1;
	out.position += written;
	return written;
}

/**
 * Prints the character `value` to the stream given by the memory_stream `out`.
 */
inline bool print(const char& value, memory_stream& out) {
	return (fputc(value, out) != EOF);
}

/**
 * Prints the int `value` to the stream given by the memory_stream `out`.
 */
inline bool print(const int& value, memory_stream& out) {
	return (fprintf(out, "%d", value) > 0);
}

/**
 * Prints the long `value` to the stream given by the memory_stream `out`.
 */
inline bool print(const long& value, memory_stream& out) {
	return (fprintf(out, "%ld", value) > 0);
}

/**
 * Prints the long long `value` to the stream given by the memory_stream `out`.
 */
inline bool print(const long long& value, memory_stream& out) {
	return (fprintf(out, "%lld", value) > 0);
}

/**
 * Prints the unsigned int `value` to the stream given by the memory_stream
 * `out`.
 */
inline bool print(const unsigned int& value, memory_stream& out) {
	return (fprintf(out, "%u", value) > 0);
}

/**
 * Prints the unsigned long `value` to the stream given by the memory_stream
 * `out`.
 */
inline bool print(const unsigned long& value, memory_stream& out) {
	return (fprintf(out, "%lu", value) > 0);
}

/**
 * Prints the unsigned long long `value` to the stream given by the
 * memory_stream `out`.
 */
inline bool print(const unsigned long long& value, memory_stream& out) {
	return (fprintf(out, "%llu", value) > 0);
}

/**
 * Prints the float `value` to the stream given by the memory_stream `out`.
 */
inline bool print(const float& value, memory_stream& out) {
	return (fprintf(out, "%f", (double) value) > 0);
}

/**
 * Prints the double `value` to the stream given by the memory_stream `out`.
 */
inline bool print(const double& value, memory_stream& out) {
	return (fprintf(out, "%lf", value) > 0);
}

/**
 * Prints the float `value` to the stream given by the memory_stream `out` with
 * the given `precision`.
 */
inline bool print(const float& value, memory_stream& out, unsigned int precision) {
	return (fprintf(out, "%.*f", precision, (double) value) > 0);
}

/**
 * Prints the double `value` to the stream given by the memory_stream `out`
 * with the given `precision`.
 */
inline bool print(const double& value, memory_stream& out, unsigned int precision) {
	return (fprintf(out, "%.*lf", precision, value) > 0);
}

/**
 * Prints the null-terminated C string `value` to the stream given by the
 * memory_stream `out`.
 */
inline bool print(const char* values, memory_stream& out) {
	return (fprintf(out, "%s", values) >= 0);
}

/**
 * A stream wrapper for reading UTF-32 characters from an underlying multibyte
 * stream (such as UTF-8).
 */
template<unsigned int BufferSize, typename Stream>
struct buffered_stream {
	Stream& underlying_stream;
	char buffer[BufferSize];
	unsigned int position;
	unsigned int length;
	mbstate_t shift;

	buffered_stream(Stream& underlying_stream) : underlying_stream(underlying_stream), position(0) {
		shift = {0};
		fill_buffer();
	}

	inline void fill_buffer() {
		length = fread(buffer, sizeof(char), BufferSize, underlying_stream);
	}

	/**
	 * Returns the next UTF-32 character (as a `char32_t`) from the stream. If
	 * there are no further bytes in the underlying stream or an error occurred,
	 * `static_cast<char32_t>(-1)` is returned.
	 */
	char32_t fgetc32() {
		static_assert(BufferSize >= MB_LEN_MAX, "BufferSize must be at least MB_LEN_MAX");

		while (true)
		{
			if (length == 0)
				return static_cast<char32_t>(-1);

			char32_t c;
			size_t status = mbrtoc32(&c, buffer + position, sizeof(char) * (length - position), &shift);
			if (status == static_cast<size_t>(-1)) {
				/* encoding error occurred */
				return static_cast<char32_t>(-1);
			} else if (status == static_cast<size_t>(-2)) {
				/* the character is valid but incomplete */
				position = 0;
				fill_buffer();
				continue;
			} else {
				if (status == 0)
					position += 1;
				else position += status;
				if (position == length) {
					position = 0;
					fill_buffer();
				}
				return c;
			}
		}
	}
};

template<unsigned int BufferSize>
struct buffered_stream<BufferSize, memory_stream> {
	memory_stream& underlying_stream;
	mbstate_t shift;

	buffered_stream(memory_stream& underlying_stream) : underlying_stream(underlying_stream) {
		shift = {0};
	}

	/**
	 * Returns the next UTF-32 character (as a `char32_t`) from the stream. If
	 * there are no further bytes in the underlying stream or an error occurred,
	 * `static_cast<char32_t>(-1)` is returned.
	 */
	char32_t fgetc32() {
		if (underlying_stream.position == underlying_stream.length)
			return static_cast<char32_t>(-1);

		char32_t c;
		size_t status = mbrtoc32(&c, underlying_stream.buffer + underlying_stream.position, sizeof(char) * (underlying_stream.length - underlying_stream.position), &shift);
		if (status == static_cast<size_t>(-1) || status == static_cast<size_t>(-2)) {
			/* encoding error occurred or the character is incomplete */
			return static_cast<char32_t>(-1);
		}

		if (status == 0)
			underlying_stream.position += 1;
		else underlying_stream.position += status;
		return c;
	}
};

/**
 * Returns the next UTF-32 character (as a `char32_t`) from the buffered_stream
 * `input`. If there are no further bytes in the underlying stream or an error
 * occurred, `static_cast<char32_t>(-1)` is returned.
 */
template<unsigned int BufferSize, typename Stream>
inline char32_t fgetc32(buffered_stream<BufferSize, Stream>& input)
{
	return input.fgetc32();
}

/**
 * A stream wrapper for reading/writing integral types as fixed-width integral
 * values. This is useful for cross-platform readability and writeability.
 */
template<typename Stream, typename BoolType = uint8_t,
	typename CharType = int8_t, typename UCharType = uint8_t,
	typename ShortType = int16_t, typename UShortType = uint16_t,
	typename IntType = int32_t, typename UIntType = uint32_t,
	typename LongType = uint64_t, typename ULongType = uint64_t,
	typename LongLongType = uint64_t, typename ULongLongType = uint64_t,
	typename FloatType = float, typename DoubleType = double>
struct fixed_width_stream {
	Stream& stream;

	fixed_width_stream(Stream& stream) : stream(stream) { }

	template<typename T, class Enable = void> struct type { };
	template<class Enable> struct type<bool, Enable> { typedef BoolType value; };
	template<class Enable> struct type<char, Enable> { typedef CharType value; };
	template<class Enable> struct type<unsigned char, Enable> { typedef UCharType value; };
	template<class Enable> struct type<short, Enable> { typedef ShortType value; };
	template<class Enable> struct type<unsigned short, Enable> { typedef UShortType value; };
	template<class Enable> struct type<int, Enable> { typedef IntType value; };
	template<class Enable> struct type<unsigned int, Enable> { typedef UIntType value; };
	template<class Enable> struct type<long, Enable> { typedef LongType value; };
	template<class Enable> struct type<unsigned long, Enable> { typedef ULongType value; };
	template<class Enable> struct type<long long, Enable> { typedef LongLongType value; };
	template<class Enable> struct type<unsigned long long, Enable> { typedef ULongLongType value; };
	template<class Enable> struct type<float, Enable> { typedef FloatType value; };
	template<class Enable> struct type<double, Enable> { typedef DoubleType value; };
};

/**
 * Reads `size(K)` bytes from `in` where `K` is the appropriate template
 * argument in the fixed_width_stream and writes them to the memory referenced
 * by `value`. This function does not perform endianness transformations.
 * \param in a fixed_width_stream.
 * \tparam T satisfies [is_fundamental](https://en.cppreference.com/w/cpp/types/is_fundamental).
 */
template<typename T, typename Stream, typename... Args,
	typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr>
inline bool read(T& value, fixed_width_stream<Stream, Args...>& in) {
	typedef typename fixed_width_stream<Stream, Args...>::template type<T>::value value_type;
	value_type c;
	if (!read(c, in.stream)) return false;
	value = (T) c;
	return true;
}

/**
 * Reads `length` elements from `in` and writes them to the native array
 * `values`. This function does not perform endianness transformations.
 * \param in a fixed_width_stream.
 * \tparam T satisfies [is_fundamental](https://en.cppreference.com/w/cpp/types/is_fundamental).
 */
template<typename T, typename Stream, typename SizeType, typename... Args,
	typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr>
inline bool read(T* values, fixed_width_stream<Stream, Args...>& in, SizeType length) {
	for (SizeType i = 0; i < length; i++)
		if (!read(values[i], in)) return false;
	return true;
}

/**
 * Writes `sizeof(K)` bytes to `out` from the memory referenced by `value`
 * where `K` is the appropriate template argument in the fixed_width_stream.
 * This function does not perform endianness transformations.
 * \param out a fixed_width_stream.
 * \tparam T satisfies [is_fundamental](https://en.cppreference.com/w/cpp/types/is_fundamental).
 */
template<typename T, typename Stream, typename... Args,
	typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr>
inline bool write(const T& value, fixed_width_stream<Stream, Args...>& out) {
	typedef typename fixed_width_stream<Stream, Args...>::template type<T>::value value_type;
	return write((value_type) value, out.stream);
}

/**
 * Writes `length` elements to `out` from the native array `values`. This
 * function does not perform endianness transformations.
 * \param out a fixed_width_stream.
 * \tparam T satisfies [is_fundamental](https://en.cppreference.com/w/cpp/types/is_fundamental).
 */
template<typename T, typename Stream, typename SizeType, typename... Args,
	typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr>
inline bool write(const T* values, fixed_width_stream<Stream, Args...>& out, SizeType length) {
	for (SizeType i = 0; i < length; i++)
		if (!write(values[i], out)) return false;
	return true;
}

/**
 * Writes the given null-terminated C string `values` to the stream `out`.
 * \tparam Stream satisfies is_writeable.
 */
template<typename Stream,
	typename std::enable_if<is_writeable<Stream>::value>::type* = nullptr>
inline bool write(const char* values, Stream& out) {
	return write(values, out, strlen(values));
}

/**
 * The default scribe implementation that provides the default behavior for
 * read/write/print functions.
 * \see [Section on scribes](#scribes).
 */
struct default_scribe { };

/* a type trait for detecting whether the function 'print' is defined with a default_scribe argument */
namespace detail {
	template<typename T, typename Stream> static auto test_default_read(int) ->
			decltype(bool(read(std::declval<T&>(), std::declval<Stream&>(), std::declval<default_scribe&>())), std::true_type{});
	template<typename T, typename Stream> static auto test_default_read(long) -> std::false_type;
	template<typename T, typename Stream> static auto test_default_write(int) ->
			decltype(bool(write(std::declval<const T&>(), std::declval<Stream&>(), std::declval<default_scribe&>())), std::true_type{});
	template<typename T, typename Stream> static auto test_default_write(long) -> std::false_type;
	template<typename T, typename Stream> static auto test_default_print(int) ->
			decltype(bool(print(std::declval<const T&>(), std::declval<Stream&>(), std::declval<default_scribe&>())), std::true_type{});
	template<typename T, typename Stream> static auto test_default_print(long) -> std::false_type;
}

template<typename T, typename Stream> struct has_default_read : decltype(core::detail::test_default_read<T, Stream>(0)){};
template<typename T, typename Stream> struct has_default_write : decltype(core::detail::test_default_write<T, Stream>(0)){};
template<typename T, typename Stream> struct has_default_print : decltype(core::detail::test_default_print<T, Stream>(0)){};

/**
 * Calls and returns `read(value, in)`, dropping the default_scribe argument.
 * \tparam Stream satisfies is_readable.
 */
template<typename T, typename Stream,
	typename std::enable_if<is_readable<Stream>::value && !has_default_read<T, Stream>::value>::type** = nullptr>
inline bool read(T& value, Stream& in, default_scribe& scribe) {
	return read(value, in);
}

/**
 * Calls and returns `write(value, out)`, dropping the default_scribe argument.
 * \tparam Stream satisfies is_writeable.
 */
template<typename T, typename Stream,
	typename std::enable_if<is_writeable<Stream>::value && !has_default_write<T, Stream>::value>::type* = nullptr>
inline bool write(const T& value, Stream& out, default_scribe& scribe) {
	return write(value, out);
}

/**
 * Calls and returns `print(value, out)`, dropping the default_scribe argument.
 * \tparam Stream satisfies is_printable.
 */
template<typename T, typename Stream,
	typename std::enable_if<is_printable<Stream>::value && !has_default_print<T, Stream>::value>::type* = nullptr>
inline auto print(const T& value, Stream& out, default_scribe& scribe) -> decltype(print(value, out)) {
	return print(value, out);
}

/**
 * A scribe that prints pointers by dereferencing the pointer and calling the
 * appropriate read/write/print function.
 * \see [Section on scribes](#scribes).
 */
struct pointer_scribe { };

/**
 * Allocates memory and stores the address in `value`, and then calls
 * `read(*value, in, reader)`, dropping the pointer_scribe argument. Note that
 * since `reader` is a variadic argument, it may be empty.
 * \tparam Stream satisfies is_readable.
 */
template<typename T, typename Stream, typename... Reader,
	typename std::enable_if<is_readable<Stream>::value>::type* = nullptr>
inline bool read(T*& value, Stream& in, const pointer_scribe& scribe, Reader&&... reader) {
	value = (T*) malloc(sizeof(T));
	if (value == NULL) {
		fprintf(stderr, "read ERROR: Out of memory.\n");
		return false;
	} else if (!read(*value, in, std::forward<Reader>(reader)...)) {
		free(value);
		return false;
	}
	return true;
}

/**
 * Calls and returns `write(*value, out, writer)`, dropping the pointer_scribe
 * argument. Note that since `writer` is a variadic argument, it may be empty.
 * \tparam Stream satisfies is_writeable.
 */
template<typename T, typename Stream, typename... Writer,
	typename std::enable_if<is_writeable<Stream>::value>::type* = nullptr>
inline bool write(const T* const value, Stream& out, const pointer_scribe& scribe, Writer&&... writer) {
	return write(*value, out, std::forward<Writer>(writer)...);
}

/**
 * Calls and returns `print(*value, out, printer)`, dropping the pointer_scribe
 * argument. Note that since `printer` is a variadic argument, it may be empty.
 * \tparam Stream satisfies is_printable.
 */
template<typename T, typename Stream, typename... Printer,
	typename std::enable_if<is_printable<Stream>::value>::type* = nullptr>
inline bool print(const T* const value, Stream& out, const pointer_scribe& scribe, Printer&&... printer) {
	return print(*value, out, std::forward<Printer>(printer)...);
}

/**
 * The default left delimitter "[" for the array print functions.
 */
char default_left_bracket[] = "[";

/**
 * The default right delimitter "]" for the array print functions.
 */
char default_right_bracket[] = "]";

/**
 * The default separator between elements ", " for the array print functions.
 */
char default_array_separator[] = ", ";

/**
 * Prints the given native array of `values` each of type `T`, where `length`
 * is the number of elements in the array. The output stream is `out`.
 * \param printer a scribe for which the function `bool print(const T&, Stream&, Printer&)`
 * 		is defined. Note that since this is a variadic argument, it may be empty.
 * \tparam Stream satisfies is_printable.
 */
template<typename T,
	const char* LeftBracket = default_left_bracket,
	const char* RightBracket = default_right_bracket,
	char const* Separator = default_array_separator,
	typename SizeType, typename Stream, typename... Printer,
	typename std::enable_if<is_printable<Stream>::value>::type* = nullptr>
bool print(const T* values, SizeType length, Stream& out, Printer&&... printer) {
	if (!print(LeftBracket, out)) return false;
	if (length == 0)
		return print(RightBracket, out);
	if (!print(values[0], out, std::forward<Printer>(printer)...)) return false;
	for (SizeType i = 1; i < length; i++) {
		if (!print(Separator, out) || !print(values[i], out, std::forward<Printer>(printer)...))
			return false;
	}
	return print(RightBracket, out);
}

/**
 * Prints the given native static array of `values` each of type `T`, where `N`
 * is the number of elements in the array. The output stream is `out`.
 * \param printer a scribe for which the function `bool print(const T&, Stream&, Printer&)`
 * 		is defined. Note that since this is a variadic argument, it may be empty.
 * \tparam Stream satisfies is_printable.
 */
template<typename T, size_t N,
	const char* LeftBracket = default_left_bracket,
	const char* RightBracket = default_right_bracket,
	char const* Separator = default_array_separator,
	typename Stream, typename... Printer,
	typename std::enable_if<is_printable<Stream>::value>::type* = nullptr>
bool print(const T (&values)[N], Stream& out, Printer&&... printer) {
	if (!print(LeftBracket, out)) return false;
	if (N == 0)
		return print(RightBracket, out);
	if (!print(values[0], out, std::forward<Printer>(printer)...)) return false;
	for (size_t i = 1; i < N; i++) {
		if (!print(Separator, out) || !print(values[i], out, std::forward<Printer>(printer)...))
			return false;
	}
	return print(RightBracket, out);
}

/**
 * Reads an array of `length` elements from `in` and stores the result in the
 * given native array `a`. This function assumes `a` has sufficient capacity.
 * \param reader a scribe for which the function `bool read(T&, Stream&, Reader&&...)`
 * 		is defined. Note that since this is a variadic argument, it may be empty.
 * \tparam Stream satisfies is_readable.
 */
template<typename T, typename Stream, typename SizeType, typename... Reader,
	typename std::enable_if<is_readable<Stream>::value>::type* = nullptr>
inline bool read(T* a, Stream& in, SizeType length, Reader&&... reader) {
	for (SizeType i = 0; i < length; i++)
		if (!read(a[i], in, std::forward<Reader>(reader)...)) return false;
	return true;
}

/**
 * Reads an array of `N` elements from `in` and stores the result in the given
 * native array `a`.
 * \param reader a scribe for which the function `bool read(T&, Stream&, Reader&&...)`
 * 		is defined. Note that since this is a variadic argument, it may be empty.
 * \tparam Stream satisfies is_readable.
 */
template<typename T, size_t N, typename Stream, typename... Reader,
	typename std::enable_if<is_readable<Stream>::value>::type* = nullptr>
inline bool read(T (&a)[N], Stream& in, Reader&&... reader) {
	for (size_t i = 0; i < N; i++)
		if (!read(a[i], in, std::forward<Reader>(reader)...)) return false;
	return true;
}

/**
 * Reads a core::array structure from `in` and stores the result in `a`.
 * \param a an uninitialized core::array structure. This function initializes
 * 		`a`, and the caller is responsible for its memory and must call free
 * 		to release its memory resources.
 * \param reader a scribe for which the function `bool read(T&, Stream&, Reader&&...)`
 * 		is defined. Note that since this is a variadic argument, it may be empty.
 * \tparam Stream satisfies is_readable.
 */
template<typename T, typename Stream, typename... Reader,
	typename std::enable_if<is_readable<Stream>::value>::type* = nullptr>
bool read(array<T>& a, Stream& in, Reader&&... reader) {
	size_t length;
	if (!read(length, in))
		return false;
	size_t capacity = ((size_t) 1) << (core::log2(length == 0 ? 1 : length) + 1);
	a.data = (T*) malloc(sizeof(T) * capacity);
	if (a.data == NULL) return false;
	if (!read(a.data, in, length, std::forward<Reader>(reader)...)) {
		free(a.data);
		return false;
	}
	a.length = length;
	a.capacity = capacity;
	return true;
}

/**
 * Writes the given native array `a` of elements to `out`, each of type `T`,
 * where the number of elements is given by `length`.
 * \param writer a scribe for which the function `bool write(const T&, Stream&, Writer&&...)`
 * 		is defined. Note that since this is a variadic argument, it may be empty.
 * \tparam Stream satisfies is_writeable.
 */
template<typename T, typename Stream, typename SizeType, typename... Writer,
	typename std::enable_if<is_writeable<Stream>::value>::type* = nullptr>
inline bool write(const T* a, Stream& out, SizeType length, Writer&&... writer) {
	for (SizeType i = 0; i < length; i++)
		if (!write(a[i], out, std::forward<Writer>(writer)...)) return false;
	return true;
}

/**
 * Writes the given native array `a` of elements to `out`, each of type `T`,
 * where the number of elements is given by `N`.
 * \param writer a scribe for which the function `bool write(const T&, Stream&, Writer&&...)`
 * 		is defined. Note that since this is a variadic argument, it may be empty.
 * \tparam Stream satisfies is_writeable.
 */
template<typename T, size_t N, typename Stream, typename... Writer,
	typename std::enable_if<is_writeable<Stream>::value>::type* = nullptr>
inline bool write(const T (&a)[N], Stream& out, Writer&&... writer) {
	for (size_t i = 0; i < N; i++)
		if (!write(a[i], out, std::forward<Writer>(writer)...)) return false;
	return true;
}

/**
 * Writes the given core::array structure `a` of elements to `out`, each of type `T`.
 * \param writer a scribe for which the function `bool write(const T&, Stream&, Writer&&...)`
 * 		is defined. Note that since this is a variadic argument, it may be empty.
 * \tparam Stream satisfies is_writeable.
 */
template<typename T, typename Stream, typename... Writer,
	typename std::enable_if<is_writeable<Stream>::value>::type* = nullptr>
bool write(const array<T>& a, Stream& out, Writer&&... writer) {
	return write(a.length, out)
		&& write(a.data, out, a.length, std::forward<Writer>(writer)...);
}

/**
 * Prints the given core::array structure `a` of elements to `out`, each of type `T`.
 * \param printer a scribe for which the function `bool print(const T&, Stream&, Printer&&...)`
 * 		is defined. Note that since this is a variadic argument, it may be empty.
 * \tparam Stream satisfies is_printable.
 */
template<typename T,
	char const* LeftBracket = default_left_bracket,
	char const* RightBracket = default_right_bracket,
	char const* Separator = default_array_separator,
	typename Stream, typename... Printer,
	typename std::enable_if<is_printable<Stream>::value>::type* = nullptr>
inline bool print(const array<T>& a, Stream&& out, Printer&&... printer) {
	return print<T, LeftBracket, RightBracket, Separator>(a.data, a.length, out, std::forward<Printer>(printer)...);
}

/**
 * Reads a core::hash_set structure `set` from `in`.
 * \param set an uninitialized core::hash_set structure. This function
 * 		initializes `set`, and the caller is responsible for its memory and
 * 		must call free to release its memory resources.
 * \param alloc_keys a memory allocation function with prototype
 * 		`void* alloc_keys(size_t count, size_t size)` that allocates space for
 * 		`count` items, each with size `size`, and initializes them such that
 * 		core::is_empty() returns `true` for each element.
 * \param reader a scribe for which the function `bool read(T&, Stream&, Reader&&...)`
 * 		is defined. Note that since this is a variadic argument, it may be empty.
 * \tparam Stream satisfies is_readable.
 */
template<typename T, typename Stream, typename... Reader,
	typename std::enable_if<is_readable<Stream>::value>::type* = nullptr>
bool read(hash_set<T>& set, Stream& in, alloc_keys_func alloc_keys, Reader&&... reader) {
	unsigned int length;
	if (!read(length, in)) return false;

	set.size = 0;
	set.capacity = 1 << (core::log2(RESIZE_THRESHOLD_INVERSE * (length == 0 ? 1 : length)) + 1);
	set.keys = (T*) alloc_keys(set.capacity, sizeof(T));
	if (set.keys == NULL) return false;

	for (unsigned int i = 0; i < length; i++) {
		T& key = *((T*) alloca(sizeof(T)));
		if (!read(key, in, std::forward<Reader>(reader)...)) return false;
		move(key, set.keys[set.index_to_insert(key)]);
		set.size++;
	}
	return true;
}

/**
 * Reads a core::hash_set structure `set` from `in`. The keys in the hash_set
 * are allocated using [calloc](http://en.cppreference.com/w/c/memory/calloc).
 * \param set an uninitialized core::hash_set structure. This function
 * 		initializes `set`, and the caller is responsible for its memory and
 * 		must call free to release its memory resources.
 * \param reader a scribe for which the function `bool read(T&, Stream&, Reader&&...)`
 * 		is defined. Note that since this is a variadic argument, it may be empty.
 * \tparam Stream satisfies is_readable.
 */
template<typename T, typename Stream, typename... Reader,
	typename std::enable_if<is_readable<Stream>::value>::type* = nullptr>
inline bool read(hash_set<T>& set, Stream& in, Reader&&... reader) {
	return read(set, in, calloc, std::forward<Reader>(reader)...);
}

/**
 * Writes the given core::hash_set structure `set` to `out`.
 * \param writer a scribe for which the function `bool write(const T&, Stream&, Writer&&...)`
 * 		is defined. Note that since this is a variadic argument, it may be empty.
 * \tparam Stream satisfies is_writeable.
 */
template<typename T, typename Stream, typename... Writer,
	typename std::enable_if<is_writeable<Stream>::value>::type* = nullptr>
bool write(const hash_set<T>& set, Stream& out, Writer&&... writer) {
	if (!write(set.size, out)) return false;
	for (unsigned int i = 0; i < set.capacity; i++) {
		if (is_empty(set.keys[i])) continue;
		if (!write(set.keys[i], out, std::forward<Writer>(writer)...)) return false;
	}
	return true;
}

/**
 * Reads a core::hash_map structure `map` from `in`.
 * \param map an uninitialized core::hash_map structure. This function
 * 		initializes `map`, and the caller is responsible for its memory and
 * 		must call free to release its memory resources.
 * \param alloc_keys a memory allocation function with prototype
 * 		`void* alloc_keys(size_t count, size_t size)` that allocates space for
 * 		`count` items, each with size `size`, and initializes them such that
 * 		core::is_empty() returns `true` for each element of type `K`.
 * \param key_reader a scribe for which the function `bool read(K&, Stream&, KeyReader&)` is defined.
 * \param value_reader a scribe for which the function `bool read(V&, Stream&, ValueReader&)` is defined.
 * \tparam Stream satisfies is_readable.
 */
template<typename K, typename V, typename Stream,
	typename KeyReader, typename ValueReader,
	typename std::enable_if<is_readable<Stream>::value>::type* = nullptr>
bool read(hash_map<K, V>& map,
	Stream& in, alloc_keys_func alloc_keys,
	KeyReader& key_reader, ValueReader& value_reader)
{
	unsigned int length;
	if (!read(length, in)) return false;

	map.table.size = 0;
	map.table.capacity = 1 << (core::log2(RESIZE_THRESHOLD_INVERSE * (length == 0 ? 1 : length)) + 1);
	map.table.keys = (K*) alloc_keys(map.table.capacity, sizeof(K));
	if (map.table.keys == NULL) return false;
	map.values = (V*) malloc(sizeof(V) * map.table.capacity);
	if (map.values == NULL) {
		free(map.table.keys);
		return false;
	}

	for (unsigned int i = 0; i < length; i++) {
		K& key = *((K*) alloca(sizeof(K)));
		if (!read(key, in, key_reader)) return false;

		bool contains;
		unsigned int bucket;
		map.get(key, contains, bucket);
		if (!read(map.values[bucket], in, value_reader))
			return false;
		if (!contains) {
			move(key, map.table.keys[bucket]);
			map.table.size++;
		} else {
			fprintf(stderr, "read WARNING: Serialized hash_map contains duplicates.\n");
		}
	}
	return true;
}

/**
 * Reads a core::hash_map structure `map` from `in`.
 * \param map an uninitialized core::hash_map structure. This function
 * 		initializes `map`, and the caller is responsible for its memory and
 * 		must call free to release its memory resources.
 * \param alloc_keys a memory allocation function with prototype
 * 		`void* alloc_keys(size_t count, size_t size)` that allocates space for
 * 		`count` items, each with size `size`, and initializes them such that
 * 		core::is_empty() returns `true` for each element of type `K`.
 * \param key_reader a scribe for which the function `bool read(K&, Stream&, KeyReader&)` is defined.
 * \tparam Stream satisfies is_readable.
 */
template<typename K, typename V, typename Stream, typename KeyReader,
	typename std::enable_if<is_readable<Stream>::value>::type* = nullptr>
inline bool read(hash_map<K, V>& map, Stream& in,
		KeyReader& key_reader,
		alloc_keys_func alloc_keys = calloc)
{
	default_scribe scribe;
	return read(map, in, alloc_keys, key_reader, scribe);
}

/**
 * Reads a core::hash_map structure `map` from `in`.
 * \param map an uninitialized core::hash_map structure. This function
 * 		initializes `map`, and the caller is responsible for its memory and
 * 		must call free to release its memory resources.
 * \param alloc_keys a memory allocation function with prototype
 * 		`void* alloc_keys(size_t count, size_t size)` that allocates space for
 * 		`count` items, each with size `size`, and initializes them such that
 * 		core::is_empty() returns `true` for each element of type `K`.
 * \tparam Stream satisfies is_readable.
 */
template<typename K, typename V, typename Stream,
	typename std::enable_if<is_readable<Stream>::value>::type* = nullptr>
inline bool read(hash_map<K, V>& map, Stream& in,
		alloc_keys_func alloc_keys = calloc)
{
	default_scribe scribe;
	return read(map, in, alloc_keys, scribe, scribe);
}

/**
 * Writes the core::hash_map structure `map` to `out`.
 * \param key_writer a scribe for which the function `bool write(const K&, Stream&, KeyWriter&)` is defined.
 * \param value_writer a scribe for which the function `bool write(const V&, Stream&, ValueWriter&)` is defined.
 * \tparam Stream satisfies is_writeable.
 */
template<typename K, typename V, typename Stream,
	typename KeyWriter, typename ValueWriter,
	typename std::enable_if<is_writeable<Stream>::value>::type* = nullptr>
bool write(const hash_map<K, V>& map, Stream& out,
		KeyWriter& key_writer, ValueWriter& value_writer)
{
	if (!write(map.table.size, out)) return false;
	for (unsigned int i = 0; i < map.table.capacity; i++) {
		if (is_empty(map.table.keys[i])) continue;
		if (!write(map.table.keys[i], out, key_writer)
		 || !write(map.values[i], out, value_writer))
			return false;
	}
	return true;
}

/**
 * Writes the core::hash_map structure `map` to `out`.
 * \param key_writer a scribe for which the function `bool write(const K&, Stream&, KeyWriter&)` is defined.
 * \tparam Stream satisfies is_writeable.
 */
template<typename K, typename V, typename Stream, typename KeyWriter,
	typename std::enable_if<is_writeable<Stream>::value>::type* = nullptr>
inline bool write(const hash_map<K, V>& map, Stream& out, KeyWriter& key_writer) {
	default_scribe scribe;
	return write(map, out, key_writer, scribe);
}

/**
 * Writes the core::hash_map structure `map` to `out`.
 * \tparam Stream satisfies is_writeable.
 */
template<typename K, typename V, typename Stream,
	typename std::enable_if<is_writeable<Stream>::value>::type* = nullptr>
inline bool write(const hash_map<K, V>& map, Stream& out) {
	default_scribe scribe;
	return write(map, out, scribe, scribe);
}

/**
 * Reads a core::array_map structure `map` from `in`.
 * \param map an uninitialized core::array_map structure. This function
 * 		initializes `map`, and the caller is responsible for its memory and
 * 		must call free to release its memory resources.
 * \param key_reader a scribe for which the function `bool read(K&, Stream&, KeyReader&)` is defined.
 * \param value_reader a scribe for which the function `bool read(V&, Stream&, ValueReader&)` is defined.
 * \tparam Stream satisfies is_readable.
 */
template<typename K, typename V, typename Stream,
	typename KeyReader, typename ValueReader,
	typename std::enable_if<is_readable<Stream>::value>::type* = nullptr>
bool read(array_map<K, V>& map, Stream& in,
		KeyReader& key_reader, ValueReader& value_reader)
{
	size_t length;
	if (!read(length, in)) return false;

	map.size = 0;
	map.capacity = 1 << (core::log2(length == 0 ? 1 : length) + 1);
	map.keys = (K*) malloc(sizeof(K) * map.capacity);
	if (map.keys == NULL) return false;
	map.values = (V*) malloc(sizeof(V) * map.capacity);
	if (map.values == NULL) {
		free(map.keys);
		return false;
	}

	for (unsigned int i = 0; i < length; i++) {
		if (!read(map.keys[i], in, key_reader)) return false;
		if (!read(map.values[i], in, value_reader)) return false;
		map.size++;
	}
	return true;
}

/**
 * Reads a core::array_map structure `map` from `in`.
 * \param map an uninitialized core::array_map structure. This function
 * 		initializes `map`, and the caller is responsible for its memory and
 * 		must call free to release its memory resources.
 * \param key_reader a scribe for which the function `bool read(K&, Stream&, KeyReader&)` is defined.
 * \tparam Stream satisfies is_readable.
 */
template<typename K, typename V, typename Stream, typename KeyReader,
	typename std::enable_if<is_readable<Stream>::value>::type* = nullptr>
inline bool read(array_map<K, V>& map, Stream& in, KeyReader& key_reader) {
	default_scribe scribe;
	return read(map, in, key_reader, scribe);
}

/**
 * Reads a core::array_map structure `map` from `in`.
 * \param map an uninitialized core::array_map structure. This function
 * 		initializes `map`, and the caller is responsible for its memory and
 * 		must call free to release its memory resources.
 * \tparam Stream satisfies is_readable.
 */
template<typename K, typename V, typename Stream,
	typename std::enable_if<is_readable<Stream>::value>::type* = nullptr>
inline bool read(array_map<K, V>& map, Stream& in) {
	default_scribe scribe;
	return read(map, in, scribe, scribe);
}

/**
 * Writes the given core::array_map structure `map` to `out`.
 * \param key_writer a scribe for which the function `bool write(const K&, Stream&, KeyWriter&)` is defined.
 * \param value_writer a scribe for which the function `bool write(const V&, Stream&, ValueWriter&)` is defined.
 * \tparam Stream satisfies is_writeable.
 */
template<typename K, typename V, typename Stream,
	typename KeyWriter, typename ValueWriter,
	typename std::enable_if<is_writeable<Stream>::value>::type* = nullptr>
bool write(const array_map<K, V>& map, Stream& out,
		KeyWriter& key_writer, ValueWriter& value_writer)
{
	if (!write(map.size, out)) return false;
	for (unsigned int i = 0; i < map.size; i++) {
		if (!write(map.keys[i], out, key_writer)) return false;
		if (!write(map.values[i], out, value_writer)) return false;
	}
	return true;
}

/**
 * Writes the given core::array_map structure `map` to `out`.
 * \param key_writer a scribe for which the function `bool write(const K&, Stream&, KeyWriter&)` is defined.
 * \tparam Stream satisfies is_writeable.
 */
template<typename K, typename V, typename Stream, typename KeyWriter,
	typename std::enable_if<is_writeable<Stream>::value>::type* = nullptr>
inline bool write(const array_map<K, V>& map, Stream& out, KeyWriter& key_writer) {
	default_scribe scribe;
	return write(map, out, key_writer, scribe);
}

/**
 * Writes the given core::array_map structure `map` to `out`.
 * \tparam Stream satisfies is_writeable.
 */
template<typename K, typename V, typename Stream,
	typename std::enable_if<is_writeable<Stream>::value>::type* = nullptr>
inline bool write(const array_map<K, V>& map, Stream& out) {
	default_scribe scribe;
	return write(map, out, scribe, scribe);
}

/**
 * Reads a core::pair structure `p` from `stream` by calling `read(p.key, stream)` and `read(p.value, stream)`.
 * \tparam Stream satisfies is_readable.
 */
template<typename K, typename V, typename Stream,
	typename std::enable_if<is_readable<Stream>::value>::type* = nullptr>
inline bool read(pair<K, V>& p, Stream& stream)
{
	return read(p.key, stream) && read(p.value, stream);
}

/**
 * Writes the given core::pair structure `p` to `stream` by calling
 * `write(p.key, stream)` and `write(p.value, stream)`.
 * \tparam Stream satisfies is_writeable.
 */
template<typename K, typename V, typename Stream,
	typename std::enable_if<is_writeable<Stream>::value>::type* = nullptr>
inline bool write(const pair<K, V>& p, Stream& stream)
{
	return write(p.key, stream) && write(p.value, stream);
}

} /* namespace core */

#endif /* IO_H_ */
