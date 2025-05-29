/**
 * \file lex.h
 *
 * This file implements common functionality for lexical analysis, such as
 * string comparison, tokenization (splitting strings by whitespace), and
 * parsing arithmetic types from strings.
 *
 * <!-- Created on: Jan 6, 2015
 *          Author: asaparov -->
 */

#ifndef LEX_H_
#define LEX_H_

#include "map.h"
#include "utility.h"

#include <ctype.h>

#if defined(__APPLE__)
#include <cwchar>

inline size_t c32rtomb(char* s, char32_t wc, mbstate_t* ps) {
	return wcrtomb(s, (wchar_t) wc, ps);
}

#else /* __APPLE__ */
#include <cuchar>
#endif /* __APPLE__ */


namespace core {

/**
 * Compares the strings given by the array<char> structure `first` and the
 * null-terminated C string `second`.
 * \returns `true` if the strings are equivalent, and `false` otherwise.
 */
/* TODO: test with benchmarks whether we should inline these functions */
inline bool compare_strings(const array<char>& first, const char* second) {
	for (unsigned int i = 0; i < first.length; i++) {
		if (first[i] != second[i])
			return false;
	}
	if (second[first.length] != '\0')
		return false;
	return true;
}

/**
 * Compares the strings given by the core::string `first` and the native char
 * array `second` whose length is given by `second_length`.
 * \returns `true` if the strings are equivalent, and `false` otherwise.
 */
inline bool compare_strings(const string& first, const char* second, unsigned int second_length) {
	if (first.length != second_length)
		return false;
	for (unsigned int i = 0; i < first.length; i++) {
		if (first[i] != second[i])
			return false;
	}
	return true;
}

/**
 * Tokenizes the given native char array `str` with length `length`, assigning
 * to each unique token an `unsigned int` identifier. These identifiers are
 * stored in the core::hash_map `names`. The tokenized identifiers are added to
 * the core::array `tokens`.
 */
bool tokenize(const char* str, unsigned int length,
		array<unsigned int>& tokens, hash_map<string, unsigned int>& names)
{
	bool whitespace = true;
	unsigned int token_start = 0;
	for (unsigned int i = 0; i < length; i++) {
		if (whitespace) {
			if (!isspace(str[i])) {
				token_start = i;
				whitespace = false;
			}
		} else {
			if (isspace(str[i])) {
				unsigned int id;
				if (!get_token(string(str + token_start, i - token_start), id, names)
				 || !tokens.add(id))
					return false;
				whitespace = true;
			}
		}
	}

	if (!whitespace) {
		unsigned int id;
		if (!get_token(string(str + token_start, length - token_start), id, names)
		 || !tokens.add(id))
			return false;
	}
	return true;
}

/**
 * Attempts to parse the string given by `token` as a `double`.
 * \tparam CharArray a string type that implements two fields: (1) `data` which
 * 		returns the underlying `char*` array, and (2) `length` which returns
 * 		the length of the string.
 * \returns `true` if successful, or `false` if there is insufficient memory or
 * 		`token` is not an appropriate string representation of a floating-point
 * 		number.
 */
template<typename CharArray>
inline bool parse_float(const CharArray& token, double& value) {
	char* buffer = (char*) malloc(sizeof(char) * (token.length + 1));
	if (buffer == NULL) {
		fprintf(stderr, "parse_float ERROR: Unable to allocate temporary string buffer.\n");
		return false;
	}
	memcpy(buffer, token.data, sizeof(char) * token.length);
	buffer[token.length] = '\0';

	char* end_ptr;
	value = strtod(buffer, &end_ptr);
	if (*end_ptr != '\0') {
		free(buffer);
		return false;
	}
	free(buffer);
	return true;
}

/**
 * Attempts to parse the string given by `token` as an `unsigned int`.
 * \tparam CharArray a string type that implements two fields: (1) `data` which
 * 		returns the underlying `char*` array, and (2) `length` which returns
 * 		the length of the string.
 * \returns `true` if successful, or `false` if there is insufficient memory or
 * 		`token` is not an appropriate string representation of a unsigned
 * 		integer.
 */
template<typename CharArray>
inline bool parse_uint(const CharArray& token, unsigned int& value, unsigned int base = 0) {
	char* buffer = (char*) malloc(sizeof(char) * (token.length + 1));
	if (buffer == NULL) {
		fprintf(stderr, "parse_uint ERROR: Unable to allocate temporary string buffer.\n");
		return false;
	}
	memcpy(buffer, token.data, sizeof(char) * token.length);
	buffer[token.length] = '\0';

	char* end_ptr;
	value = strtoul(buffer, &end_ptr, base);
	if (*end_ptr != '\0') {
		free(buffer);
		return false;
	}
	free(buffer);
	return true;
}

/**
 * Attempts to parse the string given by `token` as an `unsigned int`.
 * \tparam CharArray a string type that implements two fields: (1) `data` which
 * 		returns the underlying `char*` array, and (2) `length` which returns
 * 		the length of the string.
 * \returns `true` if successful, or `false` if there is insufficient memory or
 * 		`token` is not an appropriate string representation of a unsigned
 * 		integer.
 */
template<typename CharArray>
inline bool parse_ulonglong(const CharArray& token, unsigned long long& value) {
	char* buffer = (char*) malloc(sizeof(char) * (token.length + 1));
	if (buffer == NULL) {
		fprintf(stderr, "parse_ulonglong ERROR: Unable to allocate temporary string buffer.\n");
		return false;
	}
	memcpy(buffer, token.data, sizeof(char) * token.length);
	buffer[token.length] = '\0';

	char* end_ptr;
	value = strtoull(buffer, &end_ptr, 0);
	if (*end_ptr != '\0') {
		free(buffer);
		return false;
	}
	free(buffer);
	return true;
}

/**
 * Attempts to parse the string given by `token` as an `int`.
 * \tparam CharArray a string type that implements two fields: (1) `data` which
 * 		returns the underlying `char*` array, and (2) `length` which returns
 * 		the length of the string.
 * \returns `true` if successful, or `false` if there is insufficient memory or
 * 		`token` is not an appropriate string representation of a integer.
 */
template<typename CharArray>
inline bool parse_int(const CharArray& token, int& value) {
	char* buffer = (char*) malloc(sizeof(char) * (token.length + 1));
	if (buffer == NULL) {
		fprintf(stderr, "parse_int ERROR: Unable to allocate temporary string buffer.\n");
		return false;
	}
	memcpy(buffer, token.data, sizeof(char) * token.length);
	buffer[token.length] = '\0';

	char* end_ptr;
	value = strtol(buffer, &end_ptr, 0);
	if (*end_ptr != '\0') {
		free(buffer);
		return false;
	}
	free(buffer);
	return true;
}

/**
 * Attempts to parse the string given by `token` as a `long`.
 * \tparam CharArray a string type that implements two fields: (1) `data` which
 * 		returns the underlying `char*` array, and (2) `length` which returns
 * 		the length of the string.
 * \returns `true` if successful, or `false` if there is insufficient memory or
 * 		`token` is not an appropriate string representation of a long.
 */
template<typename CharArray>
inline bool parse_long(const CharArray& token, long& value) {
	char* buffer = (char*) malloc(sizeof(char) * (token.length + 1));
	if (buffer == NULL) {
		fprintf(stderr, "parse_longlong ERROR: Unable to allocate temporary string buffer.\n");
		return false;
	}
	memcpy(buffer, token.data, sizeof(char) * token.length);
	buffer[token.length] = '\0';

	char* end_ptr;
	value = strtol(buffer, &end_ptr, 0);
	if (*end_ptr != '\0') {
		free(buffer);
		return false;
	}
	free(buffer);
	return true;
}

/**
 * Attempts to parse the string given by `token` as a `long`.
 * \tparam CharArray a string type that implements two fields: (1) `data` which
 * 		returns the underlying `char*` array, and (2) `length` which returns
 * 		the length of the string.
 * \returns `true` if successful, or `false` if there is insufficient memory or
 * 		`token` is not an appropriate string representation of a long.
 */
template<typename CharArray>
inline bool parse_long_long(const CharArray& token, long long& value) {
	char* buffer = (char*)malloc(sizeof(char) * (token.length + 1));
	if (buffer == NULL) {
		fprintf(stderr, "parse_longlong ERROR: Unable to allocate temporary string buffer.\n");
		return false;
	}
	memcpy(buffer, token.data, sizeof(char) * token.length);
	buffer[token.length] = '\0';

	char* end_ptr;
	value = strtoll(buffer, &end_ptr, 0);
	if (*end_ptr != '\0') {
		free(buffer);
		return false;
	}
	free(buffer);
	return true;
}

/**
 * Attempts to parse the string given by `token` as an `unsigned int`.
 * \param base if `0`, the numeric base of the integer is detected
 * 		automatically in the same way as [strtoul](http://en.cppreference.com/w/cpp/string/byte/strtoul).
 * 		Otherwise, the numeric base can be specified explicitly.
 * \returns `true` if successful, or `false` if `token` is not an appropriate
 * 		string representation of an unsigned integer.
 */
template<size_t N>
inline bool parse_uint(const char (&token)[N], unsigned int& value, unsigned int base = 0) {
	char buffer[N + 1];
	for (unsigned int i = 0; i < N; i++)
		buffer[i] = token[i];
	buffer[N] = '\0';

	char* end_ptr;
	value = strtoul(buffer, &end_ptr, base);
	return (*end_ptr == '\0');
}

/**
 * Represents a position in a file. This structure is typically used to provide
 * informative errors during lexical analysis of data from a file.
 */
struct position {
	/**
	 * The line number of the position in the file.
	 */
	unsigned int line;

	/**
	 * The column number of the position in the file.
	 */
	unsigned int column;

	/**
	 * Constructs the position structure with the given `line` and `column`.
	 */
	position(unsigned int line, unsigned int column) :
		line(line), column(column) { }

	/**
	 * Constructs the position structure by copying from `p`.
	 */
	position(const position& p) :
		line(p.line), column(p.column) { }

	/**
	 * Returns a position with the column number increased by `i`.
	 */
	position operator + (unsigned int i) const {
		return position(line, column + i);
	}

	/**
	 * Returns a position with the column number decreased by `i`.
	 */
	position operator - (unsigned int i) const {
		return position(line, column - i);
	}

	/**
	 * Copies the position structure from `src` into `dst`.
	 */
	static inline bool copy(const position& src, position& dst) {
		dst.line = src.line;
		dst.column = src.column;
		return true;
	}
};

/**
 * A structure representing a single token during lexical analysis. This
 * structure is generic, intended for use across multiple lexical analyzers.
 */
template<typename TokenType>
struct lexical_token {
	/**
	 * The generic type of this token.
	 */
	TokenType type;

	/**
	 * The start position (inclusive) of the token in the source file.
	 */
	position start;

	/**
	 * The end position (exclusive) of the token in the source file.
	 */
	position end;

	/**
	 * An (optional) string representing the contents of the token.
	 */
	string text;
};

/**
 * Prints the given lexical_token `token` to the output `stream`.
 * \tparam Printer a scribe type for which the functions
 * 		`print(const TokenType&, Stream&, Printer&)` and
 * 		`print(const core::string& s, Stream&, Printer&)` are defined.
 */
template<typename TokenType, typename Stream, typename Printer>
bool print(const lexical_token<TokenType>& token, Stream& stream, Printer& printer) {
	bool success = true;
	success &= print(token.type, stream, printer);
	if (!is_empty(token.text)) {
		success &= print('(', stream);
		success &= print(token.text, stream, printer);
		success &= print(')', stream);
	}
	return success;
}

/**
 * Reports an error with the given message `error` as a null-terminated C
 * string at the given source file position `pos` to [stderr](http://en.cppreference.com/w/cpp/io/c).
 */
inline void read_error(const char* error, const position& pos) {
	fprintf(stderr, "ERROR at %d:%d: %s.\n", pos.line, pos.column, error);
}

/**
 * Constructs a lexical_token with the given `start` and `end` positions, and
 * TokenType `type`, with an empty lexical_token::text message and appends it
 * to the `tokens` array.
 */
template<typename TokenType>
bool emit_token(array<lexical_token<TokenType>>& tokens,
	const position& start, const position& end, TokenType type)
{
	if (!tokens.ensure_capacity(tokens.length + 1)) {
		fprintf(stderr, "emit_token ERROR: Unable to create token.\n");
		return false;
	}

	lexical_token<TokenType>& new_token = tokens[(unsigned int) tokens.length];
	new_token.text.data = NULL;
	new_token.type = type;
	new_token.start = start;
	new_token.end = end;
	tokens.length++;
	return true;
}

/**
 * Constructs a lexical_token with the given `start` and `end` positions, and
 * TokenType `type`, with lexical_token::text copied from `token` and appends
 * it to the `tokens` array.
 */
template<typename TokenType>
bool emit_token(
	array<lexical_token<TokenType>>& tokens, array<char>& token,
	const position& start, const position& end, TokenType type)
{
	if (!tokens.ensure_capacity(tokens.length + 1)) {
		fprintf(stderr, "emit_token ERROR: Unable to create token.\n");
		return false;
	}

	lexical_token<TokenType>& new_token = tokens[(unsigned int) tokens.length];
	if (!init(new_token.text, token.data, (unsigned int) token.length)) {
		fprintf(stderr, "emit_token ERROR: Unable to create string.\n");
		return false;
	}
	new_token.type = type;
	new_token.start = start;
	new_token.end = end;
	tokens.length++;
	token.clear();
	return true;
}

/**
 * Frees every element in the given `tokens` array. This function does not free
 * the array itself.
 */
template<typename TokenType>
void free_tokens(array<lexical_token<TokenType>>& tokens) {
	for (unsigned int i = 0; i < tokens.length; i++)
		if (tokens[i].text.data != NULL)
			core::free(tokens[i].text);
}

/**
 * Inspects the element at the given `index` in the `tokens` array. If `index`
 * is not out of bounds, and the token at that index has type that matches the
 * given `type`, the function returns `true`. Otherwise, an error message is
 * printed to [stderr](http://en.cppreference.com/w/cpp/io/c) indicating that
 * the expected token was missing, with its `name` as part of the error
 * message, and `false` is returned.
 */
template<typename TokenType>
bool expect_token(const array<lexical_token<TokenType>>& tokens,
	const unsigned int& index, TokenType type, const char* name)
{
	FILE* out = stderr;
	if (index == tokens.length) {
		/* unexpected end of input */
		fprintf(out, "ERROR: Unexpected end of input. Expected %s.\n", name);
		return false;
	}
	else if (tokens[index].type != type) {
		fprintf(out, "ERROR at %d:%d: Unexpected token ", tokens[index].start.line, tokens[index].start.column);
		print(tokens[index].type, out);
		fprintf(out, ". Expected %s.\n", name);
		return false;
	}
	return true;
}

/**
 * Appends the given wide character `next` to the char array `token` which
 * represents a multi-byte string.
 */
inline bool append_to_token(
	array<char>& token, char32_t next, mbstate_t& shift)
{
	if (!token.ensure_capacity(token.length + MB_CUR_MAX))
		return false;
	size_t written = c32rtomb(token.data + token.length, next, &shift);
	if (written == static_cast<size_t>(-1))
		return false;
	token.length += written;
	return true;
}

} /* namespace core */

#endif /* LEX_H_ */
