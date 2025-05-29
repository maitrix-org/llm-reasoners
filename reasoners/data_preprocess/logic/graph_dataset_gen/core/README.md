Documentation and code examples are available at <https://asaparov.org/docs/core>. The repository is located at <https://github.com/asaparov/core>.

This repository is a small collection of general-purpose data structures and algorithms. Originally, I developed this library to explore and tinker with different programming paradigms and design patterns. Since then, the code has evolved into a collection of generic data structures, using [templates](http://en.cppreference.com/w/cpp/language/templates) for polymorphism.

### Usage and Dependencies

To use the code, simply download the files into a folder named "core". Add this folder to the include path for the compiler, for example by using the `-I` flag.

The code makes use of the [xxhash](https://github.com/Cyan4973/xxHash) library as its default hash function implementation, but the required files are included in the repository. Otherwise, there are no dependencies on external libraries. The code makes use of `C++11` and is regularly tested with `gcc 11` but I have previously compiled it with `gcc 4.8`, `clang 4.0`, and `Microsoft Visual C++ 14.0 (2015)`. The code is intended to be platform-independent, so please create an issue if there are any compilation bugs.

### Overview

The library currently implements the following data structures:
 - [array](https://asaparov.org/docs/core/array.h.html#struct%20array) is a self-expanding sequential array,
 - [hash_set](https://asaparov.org/docs/core/map.h.html#struct%20hash_set) is an unordered associative set of unique elements, implemented using a hashtable,
 - [hash_map](https://asaparov.org/docs/core/map.h.html#struct%20hash_map) is an unordered associative map of key-value pairs, where the keys are unique, implemented using a hashtable,
 - [array_map](https://asaparov.org/docs/core/map.h.html#struct%20array_map) is a sequential map of key-value pairs, where the keys are unique.

In addition, the library implements a handful of sorting algorithms in [array.h](https://asaparov.org/docs/core/array.h.html). Set operations, such as union, intersection, and subtraction, can also be found in [array.h](https://asaparov.org/docs/core/array.h.html), which operate on sets represented as sorted arrays. The library implements serialization/deserialization with the functions `read`, `write`, and `print` for all data structures using a very regular pattern involving ["scribes"](https://asaparov.org/docs/core/io.h.html#scribes) as described in [io.h](https://asaparov.org/docs/core/io.h.html). Functions and structures useful for lexical analysis are provided in [lex.h](https://asaparov.org/docs/core/lex.h.html). [random.h](https://asaparov.org/docs/core/random.h.html) provides pseudo-random number generation and sampling from a handful of built-in distributions.

The implementations of every data structure are transparent, with `public` visibility for the underlying fields, which gives the user more control over data structure behavior. All procedures and structures do *not* use exceptions. Rather, errors are returned, typically as a `bool` type, where `true` indicates success and `false` indicates failure. I observed, in practice, that the overhead of this kind of error handling is negligible when no errors are thrown, and the overhead is much smaller than exception handling when errors do occur. The library does not provide automatic memory management, and expects the user to manage memory appropriately. The `free` function is implemented for all data structures, but it does not automatically deallocate child objects in the container classes. The library does not use any thread synchronization methods to guarantee thread safety, and the user must implement their own synchronization or avoid simultaneous writes to data structures.
