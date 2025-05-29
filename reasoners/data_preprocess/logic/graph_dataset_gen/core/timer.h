/**
 * \file timer.h
 *
 * <!-- Author: asaparov -->
 */

#ifndef TIMER_H_
#define TIMER_H_

#if defined(_WIN32) /* on Windows */
#define _WINSOCKAPI_
#include <Windows.h>
#undef max
#undef min
#elif defined(__APPLE__) /* on Mac */
#include <mach/mach_time.h>
#else /* on Linux */
#include <time.h>
#endif


namespace core {


#if defined(__APPLE__)
struct timebase_info {
	mach_timebase_info_data_t info;

	timebase_info() {
		info = {0, 0};
		mach_timebase_info(&info);
	}
};

inline timebase_info& get_timebase() {
	static timebase_info timebase;
	return timebase;
}
#endif


/**
 * A simple structure that measures time.
 */
struct timer
{
#if defined(_WIN32) /* on Windows */
	ULONGLONG start_time;
#elif defined(__APPLE__) /* on Mac */
	uint64_t start_time;
#else /* on Linux */
	timespec start_time;
#endif

	/**
	 * Constructor that starts the timer.
	 */
	timer() {
		start();
	}

	/**
	 * Starts the timer.
	 */
	inline void start() {
#if defined(_WIN32) /* on Windows */
		start_time = GetTickCount64();
#elif defined(__APPLE__) /* on Mac */
		start_time = mach_absolute_time();
#else /* on Linux */
		clock_gettime(CLOCK_MONOTONIC, &start_time);
#endif
	}

	/**
	 * Returns the number of milliseconds elapsed since the timer was last started.
	 */
	inline unsigned long long milliseconds() {
#if defined(_WIN32) /* on Windows */
		ULONGLONG end_time = GetTickCount64();
		return (unsigned long long) (end_time - start_time);
#elif defined(__APPLE__) /* on Mac */
		uint64_t end_time = mach_absolute_time();
		return (end_time - start_time) * get_timebase().info.numer / get_timebase().info.denom / 1000000;
#else /* on Linux */
		timespec end_time;
		clock_gettime(CLOCK_MONOTONIC, &end_time);
		return (unsigned long long) (end_time.tv_sec - start_time.tv_sec) * 1000 + (end_time.tv_nsec - start_time.tv_nsec) / 1000000;
#endif
	}

	/**
	 * Returns the number of nanoseconds elapsed since the timer was last started.
	 */
	inline double nanoseconds() {
#if defined(_WIN32) /* on Windows */
		ULONGLONG end_time = GetTickCount64();
		return (end_time - start_time) * 1.0e6;
#elif defined(__APPLE__) /* on Mac */
		uint64_t end_time = mach_absolute_time();
		return (double) (end_time - start_time) * get_timebase().info.numer / get_timebase().info.denom;
#else /* on Linux */
		timespec end_time;
		clock_gettime(CLOCK_MONOTONIC, &end_time);
		return (end_time.tv_sec - start_time.tv_sec) * 1.0e9 + (end_time.tv_nsec - start_time.tv_nsec);
#endif
	}
};

/**
 * On Windows and Mac, this returns the number of milliseconds elapsed since
 * the system was started. On Linux, it returns the system's best estimate of
 * the current time of day.
 */
inline unsigned long long milliseconds() {
#if defined(_WIN32) /* on Windows */
	return GetTickCount64();
#elif defined(__APPLE__) /* on Mac */
	return mach_absolute_time() * get_timebase().info.numer / get_timebase().info.denom / 1000000;
#else /* on Linux */
	timespec time;
	clock_gettime(CLOCK_REALTIME, &time);
	return (unsigned long long) time.tv_sec * 1000 + time.tv_nsec / 1000000;
#endif
}


} /* namespace core */

#endif /* TIMER_H_ */
