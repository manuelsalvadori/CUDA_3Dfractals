#ifndef __MATH_UTIL__
#define __MATH_UTIL__

// Should be much more precise with large b
inline double __host__ __device__ fastPrecisePow(double a, double b) {
	union {
		double d;
		int x[2];
	} u = { a };
	u.x[1] = (int)(b * (u.x[1] - 1072632447) + 1072632447);
	u.x[0] = 0;
	return u.d;
}


#endif /*__MATH_UTIL__*/