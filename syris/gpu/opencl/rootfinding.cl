/*
 * Newton-Raphson and bisection combined polynomial root finder.
 *
 * Requires definition of vfloat data type, which defines single or double
 * precision for vfloating point numbers.
 */

/* Machine epsilon for single precision floats. */
#define EPSILON 1.19209e-07
#define POLY_DEG 4
#define MAX_ITERATIONS 30
#define is_close(a, b, c) (fabs((a) - (b)) < (c))
#define sgn(a) (((a) == 0) ? 0 : (((a) > 0) ? 1 : -1))
#define halve(a, b) (((a) + (b)) / 2.0)

void fill_nan(vfloat *array, int start, int end) {
	/*
	 * Fill a vfloat *array* with NANs in the interval <*start*, *end*).
	 */
	int i;

	for (i = start; i < end; i++) {
		array[i] = NAN;
	}
}

vfloat polynomial(const vfloat *coeffs, unsigned int degree, vfloat x) {
	/*
	 * Evaluate a polynomial with coefficients *coeffs* and *degree* at
	 * point *x*.
	 */
	vfloat res = coeffs[degree], pow_x = 1;
	int i;

	for (i = degree - 1; i >= 0; i--) {
		pow_x *= x;
		res += pow_x * coeffs[i];
	}

	return res;
}

vfloat derivative(const vfloat *coeffs, unsigned int degree, vfloat x) {
	/*
	 * Evaluate a polynomial's derivative with coefficients *coeffs*
	 * and *degree* at point *x*.
	 */
	if (degree == 0) {
		return 0;
	}
	vfloat res = coeffs[degree - 1], pow_x = 1;
	int i;

	for (i = degree - 2; i >= 0; i--) {
		pow_x *= x;
		res += (degree - i) * pow_x * coeffs[i];
	}

	return res;
}

void derivate(const vfloat *coeffs, vfloat *d_coeffs,
								unsigned int degree) {
	/*
	 * Get derivative of a polynomial specified by its coefficients
	 * *coeffs* and *degree*. Store the new coefficients in *d_coeffs*.
	 */
	int i;
	if (degree == 0) {
		d_coeffs[0] = 0;
		return;
	}

	for (i = 0; i < degree; i++) {
		d_coeffs[i] = (degree - i) * coeffs[i];
	}
}

vfloat get_prognose(const vfloat *coeffs, unsigned int degree,
											vfloat start) {
	/* The Newton-Raphson iteraion prognose:
	 *
	 * f(x_i+1) = f(x_i) - f(x_i) / f'(x_i).
	 */
	return start - polynomial(coeffs, degree, start) /
					derivative(coeffs, degree, start);
}

int is_extreme(const vfloat *previous_coeffs, const vfloat *coeffs,
						const vfloat *next_coeffs, unsigned int degree,
						vfloat point, vfloat left_end, vfloat right_end,
						vfloat eps) {
	/* Check if a *point* is an extremum. Make use of the fact that
	 * if the derivative sign changes in point - eps and points + eps,
	 * the point must be an extremum.
	 */
	const vfloat *left_coeffs = !isnan(previous_coeffs[0]) &&
					point - eps < left_end ? previous_coeffs : coeffs;
	const vfloat *right_coeffs = !isnan(next_coeffs[0]) &&
						point + eps > right_end ? next_coeffs : coeffs;

	return sgn(derivative(left_coeffs, degree, point - eps)) != \
			sgn(derivative(right_coeffs, degree, point + eps));
}

int in_interval(vfloat value, const vfloat2 *interval) {
	/* Check if x is in the left-opened and right-cloed interval:
	 *
	 * x in (left, right>.
	 */
	return interval->x < value && value <= interval->y;
}

vfloat bisect(const vfloat *coeffs, unsigned int degree,
								vfloat x_0, vfloat2 *interval) {
	vfloat x_1;

	if (sgn(polynomial(coeffs, degree, interval->x)) ==
			sgn(polynomial(coeffs, degree, x_0))) {
		x_1 = halve(x_0, interval->y);
		interval->x = x_0;
	} else {
		x_1 = halve(interval->x, x_0);
		interval->y = x_0;
	}

	return x_1;
}

vfloat find_root(const vfloat *coeffs, unsigned int degree,
						const vfloat2 *interval, vfloat eps) {
	/* Interval Newton-Raphson root finder combined with bisection.
	 * The function proceeds with NR iterations and when the next guessed
	 * point gets beyond the interval end points we switch to bisection
	 * and contiune with NR after the guess is in the given interval again.
	 */
	vfloat x_1;
	vfloat2 loc_interval = *interval;
	vfloat x_0 = halve(loc_interval.x, loc_interval.y);
	int i;

	for (i = 0; i < MAX_ITERATIONS; i++) {
		x_1 = get_prognose(coeffs, degree, x_0);

		if (!in_interval(x_1, &loc_interval)) {
			x_1 = bisect(coeffs, degree, x_0, &loc_interval);
		}

		if (is_close(x_0, x_1, eps) || isnan(x_1)) {
			break;
		}
		x_0 = x_1;
	}

	if (i == MAX_ITERATIONS) {
		x_1 = NAN;
	}

	return x_1;
}

vfloat get_interval_root(const vfloat *previous_coeffs,
		const vfloat *coeffs, const vfloat *next_coeffs,
		unsigned int degree, const vfloat2 *interval, vfloat left_end,
		vfloat right_end, vfloat eps) {
	/*
	 * Find a polynomial root in a given *interval*.
	 */
	vfloat result = NAN;

	vfloat left = polynomial(coeffs, degree, interval->x);
	vfloat right = polynomial(coeffs, degree, interval->y);

	if ((left < 0) ^ (right < 0)) {
		/* If both ends of the interval are below or above zero, there
		 * cannot be a root. */
		result = find_root(coeffs, degree, interval, eps);
	}

	if (!in_interval(result, interval)) {
		/* If the root lies on the left interval neglect it, because
		 * it was taken into account by previous interval. At the
		 * very first interval the root cannot lie on the left end
		 * because the influence interval of a metaball starts at
		 * zero, thus cannot have a root on its left end (the root)
		 * is located where f(x) = 1, the radius of the metaball. */
		result = NAN;
	}

	if (degree == POLY_DEG && is_extreme(previous_coeffs, coeffs,
			next_coeffs, degree, result, left_end, right_end, EPSILON)) {
		result = NAN;
	}

	return result;
}

vfloat get_linear_root(const vfloat *coeffs, vfloat eps) {
	/*
	 * Get the root of first order polynomial defined by *coeffs*.
	 */
	if (is_close(coeffs[0], 0, eps)) {
		return NAN;
	} else {
		return - coeffs[1] / coeffs[0];
	}
}

void get_interval_roots(const vfloat *previous_coeffs,
							const vfloat *coeffs, const vfloat *next_coeffs,
							unsigned int degree, vfloat left_end,
							vfloat right_end, vfloat *intervals,
							vfloat *results, vfloat eps) {
	/* Find roots in given *intervals*. */
	int i = 1, count = 0;
	vfloat2 interval;

	if (degree == 1) {
		results[0] = get_linear_root(coeffs, eps);
	} else {
		while (!isnan(intervals[i]) && i < degree + 1) {
			interval = (vfloat2)(intervals[i - 1], intervals[i]);
			if (!is_close(interval.x, interval.y, EPSILON)) {
				results[count] = get_interval_root(previous_coeffs, coeffs,
					next_coeffs, degree, &interval, left_end, right_end, eps);
				count++;
			}
			i++;
		}

		if (count < degree && !isnan(next_coeffs[0]) &&
				sgn(polynomial(coeffs, degree, intervals[i - 1])) !=
				sgn(polynomial(next_coeffs, degree, intervals[i - 1]))) {
			/* Non-continuous function, compensate for the fact that
			 * at right end the two consecutive polynomials do not go to
			 * zero, but have different signs, which means that somewhere
			 * inbetween there must be a root, assume that to be exactly
			 * in the right end point. */
			results[count] = intervals[i - 1];
			count++;
		}

		/* Fill the rest with NAN.*/
		if (count < degree + 1) {
			fill_nan(results, count, degree + 1);
		}
	}
}

void split_intervals(vfloat left_end, vfloat right_end,
		const vfloat *results, vfloat *intervals, unsigned int degree) {
	/* A polynomial of a degree n can have maximum n roots. Append
	 * the original interval and sort the array. */
	int i;
	const vfloat2 interval = (vfloat2)(left_end, right_end);

	for (i = 0; i < degree; i++) {
		if (results[i] <  left_end) {
			intervals[i] = left_end;
		} else if (results[i] > right_end) {
			intervals[i] = right_end;
		} else {
			intervals[i] = results[i];
		}
	}

	intervals[degree] = left_end;
	intervals[degree + 1] = right_end;

	vf_sort(intervals, POLY_DEG + 1);
}

void get_roots(const vfloat *previous_coeffs, const vfloat *coeffs,
				const vfloat *next_coeffs, unsigned int degree,
				vfloat left_end, vfloat right_end,
				vfloat *results, vfloat eps) {
	/* Get polynomial roots in an interval bounded by *left_end* and
	 * *right_end*. *previous_coeffs* are the polynomial coefficients
	 * in the previous interval, *coeffs* in the current one and
	 * *next_coeffs* in the following interval. The previous and next
	 * coefficients are needed for boundary root treatment. The roots
	 * are stored in *results* which must be of size degree + 1.
	 * A polynomial of degree d can have maximum d roots, the + 1
	 * is needed for non-continuous interval transitions introduced
	 * by errors in estimating the polynomials. In the case of
	 * non-continuous transition, there can be d roots and one more
	 * if the polynomial in the next interval has different sign in
	 * the right end point than the current polynomial. That is,
	 * sgn(f(right)) != sgn(g(right)),
	 * where f and g are current and the next coefficients. *eps*
	 * determines the precision of the root finding procedure.
	 */
	vfloat intervals[POLY_DEG + 1];
	vfloat d_coeffs[POLY_DEG][POLY_DEG + 1];
	const vfloat empty[] = {NAN};
	int i;

	for (i = 0; i < POLY_DEG + 1; i++) {
		results[i] = NAN;
		intervals[i] = NAN;
	}

	switch(degree) {
	case 0: break;
	case 1: results[0] = get_linear_root((const vfloat *)coeffs, eps); break;
	default:
		intervals[0] = left_end;
		intervals[1] = right_end;
		derivate(coeffs, d_coeffs[0], degree);
		for (i = 1; i < degree - 1; i++) {
			derivate(d_coeffs[i - 1], d_coeffs[i], degree - i);
		}

		/* Find stationary points for polynomial derivatives. */
		for (i = degree - 2; i >= 0; i--) {
			get_interval_roots(empty, (const vfloat *)d_coeffs[i], empty,
					degree - 1 - i, left_end, right_end, intervals,
					results, eps);
			split_intervals(left_end, right_end, (const vfloat *)results,
								intervals, degree - 1 - i);
		}

		/* Calculate roots of f in the intervals given by its stationary
		 * points. */
		get_interval_roots(previous_coeffs, coeffs, next_coeffs, degree,
								left_end, right_end, intervals, results, eps);
		vf_sort(results, degree + 1);
		break;
	}
}


/* Testing and helper functions. */

void _copy_global_local(__global vfloat *gl_array, vfloat *loc_array,
							unsigned int size, bool gl_to_loc) {
	int i;

	for (i = 0; i < size; i++) {
		if (gl_to_loc) {
			loc_array[i] = gl_array[i];
		} else {
			gl_array[i] = loc_array[i];
		}
	}
}

__kernel void polynomial_eval_kernel(__global vfloat *out,
								__global vfloat *coeffs,
								const int degree,
								const vfloat x) {
	vfloat l_coeffs[POLY_DEG + 1];
	_copy_global_local(coeffs, l_coeffs, POLY_DEG + 1, true);

	out[0] = polynomial(l_coeffs, degree, x);
}

__kernel void derivative_eval_kernel(__global vfloat *out,
								__global vfloat *coeffs,
								const int degree,
								const vfloat x) {
	vfloat l_coeffs[POLY_DEG + 1];
	_copy_global_local(coeffs, l_coeffs, POLY_DEG + 1, true);

	out[0] = derivative(l_coeffs, degree, x);
}

__kernel void derivate_kernel(__global vfloat *out,
								__global vfloat *coeffs,
								const int degree) {
	vfloat l_coeffs[POLY_DEG + 1];
	vfloat d_coeffs[POLY_DEG + 1];

	_copy_global_local(coeffs, l_coeffs, POLY_DEG + 1, true);
	derivate(l_coeffs, d_coeffs, degree);
	_copy_global_local(out, d_coeffs, POLY_DEG + 1, false);
}

__kernel void roots_kernel(__global vfloat *out,
					__constant vfloat *previous_coeffs,
					__constant vfloat *coeffs,
					__constant vfloat *next_coeffs,
					const vfloat2 interval,
					const vfloat eps) {
	vfloat results[POLY_DEG + 1];
	vfloat l_previous_coeffs[POLY_DEG + 1],
			l_coeffs[POLY_DEG + 1],
			l_next_coeffs[POLY_DEG + 1];
	int i;

	for (i = 0; i < POLY_DEG + 1; i++) {
		l_previous_coeffs[i] = previous_coeffs[i];
		l_coeffs[i] = coeffs[i];
		l_next_coeffs[i] = next_coeffs[i];
	}
	get_roots(l_previous_coeffs, l_coeffs, l_next_coeffs, POLY_DEG,
			interval.x, interval.y, results, eps);

	_copy_global_local(out, results, POLY_DEG + 1, false);
}
