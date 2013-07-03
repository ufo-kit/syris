/*
 * Bisection and Newton-Raphson combination for root finding.
 *
 * Currently only 2 and 4-order polynomials are supported.
 *
 * Requires definition of vfloat data type, which defines single or double
 * precision for floating point numbers.
 */

#define ZERO_EPS 1e-21f
#define EPS 1.0e-7f
#define MAX_ITERATIONS 20

typedef enum {
	LEFT, RIGHT
} direction;

vfloat polynomial(const vfloat *p, const int degree, vfloat x) {
	vfloat x_2, res;

	switch (degree) {
		case 2: res = p[0]*x*x + p[1]*x + p[2]; break;
		case 4:
			x_2 = x*x;
			res = p[0]*x_2*x_2 + p[1]*x_2*x + p[2]*x_2 + p[3]*x + p[4];
			break;
		default: res = NAN; break;
	}

	return res;
}

vfloat derivative(const vfloat *p, const int degree, vfloat x) {
	vfloat x_2, res;

	switch (degree) {
		case 2: res = 2*p[0]*x + p[1]; break;
		case 4:
			x_2 = x*x;
			res = 4*p[0]*x_2*x + 3*p[1]*x_2 + 2*p[2]*x + p[3];
			break;
		default: res = NAN; break;
	}

	return res;
}

vfloat halve(const vfloat2 *inter) {
	return (inter->x + inter->y)/2.0f;
}

int sgn(vfloat x) {
	if (-EPS < x && x < EPS) {
		return 0;
	} else {
		if (x < 0.0f) {
			return -1;
		} else {
			return 1;
		}
	}
}

int is_concave(const vfloat *p, const int degree, vfloat2 *inter, vfloat2 *px) {
	return 2*polynomial(p, degree, halve(inter)) > px->x + px->y;
}

void bisect(const vfloat *p, const int degree, vfloat2 *inter, vfloat2 *px) {
	vfloat xm = halve(inter);
	vfloat pxm = polynomial(p, degree, xm);

	if (-EPS < pxm && pxm < EPS) {
		/* we hit the root */
		inter->x = xm;
		inter->y = NAN;
		px->x = pxm;
		px->y = NAN;
	} else if (sgn(pxm) == sgn(px->x)) {
		inter->x = xm;
		px->x = pxm;
	} else {
		inter->y = xm;
		px->y = pxm;
	}
}

vfloat find_next_same_sgn(const vfloat *p,
						const int degree,
						const vfloat2 *inter,
						int sgn_trig,
						vfloat *pxm_res,
						direction d) {
	vfloat2 tmp_inter;
	vfloat xm = halve(inter);
	vfloat pxm = polynomial(p, degree, xm);
	int sxm = sgn(pxm), i = 0;

	if (d == RIGHT) {
		while (sxm != sgn_trig && sxm != 0) {
			tmp_inter.x = inter->x;
			tmp_inter.y = xm;
			xm = halve(&tmp_inter);
			pxm = polynomial(p, degree, xm);
			sxm = sgn(pxm);
			if (i > MAX_ITERATIONS) {
				*pxm_res = pxm;
				return xm;
			}
			i++;
		}
	} else {
		while (sxm != sgn_trig && sxm != 0) {
			tmp_inter.x = xm;
			tmp_inter.y = inter->y;
			xm = halve(&tmp_inter);
			pxm = polynomial(p, degree, xm);
			sxm = sgn(pxm);
			if (i > MAX_ITERATIONS) {
				*pxm_res = pxm;
				return xm;
			}
			i++;
		}
	}

	/* Set the middle point evaluation to the new value. */
	*pxm_res = pxm;

	return xm;
}

vfloat get_xn(const vfloat *p, const int degree, vfloat x0, vfloat px0) {
	vfloat der = derivative(p, degree, x0);

	if (-EPS < der && der < EPS) {
		return NAN;
	} else {
		return x0 - px0/derivative(p, degree, x0);
	}
}

vfloat newton_raphson(const vfloat *p, const int degree, vfloat x0) {
	vfloat px0 = polynomial(p, degree, x0);
	vfloat x1 = get_xn(p, degree, x0, px0);
	int i = 1;

	while (fabs(x1 - x0) > EPS && !isnan(x1)) {
		x0 = x1;
		px0 = polynomial(p, degree, x0);
		x1 = get_xn(p, degree, x0, px0);
		if (i > MAX_ITERATIONS) {
			/* Maximum iterations limit reached. We will have to live with as
			 * precise result as we get by that many iterations.
			 */
			break;
		}
		i++;
	}

	return x1;
}

void converging_interval1(const vfloat *p, const int degree,
								vfloat2 *inter, vfloat2 px) {
	vfloat xn0, xn1;
	int i = 0;

	if (sgn(px.x) == 0) {
		inter->y = NAN;
		return;
	}

	if (sgn(px.y) == 0) {
		inter->x = inter->y;
		inter->y = NAN;
		return;
	}

	if (sgn(px.x) == sgn(px.y)) {
		/* Both end points lie "above" or "below" the function. */
		inter->x = NAN;
		inter->y = NAN;
		return;
	}

	xn0 = get_xn(p, degree, inter->x, px.x);
	xn1 = get_xn(p, degree, inter->y, px.y);
	/* The NAN checks are necessary for the Newton-Raphson iteration, because
	it does not converge if the derivative is zero at any point between the
	convergence end points. */
	while (xn0 < inter->x || xn0 > inter->y ||
			xn1 < inter->x || xn1 > inter->y ||
			isnan(xn0) || isnan(xn1)) {
		bisect(p, degree, inter, &px);
		if (isnan(inter->y)) {
			/* root hit propagation */
			return;
		}
		/* Make sure that the next Newton-Raphson iteration steps lead to the
		 * same interval, this way the convergence of the Newton-Raphson
		 * iteration is guaranteed. */
		xn0 = get_xn(p, degree, inter->x, px.x);
		xn1 = get_xn(p, degree, inter->y, px.y);
		if (i > MAX_ITERATIONS) {
			break;
		}
		i++;
	}
}

int roots_exist(int concave, vfloat sx0, vfloat sx1) {
	if (concave) {
		if (sx0 > 0 && sx1 > 0) {
			/* Both end points lie "above" the function. */
			return 0;
		}
	} else if (sx0 < 0 && sx1 < 0) {
		/* Both lie "below" the function. */
		return 0;
	}

	return 1;
}

void process_different_heights(const vfloat *p,
								const int degree,
								vfloat2 *inter,
								int concave,
								vfloat2 *px,
								vfloat4 *res) {
	vfloat sdx0, sdx1, xm, pxm;
	int sgn_trig;

	if (sgn(px->x) != 0 && sgn(px->y) != 0) {
		/* The end points lie in different half planes divided by zero,
		 * thus there is just one intersection between them. This follows
		 * from the fact that there is only one stationary point.
		 */
		converging_interval1(p, degree, inter, *px);
		res->x = inter->x;
		res->y = inter->y;
		res->z = NAN;
		res->w = NAN;
		return;
	} else {
        /* One of the end points is root, we need to take care that this
         * root is included in the result. First determine the sgn(f(x))
         * we are looking for, if the function is concave it means that
         * one of the endpoints is "below" the stationary point, thus the
         * sign we are looking for is 1, -1 otherwise. By getting the point
         * xm where sgn(f(xm)) = sgn(f(x)) we determine the start or end
         * of the interval which needs to be checked for the other root.
         */
		sdx0 = sgn(derivative(p, degree, inter->x));
		sdx1 = sgn(derivative(p, degree, inter->y));
		/* Make sure we do not have zero derivatives in the next section. */
		if (concave) {
			sgn_trig = 1;
			if (sdx0 == 0) {
				sdx0 = 1;
			}
			if (sdx1 == 0) {
				sdx1 = -1;
			}
		} else {
			sgn_trig = -1;
			if (sdx0 == 0) {
				sdx0 = -1;
			}
			if (sdx1 == 0) {
				sdx1 = 1;
			}
		}

		if (sgn(px->x) == 0) {
			/* Left end point might be a root. */
			if (sdx0 == 0) {
				/* But it is a tangent. */
				res->x = NAN;
				res->y = NAN;
				res->z = NAN;
				res->w = NAN;
				return;
			} else {
				/* Left end point is a root. */
				if (sdx0 == sdx1) {
					/* Stationary point not reached, only one root. */
					res->x = inter->x;
					res->y = NAN;
					res->z = NAN;
					res->w = NAN;
					return;
				} else {
					xm = find_next_same_sgn(p, degree, inter, sgn_trig,
															&pxm, RIGHT);
					res->x = inter->x;
					res->y = NAN;

					inter->x = xm;
					px->x = pxm;
					converging_interval1(p, degree, inter, *px);
					res->z = inter->x;
					res->w = inter->y;
					return;
				}
			}
		}
		if (sgn(px->y) == 0) {
			/* Right end point might be a root. */
			if (sdx1 == 0) {
				/* But it is a tangent. */
				res->x = NAN;
				res->y = NAN;
				res->z = NAN;
				res->w = NAN;
				return;
			} else {
				/* Right end point is a root. */
				if (sdx1 == sdx0) {
					/* Stationary point not reached, only one root. */
					res->x = inter->y;
					res->y = NAN;
					res->z = NAN;
					res->w = NAN;
					return;
				} else {
					xm = find_next_same_sgn(p, degree, inter, sgn_trig,
															&pxm, LEFT);
					res->z = inter->y;
					res->w = NAN;

					inter->y = xm;
					px->y = pxm;
					converging_interval1(p, degree, inter, *px);
					res->x = inter->x;
					res->y = inter->y;
					return;
				}
			}
		}
	}
}

void process_middle_root(const vfloat *p,
					const int degree, vfloat2 *inter, int concave,
					vfloat xm, vfloat2 *px, int sdx0, int sdxm, vfloat4 *res) {
	vfloat next_point, pxnp;

	if (sdxm == 0) {
		/* The derivation is zero, thus the root is the tangent to the
		 * function at its stationary point -> no actual thickness.
		 */
		res->x = NAN;
		res->y = NAN;
		res->z = NAN;
		res->w = NAN;
		return;
	} else {
		if (sdx0 == 0) {
			/* Derivative at the first end point is inconclusive. */
			if (concave) {
				sdx0 = 1;
			} else {
				sdx0 = -1;
			}
		}
		if (sdxm == sdx0) {
			/* Not yet at the stationary point, check the other half.
			 * Find the leftmost point with the same sign as this one.
			 */
			inter->x = xm;
			next_point = find_next_same_sgn(p, degree, inter, sdxm,
														&pxnp, RIGHT);
			inter->x = next_point;
			px->x = pxnp;
			converging_interval1(p, degree, inter, *px);

			res->x = xm;
			res->y = NAN;
			res->z = inter->x;
			res->w = inter->y;
			return;
		} else {
			/* Past stationary point, check the first half. */
			inter->y = xm;
			next_point = find_next_same_sgn(p, degree, inter, sdx0,
															&pxnp, LEFT);
			inter->y = next_point;
			px->y = pxnp;
			converging_interval1(p, degree, inter, *px);

			res->x = inter->x;
			res->y = inter->y;
			res->z = xm;
			res->w = NAN;
			return;
		}
	}
}

void converging_interval2(const vfloat *p,
							const int degree,
							vfloat2 *inter,
							vfloat2 *px,
							vfloat4 *res) {
	vfloat px0, px1, xm, pxm, tmp;
	int sx0, sx1, sdx0, sxm, sdxm, concave;
	vfloat2 tmp_inter;

	px0 = polynomial(p, degree, inter->x);
	px1 = polynomial(p, degree, inter->y);
	sx0 = sgn(px0);
	sx1 = sgn(px1);

	if (sx0 == 0 && sx1 == 0) {
		/* Both end points are roots. */
		res->x = inter->x;
		res->y = NAN;
		res->z = inter->y;
		res->w = NAN;
		return;
	}

	concave = is_concave(p, degree, inter, px);

	/* Check if there are roots at all. */
	if (!roots_exist(concave, sx0, sx1)) {
		res->x = NAN;
		res->y = NAN;
		res->z = NAN;
		res->w = NAN;
		return;
	}

	/* End points are at different heights. */
	if (sx0 != sx1) {
		process_different_heights(p, degree, inter, concave, px, res);
		return;
	}

	sdx0 = sgn(derivative(p, degree, inter->x));
	if (sdx0 == 0) {
		/* Derivative at the first end point is inconclusive. */
		if (concave) {
			sdx0 = 1;
		} else {
			sdx0 = -1;
		}
	}

	while (1) {
		xm = halve(inter);
		pxm = polynomial(p, degree, xm);
		sxm = sgn(pxm);
		sdxm = sgn(derivative(p, degree, xm));
		if (sxm == 0) {
			/* We hit one of the roots, determine the interval which needs to
			 * be checked for the other one. This is done by checking the
			 * derivative of the function at this point and comparing to the
			 * first end point. If they are the same the function has not yet
			 * get past the stationary point so examine the interval <xm, x1>
			 */
            process_middle_root(p, degree, inter, concave,
            						xm, px, sdx0, sdxm, res);
            return;
		}
		if (sxm != sx0) {
			/* The halving process found a point where the function has
			 * different sign from the end points and the derivative is
			 * the same, thus the interval is monotonic with one root inside it.
			 */
			tmp_inter.x = inter->x;
			tmp_inter.y = xm;
			inter->x = xm;
			tmp = px->y;
			px->y = pxm;
			converging_interval1(p, degree, &tmp_inter, *px);
			px->x = pxm;
			px->y = tmp;
			converging_interval1(p, degree, inter, *px);

			res->x = tmp_inter.x;
			res->y = tmp_inter.y;
			res->z = inter->x;
			res->w = inter->y;
			return;
		} else {
            /* Otherwise continue the subdivision until the two end points are
             * too close meaning there are no roots in this interval.
             */
			if (sdx0 != sdxm) {
                /* The derivative signs of the first end point and the middle
                 * point are different, so the stationary point is somewhere
                 * between them.
                 */
				inter->y = xm;
				px->x = pxm;
			} else {
				/* Stationary point in the middle and second end point interval.
				 */
				inter->x = xm;
				px->x = pxm;
                /* Get new values for tests in another iteration. The middle
                 * point will be the new start point, thus recalculate
                 * the start point values.
                 */
				sx0 = sgn(polynomial(p, degree, xm));
				sdx0 = sdxm;
			}
			if (inter->y - inter->x < EPS) {
                /* We hit the stationary point without intersecting the
                 * function, thus there are no roots.
                 */
				res->x = NAN;
				res->y = NAN;
				res->z = NAN;
				res->w = NAN;
				return;
			}
		}
	}
}

void normalize_polynomial(const vfloat *p_orig, vfloat *out, int degree) {
	/* find non-zero minimum */
	vfloat min_val = MAXFLOAT;
	int i;

	for (i = 0; i <= degree; i++) {
		if ((-ZERO_EPS > p_orig[i] || p_orig[i] > ZERO_EPS)  &&
												fabs(p_orig[i]) < min_val) {
			min_val = fabs(p_orig[i]);
		}
	}

	if (min_val != MAXFLOAT) {
		for (i = 0; i <= degree; i++) {
			out[i] = p_orig[i]/min_val;
		}
	}
}

void roots(const vfloat *p_orig, const int degree,
			const vfloat2 *const_inter, vfloat2 *res) {
	vfloat x0, p[5];
	vfloat2 inter, px;
	vfloat4 two_root_inter;
	inter = *const_inter;

	/* Get normalized polynomial coefficients. */
	normalize_polynomial(p_orig, p, degree);

	px.x = polynomial(p, degree, inter.x);
	px.y = polynomial(p, degree, inter.y);

	if (sgn(px.x) != sgn(px.y) && sgn(px.x) != 0 && sgn(px.y) != 0) {
		/* One possible root. */
        /* Derivative signs are equal, maximum one root. If both derivatives
         * are zero, the monotonicity is inconclusive, thus continue with
         * 2-root finding.
         * Check if there is root at all, the signs must change between the
         * end points.
         */
		converging_interval1(p, degree, &inter, px);
		if (isnan(inter.y)) {
			/* Root hit by bisection already. */
			res->x = inter.x;
		} else {
			/* Get a starting guess for Newton-Raphson iterations. */
			x0 = halve(&inter);
			if (!isnan(x0)) {
				res->x = newton_raphson(p, degree, x0);
			}
		}
		/* Max one root, so the other is just set to NAN. */
		res->y = NAN;
	} else {
		/* Two possible roots. */
		converging_interval2(p, degree, &inter, &px, &two_root_inter);
		if (!isnan(two_root_inter.x)) {
			/* First root. */
			if (isnan(two_root_inter.y)) {
				/* Root hit before NR iterations. */
				res->x = two_root_inter.x;
			} else {
				inter.x = two_root_inter.x;
				inter.y = two_root_inter.y;
				x0 = halve(&inter);
				res->x = newton_raphson(p, degree, x0);
			}
		} else {
			/* No first root. */
			res->x = NAN;
		}
		if (!isnan(two_root_inter.z)) {
			/* Second root. */
			if (isnan(two_root_inter.w)) {
				/* Root hit before NR iterations. */
				res->y = two_root_inter.z;
			} else {
				inter.x = two_root_inter.z;
				inter.y = two_root_inter.w;
				x0 = halve(&inter);
				res->y = newton_raphson(p, degree, x0);
			}
		} else {
			/* No second root. */
			res->y = NAN;
		}
	}
}





__kernel void roots_kernel(__constant vfloat *p,
							const vfloat2 interval,
							const int degree,
							__global vfloat *out) {
	int ix = get_global_id(0);
	int i;
	vfloat ip[5];
	vfloat2 tmp_inter, res;
	vfloat4 c2res;

	for (i = 0; i < 5; i++) {
		ip[i] = p[i];
	}


	roots(ip, degree, &interval, &res);

	out[ix] = p[0];
	out[ix+1] = p[4];
}




