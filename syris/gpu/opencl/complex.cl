/*
 * Complex arithmetic.
 */

typedef vfloat2 vcomplex;


/*
 * Complex addition.
 */
vcomplex vc_add(vcomplex *a, vcomplex *b) {
	return *a + *b;
}

/*
 * Complex subtraction.
 */
vcomplex vc_sub(vcomplex *a, vcomplex *b) {
	return *a - *b;
}

/*
 * Complex multiplication.
 */
vcomplex vc_mul(vcomplex *a, vcomplex *b) {
	return (vcomplex)(a->x * b->x - a->y * b->y, a->y * b->x + a->x * b->y);
}

/*
 * Complex division.
 */
vcomplex vc_div(vcomplex *a, vcomplex *b) {
	return (vcomplex)
			((a->x * b->x + a->y * b->y) / (b->x * b->x + b->y * b->y),
			(a->y * b->x - a->x * b->y) / (b->x * b->x + b->y * b->y));
}


/*
 * Complex addition kernel.
 */
__kernel void vc_add_kernel(__global vcomplex *in_0,
								__global vcomplex *in_1,
								__global vcomplex *out) {
	int ix = get_global_id(0);
	vcomplex a, b;

	a = in_0[ix];
	b = in_1[ix];

	out[ix] = vc_add(&a, &b);
}

/*
 * Complex subtraction kernel.
 */
__kernel void vc_sub_kernel(__global vcomplex *in_0,
								__global vcomplex *in_1,
								__global vcomplex *out) {
	int ix = get_global_id(0);
	vcomplex a, b;

	a = in_0[ix];
	b = in_1[ix];

	out[ix] = vc_sub(&a, &b);
}

/*
 * Complex multiplication kernel.
 */
__kernel void vc_mul_kernel(__global vcomplex *in_0,
								__global vcomplex *in_1,
								__global vcomplex *out) {
	int ix = get_global_id(0);
	vcomplex a, b;

	a = in_0[ix];
	b = in_1[ix];

	out[ix] = vc_mul(&a, &b);
}

/*
 * Complex division kernel.
 */
__kernel void vc_div_kernel(__global vcomplex *in_0,
								__global vcomplex *in_1,
								__global vcomplex *out) {
	int ix = get_global_id(0);
	vcomplex a, b;

	a = in_0[ix];
	b = in_1[ix];

	out[ix] = vc_div(&a, &b);
}
