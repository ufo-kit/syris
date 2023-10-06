/*
 * Copyright (C) 2013-2023 Karlsruhe Institute of Technology

 * This file is part of syris.

 * This library is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.

 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.

 * You should have received a copy of the GNU Lesser General Public
 * License along with this library. If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * Complex arithmetic on OpenCL.
 */

typedef vfloat2 vcomplex;


/*
 * Complex multiplication.
 */
vcomplex vc_mul(const vcomplex *a, const vcomplex *b) {
	return (vcomplex)(a->x * b->x - a->y * b->y, a->y * b->x + a->x * b->y);
}

/*
 * Complex division.
 */
vcomplex vc_div(const vcomplex *a, const vcomplex *b) {
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

	out[ix] = in_0[ix] + in_1[ix];
}

/*
 * Complex subtraction kernel.
 */
__kernel void vc_sub_kernel(__global vcomplex *in_0,
								__global vcomplex *in_1,
								__global vcomplex *out) {
	int ix = get_global_id(0);

	out[ix] = in_0[ix] - in_1[ix];
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
