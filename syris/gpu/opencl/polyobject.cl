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
 * Polynomial representation of a metaobject on OpenCL.
 *
 * Requires definition of vfloat data type, which defines single or double
 * precision for floating point numbers.
 */

#define POLY_COEFFS_NUM 5

typedef struct _poly_object {
	vfloat coeffs[POLY_COEFFS_NUM];
	vfloat2 interval;
} __attribute__((packed)) poly_object;


/*
 * init polynomial object made of coefficients (first 5) and interval
 * of the object influence (last 2).
 *
 */
void init_poly_object(global poly_object *po, vfloat coeffs[POLY_COEFFS_NUM+2]) {
	int i;

	for (i = 0; i < POLY_COEFFS_NUM; i++) {
		po->coeffs[i] = coeffs[i];
	}
	po->interval = (vfloat2)(coeffs[POLY_COEFFS_NUM], coeffs[POLY_COEFFS_NUM+1]);
}
