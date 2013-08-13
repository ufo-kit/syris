/*
 * Metaballs on OpenCL.
 *
 * Requires definition of vfloat data type, index defines single or double
 * precision for floating point numbers.
 */

/* Maximum number of objects. They are passed in constant memory block
 * so their number is limited. */
#define MAX_OBJECTS 100
#define left_index(a) (((a) == 0) ? 2 : (((a) - 1) % 3))
#define current_index(a) ((a) % 3)
#define right_index(a) (((a) + 1) % 3)

typedef struct _object {
	OBJECT_TYPE type;
	vfloat radius;
	/* CPU pre-computed constants. */
	vfloat2 constants;
	/* Backward affine transformation matrix. */
	vfloat16 trans_matrix;
} __attribute__((packed)) object;


/**
 * Get squared radius based on quadratic equation. First compute
 * the discriminant, then, based on the fact that r = (x2 - x1) / 2,
 * formula simplification and the fact that we need squared radius we can never
 * get nan because we do not need the square root of the discriminant.
 */
vfloat get_squared_radius(vfloat a, vfloat b, vfloat c) {
	vfloat d = b*b - 4*a*c;

	return d/(4*a*a);
}

vfloat2 solve_quadric(vfloat a, vfloat b, vfloat c) {
	vfloat d = b*b - 4*a*c;

	return (vfloat2)((-b - sqrt(d))/(2*a), (-b + sqrt(d))/(2*a));
}

void add_coeffs(vfloat *result, vfloat *src, vfloat *toadd, int size) {
	int i;

	for (i = 0; i < size; i++) {
		result[i] = src[i] + toadd[i];
	}
}

void subtract_coeffs(vfloat *result, vfloat *src, vfloat *tosub, int size) {
	int i;

	for (i = 0; i < size; i++) {
		result[i] = src[i] - tosub[i];
	}
}

/* Meta objects equations. */

void get_meta_ball_equation(vfloat *result, __constant object *o, vfloat2 *p,
								vfloat z_middle, vfloat coeff, vfloat eps) {
	vfloat r_2, R_2;				/* outer squared radius */
	vfloat cz; 				/* z-center of the fallof curve */
	vfloat2 intersection;	/* intersection of the influence area (R_2 is the
							 * radius) and a ray */


	vfloat ax = o->trans_matrix.s2;
	vfloat ay = o->trans_matrix.s6;
	vfloat az = o->trans_matrix.sa;

	// TODO: use dot() instead?
	vfloat kx = o->trans_matrix.s0*p->x + o->trans_matrix.s1*p->y +
				o->trans_matrix.s3;
	vfloat ky = o->trans_matrix.s4*p->x + o->trans_matrix.s5*p->y +
				o->trans_matrix.s7;
	vfloat kz = o->trans_matrix.s8*p->x + o->trans_matrix.s9*p->y +
				o->trans_matrix.sb;

	/* We need to calculate real intersection points, not just radius because
	 * thanks to object scaling and rotating the center of the falloff curve
	 * might be different than the one of the whole object. */
	intersection = solve_quadric(o->constants.y,
						2*kx*ax + 2*ky*ay + 2*kz*az,
						kx*kx + ky*ky + kz*kz -
						4 * o->radius * o->radius);

	if (isnan(intersection.x) ||
			is_close(intersection.y - intersection.x, 0.0, eps)) {
		/* no influence in this pixel */
		result[5] = NAN; /* for further tests for intersection */
		return;
	}

	z_middle *= coeff;
	/* determine R = r + influence for given x,y coordinates */
	R_2 = coeff * (intersection.y - intersection.x) / 2.0;
	/* now square it */
	R_2 = R_2 * R_2;
	/* shift center in z direction in order to minimize distance from 0.
	 * Precomputation for all objects done by host. */
	cz = coeff * (intersection.x + intersection.y)/2.0 - z_middle;
	result[5] = coeff * intersection.x - z_middle;
	result[6] = coeff * intersection.y - z_middle;

	/* Determine r for given x,y coordinates. This cannot be a regular quadric
	 * solver because r can be negative in order to correctly model
	 * the falloff. */
	r_2 = get_squared_radius(o->constants.y,
						2*kx*ax + 2*ky*ay + 2*kz*az,
						kx*kx + ky*ky + kz*kz - o->radius*o->radius);

	/* since (R^2 - r^2)^2 = c, r^2 = R^2 - sqrt(1/c),
	 * where c = 1/(R^2 - r^2)^2 calculated to fit f(r) = 1.0. */
	result[7] = r_2;//R_2 - sqrt(1.0/o->constants.x);

	/* final quartic coefficients */
	result[0] = o->constants.x;
	result[1] = o->constants.x*(-4*cz);
	result[2] = o->constants.x*(-2*R_2 + 6*cz*cz);
	result[3] = o->constants.x*(4*R_2*cz - 4*cz*cz*cz);
	result[4] = o->constants.x*(R_2*R_2 - 2*R_2*cz*cz + cz*cz*cz*cz);
}

void get_meta_object_equation(vfloat *result, __constant object *o,
			vfloat2 *p, const vfloat z_middle, vfloat coeff, vfloat eps) {
	switch (o->type) {
		case METABALL:
			get_meta_ball_equation(result, o, p, z_middle, coeff, eps);
			break;
		case METACUBE:
//			get_meta_cube_equation(result, o, p, z_middle, eps);
			break;
		default:
			break;
	}
}

bool is_root_valid(const vfloat *coeffs, unsigned int degree,
					vfloat root, int last_derivative_sgn) {
	/*
	 * The next root is valid if the derivative sign is different
	 * from the current one index means the polynomial must have crossed
	 * zero somewhere inbetween the two roots.
	 */
	int sgn = sgn(derivative(coeffs, degree, root));

	return last_derivative_sgn == -2 || sgn != last_derivative_sgn ||
			sgn == 0 && last_derivative_sgn == 0;
}

int next_root(const vfloat *coeffs, unsigned int degree,
					int index, vfloat *roots, int *last_derivative_sgn) {
	/*
	 * Find the index of the next valid root and push the last processed
	 * root to be the found one.
	 */
	while (index < degree + 1 && !isnan(roots[index]) &&
			!is_root_valid(coeffs, degree, roots[index],
								*last_derivative_sgn)) {
		/* Not valid but update the last values. */
		*last_derivative_sgn = sgn(derivative(coeffs, degree, roots[index]));
		index++;
	}

	if (index < degree + 1 && !isnan(roots[index])) {
		*last_derivative_sgn = sgn(derivative(coeffs, degree, roots[index]));
	}

	return index;
}

vfloat get_thickness(const vfloat *coeffs, unsigned int degree,
				vfloat *roots, vfloat *previous, int *last_derivative_sgn) {
	/*
	 * Get the projected thickness from give *roots*. *previous* is the
	 * root index has not yet been coupled with another one, saved
	 * from passed intervals. *last_derivative_sgn* is the sign of the
	 * derivative of the polynomial at the last processed root.
	 * The *roots* are extrema-free.
	 */
	vfloat thickness = 0.0;
	unsigned int next;
	unsigned int i = next_root(coeffs, degree, 0, roots, last_derivative_sgn);

	if (i == degree) {
		return 0.0;
	}

	if (!isnan(*previous) && !isnan(roots[i])) {
		thickness += roots[i] - *previous;
		*previous = NAN;
		i = next_root(coeffs, degree, i + 1, roots, last_derivative_sgn);
	}

	while (i < degree + 1 && !isnan(roots[i])) {
		next = next_root(coeffs, degree, i + 1, roots, last_derivative_sgn);
		if (next < degree + 1 && !isnan(roots[next])) {
			thickness += roots[next] - roots[i];
			*previous = NAN;
		} else {
			*previous = roots[i];
			break;
		}
		i = next_root(coeffs, degree, next + 1, roots, last_derivative_sgn);
	}

	return thickness;
}

__kernel void thickness_add_kernel(__global vfloat4 *out,
									__global vfloat *coeffs,
									__global vfloat *roots,
									vfloat previous,
									int last_derivative_sgn) {
	vfloat l_roots[POLY_DEG + 1];
	vfloat l_coeffs[POLY_DEG + 1];
	unsigned int i;

	_copy_global_local(roots, l_roots, POLY_DEG + 1, true);
	_copy_global_local(coeffs, l_coeffs, POLY_DEG + 1, true);

	vfloat thickness = get_thickness(l_coeffs, POLY_DEG, l_roots,
			&previous, &last_derivative_sgn);

	out[0] = (vfloat4)(thickness, previous, last_derivative_sgn, 0);
}

vfloat *adjust_coefficients(vfloat coeffs[3][POLY_DEG + 1],
		vfloat *source_coeffs, poly_object *objects[MAX_OBJECTS],
		vfloat left_end, unsigned int *object_index, unsigned int index,
		unsigned int size, bool addition) {
	/*
	 * Add or subtract coefficients from current coefficients.
	 */
	while (*object_index < size &&
			(addition ? objects[*object_index]->interval.x :
						objects[*object_index]->interval.y) <= left_end) {
		if (addition) {
			add_coeffs(coeffs[index == 0 ? 0 : current_index(index)],
					source_coeffs, objects[*object_index]->coeffs,
					POLY_COEFFS_NUM);
		} else {
			subtract_coeffs(coeffs[index == 0 ? 0 : current_index(index)],
					source_coeffs, objects[*object_index]->coeffs,
					POLY_COEFFS_NUM);
		}
		/* Source coefficient are current coefficients from now on. */
		source_coeffs = coeffs[index == 0 ? 0 : current_index(index)];
		(*object_index)++;
	}

	return source_coeffs;
}


void update_coefficients(vfloat coeffs[3][POLY_DEG + 1],
			poly_object *left[MAX_OBJECTS], poly_object *right[MAX_OBJECTS],
			vfloat left_end, unsigned int *left_index,
			unsigned int *right_index, unsigned int index,
			unsigned int size) {
	/*
	 * Add the upcoming interval coefficients and remove the coefficients from
	 * past intervals.
	 */
	vfloat *src_coeffs = coeffs[index == 0 ? 0 : left_index(index)];

	/* If we add some coefficient, current coefficients become the source
	 * coefficients, otherwise we would discard the changes in addition. */
	src_coeffs = adjust_coefficients(coeffs, src_coeffs, left, left_end,
											left_index, index, size, true);
	adjust_coefficients(coeffs, src_coeffs, right, left_end, right_index,
														index, size, false);
}

__kernel void thickness(__global vfloat *out,
						__constant object *objects,
						const int num_objects,
						const int2 gl_offset,
						const int4 roi,
						const vfloat size_coeff,
						const vfloat z_middle,
						const vfloat2 pixel_size,
						const int clear) {
	int ix = get_global_id(0);
	int iy = get_global_id(1);
	unsigned int mem_index = get_global_size(0) * iy + ix;

	/* +2 for interval beginning and end, +1 for radius^2 storage
	 * for case there is only one metaobject at this pixel.
	 */
	vfloat poly[POLY_COEFFS_NUM + 3];
	vfloat coeffs[3][POLY_DEG + 1];
	vfloat left_end, right_end, previous_end;
	/* + 1 root for non-continuous function compensation. */
	vfloat roots[POLY_DEG + 1];
	vfloat previous = NAN, thickness = 0;
	int last_derivative_sgn = -2;
	unsigned int left_index = 0, right_index = 0, size = 0, i, index = 0;
    poly_object pobjects[MAX_OBJECTS];
    poly_object *left[MAX_OBJECTS];
    poly_object *right[MAX_OBJECTS];

    coeffs[0][0] = NAN;
    coeffs[1][0] = NAN;
    coeffs[2][0] = NAN;

	vfloat2 obj_coords;

	if (roi.x <= ix + gl_offset.x && ix + gl_offset.x < roi.z &&
			roi.y <= iy + gl_offset.y && iy + gl_offset.y < roi.w) {
		/* We are in the FOV. */
		/* transform pixel to object point in mm first */
		obj_coords.x = (ix + gl_offset.x) * pixel_size.x;
		obj_coords.y = (iy + gl_offset.y) * pixel_size.y;

		for (i = 0; i < num_objects; i++) {
			get_meta_object_equation(poly, &objects[i], &obj_coords,
					z_middle, size_coeff, pixel_size.x);
			if (!isnan(poly[5])) {
				init_poly_object(&pobjects[i], poly);
				po_add(left, &pobjects[i], size, X_SORT);
				po_add(right, &pobjects[i], size, Y_SORT);
				size++;
			}
		}

		switch (size) {
			case 0:
				/* nothing in this pixel */
				if (clear) {
					out[mem_index] = 0.0;
				}
				break;
			case 1:
				/* only one metaobject in this pixel */
				if (poly[7] < 0.0) {
					poly[7] = 0.0;
				}
				if (clear) {
					out[mem_index] = sqrt(poly[7])*2;
				} else {
					out[mem_index] = out[mem_index] + 2 * sqrt(poly[7]);
				}
				break;
			default:
				/* more than one metaobject, we need to solve the quartic */
				po_sort(left, size, X_SORT);
				po_sort(right, size, Y_SORT);

				for (i = 0; i < POLY_DEG + 1; i++) {
					coeffs[0][i] = 0.0;
				}
				coeffs[0][4] = -1;

				/* intervals */
				left_end = left[left_index]->interval.x;
				previous_end = left_end;
				while (right_index < size) {
					update_coefficients(coeffs, left, right, left_end,
							&left_index, &right_index, index, size);

					if (left_index < size && left[left_index]->interval.x <
											right[right_index]->interval.y) {
						right_end = left[left_index]->interval.x;
					} else if(right_index < size) {
						right_end = right[right_index]->interval.y;
					} else {
						coeffs[right_index(index - 1)][0] = NAN;
					}

					if (index > 0) {
						get_roots(coeffs[left_index(index -  1)],
									coeffs[current_index(index - 1)],
									coeffs[right_index(index - 1)],
									POLY_DEG, previous_end,
									left_end, roots, pixel_size.x);
						thickness += get_thickness(
								(const vfloat *)
								coeffs[current_index(index - 1)],
								POLY_DEG, roots, &previous,
								&last_derivative_sgn) /
								size_coeff;
					}

//					if (index == 9) {
//						out[mem_index].s0 = previous_end;
//						out[mem_index].s1 = left_end;
//						unsigned int j = current_index(index - 1);
//						vfloat *cfs = coeffs[j];
////						cfs = right[right_index]->coeffs;
//						out[mem_index].s2 = cfs[0];
//						out[mem_index].s3 = cfs[1];
//						out[mem_index].s4 = cfs[2];
//						out[mem_index].s5 = cfs[3];
//						out[mem_index].s6 = cfs[4];
//						out[mem_index].s7 = roots[0];
//						out[mem_index].s8 = roots[1];
//						out[mem_index].s9 = roots[2];
//						out[mem_index].sA = roots[3];
//						out[mem_index].sB = roots[4];
//						out[mem_index].sC = thickness;
//						out[mem_index].sD = last_derivative_sgn;
//						out[mem_index].sE = size;
//						break;
//					}

					if (index > 0) {
						previous_end = left_end;
					}

					left_end = right_end;
					index++;
				}
			if (clear) {
				out[mem_index] = thickness;
			} else {
				out[mem_index] = out[mem_index] + thickness;
			}
		}
	} else if (clear) {
		/* No geometry calculation in this pixel but we want it to be zero. */
		out[mem_index] = 0;
	}
}
