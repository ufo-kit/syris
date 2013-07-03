/*
 * OpenCL code for metaballs.
 *
 * Requires definition of vfloat data type, which defines single or double
 * precision for floating point numbers.
 */

/* Maximum number of objects. They are passed in constant memory block
 * so their number is limited. */
#define MAX_OBJECTS 100

typedef struct _object {
	OBJECT_TYPE type;
	vfloat radius;
	vfloat blobbiness;
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

void add_coeffs(vfloat *result, vfloat *toadd, int size) {
	int i;

	for (i = 0; i < size; i++) {
		result[i] += toadd[i];
	}
}

void subtract_coeffs(vfloat *result, vfloat *tosub, int size) {
	int i;

	for (i = 0; i < size; i++) {
		result[i] -= tosub[i];
	}
}

/*
 * Get thickness from given roots and previous value. We need to store previous
 * value because when there is only one root we need to couple it with the next
 * from the following intervals. The following cases can happen:
 * 	- both roots are not NAN and the previous is NAN -> use the roots
 * 	- both roots not NAN, but previous is not NAN too, thus currently
 * 		the isosurface is "decreasing", so we couple the previous value
 * 		with the left root and store the next root as previous
 * 	- only one root is found and previous is NAN, just store the root as
 * 		previous
 * 	- only one root is found and previous is not NAN, couple those two ->
 * 		isosurface is "decreasing"
 * 	- both roots are NAN, do nothing
 */
void get_thickness(vfloat2 *roots, vfloat *previous, vfloat *thickness) {
	if (!isnan(*previous)) {
		if (!isnan(roots->x)) {
			if (roots->x - *previous > EPS) {
				*thickness += roots->x - *previous;
				*previous = NAN;
			}
		} else if (!isnan(roots->y)) {
			if (roots->y - *previous > EPS) {
				*thickness += roots->y - *previous;
				*previous = NAN;
			}
		}
	} else {
		if (isnan(roots->x)) {
			*previous = roots->y;
		} else if (isnan(roots->y)) {
			*previous = roots->x;
		} else {
			if (roots->y - roots->x > EPS) {
				*thickness += roots->y - roots->x;
			}
			*previous = NAN;
		}
	}
}

/* Meta objects equations. */

void get_meta_ball_equation(vfloat *result,
								__constant object *o,
								vfloat2 *p,
								const vfloat z_middle) {
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
						(o->radius+o->blobbiness)*(o->radius+o->blobbiness));

	if (isnan(intersection.x)) {
		/* no influence in this pixel */
		result[5] = NAN; /* for further tests for intersection */
		return;
	}

	/* determine R = r + influence for fiven x,y coordinates */
	R_2 = (intersection.y - intersection.x) / 2.0;
	/* now square it */
	R_2 = R_2*R_2;
	/* shift center in z direction in order to minimize distance from 0.
	 * Precomputation for all objects done by host. */
	cz = (intersection.x + intersection.y)/2.0 - z_middle;
	result[5] = intersection.x - z_middle;
	result[6] = intersection.y - z_middle;

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

void get_meta_cube_equation(vfloat *result,
								__constant object *o,
								vfloat2 *p,
								const vfloat z_middle) {
	vfloat r_2, R, R_2;			/* inner and outer squared radius */
	vfloat cz; 				/* z-center of the fallof curve */
	vfloat2 intersection;	/* intersection of the influence area (R_2 is the
							 * radius) and a ray */
	vfloat cube_poly[5], coeff;
	vfloat2 guesses;
	/* TODO: calculate the guess here. CPU precalculation was dropped. */
//	guesses.x = o->constants.y;
//	guesses.y = o->constants.z;


	vfloat cx_2 = o->trans_matrix.s2;
	cx_2 = cx_2*cx_2;

	vfloat cy_2 = o->trans_matrix.s6;
	cy_2 = cy_2*cy_2;

	vfloat cz_2 = o->trans_matrix.sa;
	cz_2 = cz_2*cz_2;

	// TODO: use dot() instead?
	vfloat ax = o->trans_matrix.s0*p->x + o->trans_matrix.s1*p->y +
				o->trans_matrix.s3;
	vfloat ay = o->trans_matrix.s4*p->x + o->trans_matrix.s5*p->y +
				o->trans_matrix.s7;
	vfloat az = o->trans_matrix.s8*p->x + o->trans_matrix.s9*p->y +
				o->trans_matrix.sb;

	vfloat bx = 2*ax*o->trans_matrix.s2;
	vfloat by = 2*ay*o->trans_matrix.s6;
	vfloat bz = 2*az*o->trans_matrix.sa;
	R = (o->radius+o->blobbiness);

	cube_poly[0] = cx_2*cx_2 + cy_2*cy_2 + cz_2*cz_2;
	cube_poly[1] = 2*(bx*cx_2 + by*cy_2 + bz*cz_2);
	cube_poly[2] = 2*(ax*ax*cx_2 + ay*ay*cy_2 + az*az*cz_2) +
			bx*bx + by*by + bz*bz;
	cube_poly[3] = 2*(ax*ax*bx + ay*ay*by + az*az*bz);
	cube_poly[4] = ax*ax*ax*ax + ay*ay*ay*ay + az*az*az*az - R*R*R*R;

	/* We need to calculate real intersection points, not just radius because
	 * thanks to object scaling and rotating the center of the falloff curve
	 * might be different than the one of the whole object. */
	roots(cube_poly, 4, &guesses, &intersection);

	if (isnan(intersection.x)) {
		/* no influence in this pixel */
		result[5] = NAN; /* for further tests for intersection */
		return;
	}

	/* determine R = r + influence for fiven x,y coordinates */
	R_2 = (intersection.y - intersection.x) / 2.0;
	/* now square it */
	R_2 = R_2*R_2;

	/* shift center in z direction in order to minimize distance from 0.
	 * Precomputation for all objects done by host. */
	cz = (intersection.x + intersection.y)/2.0 - z_middle;
	result[5] = intersection.x - z_middle;
	result[6] = intersection.y - z_middle;

	/* TODO: why do the coefficients get mixed up while computing roots???
	 * They are declared there as const!
	 */
	cube_poly[0] = cx_2*cx_2 + cy_2*cy_2 + cz_2*cz_2;
	cube_poly[1] = 2*(bx*cx_2 + by*cy_2 + bz*cz_2);
	cube_poly[2] = 2*(ax*ax*cx_2 + ay*ay*cy_2 + az*az*cz_2) +
			bx*bx + by*by + bz*bz;
	cube_poly[3] = 2*(ax*ax*bx + ay*ay*by + az*az*bz);
	cube_poly[4] = ax*ax*ax*ax + ay*ay*ay*ay + az*az*az*az -
			o->radius*o->radius*o->radius*o->radius;
	roots(cube_poly, 4, &guesses, &intersection);

	if (isnan(intersection.x)) {
		cube_poly[0] = cx_2*cx_2 + cy_2*cy_2 + cz_2*cz_2;
		cube_poly[1] = 2*(bx*cx_2 + by*cy_2 + bz*cz_2);
		cube_poly[2] = 2*(ax*ax*cx_2 + ay*ay*cy_2 + az*az*cz_2) +
				bx*bx + by*by + bz*bz;
		cube_poly[3] = 2*(ax*ax*bx + ay*ay*by + az*az*bz);
		cube_poly[4] = o->radius*o->radius*o->radius*o->radius -
				(ax*ax*ax*ax + ay*ay*ay*ay + az*az*az*az);

		/* The negative radius might go beyond the bounding box limit.
		 * TODO: check and make more flexible.
		 */
		/* TODO: guesses are no longer calculated on CPU. */
//		guesses.x = 10*o->constants.y;
//		guesses.y = 10*o->constants.z;
		roots(cube_poly, 4, &guesses, &intersection);
		r_2 = (intersection.y - intersection.x) / 2.0;
		r_2 = -r_2*r_2;
	} else {
		r_2 = (intersection.y - intersection.x) / 2.0;
		r_2 = r_2*r_2;
	}

	result[7] = r_2;
	coeff = 1.0/((R_2 - r_2)*(R_2 - r_2));

	/* final quartic coefficients */
	result[0] = coeff;
	result[1] = coeff*(-4*cz);
	result[2] = coeff*(-2*R_2 + 6*cz*cz);
	result[3] = coeff*(4*R_2*cz - 4*cz*cz*cz);
	result[4] = coeff*(R_2*R_2 - 2*R_2*cz*cz + cz*cz*cz*cz);
}

void get_meta_object_equation(vfloat *result,
								__constant object *o,
								vfloat2 *p,
								const vfloat z_middle) {
	switch (o->type) {
		case METABALL:
			get_meta_ball_equation(result, o, p, z_middle);
			break;
		case METACUBE:
			get_meta_cube_equation(result, o, p, z_middle);
			break;
		default:
			break;
	}
}

__kernel void thickness(__constant object *objects,
						__global vfloat *out,
						const int num_objects,
						const int2 start_point,
						const int2 end_point,
						const int width,
						const vfloat z_middle,
						const vfloat2 effective_ps,
						const vfloat2 roi_start,
						const int2 gl_offset,
						const int clear) {
	int ix = start_point.x + get_global_id(0);
	int iy = start_point.y + get_global_id(1);

	/* +2 for interval beginning and end, +1 for radius^2 storage
	 * for case there is only one metaobject at this pixel.
	 */
	vfloat poly[POLY_COEFFS_NUM+3];
	vfloat2 res = (vfloat2)(NAN,NAN), interval;
	vfloat previous = NAN, tmp, thickness = 0.0f;

	int min_index = 0, max_index = 0, size = 0, i, which = 0;
    poly_object pobjects[MAX_OBJECTS];
    poly_object *left[MAX_OBJECTS];
    poly_object *right[MAX_OBJECTS];

	vfloat2 obj_coords;

	if (start_point.x <= ix && ix < end_point.x &&
		start_point.y <= iy && iy < end_point.y) {
		/* We are in the FOV. */
		// transform pixel to object point in mm first
		obj_coords.x = roi_start.x + (ix + gl_offset.x) * effective_ps.x;
		obj_coords.y = roi_start.y + (iy + gl_offset.y) * effective_ps.y;

		for (i = 0; i < num_objects; i++) {
			get_meta_object_equation(poly, &objects[i], &obj_coords, z_middle);
			if (!isnan(poly[5])) {
				init_poly_object(&pobjects[i], poly);
				add(left, &pobjects[i], size, X_SORT);
				add(right, &pobjects[i], size, Y_SORT);
				size++;
			}
		}

		switch (size) {
			case 0:
				/* nothing in this pixel */
				if (clear) {
					out[width*iy+ix] = 0.0f;
				}
				break;
			case 1:
				/* only one metaobject in this pixel */
				if (poly[7] < 0.0) {
					poly[7] = 0.0;
				}
				if (clear) {
					out[width*iy+ix] = sqrt(poly[7])*2;
				} else {
					out[width*iy+ix] = out[width*iy+ix] + sqrt(poly[7])*2;
				}
				break;
			default:
				/* more than one metaobject, we need to solve the quartic */
				sort(left, size, X_SORT);
				sort(right, size, Y_SORT);

				for (i = 0; i < POLY_COEFFS_NUM; i++) {
					poly[i] = 0.0f;
				}
				poly[4] = -1.0f;

				/* intervals */
				while (max_index < size) {
					if (min_index < size && left[min_index]->interval.x <
											right[max_index]->interval.y) {
						interval.y = left[min_index]->interval.x;
					} else {
						interval.y = right[max_index]->interval.y;
					}

					if (min_index > 0 || max_index > 0) {
						/* skip first time when nothing is computed so far */
						// TODO: joint computation for both?
						roots(poly, 4, &interval, &res);
						get_thickness(&res, &previous, &thickness);
					}

					/* add minimums */
					while (min_index < size &&
							left[min_index]->interval.x <= interval.y) {
						add_coeffs(poly, left[min_index]->coeffs,
								POLY_COEFFS_NUM);
						min_index++;
					}
					/* remove all intervals which are passed */
					while (max_index < size &&
							right[max_index]->interval.y <= interval.y) {
						subtract_coeffs(poly, right[max_index]->coeffs,
								POLY_COEFFS_NUM);
						max_index++;
					}
					interval.x = interval.y;
					which++;
				}
			if (clear) {
				out[width*iy+ix] = thickness;
			} else {
				out[width*iy+ix] = out[width*iy+ix] + thickness;
			}
		}
	} else if (clear) {
		out[width*iy+ix] = 0.0f;
	}
}
