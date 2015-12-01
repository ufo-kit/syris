/*
 * Metaballs on OpenCL.
 *
 * Requires definition of vfloat data type, index defines single or double
 * precision for floating point numbers.
 */

#define get_left_index(a) (((a) == 0) ? 2 : (((a) - 1) % 3))
#define get_current_index(a) ((a) % 3)
#define get_right_index(a) (((a) + 1) % 3)


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

void add_coeffs(private vfloat *result, vfloat *src, global vfloat *toadd, int size) {
	int i;

	for (i = 0; i < size; i++) {
		result[i] = src[i] + toadd[i];
	}
}

void subtract_coeffs(private vfloat *result, vfloat *src, global vfloat *tosub, int size) {
	int i;

	for (i = 0; i < size; i++) {
		result[i] = src[i] - tosub[i];
	}
}

/* Meta objects equations. */

void get_meta_ball_equation(vfloat *result, __constant vfloat4 *object, vfloat2 *p, vfloat eps) {
	vfloat r_2, R_2;				/* outer squared radius */
	vfloat cz = object->z; 				/* z-center of the fallof curve */
	vfloat2 intersection;	/* intersection of the influence area (R_2 is the
							 * radius) and a ray */
	vfloat dist_2, tmp, falloff_const;

    dist_2 = (object->x - p->x) * (object->x - p->x) + (object->y - p->y) * (object->y - p->y);
    if (dist_2 > 4 * object->w * object->w) {
		/* no influence in this pixel */
        result[5] = NAN;
        return;
    }

    tmp = sqrt(4 * object->w * object->w - dist_2);
    intersection.x = object->z - tmp;
    intersection.y = object->z + tmp;

    /* influence region = 2 * r, thus the coefficient guaranteeing
     * f(r) = 1 is
     * 1 / (R^2 - r^2)^2 = 1 / (4r^2 - r^2)^2 = 1 / 9r^4 */

    falloff_const = 1.0 / (9 * object->w * object->w * object->w * object->w);

	/* determine R = r + influence for given x,y coordinates */
	R_2 = (intersection.y - intersection.x) / 2.0;
	/* now square it */
	R_2 = R_2 * R_2;
	result[5] = intersection.x;
	result[6] = intersection.y;


	/* since (R^2 - r^2)^2 = c, r^2 = R^2 - sqrt(1/c),
	 * where c = 1/(R^2 - r^2)^2 calculated to fit f(r) = 1.0. */
	/* final quartic coefficients */
	result[0] = falloff_const;
	result[1] = falloff_const*(-4 * cz);
	result[2] = falloff_const*(-2 * R_2 + 6 * cz * cz);
	result[3] = falloff_const*(4 * R_2 * cz - 4 * cz * cz * cz);
	result[4] = falloff_const*(R_2 * R_2 - 2 * R_2 * cz * cz + cz * cz * cz * cz);
	/* Keep the squared distance for fast intersection calculation in case there is only
	 * one object which intersects the ray. */
	result[7] = dist_2;
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
			(sgn == 0 && last_derivative_sgn == 0);
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

/**
 * Get ray-metaballs intersections based on given roots.
 * @intersections is the array where the intersections will be stored
 * @coeffs are the polynomial coefficients
 * @degree is the polynomial degree
 * @roots are the polynomial roots
 * @previous is the leftover root from previous runs
 * @last_derivative_sgn is the sign of the last derivative
 */
void get_intersections(vfloat intersections[POLY_DEG + 1], const vfloat *coeffs, unsigned int degree,
				vfloat *roots, vfloat *previous, int *last_derivative_sgn) {
	uint j;
	unsigned int next;
	unsigned int i = next_root(coeffs, degree, 0, roots, last_derivative_sgn);

	for (j = 0; j < POLY_DEG + 1; j++) {
	    intersections[j] = NAN;
	}
	j = 0;

	if (i == degree) {
	    return;
	}

    if (!isnan(*previous) && !isnan(roots[i])) {
        intersections[j++] = *previous;
        intersections[j++] = roots[i];
        *previous = NAN;
        i = next_root(coeffs, degree, i + 1, roots, last_derivative_sgn);
    }

    while (i < degree + 1 && !isnan(roots[i])) {
        next = next_root(coeffs, degree, i + 1, roots, last_derivative_sgn);
        if (next < degree + 1 && !isnan(roots[next])) {
            intersections[j++] = roots[i];
            intersections[j++] = roots[next];
            *previous = NAN;
        } else {
            *previous = roots[i];
            break;
        }
        i = next_root(coeffs, degree, next + 1, roots, last_derivative_sgn);
    }
}

/**
 * Get the projected thickness based on ray-metaball intersections.
 * @intersections are the ray-metaball intersections
 */
vfloat get_thickness(vfloat intersections[POLY_DEG + 1]) {
    int i = 0;
    vfloat thickness = 0.0;

    while (!isnan(intersections[i])) {
        thickness += intersections[i + 1] - intersections[i];
        i += 2;
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
	vfloat intersections[POLY_DEG + 1];
	vfloat thickness;
	unsigned int i;

	_copy_global_local(roots, l_roots, POLY_DEG + 1, true);
	_copy_global_local(coeffs, l_coeffs, POLY_DEG + 1, true);

	get_intersections(intersections, l_coeffs, POLY_DEG, l_roots,
			&previous, &last_derivative_sgn);
	thickness = get_thickness(intersections);

	out[0] = (vfloat4)(thickness, previous, last_derivative_sgn, 0);
}

vfloat *adjust_coefficients(private vfloat coeffs[3][POLY_DEG + 1],
		vfloat *source_coeffs, global poly_object *objects,
		global ushort *sorted, vfloat left_end, unsigned int *object_index,
		uint offset, uint index, unsigned int size, bool addition) {
	/*
	 * Add or subtract coefficients from current coefficients.
	 */
	while (*object_index < size &&
			(addition ? objects[offset + sorted[offset + *object_index]].interval.x :
						objects[offset + sorted[offset + *object_index]].interval.y) <= left_end) {
		if (addition) {
			add_coeffs(coeffs[index == 0 ? 0 : get_current_index(index)],
					source_coeffs, objects[offset + sorted[offset + *object_index]].coeffs,
					POLY_COEFFS_NUM);
		} else {
			subtract_coeffs(coeffs[index == 0 ? 0 : get_current_index(index)],
					source_coeffs, objects[offset + sorted[offset + *object_index]].coeffs,
					POLY_COEFFS_NUM);
		}
		/* Source coefficient are current coefficients from now on. */
		source_coeffs = coeffs[index == 0 ? 0 : get_current_index(index)];
		(*object_index)++;
	}

	return source_coeffs;
}


void update_coefficients(private vfloat coeffs[3][POLY_DEG + 1],
			global ushort *left, global ushort *right, global poly_object *objects,
			uint offset,
			vfloat left_end, unsigned int *left_index,
			unsigned int *right_index, unsigned int index,
			unsigned int size) {
	/*
	 * Add the upcoming interval coefficients and remove the coefficients from
	 * past intervals.
	 */
	vfloat *src_coeffs = coeffs[index == 0 ? 0 : get_left_index(index)];

	/* If we add some coefficient, current coefficients become the source
	 * coefficients, otherwise we would discard the changes in addition. */
	src_coeffs = adjust_coefficients(coeffs, src_coeffs, objects, left, left_end,
											left_index, offset, index, size, true);
	adjust_coefficients(coeffs, src_coeffs, objects, right, left_end, right_index,
														offset, index, size, false);
}


/*
 * Metaballs calculation via polynomial root finding. One can either
 * calculate the projected thickness or the real intersections. The output
 * buffer must have correct size depending on the output type, i.e. for
 * intersections w x h x MAX_OBJECTS x 2. In case the output are the
 * intersections, after the last valid intersection INFINITY follows.
 * @out_thickness: 1 for projected thickness, 0 for intersections.
 */
__kernel void metaballs(__global vfloat *out,
						__constant vfloat4 *objects,
						global poly_object *pobjects,
						global ushort *left,
						global ushort *right,
						const int num_objects,
						const int2 gl_offset,
						const int4 roi,
						const vfloat2 pixel_size,
						const int out_thickness) {
	int ix = get_global_id(0);
	int iy = get_global_id(1);
	unsigned int mem_index = get_global_size(0) * iy + ix;
	uint obj_offset = mem_index * MAX_OBJECTS;

	/* +2 for interval beginning and end, +1 for radius^2 storage
	 * for case there is only one metaobject at this pixel.
	 */
	vfloat poly[POLY_COEFFS_NUM + 3];
	vfloat coeffs[3][POLY_DEG + 1];
	vfloat left_end, right_end, previous_end;
	/* + 1 root for non-continuous function compensation. */
	vfloat roots[POLY_DEG + 1], intersections[POLY_DEG + 1];
	vfloat previous = NAN, thickness = 0, radius;
	int last_derivative_sgn = -2, last_valid_object;
	unsigned int left_index = 0, right_index = 0, size = 0, i, index = 0;
	unsigned int intersection_index = 0, inter_i;
	vfloat2 obj_coords;

    coeffs[0][0] = NAN;
    coeffs[1][0] = NAN;
    coeffs[2][0] = NAN;

	if (roi.x <= ix + gl_offset.x && ix + gl_offset.x < roi.z &&
			roi.y <= iy + gl_offset.y && iy + gl_offset.y < roi.w) {
		/* We are in the FOV. */
		/* transform pixel to object point in mm first */
		obj_coords.x = (ix + gl_offset.x) * pixel_size.x;
		obj_coords.y = (iy + gl_offset.y) * pixel_size.y;

		for (i = 0; i < num_objects; i++) {
			get_meta_ball_equation(poly, &objects[i], &obj_coords, pixel_size.x);
			if (!isnan(poly[5])) {
				init_poly_object(&pobjects[obj_offset + size], poly);
				po_add(left, pobjects, obj_offset, size, X_SORT);
				po_add(right, pobjects, obj_offset, size, Y_SORT);
				last_valid_object = i;
				size++;
			}
            /* If the size is greater than the maximum supported object size return NAN */
            if (size == MAX_OBJECTS) {
                if (out_thickness) {
                    out[mem_index] = NAN;
                }
                else {
                    out[2 * obj_offset] = NAN;
                }
                return;
            }
		}

		switch (size) {
			case 0:
				/* nothing in this pixel */
				if (out_thickness) {
                    out[mem_index] = 0.0;
                }
                else {
                    out[2 * obj_offset] = INFINITY;
                }
				break;
			case 1:
				/* only one metaobject in this pixel */
				radius = objects[last_valid_object].w * objects[last_valid_object].w - poly[7];
				if (radius < 0.0) {
					radius = 0.0;
				}
                if (out_thickness) {
                    out[mem_index] = 2 * sqrt(radius);
                }
                else {
                    if (objects[last_valid_object].z + sqrt(radius) > 0) {
                        out[2 * obj_offset] = objects[last_valid_object].z - sqrt(radius);
                        out[2 * obj_offset + 1] = objects[last_valid_object].z + sqrt(radius);
                    }
                    else {
                        out[2 * obj_offset] = INFINITY;
                    }
                }
				break;
			default:
				/* more than one metaobject, we need to solve the quartic */
				po_sort(left, pobjects, obj_offset, size, X_SORT);
				po_sort(right, pobjects, obj_offset, size, Y_SORT);

				for (i = 0; i < POLY_DEG + 1; i++) {
					coeffs[0][i] = 0.0;
				}
				coeffs[0][4] = -1;

				/* intervals */
				left_end = pobjects[obj_offset + left[obj_offset + left_index]].interval.x;
				previous_end = left_end;
				while (right_index < size) {
					update_coefficients(coeffs, left, right, pobjects, obj_offset,
					                    left_end, &left_index, &right_index, index, size);

					if (left_index < size &&
					    pobjects[obj_offset + left[obj_offset + left_index]].interval.x <
						pobjects[obj_offset + right[obj_offset + right_index]].interval.y) {
						right_end = pobjects[obj_offset + left[obj_offset + left_index]].interval.x;
					} else if(right_index < size) {
						right_end = pobjects[obj_offset + right[obj_offset + right_index]].interval.y;
					} else {
						coeffs[get_right_index(index - 1)][0] = NAN;
					}

					if (index > 0) {
						get_roots(coeffs[get_left_index(index -  1)],
									coeffs[get_current_index(index - 1)],
									coeffs[get_right_index(index - 1)],
									POLY_DEG, previous_end,
									left_end, roots, pixel_size.x);
						get_intersections(intersections,
								(const vfloat *)
								coeffs[get_current_index(index - 1)],
								POLY_DEG, roots, &previous,
								&last_derivative_sgn);
						if (out_thickness) {
                            thickness += get_thickness(intersections);
                        }
                        else {
                            inter_i = 0;
                            while(!isnan(intersections[inter_i])) {
                                out[2 * obj_offset + intersection_index] = intersections[inter_i];
                                inter_i++;
                                intersection_index++;
                            }
                        }
					}
					if (index > 0) {
						previous_end = left_end;
					}

					left_end = right_end;
					index++;
				}
			if (out_thickness) {
                out[mem_index] = thickness;
            }
            else if (intersection_index < 2 * MAX_OBJECTS) {
                out[2 * obj_offset + intersection_index] = INFINITY;
            }
		}
	} else {
		/* No geometry calculation in this pixel but we want it to be zero. */
		if (out_thickness) {
            out[mem_index] = 0;
        }
        else {
            out[2 * obj_offset] = INFINITY;
        }
	}
}


/**
 * A naive metaballs projections calculation kernel.
 * @thickness: the projected thickness 2D buffer
 * @objects: (x, y, z, radius) tuples representing the metaballs
 * @num_objects: number of metaballs
 * @z_range: start and end of the ray in the z-direction in physical units
 * @pixel_size: pixel size in physical units
 */

__kernel void naive_metaballs(__global vfloat *thickness,
                        __constant vfloat4 *objects,
                        const uint num_objects,
                        const vfloat2 z_range,
                        const vfloat pixel_size,
                        uint out_thickness) {
    int ix = get_global_id(0);
    int iy = get_global_id(1);
	uint mem_index = get_global_size(0) * iy + ix;
	uint obj_offset = mem_index * MAX_OBJECTS;
    int m, inside, num_transitions, intersection_index = 0;
    vfloat z, isosurface, dist, start, result;
    vfloat4 point;

    point.x = ix * pixel_size;
    point.y = iy * pixel_size;
    inside = 0;
    result = 0.0;

    for (z = z_range.x; z <= z_range.y; z += pixel_size) {
        isosurface = 0.0;
        point.z = z;
        for (m = 0; m < num_objects; m++) {
            dist = sqrt((objects[m].x - point.x) * (objects[m].x - point.x) +
                        (objects[m].y - point.y) * (objects[m].y - point.y) +
                        (objects[m].z - point.z) * (objects[m].z - point.z));
            if (dist <= 2 * objects[m].w) {
                isosurface += (4 * objects[m].w * objects[m].w - dist * dist) *
                            (4 * objects[m].w * objects[m].w - dist * dist) /
                            (9 * objects[m].w * objects[m].w * objects[m].w * objects[m].w);
            }
        }
        if (inside == 0 && isosurface >= 1.0) {
            inside = 1;
            if (out_thickness) {
                start = z;
            }
            else {
                thickness[2 * obj_offset + intersection_index] = z;
                intersection_index++;
                if (intersection_index == 2 * MAX_OBJECTS) {
                    thickness[2 * obj_offset] = NAN;
                    return;
                }
            }
        }
        if (inside == 1 && isosurface < 1.0) {
            inside = 0;
            if (out_thickness) {
                result += z - start;
            }
            else {
                thickness[2 * obj_offset + intersection_index] = z;
                intersection_index++;
            }
        }
    }

    if (out_thickness) {
        thickness[mem_index] = result;
    }
    else if (intersection_index < 2 * MAX_OBJECTS) {
        thickness[2 * obj_offset + intersection_index] = INFINITY;
    }
}


/**
  * Metaball intersections to slice conversion kernel.
  */
__kernel void intersections_to_slice(__global uchar *slice,
                                     __global vfloat *intersections,
                                     const uint height,
                                     const vfloat z_start,
                                     const vfloat pixel_size) {
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int width = get_global_size(0);
	uint mem_index = width * iy + ix;
	uint offset = (height * width + ix) * 2 * MAX_OBJECTS;
	uint i = 0;
	uchar inside = 0;
	vfloat value = intersections[offset + i];
    vfloat point = iy * pixel_size + z_start;

    while (!isinf(value) && i < 2 * MAX_OBJECTS) {
        if (value <= point && point <= intersections[offset + i + 1]) {
            inside = 1;
        }
        i += 2;
        value = intersections[offset + i];
    }

    slice[mem_index] = inside;
}
