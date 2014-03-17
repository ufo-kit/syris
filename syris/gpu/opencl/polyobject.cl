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
void init_poly_object(poly_object *po, vfloat coeffs[POLY_COEFFS_NUM+2]) {
	int i;

	for (i = 0; i < POLY_COEFFS_NUM; i++) {
		po->coeffs[i] = coeffs[i];
	}
	po->interval = (vfloat2)(coeffs[POLY_COEFFS_NUM], coeffs[POLY_COEFFS_NUM+1]);
}
