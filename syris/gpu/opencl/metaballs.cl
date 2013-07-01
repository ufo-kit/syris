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
