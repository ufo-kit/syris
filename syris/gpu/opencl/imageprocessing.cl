/*
 * Image processing routines on OpenCL.
 *
 * Requires definition of vfloat data type, which defines single or double
 * precision for floating point numbers.
 */


/*
 * 2D Gaussian in real space.
 */
__kernel void gauss_2d(__global vfloat *out, const vfloat2 sigma, const vfloat2 pixel_size) {
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    vfloat x = ix < width / 2 + width % 2 ? ix * pixel_size.x : (ix - width) * pixel_size.x;
    vfloat y = iy < height / 2 + height % 2 ? iy * pixel_size.y : (iy - height) * pixel_size.y;

    out[iy * width + ix] = exp (- x * x / (2 * sigma.x * sigma.x) - y * y / (2 * sigma.y * sigma.y));
}

/*
 * 2D Gaussian in Fourier space.
 */
__kernel void gauss_2d_f(__global vfloat *out, const vfloat2 sigma, const vfloat2 pixel_size) {
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    vfloat i = (ix < width / 2 + width % 2 ? ix / pixel_size.x :
                (ix - width) / pixel_size.x) / ((float) width);
    vfloat j = (iy < height / 2 + height % 2 ? iy / pixel_size.y :
                (iy - height) / pixel_size.y) / ((float) height);

	/* Fourier transform of a Gaussian is a stretched Gaussian.
	 * We assume a Gaussian with standard deviation c to be
	 *
	 * $g(x) = \frac{1}{c * sqrt(2 * \pi)} e ^ {\frac{- x ^ 2} {2 * c ^ 2}}$
	 *
	 * Given the Fourier transform
	 *
	 * $F(xi) = \int f(x) e ^ {- 2 * \pi * x * xi * i} \, \dif x,$
	 *
	 * the fourier transform of a gaussian g(x) is
	 *
	 * $G(xi) = e ^ {- 2 * \pi ^ 2 * c ^ 2 * xi ^ 2}.$
	 *
	 * The description is for brevity for 1D case.
	 */
    out[iy * width + ix] = exp(- 2 * M_PI * M_PI * (sigma.x * sigma.x * i * i +
                                                    sigma.y * sigma.y * j * j));
}

/*
 * Sum image over a given region.
 */
__kernel void sum(__global vfloat *out,
                        __global vfloat *in,
                        const int2 region,
                        const int orig_width,
                        const int2 offset,
                        const int average) {
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int i, j, width;
    vfloat value = 0.0;

    for (j = 0; j < region.y; j++) {
    	for (i = 0; i < region.x; i++) {
    		value += in[orig_width * (iy * region.y + offset.y + j) +
    		            					ix * region.x + i + offset.x];
        }
    }

    if (average) {
    	value /= (vfloat)(region.x * region.y);
    }

    out[iy * get_global_size(0) + ix] = value;
}

/*
 * Rescale an image.
 */
__kernel void rescale (read_only image2d_t input,
                       __global float *output,
                       const sampler_t sampler,
                       const vfloat2 factor)
{
    int ix = get_global_id (0);
    int iy = get_global_id (1);

    output[iy * get_global_size(0) + ix] = read_imagef(input, sampler, (float2)
                                                       (ix / factor.x + 0.5f,
                                                        iy / factor.y + 0.5f)).x;
}
