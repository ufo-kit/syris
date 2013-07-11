/*
 * Image processing routines on OpenCL.
 *
 * Requires definition of vfloat data type, which defines single or double
 * precision for floating point numbers.
 */


/*
 * 2D normalized Gaussian in Fourier space.
 */
__kernel void gauss_2_f(__global vcomplex *out,
						const float2 sigma,
						const vfloat pixel_size) {
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int n = get_global_size(0);
    vfloat i, j;

	/* Map image coordinates to Fourier coordinates. */
	i = -0.5 + ((vfloat) ix) / n;
	j = -0.5 + ((vfloat) iy) / n;

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
    out[n * ((iy + n / 2) % n) + ((ix + n / 2) % n)] = (vcomplex)
    		(exp(- 2 * M_PI * M_PI / (pixel_size * pixel_size) *
				(sigma.x * sigma.x * i * i + sigma.y * sigma.y * j * j)), 0);
}
