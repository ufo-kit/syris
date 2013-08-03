/*
 * Physical calculations in OpenCL.
 *
 * Requires definition of vfloat data type, which defines single or double
 * precision for floating point numbers. Also requires vcomplex data type
 * for working with complex numbers.
 */

/*
 * Fresnel approximated wavefield propagation.
 */
__kernel void propagator(__global vcomplex *out,
							const vfloat distance,
							const vfloat lam,
							const vfloat pixel_size,
							const vcomplex phase_factor) {

	int ix = get_global_id(0);
	int iy = get_global_id(1);
	int n = get_global_size(0);
	vfloat i, j, tmp;
	vcomplex result, c_tmp;

	/* Map image coordinates to Fourier coordinates. */
	i = -0.5 + ((vfloat) ix) / n;
	j = -0.5 + ((vfloat) iy) / n;

	/*
	 * Fresnel propagator in the Fourier domain:
	 *
	 * \begin{align}
	 * F(i, j) = e ^ {\frac{2 * pi * distance * i} {lam}} *
	 * e ^ {- i * \pi * lam * distance * (i ^ 2 + j ^ 2)}.
	 * \end{align}
	 */
	tmp = - M_PI * lam * distance * (i * i + j * j) /
			(pixel_size * pixel_size);
	if (phase_factor.x == 0 && phase_factor.y == 0) {
		result = (vcomplex)(cos(tmp), sin(tmp));
	} else {
		c_tmp = (vcomplex)(cos(tmp), sin(tmp));
		result = vc_mul(&phase_factor, &c_tmp);
	}

	/* Lowest frequencies are in the corners. */
	out[n * ((iy + n / 2) % n) + ((ix + n / 2) % n)] = result;
}
