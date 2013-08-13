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


/*
 * Calculate the exponent of a transmission function T(x,y). The exponent
 * will be evaluated when all coefficients and thicknesses are taken
 * into account.
 */
__kernel void transmission_add(__global vcomplex *transmissions,
								__global vfloat *thickness,
								const vcomplex refractive_index,
								const int clear) {
	int ix = get_global_id(0);
	int iy = get_global_id(1);
	int mem_index = iy * get_global_size(0) + ix;

	if (clear) {
		transmissions[mem_index] = (vcomplex)(
							thickness[mem_index] * refractive_index.x,
							thickness[mem_index] * refractive_index.y);
	} else {
		transmissions[mem_index] = (vcomplex)(transmissions[mem_index].x +
					thickness[mem_index] * refractive_index.x,
					transmissions[mem_index].y +
					thickness[mem_index] * refractive_index.y);
	}
}

/*
 * Transfer function T(x,y), which defines photon absorption
 * and phase changes. It depends on wavelength and material.
 */
__kernel void transfer(__global vcomplex *transmission_coeffs,
						const vfloat lambda) {
	int ix = get_global_id(0);
	int iy = get_global_id(1);
	int width = get_global_size(0);

	vfloat phase, absorp, e_a, k;
	k = -2 * M_PI / lambda;

	/* Imaginary part - phase. */
	phase = k * (transmission_coeffs[iy * width + ix].x);
	/* Real part - absorption. */
	absorp = k * (transmission_coeffs[iy * width + ix].y);
	e_a = exp(absorp);

	transmission_coeffs[iy * width + ix] = (vcomplex)(e_a * cos(phase),
														e_a * sin(phase));
}
