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
    int patch = get_global_size (0) - 1;
	int n = 2 * patch;
	vfloat i, j, tmp, sine, cosine;
	vcomplex result, c_tmp;

	/* Map image coordinates to Fourier coordinates. Sign doesn't matter
     * because of the square in the computation. */
	i = (vfloat) ix / n;
	j = (vfloat) iy / n;

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
    sine = sincos(tmp, &cosine);
	if (phase_factor.x == 0 && phase_factor.y == 0) {
		result = (vcomplex)(cosine, sine);
	} else {
		c_tmp = (vcomplex)(cosine, sine);
		result = vc_mul(&phase_factor, &c_tmp);
	}

    /* Fill all quadrants */
	out[iy * n + ix] = result;
    if (0 < ix && ix < patch) {
        out[iy * n + n - ix] = result;
        if (0 < iy && iy < patch) {
            out[(n - iy) * n + n - ix] = result;
        }
    }
    if (0 < iy && iy < patch) {
	    out[(n - iy) * n + ix] = result;
    }
}


/**
  * Compute object transfer function.
  * @wavefield: resulting complex wavefield
  * @thickness: projected thickness
  * @refractive_index: refractive index of the material
  * @wavelength: wavelength of the refractive index
  */
__kernel void transfer(__global vcomplex *wavefield,
                       __global vfloat *thickness,
                       const vcomplex refractive_index,
                       const vfloat wavelength) {
	int ix = get_global_id(0);
	int iy = get_global_id(1);
	int mem_index = iy * get_global_size(0) + ix;
    vfloat sine, cosine;
    vfloat exponent = - 2 * M_PI * thickness[mem_index] / wavelength;
    vfloat exp_absorp = exp(exponent * refractive_index.y);
    vfloat phase = exponent * refractive_index.x;

    sine = sincos(phase, &cosine);
    wavefield[mem_index] = (vcomplex)(exp_absorp * cosine,
                                      exp_absorp * sine);
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
__kernel void transfer_coeffs(__global vcomplex *transmission_coeffs,
						const vfloat lambda) {
	int ix = get_global_id(0);
	int iy = get_global_id(1);
	int width = get_global_size(0);
    vfloat sine, cosine;

	vfloat phase, absorp, e_a, k;
	k = -2 * M_PI / lambda;

	/* Imaginary part - phase. */
	phase = k * (transmission_coeffs[iy * width + ix].x);
	/* Real part - absorption. */
	absorp = k * (transmission_coeffs[iy * width + ix].y);
	e_a = exp(absorp);
    sine = sincos(phase, &cosine);

	transmission_coeffs[iy * width + ix] = (vcomplex)(e_a * cosine,
														e_a * sine);
}

/*
 * Make flat field wavefield out of a vertical profile of intensities.
 */
__kernel void make_flat(__global vcomplex *output,
                        __global vfloat *input) {
	int ix = get_global_id(0);
	int iy = get_global_id(1);

    output[iy * get_global_size(0) + ix] = (vfloat2)(sqrt(input[iy]), 0);
}
