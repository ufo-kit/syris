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
    wavefield[mem_index] = (vcomplex)(exp_absorp * cosine, exp_absorp * sine);
}


/*
 * Calculate the exponent of a transmission function T(x,y). The exponent
 * will be evaluated when all coefficients and thicknesses are taken
 * into account.
 */
__kernel void transmission_add(__global vcomplex *transmissions,
							   __global vfloat *thickness,
							   const vcomplex refractive_index,
							   const vfloat wavenumber,
							   const int clear) {
	int ix = get_global_id(0);
	int iy = get_global_id(1);
	int mem_index = iy * get_global_size(0) + ix;
    vcomplex current = (vcomplex)(-wavenumber * thickness[mem_index] * refractive_index.y,
							      -wavenumber * thickness[mem_index] * refractive_index.x);

	if (clear) {
		transmissions[mem_index] = current;
	} else {
		transmissions[mem_index] = transmissions[mem_index] + current;
	}
}


/*
 * Make flat field wavefield out of a vertical profile of intensities.
 */
__kernel void make_flat(__global vcomplex *output,
                        __global vfloat *input,
                        const vfloat3 center,
                        const vfloat2 pixel_size,
                        const vfloat z,
                        const vfloat lambda,
                        const int exponent,
                        const int phase,
                        const int parabola) {
	int ix = get_global_id(0);
	int iy = get_global_id(1);
    vfloat x, y, c_phi, s_phi, r, real, imag;
    vfloat amplitude = sqrt(input[iy]);
    amplitude = sqrt(input[iy]);

    if (phase) {
        x = ix * pixel_size.x - center.x;
        y = iy * pixel_size.y - center.y;
        if (parabola) {
            r = (x * x + y * y) / (2 * z);
        }
        else {
            r = sqrt(x * x + y * y + z * z);
        }
    } else {
        r = 0;
    }

    if (exponent) {
        real = log(amplitude);
        imag = 2 * M_PI / lambda * r;
    } else {
        s_phi = sincos(2 * M_PI / lambda * r, &c_phi);
        real = amplitude * c_phi;
        imag = amplitude * s_phi;
    }

    output[iy * get_global_size(0) + ix] = (vfloat2)(real, imag);
}

/*
 * Check the sampling of a transfer function
 */
__kernel void check_transmission_function(__global vfloat *exponent,
                                          __global bool *out,
                                          const int width,
                                          const int height) {
	int ix = 2 * get_global_id(0);
	int iy = 2 * get_global_id(1);
    int index = iy * width + ix;
    int dx, dy;
    vfloat current;

    if (exponent[index] != 0) {
        for (dx = -1; dx < 2; dx++) {
            if (ix + dx >= 0 && ix + dx < width) {
                for (dy = -1; dy < 2; dy++) {
                    if (iy + dy >= 0 && iy + dy < height) {
                        current = exponent[(iy + dy) * width + ix + dx];
                        if (current != 0 && fabs(current - exponent[index]) >= M_PI) {
                            out[index] = true;
                        }
                    }
                }
            }
        }
    }
}
