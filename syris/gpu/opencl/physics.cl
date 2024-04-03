/*
 * Copyright (C) 2013-2023 Karlsruhe Institute of Technology

 * This file is part of syris.

 * This library is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.

 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.

 * You should have received a copy of the GNU Lesser General Public
 * License along with this library. If not, see <http://www.gnu.org/licenses/>.
 */

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
							const vfloat2 pixel_size,
							const vcomplex phase_factor,
                            const int fresnel,
                            const int fourier) {

	int ix = get_global_id(0);
	int iy = get_global_id(1);
    int patch = get_global_size (0) - 1;
	int n = 2 * patch;
	vfloat i, j, tmp, sine, cosine;
	vcomplex result, c_tmp;

    if (fourier) {
        /* Map image coordinates to Fourier coordinates. Sign doesn't matter
         * because of the square in the computation. */
        i = (vfloat) ix / n;
        j = (vfloat) iy / n;
        vfloat coords = (i * i / (pixel_size.x * pixel_size.x) + j * j / (pixel_size.y * pixel_size.y));
        if (fresnel) {
            tmp = - M_PI * lam * distance * coords;
        } else {
            tmp = 2 * M_PI / lam * distance * sqrt(1 - lam * lam * coords);
        }
    } else {
        vfloat coords = (ix * ix * pixel_size.x * pixel_size.x + iy * iy * pixel_size.y * pixel_size.y);
        tmp = M_PI / (lam * distance) * coords;
    }
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

void make_flat_field(__global vcomplex *output,
                     int ix,
                     int iy,
                     const vfloat flux,
                     const vfloat3 *center,
                     const vfloat2 *pixel_size,
                     const vfloat z,
                     const vfloat lambda,
                     const int exponent,
                     const int phase,
                     const int parabola)
{
    vfloat2 result;
    vfloat x, y, c_phi, s_phi, r, real, imag;
    vfloat amplitude = sqrt(flux);

    if (phase) {
        x = ix * pixel_size->x - center->x;
        y = iy * pixel_size->y - center->y;
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
 * Make flat field wavefield out of a scalar intensity value.
 */
__kernel void make_flat_from_scalar(__global vcomplex *output,
                                    const vfloat flux,
                                    const vfloat3 center,
                                    const vfloat2 pixel_size,
                                    const vfloat z,
                                    const vfloat lambda,
                                    const int exponent,
                                    const int phase,
                                    const int parabola) {
	int ix = get_global_id(0);
	int iy = get_global_id(1);

    make_flat_field (output, ix, iy, flux, &center, &pixel_size, z,
                     lambda, exponent, phase, parabola);
}

/*
 * Make flat field wavefield out of a vertical profile of intensities.
 */
__kernel void make_flat_from_vertical_profile(__global vcomplex *output,
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

    make_flat_field (output, ix, iy, input[iy], &center, &pixel_size, z,
                     lambda, exponent, phase, parabola);
}

/*
 * Make flat field wavefield out of a 2D profile of intensities.
 */
__kernel void make_flat_from_2D_profile(__global vcomplex *output,
                                        read_only image2d_t input,
                                        const sampler_t sampler,
                                        const vfloat3 center,
                                        const vfloat2 pixel_size,
                                        const vfloat2 input_pixel_size,
                                        const vfloat z,
                                        const vfloat lambda,
                                        const int exponent,
                                        const int phase,
                                        const int parabola) {
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    float input_width = (float) get_image_width (input);
    float input_height = (float) get_image_height (input);
    float2 factor = (float2) (pixel_size.x / input_pixel_size.x,
                              pixel_size.y / input_pixel_size.y);
    float x = (ix * pixel_size.x - center.x) / input_pixel_size.x + input_width / 2.0f + 0.5f * factor.x;
    float y = (iy * pixel_size.y - center.y) / input_pixel_size.y + input_height / 2.0f + 0.5f * factor.y;

    vfloat flux = read_imagef (input, sampler, (float2) (x, y)).x * factor.x * factor.y;

    make_flat_field (output, ix, iy, flux, &center, &pixel_size, z, lambda,
                     exponent, phase, parabola);
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
