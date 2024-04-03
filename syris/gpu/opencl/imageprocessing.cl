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
 * Butterworth filter
 */
__kernel void butterworth(__global vfloat *out, const vfloat cutoff, const int order) {
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);
    int x = (ix >= width >> 1) ? ix - width : ix;
    int y = (iy >= height >> 1) ? iy - height : iy;
    vfloat dist = sqrt ((vfloat) x * x + y * y);

    out[iy * width + ix] = 1.0 / (1.0 + pown (dist / cutoff, 2 * order));
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
                                                       (ix / factor.x + 1.0f / (2 * factor.x),
                                                        iy / factor.y + 1.0f / (2 * factor.y))).x;
}

/*
 * Compute intensity of a wavefield.
 */
__kernel void compute_intensity (__global vcomplex *wavefield,
                                 __global vfloat *out)
{
    int ix = get_global_id (0);
    int iy = get_global_id (1);
    int index = iy * get_global_size(0) + ix;
    vcomplex value = wavefield[index];

    out[index] = value.x * value.x + value.y * value.y;
}
