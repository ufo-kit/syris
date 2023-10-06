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

#define EPSILON 1e-7
#define MAX_INTERSECTIONS 200

/*
 * Moeller-Trumbore intersection algorithm
 */
vfloat compute_intersection_point (const global vfloat3 *V_1,
                                  const global vfloat3 *V_2,
                                  const global vfloat3 *V_3,
                                  const vfloat3 *O,
                                  const vfloat3 *D)
{
    vfloat3 e_1, e_2, P, Q, T;
    vfloat det, inv_det, u, v, t;

    e_1 = *V_2 - *V_1;
    e_2 = *V_3 - *V_1;
    P = cross (*D, e_2);
    det = dot (e_1, P);
    //if (fabs (det) < EPSILON) {
    if (det == 0.0) {
        return -1;
    }

    inv_det = 1 / det;
    T = *O - *V_1;
    u = dot (T, P) * inv_det;
    if (u < 0 || u > 1) {
        return -1;
    }

    Q = cross (T, e_1);
    v = dot (*D, Q) * inv_det;
    if (v < 0 || u + v > 1) {
        return -1;
    }

    t = dot (e_2, Q) * inv_det;
    if (t > EPSILON) {
        return t;
    }

    return -1.0;
}


int find_leftmost (const global vfloat3 *v_1,
                   const global vfloat3 *v_2,
                   const global vfloat3 *v_3,
                   int x_0,
                   int x_1,
                   vfloat value)
{
    int i;

    while (x_0 <= x_1) {
        i = (x_0 + x_1) / 2;
        if (v_3[i].x == value) {
            while (v_3[i].x == value) {
                i--;
            }
            return i;
        } else if (v_3[i].x < value) {
            x_0 = i + 1;
        } else {
            x_1 = i - 1;
        }
    }

    return i;
}

int compute_intersections (const global vfloat3 *v_1,
                           const global vfloat3 *v_2,
                           const global vfloat3 *v_3,
                           const int num_triangles,
                           vfloat3 *O,
                           vfloat3 *D,
                           vfloat max_dx,
                           vfloat *intersections)
{
    int i, num_intersections = 0;
    vfloat current;
    // Make margins 1 px left and right
    vfloat xp = O->x + 1;
    vfloat xm = O->x - 1;
    vfloat yp = O->y + 1;
    vfloat ym = O->y - 1;
    vfloat stop = xp + max_dx;

    /* Find the index for which all the triangles have already ended in
     * x-direction */
    i = find_leftmost (v_1, v_2, v_3, 0, num_triangles, xm);
    /* Continue the search until we reach max_dx which is the largest triangle
     * in x-direction, this way we are sure that if we search until ray.x +
     * max_dx we have searched all the triangles starting to the left from the ray */
    while (i < num_triangles && v_1[i].x <= stop && v_2[i].x <= stop && v_3[i].x <= stop) {
        if (!((v_1[i].x < xm && v_2[i].x < xm && v_3[i].x < xm) ||
              (v_1[i].x > xp && v_2[i].x > xp && v_3[i].x > xp) ||
              (v_1[i].y < ym && v_2[i].y < ym && v_3[i].y < ym) ||
              (v_1[i].y > yp && v_2[i].y > yp && v_3[i].y > yp))) {
            /* There is a ray-bounding box intersection */
            current = compute_intersection_point (&v_1[i], &v_2[i], &v_3[i], O, D); 
            if (current > -1) {
                intersections[num_intersections] = current;
                num_intersections++;
                if (num_intersections == MAX_INTERSECTIONS) {
                    break;
                }
            }
        }
        i++;
    }

    vf_sort (intersections, num_intersections);

    return num_intersections;
}

vfloat project_thickness (vfloat *intersections, int num_intersections)
{
    int i, j;
    vfloat result = 0.0;

    i = 0;
    while (i < num_intersections) {
        j = i + 1;
        while (j < num_intersections && fabs (intersections[j] - intersections[i]) < EPSILON) {
            j++;
        }
        if (i < num_intersections && j < num_intersections) {
            result += fabs (intersections[j] - intersections[i]);
        }
        i = j + 1;
    }

    return result;
}



int remove_duplicates (vfloat *array, int num_elements, vfloat eps)
{
    int real_num = 0;
    int i = 0;
    int j;

    while (i < num_elements) {
        array[real_num] = array[i];
        real_num++;
        j = i + 1;
        while (j < num_elements && fabs(array[j] - array[i]) < eps) {
            j++;
        }
        i = j;
    }

    return real_num;
}


void fill_slice (vfloat *intersections,
                 global uchar *slices,
                 int num_intersections,
                 int idx,
                 int idy,
                 int width,
                 int depth)
{
    int i, j, z, z_start, z_stop;
    int slice_offset = width * depth * idy;

    for (i = 0; i < num_intersections - 1; i += 2) {
        z_start = max (0, (int) (intersections[i] + .5));
        z_stop = min (depth, (int) (intersections[i + 1] + .5));
        for (z = z_start; z < z_stop; z++) {
            slices[slice_offset + z * width + idx] = 1;
        }
    }
}

kernel void compute_thickness (const global vfloat3 *v_1,
                               const global vfloat3 *v_2,
                               const global vfloat3 *v_3,
                               global vfloat *output,
                               const int num_triangles,
                               const int image_width,
                               const int2 offset,
                               const vfloat2 mesh_offset,
                               const vfloat scale,
                               const vfloat max_dx,
                               const vfloat min_z,
                               const int supersampling)
{
    int idx = get_global_id (0);
    int idy = get_global_id (1);
    int i, j, num_intersections;
    vfloat results[16];
    vfloat x_0 = idx + offset.x + 0.5 + mesh_offset.x;
    vfloat y_0 = idy + offset.y + 0.5 + mesh_offset.y;
    vfloat3 O, D = (vfloat3)(0, 0, 1);
    vfloat intersections[MAX_INTERSECTIONS];
    O.z = min_z - 1;

    for (i = 0; i < supersampling; i++) {
        for (j = 0; j < supersampling; j++) {
            O.x = (2. * i - supersampling + 1) / (2 * supersampling) + x_0;
            O.y = (2. * j - supersampling + 1) / (2 * supersampling) + y_0;
            num_intersections = compute_intersections (v_1, v_2, v_3, num_triangles, &O, &D,
                                                       max_dx, intersections);
            if (num_intersections == MAX_INTERSECTIONS) {
                output[(idy + offset.y) * image_width + idx + offset.x] = NAN;
                return;
            }
            results[i * supersampling + j] = project_thickness (intersections, num_intersections);
        }
    }

    vf_sort (results, supersampling * supersampling);
    output[(idy + offset.y) * image_width + idx + offset.x] = scale * results[supersampling * supersampling / 2];
}

kernel void compute_slices (const global vfloat3 *v_1,
                         const global vfloat3 *v_2,
                         const global vfloat3 *v_3,
                         global uchar *output,
                         const int depth,
                         const int num_triangles,
                         const vfloat3 offset,
                         const vfloat max_dx)
{
    int idx = get_global_id (0);
    int idy = get_global_id (1);
    int width = get_global_size (0);
    int z, num_intersections;
    vfloat intersections[MAX_INTERSECTIONS];
    vfloat3 O;
    vfloat3 D = (vfloat3)(0, 0, 1);
    O.x = idx + 0.5 + offset.x;
    O.y = idy + 0.5 + offset.y;
    O.z = offset.z;

    num_intersections = compute_intersections (v_1, v_2, v_3, num_triangles, &O, &D,
                                               max_dx, intersections);
    if (num_intersections == MAX_INTERSECTIONS) {
        for (z = 0; z < depth; z++) {
            output[width * depth * idy + z * width + idx] = NAN;
        }
    } else {
        /* Make epsilon one pixel */
        num_intersections = remove_duplicates (intersections, num_intersections, EPSILON);
        fill_slice (intersections, output, num_intersections, idx, idy, width, depth);
    }
}
