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
 * Polynomials sorting using heapsort.
 *
 * Requires the normal heapsort.cl file for heap_parent function.
 */

typedef enum {
	Y_SORT, X_SORT
} sort_type;


void po_sift_down(global ushort *heap, global poly_object *objects,
                  uint offset, int start, int end, sort_type st) {
    int root = start;
    int child;
    ushort tmp;

    while (2 * root + 1 <= end) {
        child = 2 * root + 1;
        if (child + 1 <= end && (st ?
            objects[offset + heap[offset + child]].interval.x <
            objects[offset + heap[offset + child + 1]].interval.x :
            objects[offset + heap[offset + child]].interval.y <
            objects[offset + heap[offset + child + 1]].interval.y)) {
            child++;
        }
        if (child <= end && (st ?
            objects[offset + heap[offset + root]].interval.x <
            objects[offset + heap[offset + child]].interval.x :
            objects[offset + heap[offset + root]].interval.y <
            objects[offset + heap[offset + child]].interval.y)) {
            tmp = heap[offset + root];
            heap[offset + root] = heap[offset + child];
            heap[offset + child] = tmp;
            root = child;
        } else {
            return;
        }
    }
}

void po_sort(global ushort *heap, global poly_object *objects,
             uint offset, int size, sort_type st) {
    int end = size - 1;
    ushort tmp;

    while (end > 0) {
        tmp = heap[offset];
        heap[offset] = heap[offset + end];
        heap[offset + end] = tmp;
        po_sift_down(heap, objects, offset, 0, end - 1, st);
        end--;
    }
}

int heap_parent(int index) {
    return (index - 1)/2;
}

void po_add(global ushort *heap, global poly_object *objects, uint offset,
            int index, sort_type st) {
    int parent_index, next = index;
    ushort tmp;
    heap[offset + next] = index;

    while (next > 0 && (st ?
           objects[offset + heap[offset + heap_parent(next)]].interval.x <
           objects[offset + index].interval.x :
           objects[offset + heap[offset + heap_parent(next)]].interval.y <
           objects[offset + index].interval.y)) {
        parent_index = heap_parent(next);
        tmp = heap[offset + parent_index];
        heap[offset + parent_index] = heap[offset + next];
        heap[offset + next] = tmp;
        next = parent_index;
    }
}
