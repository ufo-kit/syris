/*
 * Polynomials sorting using heapsort.
 *
 * Requires the normal heapsort.cl file for heap_parent function.
 */

typedef enum {
	Y_SORT, X_SORT
} sort_type;


void po_sift_down(ushort *heap, poly_object *objects, int start, int end, sort_type st) {
    int root = start;
    int child;
    ushort tmp;

    while (2 * root + 1 <= end) {
        child = 2 * root + 1;
        if (child + 1 <= end && (st ?
            objects[heap[child]].interval.x < objects[heap[child + 1]].interval.x :
            objects[heap[child]].interval.y < objects[heap[child + 1]].interval.y)) {
            child++;
        }
        if (child <= end && (st ?
            objects[heap[root]].interval.x < objects[heap[child]].interval.x :
            objects[heap[root]].interval.y < objects[heap[child]].interval.y)) {
            tmp = heap[root];
            heap[root] = heap[child];
            heap[child] = tmp;
            root = child;
        } else {
            return;
        }
    }
}

void po_sort(ushort *heap, poly_object *objects, int size, sort_type st) {
    int end = size - 1;
    ushort tmp;

    while (end > 0) {
        tmp = heap[0];
        heap[0] = heap[end];
        heap[end] = tmp;
        po_sift_down(heap, objects, 0, end - 1, st);
        end--;
    }
}

int heap_parent(int index) {
    return (index - 1)/2;
}

void po_add(ushort *heap, poly_object *objects, int index, sort_type st) {
    int parent_index, next = index;
    ushort tmp;
    heap[next] = index;

    while (next > 0 && (st ?
           objects[heap[heap_parent(next)]].interval.x < objects[index].interval.x :
           objects[heap[heap_parent(next)]].interval.y < objects[index].interval.y)) {
        parent_index = heap_parent(next);
        tmp = heap[parent_index];
        heap[parent_index] = heap[next];
        heap[next] = tmp;
        next = parent_index;
    }
}
