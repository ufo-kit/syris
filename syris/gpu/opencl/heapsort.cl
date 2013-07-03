/*
 * Heap sort for fast sorting in O(n log(n)) time.
 */

typedef enum {
	Y_SORT, X_SORT
} sort_type;


void sift_down(poly_object **heap, int start, int end, sort_type st) {
    int root = start;
    int child;
    poly_object *tmp;

    while (root*2 + 1 <= end) {
        child = root*2 + 1;
        if (child + 1 <= end && (st ?
        		heap[child]->interval.x < heap[child + 1]->interval.x :
        		heap[child]->interval.y < heap[child + 1]->interval.y)) {
            child++;
        }
        if (child <= end && (st ?
	    		heap[root]->interval.x < heap[child]->interval.x :
	    		heap[root]->interval.y < heap[child]->interval.y)) {
            tmp = heap[root];
            heap[root] = heap[child];
            heap[child] = tmp;
            root = child;
        } else {
            return;
        }
    }
}

void sort(poly_object **heap, int size, sort_type st) {
    int end = size - 1;
    poly_object *tmp;

    while (end > 0) {
        tmp = heap[0];
        heap[0] = heap[end];
        heap[end] = tmp;
        sift_down(heap, 0, end - 1, st);
        end--;
    }
}

int heap_parent(int index) {
    return (index - 1)/2;
}

void add(poly_object **heap, poly_object *val, int next, sort_type st) {
    heap[next] = val;
    int parent_index;
    poly_object *tmp;

    while (next > 0 && (st ?
    		heap[heap_parent(next)]->interval.x < val->interval.x :
    		heap[heap_parent(next)]->interval.y < val->interval.y)) {
        parent_index = heap_parent(next);
        tmp = heap[parent_index];
        heap[parent_index] = heap[next];
        heap[next] = tmp;
        next = parent_index;
    }
}
