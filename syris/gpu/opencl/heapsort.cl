/*
 * Fast sorting of vfloat arrays in O(n log(n)) time. Implemented
 * by heapsort.
 *
 * Requires definition of vfloat data type, which defines single or double
 * precision for vfloating point numbers.
 */

void vf_swap(vfloat *array, int i, int j) {
	vfloat tmp;

	tmp = array[i];
	array[i] = array[j];
	array[j] = tmp;
}

void vf_sift_down(vfloat *heap, int start, int end) {
    int root = start;
    int child;

    while (root*2 + 1 <= end) {
        child = root*2 + 1;
        if (child + 1 <= end && (heap[child] < heap[child + 1] ||
        								isnan(heap[child + 1]))) {
            child++;
        }
        if (child <= end && (heap[root] < heap[child] || isnan(heap[child]))) {
        	vf_swap(heap, root, child);
            root = child;
        } else {
            return;
        }
    }
}

void vf_heapify(vfloat *array, int size) {
	int start = (size - 2) / 2;

	while (start >= 0) {
		vf_sift_down(array, start, size - 1);
		start--;
	}
}

void vf_sort(vfloat *array, int size) {
	vf_heapify(array, size);
    int end = size - 1;

    while (end > 0) {
    	vf_swap(array, 0, end);
        vf_sift_down(array, 0, end - 1);
        end--;
    }
}


__kernel void sort_kernel(__global vfloat *array) {
	vfloat ar[10];
	int i;

	for (i = 0; i < 10; i++) {
		ar[i] = array[i];
	}

	vf_sort(ar, 10);

	for (i = 0; i < 10; i++) {
		array[i] = ar[i];
	}
}
