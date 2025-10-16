"Commons.cuh"

__device__ void vf_swap(FP_T *array, int i, int j) {
	FP_T tmp;

	tmp = array[i];
	array[i] = array[j];
	array[j] = tmp;
}

__device__ void vf_sift_down(FP_T *heap, int start, int end) {
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

__device__ void vf_heapify(FP_T *array, int size) {
	int start = (size - 2) / 2;

	while (start >= 0) {
		vf_sift_down(array, start, size - 1);
		start--;
	}
}

__device__ void sort(FP_T *array, int size) {
	vf_heapify(array, size);
    int end = size - 1;

    while (end > 0) {
    	vf_swap(array, 0, end);
        vf_sift_down(array, 0, end - 1);
        end--;
    }
}


extern "C" __global__ void sort_kernel(FP_T *array) {
	FP_T ar[10];
	int i;

	for (i = 0; i < 10; i++) {
		ar[i] = array[i];
	}

	sort(ar, 10);

	for (i = 0; i < 10; i++) {
		array[i] = ar[i];
	}
}
