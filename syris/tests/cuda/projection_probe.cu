// Thin entry points for exercising production device functions from Python.
// Geometry and expected results live in the tests; no reference ray caster is
// compiled here.
__device__ EpsilonParams test_eps(const FP_T *e) {
    return EpsilonParams(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8]);
}

extern "C" __global__ void test_boxes(
    int n, const FP_T4 *origins, const FP_T4 *directions,
    const FP_T4 *lower, const FP_T4 *upper, const FP_T2 *ranges,
    const FP_T *eps, int *hits)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    WatertightRay ray(origins[i], directions[i], lower[i], upper[i], test_eps(eps));
    hits[i] = ray.intersects(lower[i], upper[i], ranges[i].x, ranges[i].y);
}

extern "C" __global__ void test_triangles(
    int n, const FP_T4 *origins, const FP_T4 *directions,
    const FP_T4 *vertices, const FP_T *eps, int *hits, FP_T *distances)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    FP_T4 lo, hi;
    calculateTriangleBoundingBox(vertices[3*i], vertices[3*i+1], vertices[3*i+2], lo, hi);
    WatertightRay ray(origins[i], directions[i], lo, hi, test_eps(eps));
    HIT_T t = -1;
    hits[2*i] = ray.intersects(vertices[3*i], vertices[3*i+1], vertices[3*i+2], t, 0, 0);
    hits[2*i+1] = ray.intersects(lo, hi);
    distances[i] = t;
}

extern "C" __global__ void test_trace(
    int n, unsigned nt, const FP_T4 *origins, const FP_T4 *directions,
    FP_T4 *vertices, FP_T4 *normals, unsigned *indices,
    int *rope, int *left, FP_T4 *lo, FP_T4 *hi,
    const FP_T *eps, FP_T *thickness)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Tree tree;
    tree.nb_keys = nt; tree.indices = indices; tree.rope = rope;
    tree.left = left; tree.bboxMin = lo; tree.bboxMax = hi;
    unsigned root = nt == 1 ? 0 : nt;
    WatertightRay ray(origins[i], directions[i], lo[root], hi[root], test_eps(eps));
    thickness[2*i] = traceRay(ray, tree, vertices, 0, 0);
    thickness[2*i+1] = traceRay(ray, tree, vertices, normals, 0, 0);
}
