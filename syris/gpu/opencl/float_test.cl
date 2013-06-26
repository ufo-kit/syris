__kernel void float_test(__global vfloat *buffer) {
    int ix = get_global_id(0);
    
    buffer[ix] = (vfloat) ix;
}