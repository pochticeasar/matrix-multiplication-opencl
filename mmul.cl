#define _float_vector(x) float##x
#define float_vector(x) _float_vector(x)

typedef union floatx {
  float_vector(WI) vecfloat;
  float arrfloat[WI];
} floatx;

kernel void matrix_multiplication1(global const float *a, global const float *b,
                                   global float *c, const uint n, const uint k,
                                   const uint m) {
  uint x = get_global_id(0);
  uint y = get_global_id(1);
  float sum = 0;
  for (uint t = 0; t < k; t++) {
    sum += a[y * k + t] * b[t * n + x];
  }
  c[y * n + x] = sum;
}

kernel void matrix_multiplication2(global const float *a, global const float *b,
                                   global float *c, const uint n, const uint k,
                                   const uint m) {
  uint x = get_global_id(0);
  uint y = get_global_id(1);
  uint lx = get_local_id(0);
  uint ly = get_local_id(1);
  local float la[TS2][TS2];
  local float lb[TS2][TS2];

  float sum = 0.0f;
  for (int t = 0; t < k; t += TS2) {
    la[ly][lx] = a[y * k + t + lx];
    lb[ly][lx] = b[(t + ly) * n + x];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < TS2; i++) {
      sum += la[ly][i] * lb[i][lx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  c[y * n + x] = sum;
}

kernel void matrix_multiplication3(global const float_vector(WI) *a, global const float_vector(WI) *b,
                                   global float_vector(WI) *c, const uint n, const uint k,
                                   const uint m) {
  uint x = get_global_id(0); //TS3/ITEM
  uint y = get_global_id(1);
  uint lx = get_local_id(0);
  uint ly = get_local_id(1);
  local floatx la[TS3][TS3/WI + 1];
  local floatx lb[TS3][TS3/WI];
  float_vector(WI) sum = 0.0f;
  for (int t = 0; t < k; t += TS3) {
    la[ly][lx].vecfloat = a[(y * k + t)/WI + lx];
    lb[ly][lx].vecfloat = b[(t + ly) * n/WI+ x];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int j = 0; j < TS3/WI; j++) {
        floatx veca = la[ly][j];
        for (int i = 0; i < WI; i++) {
            floatx vecb = lb[WI * j + i][lx];
            sum += veca.arrfloat[i] * vecb.vecfloat;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  c[(y * n)/WI + x] = sum;
}
