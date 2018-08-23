pycuda::complex<float> *z, pycuda::complex<float> *q, int *iteration, int maxiter
for (int n=0; n < maxiter; n++) {
  z[i] = (z[i]*z[i])+q[i]; 
  if (abs(z[i]) > 2.0f) {
    iteration[i]=n; 
    z[i] = pycuda::complex<float>(); 
    q[i] = pycuda::complex<float>();
  }
}
