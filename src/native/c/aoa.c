#include <math.h>
#include <stdlib.h>
#include <stdio.h>
// minimal stub implementations
double aoa_gcc_phat(const float* sig, const float* ref, int n, int fs, double max_tau){ (void)sig;(void)ref;(void)n;(void)fs;(void)max_tau; return 0.0; }
int aoa_compute_tdoas(const float* signals, int n_mics, int n_samples, int fs, double max_tau, double* out_tdoas){ (void)signals;(void)n_mics;(void)n_samples;(void)fs;(void)max_tau;(void)out_tdoas; return 0; }
int aoa_estimate_position(const double* mic_xyz, int n_mics, const double* tdoas, double c, int fixed_z, double z_value, double* out_xyz){ (void)mic_xyz;(void)n_mics;(void)tdoas;(void)c;(void)fixed_z;(void)z_value; out_xyz[0]=0; out_xyz[1]=0; out_xyz[2]=0; return 0; }
