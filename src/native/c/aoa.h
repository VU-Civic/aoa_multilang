#ifndef AOA_H
#define AOA_H
#ifdef __cplusplus
extern "C" {
#endif

double aoa_gcc_phat(const float* sig, const float* ref, int n, int fs, double max_tau);
int aoa_compute_tdoas(const float* signals, int n_mics, int n_samples, int fs, double max_tau, double* out_tdoas);
int aoa_estimate_position(const double* mic_xyz, int n_mics, const double* tdoas, double c, int fixed_z, double z_value, double* out_xyz);

#ifdef __cplusplus
}
#endif
#endif
