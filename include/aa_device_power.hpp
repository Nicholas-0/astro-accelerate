#ifndef ASTRO_ACCELERATE_AA_DEVICE_POWER_HPP
#define ASTRO_ACCELERATE_AA_DEVICE_POWER_HPP

#include <cufft.h>

namespace astroaccelerate {

extern void power_gpu(cudaEvent_t event, cudaStream_t stream, int samps, int acc, cufftComplex *d_signal_fft, float *d_signal_power);

extern void simple_power_and_interbin(float2 *d_input_complex, float *d_output_power, float *d_output_interbinning, int nTimesamples, int nDMs);

extern void calculate_power_interbin_and_scalloping_removal(
    float2 *d_input_complex, 
    float *d_power_output, 
    float *d_interbin_output, 
    size_t nTimesamples, 
    size_t nDMs,
    cudaStream_t &stream
);

} // namespace astroaccelerate
  
#endif // ASTRO_ACCELERATE_AA_DEVICE_POWER_HPP

