#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "aa_params.hpp"

namespace astroaccelerate {

  //{{{ Set stretch
  __global__ void power_kernel(int half_samps, int acc, cufftComplex *d_signal_fft, float *d_signal_power) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;

    if (t < half_samps)
      d_signal_power[t + acc * ( half_samps )] = ( d_signal_fft[t + 1].x * d_signal_fft[t + 1].x + d_signal_fft[t + 1].y * d_signal_fft[t + 1].y );
  }

  //}}}

__inline__ __device__ float remove_scalloping_loss(float Xm2, float Xm1, float X0, float Xp1, float Xp2){
    return(X0 + (1.88494/2.0)*(Xm1 + Xp1) + (0.88494/2.0)*(Xm2 + Xp2));
}

__inline__ __device__ float get_power(float2 A, float norm){
    return(__fdividef( (A.x*A.x + A.y*A.y), norm*norm ));
}

__inline__ __device__ float get_interbin(float2 A, float2 B, float norm) {
    return(__fdividef( (0.616850275f*( (A.x - B.x)*(A.x - B.x) + (A.y - B.y)*(A.y - B.y) )), norm*norm ));
}

__inline__ __device__ void get_power_spectrum_bins(
    float *power, 
    float *interbin1, 
    float *interbin2, 
    float2 const* __restrict__ d_input, 
    float norm, 
    size_t pos
){
    float2 Xm2 = d_input[pos - 2];
    float2 Xm1 = d_input[pos - 1];
    float2 X0  = d_input[pos + 0];
    float2 Xp1 = d_input[pos + 1];
    float2 Xp2 = d_input[pos + 2];
    
    // Power calculation
    float Xm2p = get_power(Xm2, norm);
    float Xm1p = get_power(Xm1, norm);
    float X0p  = get_power(X0,  norm);
    float Xp1p = get_power(Xp2, norm);
    float Xp2p = get_power(Xp2, norm);
    (*power) = remove_scalloping_loss(Xm2p, Xm1p, X0p, Xp1p, Xp2p);
    
    float Im2 = Xm2p;
    float Im1 = get_interbin(Xm1, X0, norm);
    float I0  = X0p;
    float Ip1 = get_interbin(X0, Xp1, norm);
    float Ip2 = Xp1p;
    float Ip3 = get_interbin(Xp1, Xp2, norm);
    
    (*interbin1) = remove_scalloping_loss(Im2, Im1, I0, Ip1, Ip2);
    (*interbin2) = remove_scalloping_loss(Im1, I0, Ip1, Ip2, Ip3);
}

__inline__ __device__ void get_power_spectrum_bins_temp(
    float *power, 
    float *interbin1, 
    float *interbin2, 
    float2 const* __restrict__ d_input, 
    float norm, 
    size_t pos
){
    float2 X0  = d_input[pos + 0];
    float2 Xp1 = d_input[pos + 1];
    
    // Power calculation
    float X0p  = get_power(X0,  norm);
    (*power) = X0p;
    
    (*interbin1) = X0p;
    (*interbin2) = get_interbin(X0, Xp1, norm);
}


  /** \brief Calculates power spectrum, interning and removes scalloping loss */
  __global__ void PaI_and_SLR_GPU_kernel(
      float2 const* __restrict__ d_input_complex, 
      float *d_output_power, 
      float *d_output_interbinning, 
      size_t nTimesamples, 
      float norm
  ){
      size_t pos_x = blockIdx.x*blockDim.x + threadIdx.x;
      size_t block_pos = blockIdx.y*((nTimesamples>>1) + 1) + pos_x;
    
      float power = 0, interbin1 = 0, interbin2 = 0;
      if( pos_x > 1 && pos_x + 2 < ((nTimesamples>>1) + 1) ) {
          get_power_spectrum_bins(&power, &interbin1, &interbin2, d_input_complex, norm, block_pos);
          //get_power_spectrum_bins_temp(&power, &interbin1, &interbin2, d_input_complex, norm, block_pos);
      }
      
      if ( pos_x < (nTimesamples>>1) ){
          d_output_power[blockIdx.y*(nTimesamples>>1) + pos_x] = power;
          d_output_interbinning[blockIdx.y*nTimesamples + 2*pos_x] = interbin1;
          d_output_interbinning[blockIdx.y*nTimesamples + 2*pos_x + 1] = interbin2;
      }
  }

  __global__ void GPU_simple_power_and_interbin_kernel(float2 *d_input_complex, float *d_output_power, float *d_output_interbinning, int nTimesamples, float norm){
    int pos_x = blockIdx.x*blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y*((nTimesamples>>1)+1);
	
    float2 A, B;
    A.x = 0; A.y = 0; B.x = 0; B.y = 0;
	
    if ( (pos_x < (nTimesamples>>1)) && (pos_x > 0) ) {
      A = d_input_complex[pos_y + pos_x];
      B = d_input_complex[pos_y + pos_x + 1];
		
      A.x = A.x/norm;
      A.y = A.y/norm;
      B.x = B.x/norm;
      B.y = B.y/norm;
    }
	
    if ( (pos_x < (nTimesamples>>1)) ){
      d_output_power[blockIdx.y*(nTimesamples>>1) + pos_x] = A.x*A.x + A.y*A.y;
      d_output_interbinning[blockIdx.y*nTimesamples + 2*pos_x] = A.x*A.x + A.y*A.y;
      d_output_interbinning[blockIdx.y*nTimesamples + 2*pos_x + 1] = 0.616850275f*( (A.x - B.x)*(A.x - B.x) + (A.y - B.y)*(A.y - B.y) );
    }
  }

  /** Kernel wrapper function for power_kernel kernel function. */
  void call_kernel_power_kernel(const dim3 &block_size, const dim3 &grid_size, const int &smem_bytes, const cudaStream_t &stream,
				const int &half_samps, const int &acc, cufftComplex *const d_signal_fft, float *const d_signal_power) {
    power_kernel<<<block_size, grid_size, smem_bytes, stream>>>(half_samps, acc, d_signal_fft, d_signal_power);
  }

  /** Kernel wrapper function for GPU_simple_power_and_interbin_kernel kernel function. */
  void call_kernel_GPU_simple_power_and_interbin_kernel(const dim3 &grid_size, const dim3 &block_size,
							float2 *const d_input_complex, float *const d_output_power, float *const d_output_interbinning, const int &nTimesamples, const float &norm) {
    GPU_simple_power_and_interbin_kernel<<<grid_size, block_size>>>(d_input_complex, d_output_power, d_output_interbinning, nTimesamples, norm);
  }  
  
/** Wrapper for GPU kernel PaI_and_SLR_GPU_kernel. */
void call_kernel_PaI_and_SLR_GPU_kernel(
    const dim3 &grid_size, 
    const dim3 &block_size,
    const  int &smem_bytes, 
    const  cudaStream_t &stream, 
    float2 *const d_input_complex, 
    float *const d_output_power, 
    float *const d_output_interbinning,
    size_t nTimesamples, 
    float norm
) {
    PaI_and_SLR_GPU_kernel<<<grid_size, block_size, smem_bytes, stream>>>(
        d_input_complex, 
        d_output_power, 
        d_output_interbinning, 
        nTimesamples, 
        norm
    );
}

} //namespace astroaccelerate
