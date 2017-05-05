#include "../AstroAccelerate.h"

namespace astroaccelerate {

template<typename AstroAccelerateParameterType>
AstroAccelerate<AstroAccelerateParameterType>::AstroAccelerate(DedispersionStrategy  &dedispersion_strategy)
						   	      : _num_tchunks(dedispersion_strategy.get_num_tchunks())
						   	      ,_range(dedispersion_strategy.get_range())
						   	      ,_nchans(dedispersion_strategy.get_nchans())
						   	      ,_maxshift(dedispersion_strategy.get_maxshift())
						   	      ,_tsamp(dedispersion_strategy.get_tsamp())
						   	      ,_max_ndms(dedispersion_strategy.get_max_ndms())
						   	      ,_sigma_cutoff(dedispersion_strategy.get_sigma_cutoff())
						   	      ,_ndms(dedispersion_strategy.get_ndms())
						   	      ,_dmshifts(dedispersion_strategy.get_dmshifts())
						   	      ,_t_processed(dedispersion_strategy.get_t_processed())
						   	      ,_dm_low(dedispersion_strategy.get_dm_low())
						   	      ,_dm_high(dedispersion_strategy.get_dm_high())
						   	      ,_dm_step(dedispersion_strategy.get_dm_step())
						   	      ,_in_bin(dedispersion_strategy.get_in_bin())
						   	      ,_out_bin(dedispersion_strategy.get_out_bin())
                                                              ,_nsamp(dedispersion_strategy.get_nsamp())
                                                              ,_sigma_constant(dedispersion_strategy.get_sigma_constant())
                                                              ,_max_boxcar_width_in_sec(dedispersion_strategy.get_max_boxcar_width_in_sec())
                                                              ,_total_ndms(dedispersion_strategy.get_total_ndms())
                                                              ,_nboots(dedispersion_strategy.get_nboots())
                                                              ,_ntrial_bins(dedispersion_strategy.get_ntrial_bins())
                                                              ,_navdms(dedispersion_strategy.get_navdms())
                                                              ,_narrow(dedispersion_strategy.get_narrow())
                                                              ,_wide(dedispersion_strategy.get_wide())
                                                              ,_nsearch(dedispersion_strategy.get_nsearch())
                                                              ,_aggression(dedispersion_strategy.get_aggression())
{
	//
	_multi_file = 0;
	_enable_debug = 0;
	_enable_analysis = 1;
	_enable_periodicity = 0;
	_enable_acceleration = 0;
	_enable_fdas_custom_fft = 1;
	_enable_fdas_inbin = 0;
	_enable_fdas_norm = 1;
	_output_dmt = 0;
	_enable_zero_dm = 0;
	_enable_zero_dm_with_outliers = 0;
	_enable_rfi = 0;
	_candidate_algorithm = 0;
	//
	_gpu_input_size = 0;
	_d_input = nullptr;
	_gpu_output_size = 0;
	_d_output = nullptr;
}

template<typename AstroAccelerateParameterType>
AstroAccelerate<AstroAccelerateParameterType>::~AstroAccelerate()
{
}

template<typename AstroAccelerateParameterType>
void AstroAccelerate<AstroAccelerateParameterType>::allocate_memory_gpu()
{
	int time_samps = _t_processed[0][0] + _maxshift;
	_gpu_input_size = (size_t)time_samps * (size_t)_nchans * sizeof(unsigned short);

	cudaError_t rc1 = ( cudaMalloc((void **)&_d_input, _gpu_input_size) );
	if (rc1 != cudaSuccess)
	{
		throw std::bad_alloc();
	}

	try
	{
		if (_nchans < _max_ndms)
	    {
		    _gpu_output_size = (size_t)time_samps * (size_t)_max_ndms * sizeof(float);
	    }
	    else
	    {
		    _gpu_output_size = (size_t)time_samps * (size_t)_nchans * sizeof(float);
	    }
	}
	catch(...)
	{
		cudaFree(_d_input);
		throw;
	}

	cudaError_t rc2 = ( cudaMalloc((void **)&_d_output, _gpu_output_size) );
	if (rc2 != cudaSuccess)
	{
		throw std::bad_alloc();
	}
	try
	{
		cudaMemset(_d_output, 0, _gpu_output_size);
	}
	catch(...)
	{
		cudaFree(_d_output);
		throw;
	}
}

template<typename AstroAccelerateParameterType>
void AstroAccelerate<AstroAccelerateParameterType>::run_dedispersion_sps(unsigned device_id
									,unsigned char *input_buffer
								        ,DmTime<float> &output_buffer
									,std::vector<float> &output_sps
									)
{
        //GpuTimer timer;
        //timer.Start();
	unsigned short *input_buffer_cast = nullptr;
	size_t inputsize = _nsamp * _nchans * sizeof(unsigned short);
	input_buffer_cast = (unsigned short *) malloc(inputsize);
    if (input_buffer_cast == nullptr)
        throw std::bad_alloc();
 	try
        {
	    // cast to unsigned short
	    for (int i = 0; i < _nchans; ++i)
            {
                for (int j = 0; i < _nsamp; ++j)
                {
                    input_buffer_cast[ (i*_nsamp) + j] = (unsigned short)(input_buffer_cast[ (i*_nsamp) + j]);
                }
            }
            //timer.Stop();
            //float time = timer.Elapsed() / 1000;
            //printf("\n:Time to cast: %g (GPU estimate)",  time);

            // call dd + sps
            run_dedispersion_sps(device_id
                    ,input_buffer_cast
                    ,output_buffer
                    ,output_sps
                    );
        
        } 
        catch(...)
        {
            free(input_buffer_cast);
            throw;
        }

        // free mem
	free(input_buffer_cast);
}

template<typename AstroAccelerateParameterType>
void AstroAccelerate<AstroAccelerateParameterType>::run_dedispersion_sps(unsigned device_id
									,unsigned short *input_buffer
									,DmTime<float> &output_buffer
									,std::vector<float> &output_sps
									)
{
	//
	long int inc = 0;
	float tstart_local = 0.0f;

	// allocate memory gpu
	allocate_memory_gpu();

	//printf("\nDe-dispersing...\n");
	GpuTimer timer;
	timer.Start();

	float tsamp_original = _tsamp;
	int maxshift_original = _maxshift;

	//float *out_tmp;
	//out_tmp = (float *) malloc(( _t_processed[0][0] + _maxshift ) * _max_ndms * sizeof(float));
	//memset(out_tmp, 0.0f, _t_processed[0][0] + _maxshift * _max_ndms * sizeof(float));

	// assume we store size of the output size
	// so max number of candidates is a quarter of it
	size_t max_peak_size = (size_t) (_nsamp * _nchans/4);
	output_sps.resize(max_peak_size*4);
        
        size_t peak_pos=0;

	for (int t = 0; t < _num_tchunks; ++t)
	{
		//printf("\nt_processed:\t%d, %d", _t_processed[0][t], t);

		load_data(-1, _in_bin, _d_input, &input_buffer[(long int) ( inc * _nchans )], _t_processed[0][t], _maxshift, _nchans, _dmshifts);


		if (_enable_zero_dm)
			zero_dm(_d_input, _nchans, _t_processed[0][t]+_maxshift);

		if (_enable_zero_dm_with_outliers)
			zero_dm_outliers(_d_input, _nchans, _t_processed[0][t]+_maxshift);

		corner_turn(_d_input, _d_output, _nchans, _t_processed[0][t] + _maxshift);

		if (_enable_rfi)
	 		rfi_gpu(_d_input, _nchans, _t_processed[0][t]+_maxshift);

		int oldBin = 1;
		for (int dm_range = 0; dm_range < _range; ++dm_range)
		{
			//printf("\n\n%f\t%f\t%f\t%d", _dm_low[dm_range], _dm_high[dm_range], _dm_step[dm_range], _ndms[dm_range]), fflush(stdout);
			//printf("\nAmount of telescope time processed: %f", tstart_local);
			_maxshift = maxshift_original / _in_bin[dm_range];

			cudaDeviceSynchronize();
			load_data(dm_range, _in_bin, _d_input, &input_buffer[(long int) ( inc * _nchans )], _t_processed[dm_range][t], _maxshift, _nchans, _dmshifts);

			if (_in_bin[dm_range] > oldBin)
			{
				bin_gpu(_d_input, _d_output, _nchans, _t_processed[dm_range - 1][t] + _maxshift * _in_bin[dm_range]);
				_tsamp = _tsamp * 2.0f;
			}

			dedisperse(dm_range, _t_processed[dm_range][t], _in_bin, _dmshifts, _d_input, _d_output, _nchans,
				( _t_processed[dm_range][t] + _maxshift ), _maxshift, &_tsamp, _dm_low, _dm_high, _dm_step, _ndms);

	//		if (_enable_acceleration == 1)
	//		{
				// gpu_outputsize = ndms[dm_range] * ( t_processed[dm_range][t] ) * sizeof(float);
				//save_data(d_output, out_tmp, gpu_outputsize);

				//#pragma omp parallel for
				for (int k = 0; k < _ndms[dm_range]; ++k)
				{
					//memcpy(&output_buffer[dm_range][k][inc / inBin[dm_range]], &out_tmp[k * t_processed[dm_range][t]], sizeof(float) * t_processed[dm_range][t]);
					save_data_offset(_d_output, k * _t_processed[dm_range][t], output_buffer[dm_range][k], inc / _in_bin[dm_range], sizeof(float) * _t_processed[dm_range][t]);
				}
				//	save_data(d_output, &output_buffer[dm_range][0][((long int)inc)/inBin[dm_range]], gpu_outputsize);
	//		}

//			if (_output_dmt == 1)
//			{
				//for (int k = 0; k < ndms[dm_range]; k++)
				//	write_output(dm_range, t_processed[dm_range][t], ndms[dm_range], gpu_memory, output_buffer[dm_range][k], gpu_outputsize, dm_low, dm_high);
				//write_output(dm_range, t_processed[dm_range][t], ndms[dm_range], gpu_memory, out_tmp, gpu_outputsize, dm_low, dm_high);
//			}
			if (_enable_analysis == 1)
			{
				size_t previous_peak_pos = peak_pos;
				analysis_GPU(output_sps, &peak_pos, max_peak_size, dm_range, tstart_local,
							_t_processed[dm_range][t], _in_bin[dm_range], _out_bin[dm_range], &_maxshift, _max_ndms, _ndms,
							_sigma_cutoff, _sigma_constant, _max_boxcar_width_in_sec
							, _d_output, _dm_low, _dm_high, _dm_step, _tsamp, _candidate_algorithm);


				//#pragma omp parallel for
				for (int count = previous_peak_pos; count < peak_pos; count++)
				{
					output_sps[4*count]     = output_sps[4*count]*_dm_step[dm_range] + _dm_low[dm_range];
					output_sps[4*count + 1] = output_sps[4*count + 1]*_tsamp + tstart_local;
					//h_output_list[4*count + 2] = h_output_list[4*count + 2];
					//h_output_list[4*count + 3] = h_output_list[4*count + 3];
				}

				// This is for testing purposes and should be removed or commented out
				//analysis_CPU(dm_range, tstart_local, t_processed[dm_range][t], (t_processed[dm_range][t]+maxshift), nchans, maxshift, max_ndms, ndms, outBin, sigma_cutoff, out_tmp,dm_low, dm_high, dm_step, tsamp);

			}
			oldBin = _in_bin[dm_range];
		}

			//memset(out_tmp, 0.0f, t_processed[0][0] + maxshift * max_ndms * sizeof(float));

		inc = inc + _t_processed[0][t];
		//printf("\nINC:\t%ld", inc);
		tstart_local = ( tsamp_original * inc );
		_tsamp = tsamp_original;
		_maxshift = maxshift_original;
	}


	FILE *fp_out;
	char filename[200];

	if (peak_pos > 0)
	{
		sprintf(filename, "global_peak_analysed");
		//if ((fp_out=fopen(filename, "w")) == NULL) {
		if ((fp_out = fopen(filename, "wb")) == nullptr)
		{
			fprintf(stderr, "Error opening output file!\n");
			exit(0);
		}
		fwrite(&output_sps[0], peak_pos*sizeof(float), 4, fp_out);
		fclose(fp_out);
	}

	timer.Stop();
	float time = timer.Elapsed() / 1000;

	//printf("\n\n === OVERALL DEDISPERSION THROUGHPUT INCLUDING SYNCS AND DATA TRANSFERS ===\n");

	//printf("\n(Performed Brute-Force Dedispersion: %g (GPU estimate)",  time);
	//printf("\nAmount of telescope time processed: %f", tstart_local);
	//printf("\nNumber of samples processed: %ld", inc);
	//printf("\nReal-time speedup factor: %lf", ( tstart_local ) / time);

	cudaFree(_d_input);
	cudaFree(_d_output);
	//free(out_tmp);
	//free(input_buffer);

	double time_processed = ( tstart_local ) / tsamp_original;
	double dm_t_processed = time_processed * _total_ndms;
	double all_processed = dm_t_processed * _nchans;
	//printf("\nGops based on %.2lf ops per channel per tsamp: %f", NOPS, ( ( NOPS * all_processed ) / ( time ) ) / 1000000000.0);
	int num_reg = SNUMREG;
	float num_threads = _total_ndms * ( _t_processed[0][0] ) / ( num_reg );
	float data_size_loaded = ( num_threads * _nchans * sizeof(ushort) ) / 1000000000;
	float time_in_sec = time;
	float bandwidth = data_size_loaded / time_in_sec;
	//printf("\nDevice global memory bandwidth in GB/s: %f", bandwidth);
	//printf("\nDevice shared memory bandwidth in GB/s: %f", bandwidth * ( num_reg ));
	float size_gb = ( _nchans * ( _t_processed[0][0] ) * sizeof(float) * 8 ) / 1000000000.0;
	//printf("\nTelescope data throughput in Gb/s: %f", size_gb / time_in_sec);

	if (_enable_periodicity == 1)
	{
		//
		GpuTimer timer;
		timer.Start();
		//
		periodicity(_range, _nsamp, _max_ndms, inc, _nboots, _ntrial_bins, _navdms, _narrow, _wide, _nsearch,	_aggression, _sigma_cutoff, output_buffer, 
                _ndms, _in_bin, _dm_low, _dm_high, _dm_step, tsamp_original);
		//
		timer.Stop();
		float time = timer.Elapsed()/1000;
		//printf("\n\n === OVERALL PERIODICITY THROUGHPUT INCLUDING SYNCS AND DATA TRANSFERS ===\n");

		//printf("\nPerformed Peroidicity Location: %f (GPU estimate)", time);
		//printf("\nAmount of telescope time processed: %f", tstart_local);
		//printf("\nNumber of samples processed: %ld", inc);
		//printf("\nReal-time speedup factor: %f", ( tstart_local ) / ( time ));
	}
/*
	if (_enable_acceleration == 1)
	{
		// todo: export fdas ouput to vector once got Karel's peak finding in master

		// Input needed for fdas is output_buffer which is DDPlan
		// Assumption: gpu memory is free and available
		//
		GpuTimer timer;
		timer.Start();
		// acceleration(range, nsamp, max_ndms, inc, nboots, ntrial_bins, navdms, narrow, wide, nsearch, aggression, sigma_cutoff, output_buffer, ndms, inBin, dm_low, dm_high, dm_step, tsamp_original);
		acceleration_fdas(_range
						  ,_nsamp
						  ,_max_ndms
						  ,inc
						  ,_nboots
						  ,_ntrial_bins
						  ,_navdms
						  ,_narrow
						  ,_wide
						  ,_nsearch
						  ,_aggression
						  ,_sigma_cutoff
						  ,output_buffer
						  ,_ndms
						  ,_in_bin
						  ,_dm_low
						  ,_dm_high
						  ,_dm_step
						  ,tsamp_original
						  ,1
						  ,0
						  ,1);
		//
		timer.Stop();
		float time = timer.Elapsed()/1000;
		//printf("\n\n === OVERALL TDAS THROUGHPUT INCLUDING SYNCS AND DATA TRANSFERS ===\n");

		//printf("\nPerformed Acceleration Location: %lf (GPU estimate)", time);
		//printf("\nAmount of telescope time processed: %f", tstart_local);
		//printf("\nNumber of samples processed: %ld", inc);
		//printf("\nReal-time speedup factor: %lf", ( tstart_local ) / ( time ));
	}
*/
	//free(out_tmp);
	cudaFree(_d_input);
	cudaFree(_d_output);
}

} // namespace astroaccelerate
