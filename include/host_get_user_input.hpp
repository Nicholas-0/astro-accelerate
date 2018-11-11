#ifndef ASTRO_ACCELERATE_HOST_GET_USER_INPUT_HPP
#define ASTRO_ACCELERATE_HOST_GET_USER_INPUT_HPP

namespace astroaccelerate {

void get_user_input(FILE **fp, int argc, char *argv[], int *multi_file, int *enable_debug, int *enable_analysis, int *enable_periodicity, int *enable_acceleration, int *enable_output_ffdot_plan, int *enable_output_fdas_list, int *output_dmt, int *enable_zero_dm, int *enable_zero_dm_with_outliers, int *enable_rfi, int *enable_old_rfi, int *enable_fdas_custom_fft, int *enable_fdas_inbin, int *enable_fdas_norm, int *nboots, int *ntrial_bins, int *navdms, float *narrow, float *wide, float *aggression, int *nsearch, int **inBin, int **outBin, float *power, float *sigma_cutoff, float *sigma_constant, float *max_boxcar_width_in_sec, int *range, float **user_dm_low, float **user_dm_high, float **user_dm_step, int *candidate_algorithm, int *enable_sps_baselinenoise, float **selected_dm_low, float **selected_dm_high, int *nb_selected_dm, int *analysis_debug, int *failsafe, float *periodicity_sigma_cutoff, int *periodicity_nHarmonics);

} //namespace astroaccelerate

#endif
