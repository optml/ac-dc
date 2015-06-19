struct optimization_setting {
	bool show_objective_values_during_optimization;
	bool show_objective_value_at_the_end;
	bool show_objective_value_at_the_beggining;
	bool do_restart_step_after_kernel_run;
	float total_execution_time;
	int device_block_dim_1;
	int device_block_dim_2;
	int device_total_threads_per_block;
	int number_of_inner_iterations_per_kernel_run;
	int device_memmory_aligned_data_elements;

	optimization_setting() :
		number_of_inner_iterations_per_kernel_run(100000),
				device_block_dim_1(14), device_block_dim_2(1),
				device_total_threads_per_block(64),
				device_memmory_aligned_data_elements(16),
				show_objective_values_during_optimization(true),
				show_objective_value_at_the_end(true),
				do_restart_step_after_kernel_run(false),
				show_objective_value_at_the_beggining(true)

	{
	}
};

