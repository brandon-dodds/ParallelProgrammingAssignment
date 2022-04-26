kernel void Histogram_Normal_B(global const uchar* input_image, global int* hist_vec) {
	int id = get_global_id(0);
	//assumes that H has been initialised to 0
	int bin_index = input_image[id];//take value as a bin index

	atomic_inc(&hist_vec[bin_index]);//serial operation, not very efficient!
	//this is just a copy operation, modify to filter out the individual colour channels
}

kernel void Cumulative_Histogram(global int* histogram_vector, global int* buffer) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	global int* C;

	for (int stride = 1; stride < N; stride *= 2) {
		buffer[id] = histogram_vector[id];
		if (id >= stride)
			buffer[id] += histogram_vector[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

		C = histogram_vector; histogram_vector = buffer; buffer = C; //swap A & B between steps
	}
}

kernel void Normalise_Histogram(global int* histogram_vector, int pixelsize) {
	int id = get_global_id(0);
	histogram_vector[id] = (float)histogram_vector[id] / pixelsize * 255;
}