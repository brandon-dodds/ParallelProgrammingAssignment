kernel void Histogram_Normal_B(global const uchar* input_image, global int* hist_vec) {
	// Histogram using regular atomic functions. Generates a normal histogram.
	int id = get_global_id(0);
	
	int bin_index = input_image[id];

	atomic_inc(&hist_vec[bin_index]);
}

kernel void Cumulative_Histogram(global int* histogram_vector, global int* buffer) {
	// Uses a Hillis-Steele scan to create a cumulative histogram.
	int id = get_global_id(0);
	int N = get_global_size(0);
	global int* C;

	for (int stride = 1; stride < N; stride *= 2) {
		buffer[id] = histogram_vector[id];
		if (id >= stride)
			buffer[id] += histogram_vector[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); 

		C = histogram_vector; histogram_vector = buffer; buffer = C; 
	}
}

kernel void Normalise_Histogram(global int* histogram_vector, int pixelsize) {
	// Normalises the histogram based on the pixel size.
	int id = get_global_id(0);
	histogram_vector[id] = (float)histogram_vector[id] / pixelsize * 255;
}

kernel void back_projection(global const uchar* image, global uchar* B, global int* histogram) {
	// Back projects the intensity values onto the original image.
	int id = get_global_id(0);

	int bin_index = image[id];

	B[id] = histogram[bin_index];
}