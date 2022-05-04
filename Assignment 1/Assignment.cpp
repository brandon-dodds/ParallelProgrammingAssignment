#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

// This code is developed to produce three histograms and two images. It does this through using a simple histogram method. 
// A Hillis-Steele scan and also a normalization function to create the LUT histogram. This is then used to calculate the intensity values.
// For the final image, which is output to the screen.
using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char** argv) {

	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.pgm";
	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	try {
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input, "input");

		cl::Context context = GetContext(platform_id, device_id);

		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);
		
		cl::Event image_buffer_write;
		cl::Event histogram_buffer_write;
		cl::Event cumulative_histogram_write;
		cl::Event simple_histogram_NDRange;
		cl::Event histogram_buffer_read;
		cl::Event cumulative_NDRange;
		cl::Event cumulative_histogram_READ;
		cl::Event histogram_normalize;
		cl::Event histogram_normalize_READ;
		cl::Event normalized_histogram_WRITE;
		cl::Event reverse_image_NDRange;
		cl::Event image_output_read;

		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		std::vector<int> histogram(256, 0);
		std::vector<int> cumulative_histogram(256, 0);
		std::vector<int> normalised_histogram(256, 0);

		int pixels = image_input.size();
		size_t histogram_size_in_bites = histogram.size() * sizeof(int);

		cl::Buffer image_buffer(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer histogram_buffer(context, CL_MEM_READ_WRITE, histogram_size_in_bites);
		cl::Buffer cumulative_histogram_buffer(context, CL_MEM_READ_WRITE, histogram_size_in_bites);
		cl::Buffer normalised_histogram_buffer(context, CL_MEM_READ_WRITE, histogram_size_in_bites);
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size());

		queue.enqueueWriteBuffer(image_buffer, CL_TRUE, 0, image_input.size(), &image_input.data()[0], NULL, &image_buffer_write);
		queue.enqueueWriteBuffer(histogram_buffer, CL_TRUE, 0, histogram_size_in_bites, &histogram.data()[0], NULL, &histogram_buffer_write);
		queue.enqueueWriteBuffer(cumulative_histogram_buffer, CL_TRUE, 0, histogram_size_in_bites, &cumulative_histogram.data()[0], NULL, &cumulative_histogram_write);

		cl::Kernel histogram_simple = cl::Kernel(program, "Histogram_Normal_B");
		histogram_simple.setArg(0, image_buffer);
		histogram_simple.setArg(1, histogram_buffer);

		queue.enqueueNDRangeKernel(histogram_simple, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &simple_histogram_NDRange);

		queue.enqueueReadBuffer(histogram_buffer, CL_TRUE, 0, histogram_size_in_bites, &histogram.data()[0], NULL, &histogram_buffer_read);

		cl::Kernel histogram_cumulative = cl::Kernel(program, "Cumulative_Histogram");
		histogram_cumulative.setArg(0, histogram_buffer);
		histogram_cumulative.setArg(1, cumulative_histogram_buffer);

		queue.enqueueNDRangeKernel(histogram_cumulative, cl::NullRange, cl::NDRange(histogram.size()), cl::NullRange, NULL, &cumulative_NDRange);

		queue.enqueueReadBuffer(histogram_buffer, CL_TRUE, 0, histogram_size_in_bites, &cumulative_histogram.data()[0], NULL, &cumulative_histogram_READ);

		cl::Kernel histogram_normalise = cl::Kernel(program, "Normalise_Histogram");
		histogram_normalise.setArg(0, histogram_buffer);
		histogram_normalise.setArg(1, pixels);

		queue.enqueueNDRangeKernel(histogram_normalise, cl::NullRange, cl::NDRange(histogram.size()), cl::NullRange, NULL, &histogram_normalize);
		queue.enqueueReadBuffer(histogram_buffer, CL_TRUE, 0, histogram_size_in_bites, &normalised_histogram.data()[0], NULL, &histogram_normalize_READ);
		
		queue.enqueueWriteBuffer(normalised_histogram_buffer, CL_TRUE, 0, histogram_size_in_bites, &normalised_histogram.data()[0], NULL, &normalized_histogram_WRITE);
		std::vector<unsigned char> output_buffer(image_input.size());
		cl::Kernel reverse_image = cl::Kernel(program, "back_projection");
		reverse_image.setArg(0, image_buffer);
		reverse_image.setArg(1, dev_image_output);
		reverse_image.setArg(2, normalised_histogram_buffer);

		queue.enqueueNDRangeKernel(reverse_image, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &reverse_image_NDRange);

		std::cout << "Histogram = " << histogram << std::endl;
		std::cout << "Cumulative Histogram = " << cumulative_histogram << std::endl;
		std::cout << "Normalised Histogram = " << normalised_histogram << std::endl;

		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0], NULL, &image_output_read);

		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image, "output");

		std::cout << "Image buffer write [ns]:" <<
			image_buffer_write.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			image_buffer_write.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		std::cout << "Histogram buffer write [ns]:" <<
			histogram_buffer_write.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			histogram_buffer_write.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		std::cout << "Cumulative histogram write [ns]:" <<
			cumulative_histogram_write.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			cumulative_histogram_write.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		std::cout << "Simple histogram NDRange [ns]:" <<
			simple_histogram_NDRange.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			simple_histogram_NDRange.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		std::cout << "Histogram buffer read [ns]:" <<
			histogram_buffer_read.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			histogram_buffer_read.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		std::cout << "Cumulative histogram NDRange [ns]:" <<
			cumulative_NDRange.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			cumulative_NDRange.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		std::cout << "Cumulative histogram READ [ns]:" <<
			cumulative_histogram_READ.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			cumulative_histogram_READ.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		std::cout << "histogram normalize NDRange [ns]:" <<
			histogram_normalize.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			histogram_normalize.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		std::cout << "Histogram normalize READ [ns]:" <<
			histogram_normalize_READ.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			histogram_normalize_READ.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		std::cout << "Image Histogram WRITE[ns]:" <<
			normalized_histogram_WRITE.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			normalized_histogram_WRITE.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		std::cout << "Reverse image NDRange [ns]:" <<
			reverse_image_NDRange.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			reverse_image_NDRange.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		std::cout << "Image output READ [ns]:" <<
			image_output_read.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			image_output_read.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}
	return 0;
}