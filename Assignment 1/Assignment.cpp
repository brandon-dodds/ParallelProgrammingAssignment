#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

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

		cl::CommandQueue queue(context);

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
		size_t input_size = histogram.size() * sizeof(int);

		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, input_size);

		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, input_size, &histogram.data()[0]);

		cl::Kernel histogram_simple = cl::Kernel(program, "Histogram_Normal_B");
		histogram_simple.setArg(0, buffer_A);
		histogram_simple.setArg(1, buffer_B);

		queue.enqueueNDRangeKernel(histogram_simple, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange);

		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, input_size, &histogram.data()[0]);

		std::cout << "Histogram = " << histogram << std::endl;

		while (!disp_input.is_closed()
			&& !disp_input.is_keyESC()) {
			disp_input.wait(1);
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