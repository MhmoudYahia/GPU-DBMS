// #include <iostream>
// #include <cuda_runtime.h>

// int check()
// {
//     // Ensure the CUDA runtime is initialized
//     cudaSetDevice(0);
//     cudaFree(0); // Free some memory to initialize the runtime

//     size_t free_memory, total_memory;
//     cudaError_t error = cudaMemGetInfo(&free_memory, &total_memory);
//     if (error != cudaSuccess)
//     {
//         std::cerr << "Error getting GPU memory info: " << cudaGetErrorString(error) << std::endl;
//     }
//     else
//     {
//         std::cout << "GPU Memory - Free: " << free_memory / 1024 / 1024
//                   << " MB, Total: " << total_memory / 1024 / 1024
//                   << " MB, Used: " << (total_memory - free_memory) / 1024 / 1024
//                   << " MB" << std::endl;
//     }

//     return 0;
// }
