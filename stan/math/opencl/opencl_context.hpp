#ifndef STAN_MATH_OPENCL_OPENCL_CONTEXT_HPP
#define STAN_MATH_OPENCL_OPENCL_CONTEXT_HPP
#ifdef STAN_OPENCL

#define DEVICE_FILTER CL_DEVICE_TYPE_ALL
#ifndef OPENCL_PLATFORM_ID
#define OPENCL_PLATFORM_ID -1
#endif

#ifndef OPENCL_DEVICE_ID
#define OPENCL_DEVICE -1
#endif
#include <stan/math/opencl/matrix_cl_view.hpp>
#include <stan/math/opencl/err/check_opencl.hpp>
#include <stan/math/prim/scal/err/system_error.hpp>

#include <CL/cl.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <cmath>
#include <cerrno>

/**
 *  @file stan/math/opencl/opencl_context.hpp
 *  @brief Initialization for OpenCL:
 *    1. create context
 *    2. Find OpenCL platforms and devices available
 *    3. set up command queue
 *    4. set architecture dependent kernel parameters
 */
namespace stan {
namespace math {
namespace opencl {
/**
 * A helper function to convert an array to a cl::size_t<N>.
 * This implementation throws because cl::size_t<N> for N!=3
 * should throw.
 *
 * @param values the input array to be converted
 * @return the cl::size_t<N> converted from the input array
 */
template <int N>
inline cl::size_t<N> to_size_t(const size_t (&values)[N]) {
  throw std::domain_error("cl::size_t<N> is not supported for N != 3");
}

/**
 * A template specialization of the helper function
 * to convert an array to a cl::size_t<3>.
 *
 * @param values the input array to be converted
 * @return the cl::size_t<3> converted from the input array
 */
template <>
inline cl::size_t<3> to_size_t(const size_t (&values)[3]) {
  cl::size_t<3> s;
  for (size_t i = 0; i < 3; i++)
    s[i] = values[i];
  return s;
}
inline void choose_best_device(int& gpu_platform, int& gpu_device) {
  std::vector<cl::Platform> temp_platforms_;
  std::vector<cl::Device> temp_devices_;
  cl::Platform::get(&temp_platforms_);
  if(temp_platforms_.size() == 0) {
    system_error("OpenCL Initialization", "[Platform]", -1,
                     "No OpenCL device found");
  }
  
  int max_SMs = 0;
  for(int i = 0;i < temp_platforms_.size(); i++) {
    temp_platforms_[i].getDevices(DEVICE_FILTER, &temp_devices_);
    for(int j = 0;j < temp_devices_.size(); j++) {
      int device_SMs = temp_devices_[j].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
      int vendor_id = temp_devices_[j].getInfo<CL_DEVICE_VENDOR_ID>();
      int device_type = temp_devices_[j].getInfo<CL_DEVICE_TYPE>();
      // multiply the number of SMs for GPUs
      if(device_type == CL_DEVICE_TYPE_GPU) {
        device_SMs *= 100;
      }
      // multiply the number of SMs for NVIDIA and AMD
      // Intel has a larger number of SMs but performs worse
      if(vendor_id == 0x10de || vendor_id == 0x1002){
        device_SMs *= 100;
      }
      if(device_SMs > max_SMs) {
        max_SMs = device_SMs;
        gpu_platform = i;
        gpu_device = j;
      }
    }
  }
}
}  // namespace opencl
/**
 * The <code>opencl_context_base</code> class represents an OpenCL context
 * in the standard Meyers singleton design pattern.
 *
 * See the OpenCL specification glossary for a list of terms:
 * https://www.khronos.org/registry/OpenCL/specs/opencl-1.2.pdf.
 * The context includes the set of devices available on the host, command
 * queues, manages kernels.
 *
 * This is designed so there's only one instance running on the host.
 *
 * Some design decisions that may need to be addressed later:
 * - we are assuming a single OpenCL platform. We may want to run on multiple
 * platforms simulatenously
 * - we are assuming a single OpenCL device. We may want to run on multiple
 * devices simulatenously
 */
class opencl_context_base {
  friend class opencl_context;

 private:
  /**
   * Construct the opencl_context by initializing the
   * OpenCL context, devices, command queues, and kernel
   * groups.
   *
   * This constructor does the following:
   * 1. Gets the available platforms and selects the platform
   *  with id OPENCL_PLATFORM_ID.
   * 2. Gets the available devices and selects the device with id
   *  OPENCL_DEVICE_ID.
   * 3. Creates the OpenCL context with the device.
   * 4. Creates the OpenCL command queue for the selected device.
   * 5. Sets OpenCL device dependent kernel parameters
   * @throw std::system_error if an OpenCL error occurs.
   */
  opencl_context_base() {
    try {
      int gpu_platform = OPENCL_PLATFORM_ID;
      int gpu_device = OPENCL_DEVICE_ID;
      if(gpu_platform == -1 || gpu_device == -1){
        opencl::choose_best_device(gpu_platform, gpu_device); 
      }
      // platform
      cl::Platform::get(&platforms_);
      if (gpu_platform >= platforms_.size()) {
        system_error("OpenCL Initialization", "[Platform]", -1,
                     "CL_INVALID_PLATFORM");
      }
      platform_ = platforms_[gpu_platform];
      platform_name_ = platform_.getInfo<CL_PLATFORM_NAME>();
      platform_.getDevices(DEVICE_FILTER, &devices_);
      if (devices_.size() == 0) {
        system_error("OpenCL Initialization", "[Device]", -1,
                     "CL_DEVICE_NOT_FOUND");
      }
      if (gpu_device >= devices_.size()) {
        system_error("OpenCL Initialization", "[Device]", -1,
                     "CL_INVALID_DEVICE");
      }
      device_ = devices_[gpu_device];
      // context and queue
      cl_command_queue_properties device_properties;
      device_.getInfo<cl_command_queue_properties>(CL_DEVICE_QUEUE_PROPERTIES,
                                                   &device_properties);
      device_.getInfo<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE,
                              &max_thread_block_size_);
      std::cout << "Selected GPU: " << device_.getInfo<CL_DEVICE_NAME>()
                << " on platform " << platform_name_ << std::endl;
      context_ = cl::Context(device_);
      if (device_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
        command_queue_ = cl::CommandQueue(
            context_, device_, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, nullptr);
      } else {
        command_queue_ = cl::CommandQueue(context_, device_, 0, nullptr);
      }
      int thread_block_size_sqrt
          = static_cast<int>(sqrt(static_cast<double>(max_thread_block_size_)));
      // Does a compile time check of the maximum allowed
      // dimension of a square thread block size
      // WG size of (32,32) works on all recent GPUs but would fail on some
      // older integrated GPUs or CPUs
      if (thread_block_size_sqrt < base_opts_["THREAD_BLOCK_SIZE"]) {
        base_opts_["THREAD_BLOCK_SIZE"] = thread_block_size_sqrt;
        base_opts_["WORK_PER_THREAD"] = 1;
      }
      if (max_thread_block_size_ < base_opts_["LOCAL_SIZE_"]) {
        // must be a power of base_opts_["REDUCTION_STEP_SIZE"]
        const int p = std::log(max_thread_block_size_)
                      / std::log(base_opts_["REDUCTION_STEP_SIZE"]);
        base_opts_["LOCAL_SIZE_"]
            = std::pow(base_opts_["REDUCTION_STEP_SIZE"], p);
      }
      // Thread block size for the Cholesky
      // TODO(Steve): This should be tuned in a higher part of the stan language
      if (max_thread_block_size_ >= 256) {
        tuning_opts_.cholesky_min_L11_size = 256;
      } else {
        tuning_opts_.cholesky_min_L11_size = max_thread_block_size_;
      }
    } catch (const cl::Error& e) {
      check_opencl_error("opencl_context", e);
    }
  }

 protected:
  cl::Context context_;  // Manages the the device, queue, platform, memory,etc.
  cl::CommandQueue command_queue_;       // job queue for device, one per device
  std::vector<cl::Platform> platforms_;  // Vector of available platforms
  cl::Platform platform_;                // The platform for compiling kernels
  std::string platform_name_;  // The platform such as NVIDIA OpenCL or AMD SDK
  std::vector<cl::Device> devices_;  // All available OpenCL devices
  cl::Device device_;                // The selected OpenCL device
  std::string device_name_;          // The name of OpenCL device
  size_t max_thread_block_size_;  // The maximum size of a block of workers on
                                  // the device

  // Holds Default parameter values for each Kernel.
  typedef std::map<std::string, int> map_base_opts;
  map_base_opts base_opts_
      = {{"LOWER", static_cast<int>(matrix_cl_view::Lower)},
         {"UPPER", static_cast<int>(matrix_cl_view::Upper)},
         {"ENTIRE", static_cast<int>(matrix_cl_view::Entire)},
         {"DIAGONAL", static_cast<int>(matrix_cl_view::Diagonal)},
         {"UPPER_TO_LOWER", static_cast<int>(TriangularMapCL::UpperToLower)},
         {"LOWER_TO_UPPER", static_cast<int>(TriangularMapCL::LowerToUpper)},
         {"THREAD_BLOCK_SIZE", 32},
         {"WORK_PER_THREAD", 8},
         {"REDUCTION_STEP_SIZE", 4},
         {"LOCAL_SIZE_", 64}};
  // TODO(Steve): Make these tunable during warmup
  struct tuning_struct {
    // Used in math/opencl/cholesky_decompose
    int cholesky_min_L11_size = 256;
    int cholesky_partition = 4;
    int cholesky_size_worth_transfer = 1250;
    // Used in math/rev/mat/fun/cholesky_decompose
    int cholesky_rev_min_block_size = 512;
    int cholesky_rev_block_partition = 8;
    // used in math/opencl/multiply
    int multiply_split_upper_limit = 2000000;
    // used in math/prim/mat/fun/gp_exp_quad_cov
    double gp_exp_quad_cov_complex = 1'000'000;
    double gp_exp_quad_cov_simple = 1'250;
    // used in math/prim/mat/fun/multiply
    // and math/rev/mat/fun/multiply
    int multiply_dim_prod_worth_transfer = 2000000;
    // used in math/prim/mat/fun/mdivide_left_tri
    // and math/rev/mat/fun/mdivide_left_tri
    int tri_inverse_size_worth_transfer = 100;
    
  } tuning_opts_;

  int gpu_enabled = 1;
  
  struct cl_buffer_ {
    int used;
    const void* address;
    cl::Buffer buffer;
  };

  cl_buffer_ X_s_buf[2];
  cl_buffer_ X_r_buf[2];
  cl_buffer_ score_buf[2];
  cl_buffer_ time_buf[2];
  cl_buffer_ IDp_buf[2];
  cl_buffer_ IDs_buf[2];
  cl_buffer_ is_pbo_buf[2];

  static opencl_context_base& getInstance() {
    static opencl_context_base instance_;
    return instance_;
  }

  opencl_context_base(opencl_context_base const&) = delete;
  void operator=(opencl_context_base const&) = delete;
};

/**
 * The API to access the methods and values in opencl_context_base
 */
class opencl_context {
 public:
  opencl_context() = default;
  /**
   * Returns the description of the OpenCL platform and device that is used.
   * Devices will be an OpenCL and Platforms are a specific OpenCL implimenation
   * such as AMD SDK's or Nvidia's OpenCL implimentation.
   */
  inline std::string description() const {
    std::ostringstream msg;

    msg << "Platform ID: " << OPENCL_PLATFORM_ID << "\n";
    msg << "Platform Name: "
        << opencl_context_base::getInstance()
               .platform_.getInfo<CL_PLATFORM_NAME>()
        << "\n";
    msg << "Platform Vendor: "
        << opencl_context_base::getInstance()
               .platform_.getInfo<CL_PLATFORM_VENDOR>()
        << "\n";
    msg << "\tDevice " << OPENCL_DEVICE_ID << ": "
        << "\n";
    msg << "\t\tDevice Name: "
        << opencl_context_base::getInstance().device_.getInfo<CL_DEVICE_NAME>()
        << "\n";
    msg << "\t\tDevice Type: "
        << opencl_context_base::getInstance().device_.getInfo<CL_DEVICE_TYPE>()
        << "\n";
    msg << "\t\tDevice Vendor: "
        << opencl_context_base::getInstance()
               .device_.getInfo<CL_DEVICE_VENDOR>()
        << "\n";
    msg << "\t\tDevice Max Compute Units: "
        << opencl_context_base::getInstance()
               .device_.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
        << "\n";
    msg << "\t\tDevice Global Memory: "
        << opencl_context_base::getInstance()
               .device_.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()
        << "\n";
    msg << "\t\tDevice Max Clock Frequency: "
        << opencl_context_base::getInstance()
               .device_.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>()
        << "\n";
    msg << "\t\tDevice Max Allocateable Memory: "
        << opencl_context_base::getInstance()
               .device_.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()
        << "\n";
    msg << "\t\tDevice Local Memory: "
        << opencl_context_base::getInstance()
               .device_.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()
        << "\n";
    msg << "\t\tDevice Available: "
        << opencl_context_base::getInstance()
               .device_.getInfo<CL_DEVICE_AVAILABLE>()
        << "\n";
    return msg.str();
  }

  /**
   * Returns the description of the OpenCL platforms and devices that
   * are available. Devices will be an OpenCL and Platforms are a specific
   * OpenCL implimenation such as AMD SDK's or Nvidia's OpenCL implimentation.
   */
  inline std::string capabilities() const {
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    std::ostringstream msg;
    int platform_id = 0;
    int device_id = 0;

    msg << "Number of Platforms: " << all_platforms.size() << "\n";
    for (auto plat_iter : all_platforms) {
      cl::Platform platform(plat_iter);

      msg << "Platform ID: " << platform_id++ << "\n";
      msg << "Platform Name: " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";
      msg << "Platform Vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>()
          << "\n";

      try {
        std::vector<cl::Device> all_devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);

        for (auto device_iter : all_devices) {
          cl::Device device(device_iter);

          msg << "\tDevice " << device_id++ << ": "
              << "\n";
          msg << "\t\tDevice Name: " << device.getInfo<CL_DEVICE_NAME>()
              << "\n";
          msg << "\t\tDevice Type: " << device.getInfo<CL_DEVICE_TYPE>()
              << "\n";
          msg << "\t\tDevice Vendor: " << device.getInfo<CL_DEVICE_VENDOR>()
              << "\n";
          msg << "\t\tDevice Max Compute Units: "
              << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << "\n";
          msg << "\t\tDevice Global Memory: "
              << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << "\n";
          msg << "\t\tDevice Max Clock Frequency: "
              << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "\n";
          msg << "\t\tDevice Max Allocateable Memory: "
              << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << "\n";
          msg << "\t\tDevice Local Memory: "
              << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << "\n";
          msg << "\t\tDevice Available: "
              << device.getInfo<CL_DEVICE_AVAILABLE>() << "\n";
        }
      } catch (const cl::Error& e) {
        // if one of the platforms have no devices that match the device type
        // it will throw the error == -1 (DEVICE_NOT_FOUND)
        // other errors will throw a system error
        if (e.err() == -1) {
          msg << "\tno (OpenCL) devices in the platform with ID " << platform_id
              << "\n";
        } else {
          check_opencl_error("capabilities", e);
        }
      }
    }
    return msg.str();
  }

  /**
   * Returns the reference to the OpenCL context. The OpenCL context manages
   * objects such as the device, memory, command queue, program, and kernel
   * objects. For stan, there should only be one context, queue, device, and
   * program with multiple kernels.
   */
  inline cl::Context& context() {
    return opencl_context_base::getInstance().context_;
  }
  /**
   * Returns the reference to the active OpenCL command queue for the device.
   * One command queue will exist per device where
   * kernels are placed on the command queue and by default executed in order.
   */
  inline cl::CommandQueue& queue() {
    return opencl_context_base::getInstance().command_queue_;
  }
  /**
   * Returns a copy of the map of kernel defines
   */
  inline opencl_context_base::map_base_opts base_opts() {
    return opencl_context_base::getInstance().base_opts_;
  }
  /**
   * Returns the maximum thread block size defined by
   * CL_DEVICE_MAX_WORK_GROUP_SIZE for the device in the context. This is the
   * maximum product of thread block dimensions for a particular device. IE a
   * max workgoup of 256 would allow thread blocks of sizes (16,16), (128,2),
   * (8, 32), etc.
   */
  inline int max_thread_block_size() {
    return opencl_context_base::getInstance().max_thread_block_size_;
  }

  /**
   * Returns the thread block size for the Cholesky Decompositions L_11.
   */
  inline opencl_context_base::tuning_struct& tuning_opts() {
    return opencl_context_base::getInstance().tuning_opts_;
  }

  /**
   * Returns a vector containing the OpenCL device used to create the context
   */
  inline std::vector<cl::Device> device() {
    return {opencl_context_base::getInstance().device_};
  }

  /**
   * Returns a vector containing the OpenCL platform used to create the context
   */
  inline std::vector<cl::Platform> platform() {
    return {opencl_context_base::getInstance().platform_};
  }

  inline int gpu_enabled() {
    return opencl_context_base::getInstance().gpu_enabled;
  }

  inline void gpu_enabled(int i) {
    opencl_context_base::getInstance().gpu_enabled = i;
  }

  inline int X_s_copied(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& x) {
    const void* addr = static_cast<const void*>(&x);
    if(opencl_context_base::getInstance().X_s_buf[0].used == 0) {
      return -1;
    } else if(opencl_context_base::getInstance().X_s_buf[0].address == addr) {
      return 0;
    } else if(opencl_context_base::getInstance().X_s_buf[1].used == 0) {
      return -2;
    } else if(opencl_context_base::getInstance().X_s_buf[1].address == addr)  {
      return 1;
    }
    // this should not happen
    return -5;
  }

  inline cl::Buffer& X_s(int ind) {
    return opencl_context_base::getInstance().X_s_buf[ind].buffer;
  }
  
  inline void X_s_addr(int ind, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& x) {
    opencl_context_base::getInstance().X_s_buf[ind].address = static_cast<const void*>(&x);
    opencl_context_base::getInstance().X_s_buf[ind].used = 1;
  }
  
  inline int X_r_copied(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& x) {
    const void* addr = static_cast<const void*>(&x);
    if(opencl_context_base::getInstance().X_r_buf[0].used == 0) {
      return -1;
    } else if(opencl_context_base::getInstance().X_r_buf[0].address == addr) {
      return 0;
    } else if(opencl_context_base::getInstance().X_r_buf[1].used == 0) {
      return -2;
    } else if(opencl_context_base::getInstance().X_r_buf[1].address == addr)  {
      return 1;
    }
    // this should not happen
    return -5;
  }

  inline cl::Buffer& X_r(int ind) {
    return opencl_context_base::getInstance().X_r_buf[ind].buffer;
  }
    
  inline void X_r_addr(int ind, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& x) {
    opencl_context_base::getInstance().X_r_buf[ind].address = static_cast<const void*>(&x);
    opencl_context_base::getInstance().X_r_buf[ind].used = 1;
  }

  inline int score_copied(const vector_d& x) {
    const void* addr = static_cast<const void*>(&x);
    if(opencl_context_base::getInstance().score_buf[0].used == 0) {
      return -1;
    } else if(opencl_context_base::getInstance().score_buf[0].address == addr) {
      return 0;
    } else if(opencl_context_base::getInstance().score_buf[1].used == 0) {
      return -2;
    } else if(opencl_context_base::getInstance().score_buf[1].address == addr)  {
      return 1;
    }
    // this should not happen
    return -5;
  }

  inline cl::Buffer& score(int ind) {
    return opencl_context_base::getInstance().score_buf[ind].buffer;
  }
  
  inline void score_addr(int ind, const vector_d& x) {
    opencl_context_base::getInstance().score_buf[ind].address = static_cast<const void*>(&x);
    opencl_context_base::getInstance().score_buf[ind].used = 1;
  }

  inline int time_copied(const vector_d& x) {
    const void* addr = static_cast<const void*>(&x);
    if(opencl_context_base::getInstance().time_buf[0].used == 0) {
      return -1;
    } else if(opencl_context_base::getInstance().time_buf[0].address == addr) {
      return 0;
    } else if(opencl_context_base::getInstance().time_buf[1].used == 0) {
      return -2;
    } else if(opencl_context_base::getInstance().time_buf[1].address == addr)  {
      return 1;
    }
    // this should not happen
    return -5;
  }

  inline cl::Buffer& time(int ind) {
    return opencl_context_base::getInstance().time_buf[ind].buffer;
  }
  
  inline void time_addr(int ind, const vector_d& x) {
    opencl_context_base::getInstance().time_buf[ind].address = static_cast<const void*>(&x);
    opencl_context_base::getInstance().time_buf[ind].used = 1;
  }

  inline int IDp_copied(const std::vector<int>& x) {
    const void* addr = static_cast<const void*>(&x);
    if(opencl_context_base::getInstance().IDp_buf[0].used == 0) {
      return -1;
    } else if(opencl_context_base::getInstance().IDp_buf[0].address == addr) {
      return 0;
    } else if(opencl_context_base::getInstance().IDp_buf[1].used == 0) {
      return -2;
    } else if(opencl_context_base::getInstance().IDp_buf[1].address == addr)  {
      return 1;
    }
    // this should not happen
    return -5;
  }

  inline cl::Buffer& IDp(int ind) {
    return opencl_context_base::getInstance().IDp_buf[ind].buffer;
  }
  
  inline void IDp_addr(int ind, const std::vector<int>& x) {
    opencl_context_base::getInstance().IDp_buf[ind].address = static_cast<const void*>(&x);
    opencl_context_base::getInstance().IDp_buf[ind].used = 1;
  }

  inline int IDs_copied(const std::vector<int>& x) {
    const void* addr = static_cast<const void*>(&x);
    if(opencl_context_base::getInstance().IDs_buf[0].used == 0) {
      return -1;
    } else if(opencl_context_base::getInstance().IDs_buf[0].address == addr) {
      return 0;
    } else if(opencl_context_base::getInstance().IDs_buf[1].used == 0) {
      return -2;
    } else if(opencl_context_base::getInstance().IDs_buf[1].address == addr)  {
      return 1;
    }
    // this should not happen
    return -5;
  }

  inline cl::Buffer& IDs(int ind) {
    return opencl_context_base::getInstance().IDs_buf[ind].buffer;
  }
  
  inline void IDs_addr(int ind, const std::vector<int>& x) {
    opencl_context_base::getInstance().IDs_buf[ind].address = static_cast<const void*>(&x);
    opencl_context_base::getInstance().IDs_buf[ind].used = 1;
  }

  inline int is_pbo_copied(const std::vector<int>& x) {
    const void* addr = static_cast<const void*>(&x);
    if(opencl_context_base::getInstance().is_pbo_buf[0].used == 0) {
      return -1;
    } else if(opencl_context_base::getInstance().is_pbo_buf[0].address == addr) {
      return 0;
    } else if(opencl_context_base::getInstance().is_pbo_buf[1].used == 0) {
      return -2;
    } else if(opencl_context_base::getInstance().is_pbo_buf[1].address == addr)  {
      return 1;
    }
    // this should not happen
    return -5;
  }

  inline cl::Buffer& is_pbo(int ind) {
    return opencl_context_base::getInstance().is_pbo_buf[ind].buffer;
  }
  
  inline void is_pbo_addr(int ind, const std::vector<int>& x) {
    opencl_context_base::getInstance().is_pbo_buf[ind].address = static_cast<const void*>(&x);
    opencl_context_base::getInstance().is_pbo_buf[ind].used = 1;
  }

  inline void clear_cache() {
    opencl_context_base::getInstance().X_s_buf[0].used = 0;
    opencl_context_base::getInstance().X_s_buf[1].used = 0;
    opencl_context_base::getInstance().X_r_buf[0].used = 0;
    opencl_context_base::getInstance().X_r_buf[1].used = 0;
    opencl_context_base::getInstance().score_buf[0].used = 0;
    opencl_context_base::getInstance().score_buf[1].used = 0;
    opencl_context_base::getInstance().time_buf[0].used = 0;
    opencl_context_base::getInstance().time_buf[1].used = 0;
    opencl_context_base::getInstance().IDp_buf[0].used = 0;
    opencl_context_base::getInstance().IDp_buf[1].used = 0;
    opencl_context_base::getInstance().IDs_buf[0].used = 0;
    opencl_context_base::getInstance().IDs_buf[1].used = 0;
    opencl_context_base::getInstance().is_pbo_buf[0].used = 0;
    opencl_context_base::getInstance().is_pbo_buf[1].used = 0;
  }
};
static opencl_context opencl_context;
}  // namespace math
}  // namespace stan

#endif
#endif
