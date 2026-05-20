#include <motifcl/runtime/vulkan_backend.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace motifcl {
namespace {

using VkFlags = std::uint32_t;
using VkInstanceCreateFlags = VkFlags;
using VkStructureType = std::int32_t;
using VkResult = std::int32_t;
using VkInstance = struct VkInstance_T*;
using VkPhysicalDevice = struct VkPhysicalDevice_T*;
using PFN_vkVoidFunction = void (*)();

constexpr VkResult VK_SUCCESS = 0;
constexpr VkStructureType VK_STRUCTURE_TYPE_APPLICATION_INFO = 0;
constexpr VkStructureType VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1;

constexpr std::uint32_t vk_make_api_version(std::uint32_t variant,
                                            std::uint32_t major,
                                            std::uint32_t minor,
                                            std::uint32_t patch) {
    return (variant << 29u) | (major << 22u) | (minor << 12u) | patch;
}

struct VkApplicationInfo {
    VkStructureType sType;
    const void* pNext;
    const char* pApplicationName;
    std::uint32_t applicationVersion;
    const char* pEngineName;
    std::uint32_t engineVersion;
    std::uint32_t apiVersion;
};

struct VkInstanceCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkInstanceCreateFlags flags;
    const VkApplicationInfo* pApplicationInfo;
    std::uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    std::uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
};

using PFN_vkGetInstanceProcAddr = PFN_vkVoidFunction (*)(VkInstance instance, const char* pName);
using PFN_vkEnumerateInstanceVersion = VkResult (*)(std::uint32_t* pApiVersion);
using PFN_vkCreateInstance = VkResult (*)(const VkInstanceCreateInfo* pCreateInfo,
                                          const void* pAllocator,
                                          VkInstance* pInstance);
using PFN_vkDestroyInstance = void (*)(VkInstance instance, const void* pAllocator);
using PFN_vkEnumeratePhysicalDevices = VkResult (*)(VkInstance instance,
                                                    std::uint32_t* pPhysicalDeviceCount,
                                                    VkPhysicalDevice* pPhysicalDevices);
using PFN_vkGetPhysicalDeviceProperties = void (*)(VkPhysicalDevice physicalDevice, void* pProperties);

struct VkPhysicalDevicePropertiesPrefix {
    std::uint32_t apiVersion;
    std::uint32_t driverVersion;
    std::uint32_t vendorID;
    std::uint32_t deviceID;
    std::uint32_t deviceType;
    char deviceName[256];
};

class DynamicLibrary {
public:
#ifdef _WIN32
    using Symbol = FARPROC;
#else
    using Symbol = void*;
#endif

    DynamicLibrary() = default;
    DynamicLibrary(const DynamicLibrary&) = delete;
    DynamicLibrary& operator=(const DynamicLibrary&) = delete;

    ~DynamicLibrary() {
#ifdef _WIN32
        if (handle_) FreeLibrary(handle_);
#else
        if (handle_) dlclose(handle_);
#endif
    }

    bool open(const char* name) {
#ifdef _WIN32
        handle_ = LoadLibraryA(name);
#else
        handle_ = dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
        if (handle_) {
            name_ = name;
            return true;
        }
        return false;
    }

    Symbol symbol(const char* name) const {
#ifdef _WIN32
        return handle_ ? GetProcAddress(handle_, name) : nullptr;
#else
        return handle_ ? dlsym(handle_, name) : nullptr;
#endif
    }

    const std::string& name() const { return name_; }

private:
#ifdef _WIN32
    HMODULE handle_ = nullptr;
#else
    void* handle_ = nullptr;
#endif
    std::string name_;
};

template <typename Fn>
Fn load_symbol(const DynamicLibrary& library, const char* name) {
    const auto symbol = library.symbol(name);
    Fn fn = nullptr;
    static_assert(sizeof(fn) == sizeof(symbol), "Vulkan function pointer size mismatch");
    std::memcpy(&fn, &symbol, sizeof(fn));
    return fn;
}

template <typename Fn>
Fn load_instance_function(PFN_vkGetInstanceProcAddr get_proc, VkInstance instance, const char* name) {
    const auto symbol = get_proc(instance, name);
    Fn fn = nullptr;
    static_assert(sizeof(fn) == sizeof(symbol), "Vulkan instance function pointer size mismatch");
    std::memcpy(&fn, &symbol, sizeof(fn));
    return fn;
}

bool open_vulkan_loader(DynamicLibrary& library) {
#ifdef _WIN32
    const std::array<const char*, 1> names = {"vulkan-1.dll"};
#else
    const std::array<const char*, 2> names = {"libvulkan.so.1", "libvulkan.so"};
#endif
    return std::any_of(names.begin(), names.end(), [&](const char* name) {
        return library.open(name);
    });
}

std::string bounded_string(const char* value, std::size_t max_size) {
    std::size_t size = 0;
    while (size < max_size && value[size] != '\0') ++size;
    return std::string(value, size);
}

} // namespace

std::string vulkan_version_string(std::uint32_t version) {
    const auto major = (version >> 22u) & 0x7fu;
    const auto minor = (version >> 12u) & 0x3ffu;
    const auto patch = version & 0xfffu;
    std::ostringstream out;
    out << major << '.' << minor << '.' << patch;
    return out.str();
}

VulkanProbeResult probe_vulkan_runtime() {
    VulkanProbeResult result;

    DynamicLibrary loader;
    if (!open_vulkan_loader(loader)) {
        result.error = "Vulkan loader not found";
        return result;
    }
    result.loader_found = true;
    result.loader_path = loader.name();

    auto get_proc = load_symbol<PFN_vkGetInstanceProcAddr>(loader, "vkGetInstanceProcAddr");
    if (!get_proc) {
        result.error = "Vulkan loader is missing vkGetInstanceProcAddr";
        return result;
    }

    auto enumerate_instance_version =
        load_instance_function<PFN_vkEnumerateInstanceVersion>(get_proc, nullptr, "vkEnumerateInstanceVersion");
    if (!enumerate_instance_version) {
        enumerate_instance_version = load_symbol<PFN_vkEnumerateInstanceVersion>(loader, "vkEnumerateInstanceVersion");
    }
    result.api_version = vk_make_api_version(0, 1, 0, 0);
    if (enumerate_instance_version) {
        const VkResult rc = enumerate_instance_version(&result.api_version);
        if (rc != VK_SUCCESS) {
            result.error = "vkEnumerateInstanceVersion failed";
            return result;
        }
    }

    auto create_instance = load_instance_function<PFN_vkCreateInstance>(get_proc, nullptr, "vkCreateInstance");
    if (!create_instance) create_instance = load_symbol<PFN_vkCreateInstance>(loader, "vkCreateInstance");
    if (!create_instance) {
        result.error = "Vulkan loader is missing vkCreateInstance";
        return result;
    }

    const VkApplicationInfo app_info{
        VK_STRUCTURE_TYPE_APPLICATION_INFO,
        nullptr,
        "MotifCL",
        vk_make_api_version(0, 0, 1, 0),
        "MotifCL",
        vk_make_api_version(0, 0, 1, 0),
        std::max(result.api_version, vk_make_api_version(0, 1, 0, 0)),
    };
    const VkInstanceCreateInfo create_info{
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        nullptr,
        0,
        &app_info,
        0,
        nullptr,
        0,
        nullptr,
    };

    VkInstance instance = nullptr;
    const VkResult create_rc = create_instance(&create_info, nullptr, &instance);
    if (create_rc != VK_SUCCESS || !instance) {
        result.error = "vkCreateInstance failed";
        return result;
    }
    result.instance_created = true;

    auto destroy_instance = load_instance_function<PFN_vkDestroyInstance>(get_proc, instance, "vkDestroyInstance");
    auto enumerate_physical_devices =
        load_instance_function<PFN_vkEnumeratePhysicalDevices>(get_proc, instance, "vkEnumeratePhysicalDevices");
    auto get_physical_device_properties =
        load_instance_function<PFN_vkGetPhysicalDeviceProperties>(get_proc, instance, "vkGetPhysicalDeviceProperties");
    if (!enumerate_physical_devices) {
        result.error = "Vulkan instance is missing vkEnumeratePhysicalDevices";
        if (destroy_instance) destroy_instance(instance, nullptr);
        return result;
    }

    const VkResult enum_rc = enumerate_physical_devices(instance, &result.physical_device_count, nullptr);
    if (enum_rc != VK_SUCCESS) {
        result.error = "vkEnumeratePhysicalDevices failed";
    } else if (result.physical_device_count == 0) {
        result.error = "No Vulkan physical devices found";
    } else {
        std::vector<VkPhysicalDevice> devices(result.physical_device_count, nullptr);
        const VkResult enum_devices_rc =
            enumerate_physical_devices(instance, &result.physical_device_count, devices.data());
        if (enum_devices_rc != VK_SUCCESS) {
            result.error = "vkEnumeratePhysicalDevices device list failed";
        } else if (get_physical_device_properties) {
            result.devices.reserve(result.physical_device_count);
            for (VkPhysicalDevice device : devices) {
                struct alignas(8) PropertiesStorage {
                    std::array<std::uint8_t, 4096> bytes;
                };
                PropertiesStorage storage{};
                get_physical_device_properties(device, storage.bytes.data());
                const auto* prefix = reinterpret_cast<const VkPhysicalDevicePropertiesPrefix*>(storage.bytes.data());
                VulkanPhysicalDeviceInfo info;
                info.name = bounded_string(prefix->deviceName, sizeof(prefix->deviceName));
                info.vendor_id = prefix->vendorID;
                info.device_id = prefix->deviceID;
                info.device_type = prefix->deviceType;
                info.api_version = prefix->apiVersion;
                info.driver_version = prefix->driverVersion;
                result.devices.push_back(std::move(info));
            }
        }
    }

    if (destroy_instance) destroy_instance(instance, nullptr);
    return result;
}

} // namespace motifcl
