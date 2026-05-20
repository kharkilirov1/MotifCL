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
using VkDeviceSize = std::uint64_t;
using VkBool32 = std::uint32_t;
using VkInstanceCreateFlags = VkFlags;
using VkDeviceCreateFlags = VkFlags;
using VkDeviceQueueCreateFlags = VkFlags;
using VkBufferCreateFlags = VkFlags;
using VkBufferUsageFlags = VkFlags;
using VkMemoryPropertyFlags = VkFlags;
using VkDescriptorSetLayoutCreateFlags = VkFlags;
using VkDescriptorPoolCreateFlags = VkFlags;
using VkPipelineLayoutCreateFlags = VkFlags;
using VkShaderModuleCreateFlags = VkFlags;
using VkPipelineCreateFlags = VkFlags;
using VkCommandPoolCreateFlags = VkFlags;
using VkCommandBufferUsageFlags = VkFlags;
using VkQueueFlags = VkFlags;
using VkPipelineStageFlags = VkFlags;
using VkShaderStageFlags = VkFlags;
using VkDescriptorPoolResetFlags = VkFlags;
using VkStructureType = std::int32_t;
using VkResult = std::int32_t;
using VkInstance = struct VkInstance_T*;
using VkPhysicalDevice = struct VkPhysicalDevice_T*;
using VkDevice = struct VkDevice_T*;
using VkQueue = struct VkQueue_T*;
using VkBuffer = struct VkBuffer_T*;
using VkDeviceMemory = struct VkDeviceMemory_T*;
using VkShaderModule = struct VkShaderModule_T*;
using VkDescriptorSetLayout = struct VkDescriptorSetLayout_T*;
using VkPipelineLayout = struct VkPipelineLayout_T*;
using VkPipeline = struct VkPipeline_T*;
using VkDescriptorPool = struct VkDescriptorPool_T*;
using VkDescriptorSet = struct VkDescriptorSet_T*;
using VkCommandPool = struct VkCommandPool_T*;
using VkCommandBuffer = struct VkCommandBuffer_T*;
using PFN_vkVoidFunction = void (*)();

constexpr VkResult VK_SUCCESS = 0;
constexpr VkStructureType VK_STRUCTURE_TYPE_APPLICATION_INFO = 0;
constexpr VkStructureType VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1;
constexpr VkStructureType VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2;
constexpr VkStructureType VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3;
constexpr VkStructureType VK_STRUCTURE_TYPE_SUBMIT_INFO = 4;
constexpr VkStructureType VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5;
constexpr VkStructureType VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12;
constexpr VkStructureType VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16;
constexpr VkStructureType VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18;
constexpr VkStructureType VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 28;
constexpr VkStructureType VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30;
constexpr VkStructureType VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32;
constexpr VkStructureType VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33;
constexpr VkStructureType VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34;
constexpr VkStructureType VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35;
constexpr VkStructureType VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39;
constexpr VkStructureType VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40;
constexpr VkStructureType VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42;

constexpr VkQueueFlags VK_QUEUE_COMPUTE_BIT = 0x00000002u;
constexpr VkBufferUsageFlags VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020u;
constexpr VkMemoryPropertyFlags VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000001u;
constexpr VkMemoryPropertyFlags VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000002u;
constexpr std::int32_t VK_SHARING_MODE_EXCLUSIVE = 0;
constexpr std::int32_t VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7;
constexpr VkShaderStageFlags VK_SHADER_STAGE_COMPUTE_BIT = 0x00000020u;
constexpr std::int32_t VK_PIPELINE_BIND_POINT_COMPUTE = 1;
constexpr std::int32_t VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0;

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

struct VkDeviceQueueCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkDeviceQueueCreateFlags flags;
    std::uint32_t queueFamilyIndex;
    std::uint32_t queueCount;
    const float* pQueuePriorities;
};

struct VkDeviceCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkDeviceCreateFlags flags;
    std::uint32_t queueCreateInfoCount;
    const VkDeviceQueueCreateInfo* pQueueCreateInfos;
    std::uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    std::uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
    const void* pEnabledFeatures;
};

struct VkQueueFamilyProperties {
    VkQueueFlags queueFlags;
    std::uint32_t queueCount;
    std::uint32_t timestampValidBits;
    struct {
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t depth;
    } minImageTransferGranularity;
};

struct VkMemoryType {
    VkMemoryPropertyFlags propertyFlags;
    std::uint32_t heapIndex;
};

struct VkMemoryHeap {
    VkDeviceSize size;
    VkFlags flags;
};

struct VkPhysicalDeviceMemoryProperties {
    std::uint32_t memoryTypeCount;
    VkMemoryType memoryTypes[32];
    std::uint32_t memoryHeapCount;
    VkMemoryHeap memoryHeaps[16];
};

struct VkBufferCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkBufferCreateFlags flags;
    VkDeviceSize size;
    VkBufferUsageFlags usage;
    std::int32_t sharingMode;
    std::uint32_t queueFamilyIndexCount;
    const std::uint32_t* pQueueFamilyIndices;
};

struct VkMemoryRequirements {
    VkDeviceSize size;
    VkDeviceSize alignment;
    std::uint32_t memoryTypeBits;
};

struct VkMemoryAllocateInfo {
    VkStructureType sType;
    const void* pNext;
    VkDeviceSize allocationSize;
    std::uint32_t memoryTypeIndex;
};

struct VkDescriptorSetLayoutBinding {
    std::uint32_t binding;
    std::int32_t descriptorType;
    std::uint32_t descriptorCount;
    VkShaderStageFlags stageFlags;
    const void* pImmutableSamplers;
};

struct VkDescriptorSetLayoutCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkDescriptorSetLayoutCreateFlags flags;
    std::uint32_t bindingCount;
    const VkDescriptorSetLayoutBinding* pBindings;
};

struct VkDescriptorPoolSize {
    std::int32_t type;
    std::uint32_t descriptorCount;
};

struct VkDescriptorPoolCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkDescriptorPoolCreateFlags flags;
    std::uint32_t maxSets;
    std::uint32_t poolSizeCount;
    const VkDescriptorPoolSize* pPoolSizes;
};

struct VkDescriptorSetAllocateInfo {
    VkStructureType sType;
    const void* pNext;
    VkDescriptorPool descriptorPool;
    std::uint32_t descriptorSetCount;
    const VkDescriptorSetLayout* pSetLayouts;
};

struct VkDescriptorBufferInfo {
    VkBuffer buffer;
    VkDeviceSize offset;
    VkDeviceSize range;
};

struct VkWriteDescriptorSet {
    VkStructureType sType;
    const void* pNext;
    VkDescriptorSet dstSet;
    std::uint32_t dstBinding;
    std::uint32_t dstArrayElement;
    std::uint32_t descriptorCount;
    std::int32_t descriptorType;
    const void* pImageInfo;
    const VkDescriptorBufferInfo* pBufferInfo;
    const void* pTexelBufferView;
};

struct VkPushConstantRange {
    VkShaderStageFlags stageFlags;
    std::uint32_t offset;
    std::uint32_t size;
};

struct VkPipelineLayoutCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkPipelineLayoutCreateFlags flags;
    std::uint32_t setLayoutCount;
    const VkDescriptorSetLayout* pSetLayouts;
    std::uint32_t pushConstantRangeCount;
    const VkPushConstantRange* pPushConstantRanges;
};

struct VkShaderModuleCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkShaderModuleCreateFlags flags;
    std::size_t codeSize;
    const std::uint32_t* pCode;
};

struct VkPipelineShaderStageCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkPipelineCreateFlags flags;
    VkShaderStageFlags stage;
    VkShaderModule module;
    const char* pName;
    const void* pSpecializationInfo;
};

struct VkComputePipelineCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkPipelineCreateFlags flags;
    VkPipelineShaderStageCreateInfo stage;
    VkPipelineLayout layout;
    VkPipeline basePipelineHandle;
    std::int32_t basePipelineIndex;
};

struct VkCommandPoolCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkCommandPoolCreateFlags flags;
    std::uint32_t queueFamilyIndex;
};

struct VkCommandBufferAllocateInfo {
    VkStructureType sType;
    const void* pNext;
    VkCommandPool commandPool;
    std::int32_t level;
    std::uint32_t commandBufferCount;
};

struct VkCommandBufferBeginInfo {
    VkStructureType sType;
    const void* pNext;
    VkCommandBufferUsageFlags flags;
    const void* pInheritanceInfo;
};

struct VkSubmitInfo {
    VkStructureType sType;
    const void* pNext;
    std::uint32_t waitSemaphoreCount;
    const void* pWaitSemaphores;
    const VkPipelineStageFlags* pWaitDstStageMask;
    std::uint32_t commandBufferCount;
    const VkCommandBuffer* pCommandBuffers;
    std::uint32_t signalSemaphoreCount;
    const void* pSignalSemaphores;
};

using PFN_vkGetInstanceProcAddr = PFN_vkVoidFunction (*)(VkInstance instance, const char* pName);
using PFN_vkGetDeviceProcAddr = PFN_vkVoidFunction (*)(VkDevice device, const char* pName);
using PFN_vkEnumerateInstanceVersion = VkResult (*)(std::uint32_t* pApiVersion);
using PFN_vkCreateInstance = VkResult (*)(const VkInstanceCreateInfo* pCreateInfo,
                                          const void* pAllocator,
                                          VkInstance* pInstance);
using PFN_vkDestroyInstance = void (*)(VkInstance instance, const void* pAllocator);
using PFN_vkEnumeratePhysicalDevices = VkResult (*)(VkInstance instance,
                                                    std::uint32_t* pPhysicalDeviceCount,
                                                    VkPhysicalDevice* pPhysicalDevices);
using PFN_vkGetPhysicalDeviceProperties = void (*)(VkPhysicalDevice physicalDevice, void* pProperties);
using PFN_vkGetPhysicalDeviceQueueFamilyProperties = void (*)(VkPhysicalDevice physicalDevice,
                                                              std::uint32_t* pQueueFamilyPropertyCount,
                                                              VkQueueFamilyProperties* pQueueFamilyProperties);
using PFN_vkGetPhysicalDeviceMemoryProperties = void (*)(VkPhysicalDevice physicalDevice,
                                                         VkPhysicalDeviceMemoryProperties* pMemoryProperties);
using PFN_vkCreateDevice = VkResult (*)(VkPhysicalDevice physicalDevice,
                                        const VkDeviceCreateInfo* pCreateInfo,
                                        const void* pAllocator,
                                        VkDevice* pDevice);
using PFN_vkDestroyDevice = void (*)(VkDevice device, const void* pAllocator);
using PFN_vkGetDeviceQueue = void (*)(VkDevice device, std::uint32_t queueFamilyIndex,
                                      std::uint32_t queueIndex, VkQueue* pQueue);
using PFN_vkCreateBuffer = VkResult (*)(VkDevice device, const VkBufferCreateInfo* pCreateInfo,
                                        const void* pAllocator, VkBuffer* pBuffer);
using PFN_vkDestroyBuffer = void (*)(VkDevice device, VkBuffer buffer, const void* pAllocator);
using PFN_vkGetBufferMemoryRequirements = void (*)(VkDevice device, VkBuffer buffer,
                                                   VkMemoryRequirements* pMemoryRequirements);
using PFN_vkAllocateMemory = VkResult (*)(VkDevice device, const VkMemoryAllocateInfo* pAllocateInfo,
                                          const void* pAllocator, VkDeviceMemory* pMemory);
using PFN_vkFreeMemory = void (*)(VkDevice device, VkDeviceMemory memory, const void* pAllocator);
using PFN_vkBindBufferMemory = VkResult (*)(VkDevice device, VkBuffer buffer, VkDeviceMemory memory,
                                            VkDeviceSize memoryOffset);
using PFN_vkMapMemory = VkResult (*)(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset,
                                     VkDeviceSize size, VkFlags flags, void** ppData);
using PFN_vkUnmapMemory = void (*)(VkDevice device, VkDeviceMemory memory);
using PFN_vkCreateDescriptorSetLayout = VkResult (*)(VkDevice device,
                                                     const VkDescriptorSetLayoutCreateInfo* pCreateInfo,
                                                     const void* pAllocator,
                                                     VkDescriptorSetLayout* pSetLayout);
using PFN_vkDestroyDescriptorSetLayout = void (*)(VkDevice device, VkDescriptorSetLayout descriptorSetLayout,
                                                  const void* pAllocator);
using PFN_vkCreateDescriptorPool = VkResult (*)(VkDevice device, const VkDescriptorPoolCreateInfo* pCreateInfo,
                                                const void* pAllocator, VkDescriptorPool* pDescriptorPool);
using PFN_vkDestroyDescriptorPool = void (*)(VkDevice device, VkDescriptorPool descriptorPool,
                                             const void* pAllocator);
using PFN_vkAllocateDescriptorSets = VkResult (*)(VkDevice device,
                                                  const VkDescriptorSetAllocateInfo* pAllocateInfo,
                                                  VkDescriptorSet* pDescriptorSets);
using PFN_vkUpdateDescriptorSets = void (*)(VkDevice device, std::uint32_t descriptorWriteCount,
                                            const VkWriteDescriptorSet* pDescriptorWrites,
                                            std::uint32_t descriptorCopyCount, const void* pDescriptorCopies);
using PFN_vkCreatePipelineLayout = VkResult (*)(VkDevice device,
                                                const VkPipelineLayoutCreateInfo* pCreateInfo,
                                                const void* pAllocator, VkPipelineLayout* pPipelineLayout);
using PFN_vkDestroyPipelineLayout = void (*)(VkDevice device, VkPipelineLayout pipelineLayout,
                                             const void* pAllocator);
using PFN_vkCreateShaderModule = VkResult (*)(VkDevice device, const VkShaderModuleCreateInfo* pCreateInfo,
                                              const void* pAllocator, VkShaderModule* pShaderModule);
using PFN_vkDestroyShaderModule = void (*)(VkDevice device, VkShaderModule shaderModule,
                                           const void* pAllocator);
using PFN_vkCreateComputePipelines = VkResult (*)(VkDevice device, VkPipeline pipelineCache,
                                                  std::uint32_t createInfoCount,
                                                  const VkComputePipelineCreateInfo* pCreateInfos,
                                                  const void* pAllocator, VkPipeline* pPipelines);
using PFN_vkDestroyPipeline = void (*)(VkDevice device, VkPipeline pipeline, const void* pAllocator);
using PFN_vkCreateCommandPool = VkResult (*)(VkDevice device, const VkCommandPoolCreateInfo* pCreateInfo,
                                             const void* pAllocator, VkCommandPool* pCommandPool);
using PFN_vkDestroyCommandPool = void (*)(VkDevice device, VkCommandPool commandPool, const void* pAllocator);
using PFN_vkAllocateCommandBuffers = VkResult (*)(VkDevice device,
                                                  const VkCommandBufferAllocateInfo* pAllocateInfo,
                                                  VkCommandBuffer* pCommandBuffers);
using PFN_vkBeginCommandBuffer = VkResult (*)(VkCommandBuffer commandBuffer,
                                              const VkCommandBufferBeginInfo* pBeginInfo);
using PFN_vkEndCommandBuffer = VkResult (*)(VkCommandBuffer commandBuffer);
using PFN_vkCmdBindPipeline = void (*)(VkCommandBuffer commandBuffer, std::int32_t pipelineBindPoint,
                                       VkPipeline pipeline);
using PFN_vkCmdBindDescriptorSets = void (*)(VkCommandBuffer commandBuffer, std::int32_t pipelineBindPoint,
                                             VkPipelineLayout layout, std::uint32_t firstSet,
                                             std::uint32_t descriptorSetCount,
                                             const VkDescriptorSet* pDescriptorSets,
                                             std::uint32_t dynamicOffsetCount,
                                             const std::uint32_t* pDynamicOffsets);
using PFN_vkCmdDispatch = void (*)(VkCommandBuffer commandBuffer, std::uint32_t groupCountX,
                                   std::uint32_t groupCountY, std::uint32_t groupCountZ);
using PFN_vkQueueSubmit = VkResult (*)(VkQueue queue, std::uint32_t submitCount,
                                       const VkSubmitInfo* pSubmits, void* fence);
using PFN_vkQueueWaitIdle = VkResult (*)(VkQueue queue);

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

template <typename Fn>
Fn load_device_function(PFN_vkGetDeviceProcAddr get_proc, VkDevice device, const char* name) {
    const auto symbol = get_proc(device, name);
    Fn fn = nullptr;
    static_assert(sizeof(fn) == sizeof(symbol), "Vulkan device function pointer size mismatch");
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

const std::array<std::uint32_t, 97>& smoke_compute_spirv() {
    static const std::array<std::uint32_t, 97> words = {
        0x07230203, 0x00010000, 0x00000000, 0x0000000f, 0x00000000, 0x00020011, 0x00000001, 0x0003000e,
        0x00000000, 0x00000001, 0x0005000f, 0x00000005, 0x00000003, 0x6e69616d, 0x00000000, 0x00060010,
        0x00000003, 0x00000011, 0x00000001, 0x00000001, 0x00000001, 0x00040047, 0x00000009, 0x00000006,
        0x00000004, 0x00050048, 0x0000000a, 0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x0000000a,
        0x00000003, 0x00040047, 0x0000000d, 0x00000022, 0x00000000, 0x00040047, 0x0000000d, 0x00000021,
        0x00000000, 0x00020013, 0x00000001, 0x00030021, 0x00000002, 0x00000001, 0x00030016, 0x00000005,
        0x00000020, 0x00040015, 0x00000006, 0x00000020, 0x00000000, 0x0004002b, 0x00000006, 0x00000007,
        0x00000000, 0x0004002b, 0x00000005, 0x00000008, 0x42280000, 0x0003001d, 0x00000009, 0x00000005,
        0x0003001e, 0x0000000a, 0x00000009, 0x00040020, 0x0000000b, 0x00000002, 0x0000000a, 0x00040020,
        0x0000000c, 0x00000002, 0x00000005, 0x0004003b, 0x0000000b, 0x0000000d, 0x00000002, 0x00050036,
        0x00000001, 0x00000003, 0x00000000, 0x00000002, 0x000200f8, 0x00000004, 0x00060041, 0x0000000c,
        0x0000000e, 0x0000000d, 0x00000007, 0x00000007, 0x0003003e, 0x0000000e, 0x00000008, 0x000100fd,
        0x00010038,
    };
    return words;
}

std::uint32_t find_host_visible_coherent_memory_type(const VkPhysicalDeviceMemoryProperties& memory_properties,
                                                     std::uint32_t memory_type_bits) {
    constexpr std::uint32_t kInvalid = 0xffffffffu;
    for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
        const bool allowed = (memory_type_bits & (1u << i)) != 0;
        const auto flags = memory_properties.memoryTypes[i].propertyFlags;
        const bool wanted = (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 &&
                            (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        if (allowed && wanted) return i;
    }
    return kInvalid;
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

VulkanSmokeComputeResult run_vulkan_smoke_compute() {
    VulkanSmokeComputeResult result;

    DynamicLibrary loader;
    if (!open_vulkan_loader(loader)) {
        result.error = "Vulkan loader not found";
        return result;
    }

    auto get_proc = load_symbol<PFN_vkGetInstanceProcAddr>(loader, "vkGetInstanceProcAddr");
    if (!get_proc) {
        result.error = "Vulkan loader is missing vkGetInstanceProcAddr";
        return result;
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
        vk_make_api_version(0, 1, 0, 0),
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
    if (create_instance(&create_info, nullptr, &instance) != VK_SUCCESS || !instance) {
        result.error = "vkCreateInstance failed";
        return result;
    }

    auto destroy_instance = load_instance_function<PFN_vkDestroyInstance>(get_proc, instance, "vkDestroyInstance");
    auto enumerate_physical_devices =
        load_instance_function<PFN_vkEnumeratePhysicalDevices>(get_proc, instance, "vkEnumeratePhysicalDevices");
    auto get_physical_device_properties =
        load_instance_function<PFN_vkGetPhysicalDeviceProperties>(get_proc, instance, "vkGetPhysicalDeviceProperties");
    auto get_queue_family_properties =
        load_instance_function<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
            get_proc, instance, "vkGetPhysicalDeviceQueueFamilyProperties");
    auto get_memory_properties =
        load_instance_function<PFN_vkGetPhysicalDeviceMemoryProperties>(
            get_proc, instance, "vkGetPhysicalDeviceMemoryProperties");
    auto create_device = load_instance_function<PFN_vkCreateDevice>(get_proc, instance, "vkCreateDevice");
    auto get_device_proc = load_instance_function<PFN_vkGetDeviceProcAddr>(get_proc, instance, "vkGetDeviceProcAddr");

    auto fail = [&](const std::string& message) {
        result.error = message;
        if (destroy_instance) destroy_instance(instance, nullptr);
        return result;
    };

    if (!enumerate_physical_devices || !get_queue_family_properties || !get_memory_properties ||
        !create_device || !get_device_proc) {
        return fail("Vulkan instance is missing required device creation functions");
    }

    std::uint32_t physical_device_count = 0;
    if (enumerate_physical_devices(instance, &physical_device_count, nullptr) != VK_SUCCESS ||
        physical_device_count == 0) {
        return fail("No Vulkan physical devices found");
    }
    std::vector<VkPhysicalDevice> physical_devices(physical_device_count, nullptr);
    if (enumerate_physical_devices(instance, &physical_device_count, physical_devices.data()) != VK_SUCCESS) {
        return fail("vkEnumeratePhysicalDevices device list failed");
    }

    VkPhysicalDevice selected_physical_device = nullptr;
    std::uint32_t selected_queue_family = 0;
    for (VkPhysicalDevice physical_device : physical_devices) {
        std::uint32_t queue_family_count = 0;
        get_queue_family_properties(physical_device, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        if (queue_family_count > 0) {
            get_queue_family_properties(physical_device, &queue_family_count, queue_families.data());
        }
        for (std::uint32_t i = 0; i < queue_family_count; ++i) {
            if (queue_families[i].queueCount > 0 &&
                (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                selected_physical_device = physical_device;
                selected_queue_family = i;
                break;
            }
        }
        if (selected_physical_device) break;
    }
    if (!selected_physical_device) return fail("No Vulkan compute queue family found");

    if (get_physical_device_properties) {
        struct alignas(8) PropertiesStorage {
            std::array<std::uint8_t, 4096> bytes;
        };
        PropertiesStorage storage{};
        get_physical_device_properties(selected_physical_device, storage.bytes.data());
        const auto* prefix = reinterpret_cast<const VkPhysicalDevicePropertiesPrefix*>(storage.bytes.data());
        result.device_name = bounded_string(prefix->deviceName, sizeof(prefix->deviceName));
    }

    VkPhysicalDeviceMemoryProperties memory_properties{};
    get_memory_properties(selected_physical_device, &memory_properties);

    const float priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_create_info{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        nullptr,
        0,
        selected_queue_family,
        1,
        &priority,
    };
    const VkDeviceCreateInfo device_create_info{
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        nullptr,
        0,
        1,
        &queue_create_info,
        0,
        nullptr,
        0,
        nullptr,
        nullptr,
    };
    VkDevice device = nullptr;
    if (create_device(selected_physical_device, &device_create_info, nullptr, &device) != VK_SUCCESS || !device) {
        return fail("vkCreateDevice failed");
    }

    auto load_device = [&](auto tag, const char* name) {
        using Fn = decltype(tag);
        return load_device_function<Fn>(get_device_proc, device, name);
    };

    auto destroy_device = load_device(PFN_vkDestroyDevice{}, "vkDestroyDevice");
    auto get_device_queue = load_device(PFN_vkGetDeviceQueue{}, "vkGetDeviceQueue");
    auto create_buffer = load_device(PFN_vkCreateBuffer{}, "vkCreateBuffer");
    auto destroy_buffer = load_device(PFN_vkDestroyBuffer{}, "vkDestroyBuffer");
    auto get_buffer_memory_requirements =
        load_device(PFN_vkGetBufferMemoryRequirements{}, "vkGetBufferMemoryRequirements");
    auto allocate_memory = load_device(PFN_vkAllocateMemory{}, "vkAllocateMemory");
    auto free_memory = load_device(PFN_vkFreeMemory{}, "vkFreeMemory");
    auto bind_buffer_memory = load_device(PFN_vkBindBufferMemory{}, "vkBindBufferMemory");
    auto map_memory = load_device(PFN_vkMapMemory{}, "vkMapMemory");
    auto unmap_memory = load_device(PFN_vkUnmapMemory{}, "vkUnmapMemory");
    auto create_descriptor_set_layout =
        load_device(PFN_vkCreateDescriptorSetLayout{}, "vkCreateDescriptorSetLayout");
    auto destroy_descriptor_set_layout =
        load_device(PFN_vkDestroyDescriptorSetLayout{}, "vkDestroyDescriptorSetLayout");
    auto create_descriptor_pool = load_device(PFN_vkCreateDescriptorPool{}, "vkCreateDescriptorPool");
    auto destroy_descriptor_pool = load_device(PFN_vkDestroyDescriptorPool{}, "vkDestroyDescriptorPool");
    auto allocate_descriptor_sets = load_device(PFN_vkAllocateDescriptorSets{}, "vkAllocateDescriptorSets");
    auto update_descriptor_sets = load_device(PFN_vkUpdateDescriptorSets{}, "vkUpdateDescriptorSets");
    auto create_pipeline_layout = load_device(PFN_vkCreatePipelineLayout{}, "vkCreatePipelineLayout");
    auto destroy_pipeline_layout = load_device(PFN_vkDestroyPipelineLayout{}, "vkDestroyPipelineLayout");
    auto create_shader_module = load_device(PFN_vkCreateShaderModule{}, "vkCreateShaderModule");
    auto destroy_shader_module = load_device(PFN_vkDestroyShaderModule{}, "vkDestroyShaderModule");
    auto create_compute_pipelines = load_device(PFN_vkCreateComputePipelines{}, "vkCreateComputePipelines");
    auto destroy_pipeline = load_device(PFN_vkDestroyPipeline{}, "vkDestroyPipeline");
    auto create_command_pool = load_device(PFN_vkCreateCommandPool{}, "vkCreateCommandPool");
    auto destroy_command_pool = load_device(PFN_vkDestroyCommandPool{}, "vkDestroyCommandPool");
    auto allocate_command_buffers = load_device(PFN_vkAllocateCommandBuffers{}, "vkAllocateCommandBuffers");
    auto begin_command_buffer = load_device(PFN_vkBeginCommandBuffer{}, "vkBeginCommandBuffer");
    auto end_command_buffer = load_device(PFN_vkEndCommandBuffer{}, "vkEndCommandBuffer");
    auto cmd_bind_pipeline = load_device(PFN_vkCmdBindPipeline{}, "vkCmdBindPipeline");
    auto cmd_bind_descriptor_sets = load_device(PFN_vkCmdBindDescriptorSets{}, "vkCmdBindDescriptorSets");
    auto cmd_dispatch = load_device(PFN_vkCmdDispatch{}, "vkCmdDispatch");
    auto queue_submit = load_device(PFN_vkQueueSubmit{}, "vkQueueSubmit");
    auto queue_wait_idle = load_device(PFN_vkQueueWaitIdle{}, "vkQueueWaitIdle");

    VkBuffer buffer = nullptr;
    VkDeviceMemory memory = nullptr;
    VkDescriptorSetLayout descriptor_set_layout = nullptr;
    VkDescriptorPool descriptor_pool = nullptr;
    VkPipelineLayout pipeline_layout = nullptr;
    VkShaderModule shader_module = nullptr;
    VkPipeline pipeline = nullptr;
    VkCommandPool command_pool = nullptr;

    auto cleanup = [&]() {
        if (device) {
            if (destroy_command_pool && command_pool) destroy_command_pool(device, command_pool, nullptr);
            if (destroy_pipeline && pipeline) destroy_pipeline(device, pipeline, nullptr);
            if (destroy_shader_module && shader_module) destroy_shader_module(device, shader_module, nullptr);
            if (destroy_pipeline_layout && pipeline_layout) destroy_pipeline_layout(device, pipeline_layout, nullptr);
            if (destroy_descriptor_pool && descriptor_pool) destroy_descriptor_pool(device, descriptor_pool, nullptr);
            if (destroy_descriptor_set_layout && descriptor_set_layout) {
                destroy_descriptor_set_layout(device, descriptor_set_layout, nullptr);
            }
            if (destroy_buffer && buffer) destroy_buffer(device, buffer, nullptr);
            if (free_memory && memory) free_memory(device, memory, nullptr);
            if (destroy_device) destroy_device(device, nullptr);
        }
        if (destroy_instance) destroy_instance(instance, nullptr);
    };

    auto fail_with_cleanup = [&](const std::string& message) {
        result.error = message;
        cleanup();
        return result;
    };

    if (!destroy_device || !get_device_queue || !create_buffer || !destroy_buffer ||
        !get_buffer_memory_requirements || !allocate_memory || !free_memory ||
        !bind_buffer_memory || !map_memory || !unmap_memory ||
        !create_descriptor_set_layout || !destroy_descriptor_set_layout ||
        !create_descriptor_pool || !destroy_descriptor_pool || !allocate_descriptor_sets ||
        !update_descriptor_sets || !create_pipeline_layout || !destroy_pipeline_layout ||
        !create_shader_module || !destroy_shader_module || !create_compute_pipelines ||
        !destroy_pipeline || !create_command_pool || !destroy_command_pool ||
        !allocate_command_buffers || !begin_command_buffer || !end_command_buffer ||
        !cmd_bind_pipeline || !cmd_bind_descriptor_sets || !cmd_dispatch ||
        !queue_submit || !queue_wait_idle) {
        return fail_with_cleanup("Vulkan device is missing required compute functions");
    }

    VkQueue queue = nullptr;
    get_device_queue(device, selected_queue_family, 0, &queue);
    if (!queue) return fail_with_cleanup("vkGetDeviceQueue returned null");

    const VkBufferCreateInfo buffer_create_info{
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        nullptr,
        0,
        sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_SHARING_MODE_EXCLUSIVE,
        0,
        nullptr,
    };
    if (create_buffer(device, &buffer_create_info, nullptr, &buffer) != VK_SUCCESS || !buffer) {
        return fail_with_cleanup("vkCreateBuffer failed");
    }

    VkMemoryRequirements memory_requirements{};
    get_buffer_memory_requirements(device, buffer, &memory_requirements);
    const std::uint32_t memory_type =
        find_host_visible_coherent_memory_type(memory_properties, memory_requirements.memoryTypeBits);
    if (memory_type == 0xffffffffu) {
        return fail_with_cleanup("No host-visible coherent Vulkan memory type for smoke buffer");
    }
    const VkMemoryAllocateInfo allocate_info{
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        nullptr,
        memory_requirements.size,
        memory_type,
    };
    if (allocate_memory(device, &allocate_info, nullptr, &memory) != VK_SUCCESS || !memory) {
        return fail_with_cleanup("vkAllocateMemory failed");
    }
    if (bind_buffer_memory(device, buffer, memory, 0) != VK_SUCCESS) {
        return fail_with_cleanup("vkBindBufferMemory failed");
    }

    const VkDescriptorSetLayoutBinding binding{
        0,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        1,
        VK_SHADER_STAGE_COMPUTE_BIT,
        nullptr,
    };
    const VkDescriptorSetLayoutCreateInfo descriptor_set_layout_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        nullptr,
        0,
        1,
        &binding,
    };
    if (create_descriptor_set_layout(device, &descriptor_set_layout_info, nullptr, &descriptor_set_layout) !=
            VK_SUCCESS ||
        !descriptor_set_layout) {
        return fail_with_cleanup("vkCreateDescriptorSetLayout failed");
    }

    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    const VkDescriptorPoolCreateInfo descriptor_pool_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        nullptr,
        0,
        1,
        1,
        &pool_size,
    };
    if (create_descriptor_pool(device, &descriptor_pool_info, nullptr, &descriptor_pool) != VK_SUCCESS ||
        !descriptor_pool) {
        return fail_with_cleanup("vkCreateDescriptorPool failed");
    }

    const VkDescriptorSetAllocateInfo descriptor_set_allocate_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        nullptr,
        descriptor_pool,
        1,
        &descriptor_set_layout,
    };
    VkDescriptorSet descriptor_set = nullptr;
    if (allocate_descriptor_sets(device, &descriptor_set_allocate_info, &descriptor_set) != VK_SUCCESS ||
        !descriptor_set) {
        return fail_with_cleanup("vkAllocateDescriptorSets failed");
    }

    const VkDescriptorBufferInfo descriptor_buffer_info{buffer, 0, sizeof(float)};
    const VkWriteDescriptorSet write_descriptor{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        nullptr,
        descriptor_set,
        0,
        0,
        1,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        nullptr,
        &descriptor_buffer_info,
        nullptr,
    };
    update_descriptor_sets(device, 1, &write_descriptor, 0, nullptr);

    const VkPipelineLayoutCreateInfo pipeline_layout_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        nullptr,
        0,
        1,
        &descriptor_set_layout,
        0,
        nullptr,
    };
    if (create_pipeline_layout(device, &pipeline_layout_info, nullptr, &pipeline_layout) != VK_SUCCESS ||
        !pipeline_layout) {
        return fail_with_cleanup("vkCreatePipelineLayout failed");
    }

    const auto& shader = smoke_compute_spirv();
    const VkShaderModuleCreateInfo shader_module_info{
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        nullptr,
        0,
        shader.size() * sizeof(std::uint32_t),
        shader.data(),
    };
    if (create_shader_module(device, &shader_module_info, nullptr, &shader_module) != VK_SUCCESS || !shader_module) {
        return fail_with_cleanup("vkCreateShaderModule failed");
    }

    const VkPipelineShaderStageCreateInfo stage_info{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr,
        0,
        VK_SHADER_STAGE_COMPUTE_BIT,
        shader_module,
        "main",
        nullptr,
    };
    const VkComputePipelineCreateInfo compute_pipeline_info{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        nullptr,
        0,
        stage_info,
        pipeline_layout,
        nullptr,
        -1,
    };
    if (create_compute_pipelines(device, nullptr, 1, &compute_pipeline_info, nullptr, &pipeline) != VK_SUCCESS ||
        !pipeline) {
        return fail_with_cleanup("vkCreateComputePipelines failed");
    }

    const VkCommandPoolCreateInfo command_pool_info{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        nullptr,
        0,
        selected_queue_family,
    };
    if (create_command_pool(device, &command_pool_info, nullptr, &command_pool) != VK_SUCCESS || !command_pool) {
        return fail_with_cleanup("vkCreateCommandPool failed");
    }

    const VkCommandBufferAllocateInfo command_buffer_allocate_info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        nullptr,
        command_pool,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        1,
    };
    VkCommandBuffer command_buffer = nullptr;
    if (allocate_command_buffers(device, &command_buffer_allocate_info, &command_buffer) != VK_SUCCESS ||
        !command_buffer) {
        return fail_with_cleanup("vkAllocateCommandBuffers failed");
    }

    const VkCommandBufferBeginInfo begin_info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr,
        0,
        nullptr,
    };
    if (begin_command_buffer(command_buffer, &begin_info) != VK_SUCCESS) {
        return fail_with_cleanup("vkBeginCommandBuffer failed");
    }
    cmd_bind_pipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    cmd_bind_descriptor_sets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                             &descriptor_set, 0, nullptr);
    cmd_dispatch(command_buffer, 1, 1, 1);
    if (end_command_buffer(command_buffer) != VK_SUCCESS) {
        return fail_with_cleanup("vkEndCommandBuffer failed");
    }

    const VkSubmitInfo submit_info{
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
        nullptr,
        0,
        nullptr,
        nullptr,
        1,
        &command_buffer,
        0,
        nullptr,
    };
    if (queue_submit(queue, 1, &submit_info, nullptr) != VK_SUCCESS) {
        return fail_with_cleanup("vkQueueSubmit failed");
    }
    if (queue_wait_idle(queue) != VK_SUCCESS) {
        return fail_with_cleanup("vkQueueWaitIdle failed");
    }

    void* mapped = nullptr;
    if (map_memory(device, memory, 0, sizeof(float), 0, &mapped) != VK_SUCCESS || !mapped) {
        return fail_with_cleanup("vkMapMemory failed");
    }
    std::memcpy(&result.output, mapped, sizeof(float));
    unmap_memory(device, memory);
    if (result.output != 42.0f) {
        return fail_with_cleanup("Vulkan smoke compute returned unexpected output");
    }

    result.success = true;
    cleanup();
    return result;
}

} // namespace motifcl
