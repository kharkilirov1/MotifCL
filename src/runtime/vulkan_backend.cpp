#include <motifcl/runtime/vulkan_backend.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
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

// Embedded SPIR-V for the cached fast-dispatch kernels (see
// kernels/vulkan/*.comp and tools/gen_vulkan_spirv.py).
#include "vulkan_spirv_kernels.inc"

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
using VkPipelineCache = struct VkPipelineCache_T*;
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
constexpr VkStructureType VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO = 1000117000;
constexpr VkStructureType VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 = 1000059000;
constexpr VkStructureType VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES = 1000094000;
constexpr VkStructureType VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES = 1000177000;

constexpr VkQueueFlags VK_QUEUE_COMPUTE_BIT = 0x00000002u;
constexpr VkBufferUsageFlags VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020u;
constexpr VkMemoryPropertyFlags VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000001u;
constexpr VkMemoryPropertyFlags VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000002u;
constexpr std::int32_t VK_SHARING_MODE_EXCLUSIVE = 0;
constexpr std::int32_t VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7;
constexpr VkShaderStageFlags VK_SHADER_STAGE_COMPUTE_BIT = 0x00000020u;
constexpr std::int32_t VK_PIPELINE_BIND_POINT_COMPUTE = 1;
constexpr std::int32_t VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0;

// --- additions for the cached fast-dispatch path ---------------------------
// NOTE: the two VK_MEMORY_PROPERTY_* values above are historically off by one
// bit position (0x1 is really DEVICE_LOCAL, 0x2 is really HOST_VISIBLE); the
// legacy chooser therefore selects DEVICE_LOCAL|HOST_VISIBLE (BAR) types and
// works on the drivers we target.  New code uses the spec-correct kMem* bits.
constexpr VkMemoryPropertyFlags kMemDeviceLocalBit = 0x00000001u;
constexpr VkMemoryPropertyFlags kMemHostVisibleBit = 0x00000002u;
constexpr VkMemoryPropertyFlags kMemHostCoherentBit = 0x00000004u;

constexpr VkStructureType VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8;
constexpr VkStructureType VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO = 11;
constexpr VkStructureType VK_STRUCTURE_TYPE_MEMORY_BARRIER = 46;

constexpr VkBufferUsageFlags VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x00000001u;
constexpr VkBufferUsageFlags VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x00000002u;

constexpr std::int32_t VK_QUERY_TYPE_TIMESTAMP = 2;
constexpr VkFlags VK_QUERY_RESULT_64_BIT = 0x00000001u;
constexpr VkFlags VK_QUERY_RESULT_WAIT_BIT = 0x00000002u;

constexpr VkPipelineStageFlags VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT = 0x00000001u;
constexpr VkPipelineStageFlags VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 0x00000800u;
constexpr VkPipelineStageFlags VK_PIPELINE_STAGE_TRANSFER_BIT = 0x00001000u;
constexpr VkPipelineStageFlags VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT = 0x00002000u;

constexpr VkFlags VK_ACCESS_SHADER_READ_BIT = 0x00000020u;
constexpr VkFlags VK_ACCESS_SHADER_WRITE_BIT = 0x00000040u;
constexpr VkFlags VK_ACCESS_TRANSFER_WRITE_BIT = 0x00001000u;

constexpr VkFlags VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x00000001u;

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

struct VkExtensionProperties {
    char extensionName[256];
    std::uint32_t specVersion;
};

struct VkPhysicalDeviceFeatures {
    VkBool32 robustBufferAccess;
    VkBool32 fullDrawIndexUint32;
    VkBool32 imageCubeArray;
    VkBool32 independentBlend;
    VkBool32 geometryShader;
    VkBool32 tessellationShader;
    VkBool32 sampleRateShading;
    VkBool32 dualSrcBlend;
    VkBool32 logicOp;
    VkBool32 multiDrawIndirect;
    VkBool32 drawIndirectFirstInstance;
    VkBool32 depthClamp;
    VkBool32 depthBiasClamp;
    VkBool32 fillModeNonSolid;
    VkBool32 depthBounds;
    VkBool32 wideLines;
    VkBool32 largePoints;
    VkBool32 alphaToOne;
    VkBool32 multiViewport;
    VkBool32 samplerAnisotropy;
    VkBool32 textureCompressionETC2;
    VkBool32 textureCompressionASTC_LDR;
    VkBool32 textureCompressionBC;
    VkBool32 occlusionQueryPrecise;
    VkBool32 pipelineStatisticsQuery;
    VkBool32 vertexPipelineStoresAndAtomics;
    VkBool32 fragmentStoresAndAtomics;
    VkBool32 shaderTessellationAndGeometryPointSize;
    VkBool32 shaderImageGatherExtended;
    VkBool32 shaderStorageImageExtendedFormats;
    VkBool32 shaderStorageImageMultisample;
    VkBool32 shaderStorageImageReadWithoutFormat;
    VkBool32 shaderStorageImageWriteWithoutFormat;
    VkBool32 shaderUniformBufferArrayDynamicIndexing;
    VkBool32 shaderSampledImageArrayDynamicIndexing;
    VkBool32 shaderStorageBufferArrayDynamicIndexing;
    VkBool32 shaderStorageImageArrayDynamicIndexing;
    VkBool32 shaderClipDistance;
    VkBool32 shaderCullDistance;
    VkBool32 shaderFloat64;
    VkBool32 shaderInt64;
    VkBool32 shaderInt16;
    VkBool32 shaderResourceResidency;
    VkBool32 shaderResourceMinLod;
    VkBool32 sparseBinding;
    VkBool32 sparseResidencyBuffer;
    VkBool32 sparseResidencyImage2D;
    VkBool32 sparseResidencyImage3D;
    VkBool32 sparseResidency2Samples;
    VkBool32 sparseResidency4Samples;
    VkBool32 sparseResidency8Samples;
    VkBool32 sparseResidency16Samples;
    VkBool32 sparseResidencyAliased;
    VkBool32 variableMultisampleRate;
    VkBool32 inheritedQueries;
};

struct VkPhysicalDeviceFeatures2 {
    VkStructureType sType;
    void* pNext;
    VkPhysicalDeviceFeatures features;
};

struct VkPhysicalDevice8BitStorageFeatures {
    VkStructureType sType;
    void* pNext;
    VkBool32 storageBuffer8BitAccess;
    VkBool32 uniformAndStorageBuffer8BitAccess;
    VkBool32 storagePushConstant8;
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

using VkFence = struct VkFence_T*;
using VkQueryPool = struct VkQueryPool_T*;

struct VkFenceCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
};

struct VkMemoryBarrier {
    VkStructureType sType;
    const void* pNext;
    VkFlags srcAccessMask;
    VkFlags dstAccessMask;
};

struct VkBufferCopy {
    VkDeviceSize srcOffset;
    VkDeviceSize dstOffset;
    VkDeviceSize size;
};

struct VkQueryPoolCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    std::int32_t queryType;
    std::uint32_t queryCount;
    VkFlags pipelineStatistics;
};

// Full VkPhysicalDeviceLimits/Properties layout (Vulkan 1.0 core ABI); the
// short prefix struct above stays for name-only probes.  A size static_assert
// below guards the transcription on 64-bit builds.
struct VkPhysicalDeviceLimitsFull {
    std::uint32_t maxImageDimension1D;
    std::uint32_t maxImageDimension2D;
    std::uint32_t maxImageDimension3D;
    std::uint32_t maxImageDimensionCube;
    std::uint32_t maxImageArrayLayers;
    std::uint32_t maxTexelBufferElements;
    std::uint32_t maxUniformBufferRange;
    std::uint32_t maxStorageBufferRange;
    std::uint32_t maxPushConstantsSize;
    std::uint32_t maxMemoryAllocationCount;
    std::uint32_t maxSamplerAllocationCount;
    VkDeviceSize bufferImageGranularity;
    VkDeviceSize sparseAddressSpaceSize;
    std::uint32_t maxBoundDescriptorSets;
    std::uint32_t maxPerStageDescriptorSamplers;
    std::uint32_t maxPerStageDescriptorUniformBuffers;
    std::uint32_t maxPerStageDescriptorStorageBuffers;
    std::uint32_t maxPerStageDescriptorSampledImages;
    std::uint32_t maxPerStageDescriptorStorageImages;
    std::uint32_t maxPerStageDescriptorInputAttachments;
    std::uint32_t maxPerStageResources;
    std::uint32_t maxDescriptorSetSamplers;
    std::uint32_t maxDescriptorSetUniformBuffers;
    std::uint32_t maxDescriptorSetUniformBuffersDynamic;
    std::uint32_t maxDescriptorSetStorageBuffers;
    std::uint32_t maxDescriptorSetStorageBuffersDynamic;
    std::uint32_t maxDescriptorSetSampledImages;
    std::uint32_t maxDescriptorSetStorageImages;
    std::uint32_t maxDescriptorSetInputAttachments;
    std::uint32_t maxVertexInputAttributes;
    std::uint32_t maxVertexInputBindings;
    std::uint32_t maxVertexInputAttributeOffset;
    std::uint32_t maxVertexInputBindingStride;
    std::uint32_t maxVertexOutputComponents;
    std::uint32_t maxTessellationGenerationLevel;
    std::uint32_t maxTessellationPatchSize;
    std::uint32_t maxTessellationControlPerVertexInputComponents;
    std::uint32_t maxTessellationControlPerVertexOutputComponents;
    std::uint32_t maxTessellationControlPerPatchOutputComponents;
    std::uint32_t maxTessellationControlTotalOutputComponents;
    std::uint32_t maxTessellationEvaluationInputComponents;
    std::uint32_t maxTessellationEvaluationOutputComponents;
    std::uint32_t maxGeometryShaderInvocations;
    std::uint32_t maxGeometryInputComponents;
    std::uint32_t maxGeometryOutputComponents;
    std::uint32_t maxGeometryOutputVertices;
    std::uint32_t maxGeometryTotalOutputComponents;
    std::uint32_t maxFragmentInputComponents;
    std::uint32_t maxFragmentOutputAttachments;
    std::uint32_t maxFragmentDualSrcAttachments;
    std::uint32_t maxFragmentCombinedOutputResources;
    std::uint32_t maxComputeSharedMemorySize;
    std::uint32_t maxComputeWorkGroupCount[3];
    std::uint32_t maxComputeWorkGroupInvocations;
    std::uint32_t maxComputeWorkGroupSize[3];
    std::uint32_t subPixelPrecisionBits;
    std::uint32_t subTexelPrecisionBits;
    std::uint32_t mipmapPrecisionBits;
    std::uint32_t maxDrawIndexedIndexValue;
    std::uint32_t maxDrawIndirectCount;
    float maxSamplerLodBias;
    float maxSamplerAnisotropy;
    std::uint32_t maxViewports;
    std::uint32_t maxViewportDimensions[2];
    float viewportBoundsRange[2];
    std::uint32_t viewportSubPixelBits;
    std::size_t minMemoryMapAlignment;
    VkDeviceSize minTexelBufferOffsetAlignment;
    VkDeviceSize minUniformBufferOffsetAlignment;
    VkDeviceSize minStorageBufferOffsetAlignment;
    std::int32_t minTexelOffset;
    std::uint32_t maxTexelOffset;
    std::int32_t minTexelGatherOffset;
    std::uint32_t maxTexelGatherOffset;
    float minInterpolationOffset;
    float maxInterpolationOffset;
    std::uint32_t subPixelInterpolationOffsetBits;
    std::uint32_t maxFramebufferWidth;
    std::uint32_t maxFramebufferHeight;
    std::uint32_t maxFramebufferLayers;
    VkFlags framebufferColorSampleCounts;
    VkFlags framebufferDepthSampleCounts;
    VkFlags framebufferStencilSampleCounts;
    VkFlags framebufferNoAttachmentsSampleCounts;
    std::uint32_t maxColorAttachments;
    VkFlags sampledImageColorSampleCounts;
    VkFlags sampledImageIntegerSampleCounts;
    VkFlags sampledImageDepthSampleCounts;
    VkFlags sampledImageStencilSampleCounts;
    VkFlags storageImageSampleCounts;
    std::uint32_t maxSampleMaskWords;
    VkBool32 timestampComputeAndGraphics;
    float timestampPeriod;
    std::uint32_t maxClipDistances;
    std::uint32_t maxCullDistances;
    std::uint32_t maxCombinedClipAndCullDistances;
    std::uint32_t discreteQueuePriorities;
    float pointSizeRange[2];
    float lineWidthRange[2];
    float pointSizeGranularity;
    float lineWidthGranularity;
    VkBool32 strictLines;
    VkBool32 standardSampleLocations;
    VkDeviceSize optimalBufferCopyOffsetAlignment;
    VkDeviceSize optimalBufferCopyRowPitchAlignment;
    VkDeviceSize nonCoherentAtomSize;
};

struct VkPhysicalDeviceSparsePropertiesFull {
    VkBool32 residencyStandard2DBlockShape;
    VkBool32 residencyStandard2DMultisampleBlockShape;
    VkBool32 residencyStandard3DBlockShape;
    VkBool32 residencyAlignedMipSize;
    VkBool32 residencyNonResidentStrict;
};

struct VkPhysicalDevicePropertiesFull {
    std::uint32_t apiVersion;
    std::uint32_t driverVersion;
    std::uint32_t vendorID;
    std::uint32_t deviceID;
    std::int32_t deviceType;
    char deviceName[256];
    std::uint8_t pipelineCacheUUID[16];
    VkPhysicalDeviceLimitsFull limits;
    VkPhysicalDeviceSparsePropertiesFull sparseProperties;
};

static_assert(sizeof(void*) != 8 || sizeof(VkPhysicalDeviceLimitsFull) == 504,
              "VkPhysicalDeviceLimits transcription drifted from the Vulkan 1.0 ABI");
static_assert(sizeof(void*) != 8 || sizeof(VkPhysicalDevicePropertiesFull) == 824,
              "VkPhysicalDeviceProperties transcription drifted from the Vulkan 1.0 ABI");

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
using PFN_vkEnumerateDeviceExtensionProperties = VkResult (*)(VkPhysicalDevice physicalDevice,
                                                              const char* pLayerName,
                                                              std::uint32_t* pPropertyCount,
                                                              VkExtensionProperties* pProperties);
using PFN_vkGetPhysicalDeviceFeatures2 = void (*)(VkPhysicalDevice physicalDevice,
                                                  VkPhysicalDeviceFeatures2* pFeatures);
// vulkan1.1: physical-device properties2 chain. Used to read subgroup
// properties (subgroupSize, supportedStages, supportedOperations) so kernels
// can opt into subgroupAdd / subgroupBroadcast only when the device actually
// supports arithmetic subgroup operations in compute.
struct VkPhysicalDeviceSubgroupProperties {
    std::int32_t sType;
    void* pNext;
    std::uint32_t subgroupSize;
    std::uint32_t supportedStages;       // bitmask, VK_SHADER_STAGE_COMPUTE_BIT = 0x00000010
    std::uint32_t supportedOperations;   // bitmask, VK_SUBGROUP_FEATURE_ARITHMETIC_BIT = 0x00000004
    std::uint32_t quadOperationsInAllStages;
};
static_assert(sizeof(VkPhysicalDeviceSubgroupProperties) == 32,
              "VkPhysicalDeviceSubgroupProperties layout must match the driver's struct");
struct VkPhysicalDeviceProperties2 {
    std::int32_t sType;
    void* pNext;
    // VkPhysicalDeviceProperties is a ~1840-byte inline struct (not a pointer).
    // We only need sType + pNext + enough storage; the driver writes the inline
    // properties starting at offset 16. Keep a raw byte buffer here so the
    // driver's struct layout matches ours (pNext at offset 8, inline struct at
    // offset 16) without us having to mirror the full VkPhysicalDeviceProperties.
    std::array<std::uint8_t, 2048> properties_storage;
};
using PFN_vkGetPhysicalDeviceProperties2 = void (*)(VkPhysicalDevice physicalDevice,
                                                     VkPhysicalDeviceProperties2* pProperties);
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
using PFN_vkCreateComputePipelines = VkResult (*)(VkDevice device, VkPipelineCache pipelineCache,
                                                  std::uint32_t createInfoCount,
                                                  const VkComputePipelineCreateInfo* pCreateInfos,
                                                  const void* pAllocator, VkPipeline* pPipelines);
using PFN_vkDestroyPipeline = void (*)(VkDevice device, VkPipeline pipeline, const void* pAllocator);
// Pipeline cache: driver-side SPIR-V -> ISA compilation cache. Eliminates the
// 1-5ms cold-compile cost on first use of each kernel and persists it across
// runs when serialized to disk.
struct VkPipelineCacheCreateInfo {
    std::int32_t sType;
    const void* pNext;
    std::uint32_t flags;
    std::size_t initialDataSize;
    const void* pInitialData;
};
using PFN_vkCreatePipelineCache = VkResult (*)(VkDevice device, const VkPipelineCacheCreateInfo* pCreateInfo,
                                               const void* pAllocator, VkPipelineCache* pPipelineCache);
using PFN_vkDestroyPipelineCache = void (*)(VkDevice device, VkPipelineCache pipelineCache,
                                            const void* pAllocator);
using PFN_vkGetPipelineCacheData = VkResult (*)(VkDevice device, VkPipelineCache pipelineCache,
                                                std::size_t* pDataSize, void* pData);
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

// --- fast-dispatch path function pointers ---
using PFN_vkCreateFence = VkResult (*)(VkDevice device, const VkFenceCreateInfo* pCreateInfo,
                                       const void* pAllocator, VkFence* pFence);
using PFN_vkDestroyFence = void (*)(VkDevice device, VkFence fence, const void* pAllocator);
using PFN_vkResetFences = VkResult (*)(VkDevice device, std::uint32_t fenceCount, const VkFence* pFences);
using PFN_vkWaitForFences = VkResult (*)(VkDevice device, std::uint32_t fenceCount, const VkFence* pFences,
                                         VkBool32 waitAll, std::uint64_t timeout);
using PFN_vkResetCommandPool = VkResult (*)(VkDevice device, VkCommandPool commandPool, VkFlags flags);
using PFN_vkCmdPushConstants = void (*)(VkCommandBuffer commandBuffer, VkPipelineLayout layout,
                                        VkShaderStageFlags stageFlags, std::uint32_t offset,
                                        std::uint32_t size, const void* pValues);
using PFN_vkCmdPipelineBarrier = void (*)(VkCommandBuffer commandBuffer, VkPipelineStageFlags srcStageMask,
                                          VkPipelineStageFlags dstStageMask, VkFlags dependencyFlags,
                                          std::uint32_t memoryBarrierCount, const VkMemoryBarrier* pMemoryBarriers,
                                          std::uint32_t bufferMemoryBarrierCount, const void* pBufferMemoryBarriers,
                                          std::uint32_t imageMemoryBarrierCount, const void* pImageMemoryBarriers);
using PFN_vkCmdCopyBuffer = void (*)(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkBuffer dstBuffer,
                                     std::uint32_t regionCount, const VkBufferCopy* pRegions);
using PFN_vkCreateQueryPool = VkResult (*)(VkDevice device, const VkQueryPoolCreateInfo* pCreateInfo,
                                           const void* pAllocator, VkQueryPool* pQueryPool);
using PFN_vkDestroyQueryPool = void (*)(VkDevice device, VkQueryPool queryPool, const void* pAllocator);
using PFN_vkCmdResetQueryPool = void (*)(VkCommandBuffer commandBuffer, VkQueryPool queryPool,
                                         std::uint32_t firstQuery, std::uint32_t queryCount);
using PFN_vkCmdWriteTimestamp = void (*)(VkCommandBuffer commandBuffer, VkPipelineStageFlags pipelineStage,
                                         VkQueryPool queryPool, std::uint32_t query);
using PFN_vkGetQueryPoolResults = VkResult (*)(VkDevice device, VkQueryPool queryPool, std::uint32_t firstQuery,
                                               std::uint32_t queryCount, std::size_t dataSize, void* pData,
                                               VkDeviceSize stride, VkFlags flags);

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

const std::array<std::uint32_t, 664>& f32_matmul_1x4x4_spirv() {
    static const std::array<std::uint32_t, 664> words = {
        0x07230203, 0x00010000, 0x00000000, 0x0000007e, 0x00000000, 0x00020011, 0x00000001, 0x0003000e,
        0x00000000, 0x00000001, 0x0005000f, 0x00000005, 0x00000003, 0x6e69616d, 0x00000000, 0x00060010,
        0x00000003, 0x00000011, 0x00000001, 0x00000001, 0x00000001, 0x00040047, 0x00000017, 0x00000006,
        0x00000004, 0x00050048, 0x00000018, 0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x00000018,
        0x00000003, 0x00040047, 0x0000001b, 0x00000022, 0x00000000, 0x00040047, 0x0000001b, 0x00000021,
        0x00000000, 0x00040047, 0x0000001c, 0x00000022, 0x00000000, 0x00040047, 0x0000001c, 0x00000021,
        0x00000001, 0x00040047, 0x0000001d, 0x00000022, 0x00000000, 0x00040047, 0x0000001d, 0x00000021,
        0x00000002, 0x00020013, 0x00000001, 0x00030021, 0x00000002, 0x00000001, 0x00030016, 0x00000005,
        0x00000020, 0x00040015, 0x00000006, 0x00000020, 0x00000000, 0x0004002b, 0x00000006, 0x00000007,
        0x00000000, 0x0004002b, 0x00000006, 0x00000008, 0x00000001, 0x0004002b, 0x00000006, 0x00000009,
        0x00000002, 0x0004002b, 0x00000006, 0x0000000a, 0x00000003, 0x0004002b, 0x00000006, 0x0000000b,
        0x00000004, 0x0004002b, 0x00000006, 0x0000000c, 0x00000005, 0x0004002b, 0x00000006, 0x0000000d,
        0x00000006, 0x0004002b, 0x00000006, 0x0000000e, 0x00000007, 0x0004002b, 0x00000006, 0x0000000f,
        0x00000008, 0x0004002b, 0x00000006, 0x00000010, 0x00000009, 0x0004002b, 0x00000006, 0x00000011,
        0x0000000a, 0x0004002b, 0x00000006, 0x00000012, 0x0000000b, 0x0004002b, 0x00000006, 0x00000013,
        0x0000000c, 0x0004002b, 0x00000006, 0x00000014, 0x0000000d, 0x0004002b, 0x00000006, 0x00000015,
        0x0000000e, 0x0004002b, 0x00000006, 0x00000016, 0x0000000f, 0x0003001d, 0x00000017, 0x00000005,
        0x0003001e, 0x00000018, 0x00000017, 0x00040020, 0x00000019, 0x00000002, 0x00000018, 0x00040020,
        0x0000001a, 0x00000002, 0x00000005, 0x0004003b, 0x00000019, 0x0000001b, 0x00000002, 0x0004003b,
        0x00000019, 0x0000001c, 0x00000002, 0x0004003b, 0x00000019, 0x0000001d, 0x00000002, 0x00050036,
        0x00000001, 0x00000003, 0x00000000, 0x00000002, 0x000200f8, 0x00000004, 0x00060041, 0x0000001a,
        0x0000001e, 0x0000001b, 0x00000007, 0x00000007, 0x0004003d, 0x00000005, 0x0000001f, 0x0000001e,
        0x00060041, 0x0000001a, 0x00000020, 0x0000001c, 0x00000007, 0x00000007, 0x0004003d, 0x00000005,
        0x00000021, 0x00000020, 0x00050085, 0x00000005, 0x00000022, 0x0000001f, 0x00000021, 0x00060041,
        0x0000001a, 0x00000023, 0x0000001b, 0x00000007, 0x00000008, 0x0004003d, 0x00000005, 0x00000024,
        0x00000023, 0x00060041, 0x0000001a, 0x00000025, 0x0000001c, 0x00000007, 0x0000000b, 0x0004003d,
        0x00000005, 0x00000026, 0x00000025, 0x00050085, 0x00000005, 0x00000027, 0x00000024, 0x00000026,
        0x00050081, 0x00000005, 0x00000028, 0x00000022, 0x00000027, 0x00060041, 0x0000001a, 0x00000029,
        0x0000001b, 0x00000007, 0x00000009, 0x0004003d, 0x00000005, 0x0000002a, 0x00000029, 0x00060041,
        0x0000001a, 0x0000002b, 0x0000001c, 0x00000007, 0x0000000f, 0x0004003d, 0x00000005, 0x0000002c,
        0x0000002b, 0x00050085, 0x00000005, 0x0000002d, 0x0000002a, 0x0000002c, 0x00050081, 0x00000005,
        0x0000002e, 0x00000028, 0x0000002d, 0x00060041, 0x0000001a, 0x0000002f, 0x0000001b, 0x00000007,
        0x0000000a, 0x0004003d, 0x00000005, 0x00000030, 0x0000002f, 0x00060041, 0x0000001a, 0x00000031,
        0x0000001c, 0x00000007, 0x00000013, 0x0004003d, 0x00000005, 0x00000032, 0x00000031, 0x00050085,
        0x00000005, 0x00000033, 0x00000030, 0x00000032, 0x00050081, 0x00000005, 0x00000034, 0x0000002e,
        0x00000033, 0x00060041, 0x0000001a, 0x00000035, 0x0000001d, 0x00000007, 0x00000007, 0x0003003e,
        0x00000035, 0x00000034, 0x00060041, 0x0000001a, 0x00000036, 0x0000001b, 0x00000007, 0x00000007,
        0x0004003d, 0x00000005, 0x00000037, 0x00000036, 0x00060041, 0x0000001a, 0x00000038, 0x0000001c,
        0x00000007, 0x00000008, 0x0004003d, 0x00000005, 0x00000039, 0x00000038, 0x00050085, 0x00000005,
        0x0000003a, 0x00000037, 0x00000039, 0x00060041, 0x0000001a, 0x0000003b, 0x0000001b, 0x00000007,
        0x00000008, 0x0004003d, 0x00000005, 0x0000003c, 0x0000003b, 0x00060041, 0x0000001a, 0x0000003d,
        0x0000001c, 0x00000007, 0x0000000c, 0x0004003d, 0x00000005, 0x0000003e, 0x0000003d, 0x00050085,
        0x00000005, 0x0000003f, 0x0000003c, 0x0000003e, 0x00050081, 0x00000005, 0x00000040, 0x0000003a,
        0x0000003f, 0x00060041, 0x0000001a, 0x00000041, 0x0000001b, 0x00000007, 0x00000009, 0x0004003d,
        0x00000005, 0x00000042, 0x00000041, 0x00060041, 0x0000001a, 0x00000043, 0x0000001c, 0x00000007,
        0x00000010, 0x0004003d, 0x00000005, 0x00000044, 0x00000043, 0x00050085, 0x00000005, 0x00000045,
        0x00000042, 0x00000044, 0x00050081, 0x00000005, 0x00000046, 0x00000040, 0x00000045, 0x00060041,
        0x0000001a, 0x00000047, 0x0000001b, 0x00000007, 0x0000000a, 0x0004003d, 0x00000005, 0x00000048,
        0x00000047, 0x00060041, 0x0000001a, 0x00000049, 0x0000001c, 0x00000007, 0x00000014, 0x0004003d,
        0x00000005, 0x0000004a, 0x00000049, 0x00050085, 0x00000005, 0x0000004b, 0x00000048, 0x0000004a,
        0x00050081, 0x00000005, 0x0000004c, 0x00000046, 0x0000004b, 0x00060041, 0x0000001a, 0x0000004d,
        0x0000001d, 0x00000007, 0x00000008, 0x0003003e, 0x0000004d, 0x0000004c, 0x00060041, 0x0000001a,
        0x0000004e, 0x0000001b, 0x00000007, 0x00000007, 0x0004003d, 0x00000005, 0x0000004f, 0x0000004e,
        0x00060041, 0x0000001a, 0x00000050, 0x0000001c, 0x00000007, 0x00000009, 0x0004003d, 0x00000005,
        0x00000051, 0x00000050, 0x00050085, 0x00000005, 0x00000052, 0x0000004f, 0x00000051, 0x00060041,
        0x0000001a, 0x00000053, 0x0000001b, 0x00000007, 0x00000008, 0x0004003d, 0x00000005, 0x00000054,
        0x00000053, 0x00060041, 0x0000001a, 0x00000055, 0x0000001c, 0x00000007, 0x0000000d, 0x0004003d,
        0x00000005, 0x00000056, 0x00000055, 0x00050085, 0x00000005, 0x00000057, 0x00000054, 0x00000056,
        0x00050081, 0x00000005, 0x00000058, 0x00000052, 0x00000057, 0x00060041, 0x0000001a, 0x00000059,
        0x0000001b, 0x00000007, 0x00000009, 0x0004003d, 0x00000005, 0x0000005a, 0x00000059, 0x00060041,
        0x0000001a, 0x0000005b, 0x0000001c, 0x00000007, 0x00000011, 0x0004003d, 0x00000005, 0x0000005c,
        0x0000005b, 0x00050085, 0x00000005, 0x0000005d, 0x0000005a, 0x0000005c, 0x00050081, 0x00000005,
        0x0000005e, 0x00000058, 0x0000005d, 0x00060041, 0x0000001a, 0x0000005f, 0x0000001b, 0x00000007,
        0x0000000a, 0x0004003d, 0x00000005, 0x00000060, 0x0000005f, 0x00060041, 0x0000001a, 0x00000061,
        0x0000001c, 0x00000007, 0x00000015, 0x0004003d, 0x00000005, 0x00000062, 0x00000061, 0x00050085,
        0x00000005, 0x00000063, 0x00000060, 0x00000062, 0x00050081, 0x00000005, 0x00000064, 0x0000005e,
        0x00000063, 0x00060041, 0x0000001a, 0x00000065, 0x0000001d, 0x00000007, 0x00000009, 0x0003003e,
        0x00000065, 0x00000064, 0x00060041, 0x0000001a, 0x00000066, 0x0000001b, 0x00000007, 0x00000007,
        0x0004003d, 0x00000005, 0x00000067, 0x00000066, 0x00060041, 0x0000001a, 0x00000068, 0x0000001c,
        0x00000007, 0x0000000a, 0x0004003d, 0x00000005, 0x00000069, 0x00000068, 0x00050085, 0x00000005,
        0x0000006a, 0x00000067, 0x00000069, 0x00060041, 0x0000001a, 0x0000006b, 0x0000001b, 0x00000007,
        0x00000008, 0x0004003d, 0x00000005, 0x0000006c, 0x0000006b, 0x00060041, 0x0000001a, 0x0000006d,
        0x0000001c, 0x00000007, 0x0000000e, 0x0004003d, 0x00000005, 0x0000006e, 0x0000006d, 0x00050085,
        0x00000005, 0x0000006f, 0x0000006c, 0x0000006e, 0x00050081, 0x00000005, 0x00000070, 0x0000006a,
        0x0000006f, 0x00060041, 0x0000001a, 0x00000071, 0x0000001b, 0x00000007, 0x00000009, 0x0004003d,
        0x00000005, 0x00000072, 0x00000071, 0x00060041, 0x0000001a, 0x00000073, 0x0000001c, 0x00000007,
        0x00000012, 0x0004003d, 0x00000005, 0x00000074, 0x00000073, 0x00050085, 0x00000005, 0x00000075,
        0x00000072, 0x00000074, 0x00050081, 0x00000005, 0x00000076, 0x00000070, 0x00000075, 0x00060041,
        0x0000001a, 0x00000077, 0x0000001b, 0x00000007, 0x0000000a, 0x0004003d, 0x00000005, 0x00000078,
        0x00000077, 0x00060041, 0x0000001a, 0x00000079, 0x0000001c, 0x00000007, 0x00000016, 0x0004003d,
        0x00000005, 0x0000007a, 0x00000079, 0x00050085, 0x00000005, 0x0000007b, 0x00000078, 0x0000007a,
        0x00050081, 0x00000005, 0x0000007c, 0x00000076, 0x0000007b, 0x00060041, 0x0000001a, 0x0000007d,
        0x0000001d, 0x00000007, 0x0000000a, 0x0003003e, 0x0000007d, 0x0000007c, 0x000100fd, 0x00010038
    };
    return words;
}

void append_spirv_string(std::vector<std::uint32_t>& words, const char* text) {
    std::uint32_t packed = 0;
    std::uint32_t shift = 0;
    for (;;) {
        const auto ch = static_cast<unsigned char>(*text);
        packed |= static_cast<std::uint32_t>(ch) << shift;
        if (ch == 0) {
            words.push_back(packed);
            return;
        }
        ++text;
        shift += 8;
        if (shift == 32) {
            words.push_back(packed);
            packed = 0;
            shift = 0;
        }
    }
}

void append_spirv_string_inst(std::vector<std::uint32_t>& words,
                              std::uint16_t opcode,
                              const char* text) {
    const auto start = words.size();
    words.push_back(0);
    append_spirv_string(words, text);
    words[start] = (static_cast<std::uint32_t>(words.size() - start) << 16u) | opcode;
}

void append_spirv_inst(std::vector<std::uint32_t>& words,
                       std::uint16_t opcode,
                       const std::vector<std::uint32_t>& operands) {
    words.push_back((static_cast<std::uint32_t>(operands.size() + 1) << 16u) | opcode);
    words.insert(words.end(), operands.begin(), operands.end());
}

std::uint32_t f32_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::vector<std::uint32_t> f32_m1_matmul_spirv(std::size_t k, std::size_t n) {
    constexpr std::uint16_t op_capability = 17;
    constexpr std::uint16_t op_memory_model = 14;
    constexpr std::uint16_t op_entry_point = 15;
    constexpr std::uint16_t op_execution_mode = 16;
    constexpr std::uint16_t op_decorate = 71;
    constexpr std::uint16_t op_member_decorate = 72;
    constexpr std::uint16_t op_type_void = 19;
    constexpr std::uint16_t op_type_function = 33;
    constexpr std::uint16_t op_type_float = 22;
    constexpr std::uint16_t op_type_int = 21;
    constexpr std::uint16_t op_constant = 43;
    constexpr std::uint16_t op_type_runtime_array = 29;
    constexpr std::uint16_t op_type_struct = 30;
    constexpr std::uint16_t op_type_pointer = 32;
    constexpr std::uint16_t op_variable = 59;
    constexpr std::uint16_t op_function = 54;
    constexpr std::uint16_t op_label = 248;
    constexpr std::uint16_t op_access_chain = 65;
    constexpr std::uint16_t op_load = 61;
    constexpr std::uint16_t op_fmul = 133;
    constexpr std::uint16_t op_fadd = 129;
    constexpr std::uint16_t op_store = 62;
    constexpr std::uint16_t op_return = 253;
    constexpr std::uint16_t op_function_end = 56;

    constexpr std::uint32_t capability_shader = 1;
    constexpr std::uint32_t addressing_model_logical = 0;
    constexpr std::uint32_t memory_model_glsl450 = 1;
    constexpr std::uint32_t execution_model_gl_compute = 5;
    constexpr std::uint32_t execution_mode_local_size = 17;
    constexpr std::uint32_t decoration_array_stride = 6;
    constexpr std::uint32_t decoration_offset = 35;
    constexpr std::uint32_t decoration_buffer_block = 3;
    constexpr std::uint32_t decoration_descriptor_set = 34;
    constexpr std::uint32_t decoration_binding = 33;
    constexpr std::uint32_t storage_class_uniform = 2;

    const auto max_b_index = k * n - 1;
    const auto max_constant = std::max(max_b_index, std::max(k - 1, n - 1));

    const std::uint32_t id_void = 1;
    const std::uint32_t id_fn = 2;
    const std::uint32_t id_main = 3;
    const std::uint32_t id_label = 4;
    const std::uint32_t id_f32 = 5;
    const std::uint32_t id_u32 = 6;
    std::uint32_t next_id = 7;
    std::vector<std::uint32_t> constants(max_constant + 1);
    for (std::uint32_t value = 0; value <= max_constant; ++value) constants[value] = next_id++;
    const std::uint32_t id_runtime_array = next_id++;
    const std::uint32_t id_buffer_struct = next_id++;
    const std::uint32_t id_ptr_buffer = next_id++;
    const std::uint32_t id_ptr_f32 = next_id++;
    const std::uint32_t id_a = next_id++;
    const std::uint32_t id_b = next_id++;
    const std::uint32_t id_c = next_id++;
    auto new_id = [&]() {
        return next_id++;
    };

    std::vector<std::uint32_t> body;
    auto emit_body = [&](std::uint16_t opcode, const std::vector<std::uint32_t>& operands) {
        append_spirv_inst(body, opcode, operands);
    };
    for (std::uint32_t col = 0; col < static_cast<std::uint32_t>(n); ++col) {
        std::uint32_t acc = 0;
        for (std::uint32_t kk = 0; kk < static_cast<std::uint32_t>(k); ++kk) {
            const auto a_ptr = new_id();
            const auto a_val = new_id();
            const auto b_ptr = new_id();
            const auto b_val = new_id();
            const auto product = new_id();
            emit_body(op_access_chain, {id_ptr_f32, a_ptr, id_a, constants[0], constants[kk]});
            emit_body(op_load, {id_f32, a_val, a_ptr});
            emit_body(op_access_chain, {id_ptr_f32, b_ptr, id_b, constants[0], constants[kk * n + col]});
            emit_body(op_load, {id_f32, b_val, b_ptr});
            emit_body(op_fmul, {id_f32, product, a_val, b_val});
            if (acc == 0) {
                acc = product;
            } else {
                const auto sum = new_id();
                emit_body(op_fadd, {id_f32, sum, acc, product});
                acc = sum;
            }
        }
        const auto c_ptr = new_id();
        emit_body(op_access_chain, {id_ptr_f32, c_ptr, id_c, constants[0], constants[col]});
        emit_body(op_store, {c_ptr, acc});
    }

    std::vector<std::uint32_t> words;
    words.reserve(160 + body.size());
    words.push_back(0x07230203);
    words.push_back(0x00010000);
    words.push_back(0);
    words.push_back(next_id);
    words.push_back(0);
    append_spirv_inst(words, op_capability, {capability_shader});
    append_spirv_inst(words, op_memory_model, {addressing_model_logical, memory_model_glsl450});
    words.push_back((5u << 16u) | op_entry_point);
    words.push_back(execution_model_gl_compute);
    words.push_back(id_main);
    append_spirv_string(words, "main");
    append_spirv_inst(words, op_execution_mode, {id_main, execution_mode_local_size, 1, 1, 1});
    append_spirv_inst(words, op_decorate, {id_runtime_array, decoration_array_stride, 4});
    append_spirv_inst(words, op_member_decorate, {id_buffer_struct, 0, decoration_offset, 0});
    append_spirv_inst(words, op_decorate, {id_buffer_struct, decoration_buffer_block});
    append_spirv_inst(words, op_decorate, {id_a, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_a, decoration_binding, 0});
    append_spirv_inst(words, op_decorate, {id_b, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_b, decoration_binding, 1});
    append_spirv_inst(words, op_decorate, {id_c, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_c, decoration_binding, 2});
    append_spirv_inst(words, op_type_void, {id_void});
    append_spirv_inst(words, op_type_function, {id_fn, id_void});
    append_spirv_inst(words, op_type_float, {id_f32, 32});
    append_spirv_inst(words, op_type_int, {id_u32, 32, 0});
    for (std::uint32_t value = 0; value <= max_constant; ++value) {
        append_spirv_inst(words, op_constant, {id_u32, constants[value], value});
    }
    append_spirv_inst(words, op_type_runtime_array, {id_runtime_array, id_f32});
    append_spirv_inst(words, op_type_struct, {id_buffer_struct, id_runtime_array});
    append_spirv_inst(words, op_type_pointer, {id_ptr_buffer, storage_class_uniform, id_buffer_struct});
    append_spirv_inst(words, op_type_pointer, {id_ptr_f32, storage_class_uniform, id_f32});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_a, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_b, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_c, storage_class_uniform});
    append_spirv_inst(words, op_function, {id_void, id_main, 0, id_fn});
    append_spirv_inst(words, op_label, {id_label});
    words.insert(words.end(), body.begin(), body.end());
    append_spirv_inst(words, op_return, {});
    append_spirv_inst(words, op_function_end, {});
    return words;
}

std::vector<std::uint32_t> f32_matmul_spirv(std::size_t k, std::size_t n) {
    constexpr std::uint16_t op_capability = 17;
    constexpr std::uint16_t op_memory_model = 14;
    constexpr std::uint16_t op_entry_point = 15;
    constexpr std::uint16_t op_execution_mode = 16;
    constexpr std::uint16_t op_decorate = 71;
    constexpr std::uint16_t op_member_decorate = 72;
    constexpr std::uint16_t op_type_void = 19;
    constexpr std::uint16_t op_type_function = 33;
    constexpr std::uint16_t op_type_float = 22;
    constexpr std::uint16_t op_type_int = 21;
    constexpr std::uint16_t op_type_vector = 23;
    constexpr std::uint16_t op_constant = 43;
    constexpr std::uint16_t op_type_runtime_array = 29;
    constexpr std::uint16_t op_type_struct = 30;
    constexpr std::uint16_t op_type_pointer = 32;
    constexpr std::uint16_t op_variable = 59;
    constexpr std::uint16_t op_function = 54;
    constexpr std::uint16_t op_label = 248;
    constexpr std::uint16_t op_load = 61;
    constexpr std::uint16_t op_composite_extract = 81;
    constexpr std::uint16_t op_iadd = 128;
    constexpr std::uint16_t op_imul = 132;
    constexpr std::uint16_t op_access_chain = 65;
    constexpr std::uint16_t op_fmul = 133;
    constexpr std::uint16_t op_fadd = 129;
    constexpr std::uint16_t op_store = 62;
    constexpr std::uint16_t op_return = 253;
    constexpr std::uint16_t op_function_end = 56;

    constexpr std::uint32_t capability_shader = 1;
    constexpr std::uint32_t addressing_model_logical = 0;
    constexpr std::uint32_t memory_model_glsl450 = 1;
    constexpr std::uint32_t execution_model_gl_compute = 5;
    constexpr std::uint32_t execution_mode_local_size = 17;
    constexpr std::uint32_t decoration_array_stride = 6;
    constexpr std::uint32_t decoration_offset = 35;
    constexpr std::uint32_t decoration_buffer_block = 3;
    constexpr std::uint32_t decoration_descriptor_set = 34;
    constexpr std::uint32_t decoration_binding = 33;
    constexpr std::uint32_t decoration_built_in = 11;
    constexpr std::uint32_t built_in_global_invocation_id = 28;
    constexpr std::uint32_t storage_class_input = 1;
    constexpr std::uint32_t storage_class_uniform = 2;

    const auto max_constant = std::max(k, n);

    const std::uint32_t id_void = 1;
    const std::uint32_t id_fn = 2;
    const std::uint32_t id_main = 3;
    const std::uint32_t id_label = 4;
    const std::uint32_t id_f32 = 5;
    const std::uint32_t id_u32 = 6;
    const std::uint32_t id_v3u32 = 7;
    const std::uint32_t id_ptr_input_v3u32 = 8;
    const std::uint32_t id_global_invocation_id = 9;
    std::uint32_t next_id = 10;
    std::vector<std::uint32_t> constants(max_constant + 1);
    for (std::uint32_t value = 0; value <= max_constant; ++value) constants[value] = next_id++;
    const std::uint32_t id_runtime_array = next_id++;
    const std::uint32_t id_buffer_struct = next_id++;
    const std::uint32_t id_ptr_buffer = next_id++;
    const std::uint32_t id_ptr_f32 = next_id++;
    const std::uint32_t id_a = next_id++;
    const std::uint32_t id_b = next_id++;
    const std::uint32_t id_c = next_id++;
    auto new_id = [&]() {
        return next_id++;
    };

    std::vector<std::uint32_t> body;
    auto emit_body = [&](std::uint16_t opcode, const std::vector<std::uint32_t>& operands) {
        append_spirv_inst(body, opcode, operands);
    };

    const auto gid = new_id();
    const auto col = new_id();
    const auto row = new_id();
    emit_body(op_load, {id_v3u32, gid, id_global_invocation_id});
    emit_body(op_composite_extract, {id_u32, col, gid, 0});
    emit_body(op_composite_extract, {id_u32, row, gid, 1});

    std::uint32_t acc = 0;
    for (std::uint32_t kk = 0; kk < static_cast<std::uint32_t>(k); ++kk) {
        const auto a_row_base = new_id();
        const auto a_index = new_id();
        const auto b_row_base = new_id();
        const auto b_index = new_id();
        const auto a_ptr = new_id();
        const auto a_val = new_id();
        const auto b_ptr = new_id();
        const auto b_val = new_id();
        const auto product = new_id();
        emit_body(op_imul, {id_u32, a_row_base, row, constants[static_cast<std::uint32_t>(k)]});
        emit_body(op_iadd, {id_u32, a_index, a_row_base, constants[kk]});
        emit_body(op_imul, {id_u32, b_row_base, constants[kk], constants[static_cast<std::uint32_t>(n)]});
        emit_body(op_iadd, {id_u32, b_index, b_row_base, col});
        emit_body(op_access_chain, {id_ptr_f32, a_ptr, id_a, constants[0], a_index});
        emit_body(op_load, {id_f32, a_val, a_ptr});
        emit_body(op_access_chain, {id_ptr_f32, b_ptr, id_b, constants[0], b_index});
        emit_body(op_load, {id_f32, b_val, b_ptr});
        emit_body(op_fmul, {id_f32, product, a_val, b_val});
        if (acc == 0) {
            acc = product;
        } else {
            const auto sum = new_id();
            emit_body(op_fadd, {id_f32, sum, acc, product});
            acc = sum;
        }
    }
    const auto c_row_base = new_id();
    const auto c_index = new_id();
    const auto c_ptr = new_id();
    emit_body(op_imul, {id_u32, c_row_base, row, constants[static_cast<std::uint32_t>(n)]});
    emit_body(op_iadd, {id_u32, c_index, c_row_base, col});
    emit_body(op_access_chain, {id_ptr_f32, c_ptr, id_c, constants[0], c_index});
    emit_body(op_store, {c_ptr, acc});

    std::vector<std::uint32_t> words;
    words.reserve(180 + body.size());
    words.push_back(0x07230203);
    words.push_back(0x00010000);
    words.push_back(0);
    words.push_back(next_id);
    words.push_back(0);
    append_spirv_inst(words, op_capability, {capability_shader});
    append_spirv_inst(words, op_memory_model, {addressing_model_logical, memory_model_glsl450});
    words.push_back((6u << 16u) | op_entry_point);
    words.push_back(execution_model_gl_compute);
    words.push_back(id_main);
    append_spirv_string(words, "main");
    words.push_back(id_global_invocation_id);
    append_spirv_inst(words, op_execution_mode, {id_main, execution_mode_local_size, 1, 1, 1});
    append_spirv_inst(words, op_decorate,
                      {id_global_invocation_id, decoration_built_in, built_in_global_invocation_id});
    append_spirv_inst(words, op_decorate, {id_runtime_array, decoration_array_stride, 4});
    append_spirv_inst(words, op_member_decorate, {id_buffer_struct, 0, decoration_offset, 0});
    append_spirv_inst(words, op_decorate, {id_buffer_struct, decoration_buffer_block});
    append_spirv_inst(words, op_decorate, {id_a, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_a, decoration_binding, 0});
    append_spirv_inst(words, op_decorate, {id_b, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_b, decoration_binding, 1});
    append_spirv_inst(words, op_decorate, {id_c, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_c, decoration_binding, 2});
    append_spirv_inst(words, op_type_void, {id_void});
    append_spirv_inst(words, op_type_function, {id_fn, id_void});
    append_spirv_inst(words, op_type_float, {id_f32, 32});
    append_spirv_inst(words, op_type_int, {id_u32, 32, 0});
    append_spirv_inst(words, op_type_vector, {id_v3u32, id_u32, 3});
    append_spirv_inst(words, op_type_pointer, {id_ptr_input_v3u32, storage_class_input, id_v3u32});
    for (std::uint32_t value = 0; value <= max_constant; ++value) {
        append_spirv_inst(words, op_constant, {id_u32, constants[value], value});
    }
    append_spirv_inst(words, op_type_runtime_array, {id_runtime_array, id_f32});
    append_spirv_inst(words, op_type_struct, {id_buffer_struct, id_runtime_array});
    append_spirv_inst(words, op_type_pointer, {id_ptr_buffer, storage_class_uniform, id_buffer_struct});
    append_spirv_inst(words, op_type_pointer, {id_ptr_f32, storage_class_uniform, id_f32});
    append_spirv_inst(words, op_variable, {id_ptr_input_v3u32, id_global_invocation_id, storage_class_input});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_a, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_b, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_c, storage_class_uniform});
    append_spirv_inst(words, op_function, {id_void, id_main, 0, id_fn});
    append_spirv_inst(words, op_label, {id_label});
    words.insert(words.end(), body.begin(), body.end());
    append_spirv_inst(words, op_return, {});
    append_spirv_inst(words, op_function_end, {});
    return words;
}

[[maybe_unused]] std::vector<std::uint32_t> f32_matmul_transpose_b_spirv(std::size_t k, std::size_t n) {
    constexpr std::uint16_t op_capability = 17;
    constexpr std::uint16_t op_memory_model = 14;
    constexpr std::uint16_t op_entry_point = 15;
    constexpr std::uint16_t op_execution_mode = 16;
    constexpr std::uint16_t op_decorate = 71;
    constexpr std::uint16_t op_member_decorate = 72;
    constexpr std::uint16_t op_type_void = 19;
    constexpr std::uint16_t op_type_function = 33;
    constexpr std::uint16_t op_type_float = 22;
    constexpr std::uint16_t op_type_int = 21;
    constexpr std::uint16_t op_type_vector = 23;
    constexpr std::uint16_t op_constant = 43;
    constexpr std::uint16_t op_type_runtime_array = 29;
    constexpr std::uint16_t op_type_struct = 30;
    constexpr std::uint16_t op_type_pointer = 32;
    constexpr std::uint16_t op_variable = 59;
    constexpr std::uint16_t op_function = 54;
    constexpr std::uint16_t op_label = 248;
    constexpr std::uint16_t op_load = 61;
    constexpr std::uint16_t op_composite_extract = 81;
    constexpr std::uint16_t op_iadd = 128;
    constexpr std::uint16_t op_imul = 132;
    constexpr std::uint16_t op_access_chain = 65;
    constexpr std::uint16_t op_fmul = 133;
    constexpr std::uint16_t op_fadd = 129;
    constexpr std::uint16_t op_store = 62;
    constexpr std::uint16_t op_return = 253;
    constexpr std::uint16_t op_function_end = 56;

    constexpr std::uint32_t capability_shader = 1;
    constexpr std::uint32_t addressing_model_logical = 0;
    constexpr std::uint32_t memory_model_glsl450 = 1;
    constexpr std::uint32_t execution_model_gl_compute = 5;
    constexpr std::uint32_t execution_mode_local_size = 17;
    constexpr std::uint32_t decoration_array_stride = 6;
    constexpr std::uint32_t decoration_offset = 35;
    constexpr std::uint32_t decoration_buffer_block = 3;
    constexpr std::uint32_t decoration_descriptor_set = 34;
    constexpr std::uint32_t decoration_binding = 33;
    constexpr std::uint32_t decoration_built_in = 11;
    constexpr std::uint32_t built_in_global_invocation_id = 28;
    constexpr std::uint32_t storage_class_input = 1;
    constexpr std::uint32_t storage_class_uniform = 2;

    const auto max_constant = std::max(k, n);

    const std::uint32_t id_void = 1;
    const std::uint32_t id_fn = 2;
    const std::uint32_t id_main = 3;
    const std::uint32_t id_label = 4;
    const std::uint32_t id_f32 = 5;
    const std::uint32_t id_u32 = 6;
    const std::uint32_t id_v3u32 = 7;
    const std::uint32_t id_ptr_input_v3u32 = 8;
    const std::uint32_t id_global_invocation_id = 9;
    std::uint32_t next_id = 10;
    std::vector<std::uint32_t> constants(max_constant + 1);
    for (std::uint32_t value = 0; value <= max_constant; ++value) constants[value] = next_id++;
    const std::uint32_t id_runtime_array = next_id++;
    const std::uint32_t id_buffer_struct = next_id++;
    const std::uint32_t id_ptr_buffer = next_id++;
    const std::uint32_t id_ptr_f32 = next_id++;
    const std::uint32_t id_a = next_id++;
    const std::uint32_t id_b = next_id++;
    const std::uint32_t id_c = next_id++;
    auto new_id = [&]() {
        return next_id++;
    };

    std::vector<std::uint32_t> body;
    auto emit_body = [&](std::uint16_t opcode, const std::vector<std::uint32_t>& operands) {
        append_spirv_inst(body, opcode, operands);
    };

    const auto gid = new_id();
    const auto col = new_id();
    const auto row = new_id();
    emit_body(op_load, {id_v3u32, gid, id_global_invocation_id});
    emit_body(op_composite_extract, {id_u32, col, gid, 0});
    emit_body(op_composite_extract, {id_u32, row, gid, 1});

    std::uint32_t acc = 0;
    for (std::uint32_t kk = 0; kk < static_cast<std::uint32_t>(k); ++kk) {
        const auto a_row_base = new_id();
        const auto a_index = new_id();
        const auto b_row_base = new_id();
        const auto b_index = new_id();
        const auto a_ptr = new_id();
        const auto a_val = new_id();
        const auto b_ptr = new_id();
        const auto b_val = new_id();
        const auto product = new_id();
        emit_body(op_imul, {id_u32, a_row_base, row, constants[static_cast<std::uint32_t>(k)]});
        emit_body(op_iadd, {id_u32, a_index, a_row_base, constants[kk]});
        emit_body(op_imul, {id_u32, b_row_base, col, constants[static_cast<std::uint32_t>(k)]});
        emit_body(op_iadd, {id_u32, b_index, b_row_base, constants[kk]});
        emit_body(op_access_chain, {id_ptr_f32, a_ptr, id_a, constants[0], a_index});
        emit_body(op_load, {id_f32, a_val, a_ptr});
        emit_body(op_access_chain, {id_ptr_f32, b_ptr, id_b, constants[0], b_index});
        emit_body(op_load, {id_f32, b_val, b_ptr});
        emit_body(op_fmul, {id_f32, product, a_val, b_val});
        if (acc == 0) {
            acc = product;
        } else {
            const auto sum = new_id();
            emit_body(op_fadd, {id_f32, sum, acc, product});
            acc = sum;
        }
    }
    const auto c_row_base = new_id();
    const auto c_index = new_id();
    const auto c_ptr = new_id();
    emit_body(op_imul, {id_u32, c_row_base, row, constants[static_cast<std::uint32_t>(n)]});
    emit_body(op_iadd, {id_u32, c_index, c_row_base, col});
    emit_body(op_access_chain, {id_ptr_f32, c_ptr, id_c, constants[0], c_index});
    emit_body(op_store, {c_ptr, acc});

    std::vector<std::uint32_t> words;
    words.reserve(180 + body.size());
    words.push_back(0x07230203);
    words.push_back(0x00010000);
    words.push_back(0);
    words.push_back(next_id);
    words.push_back(0);
    append_spirv_inst(words, op_capability, {capability_shader});
    append_spirv_inst(words, op_memory_model, {addressing_model_logical, memory_model_glsl450});
    words.push_back((6u << 16u) | op_entry_point);
    words.push_back(execution_model_gl_compute);
    words.push_back(id_main);
    append_spirv_string(words, "main");
    words.push_back(id_global_invocation_id);
    append_spirv_inst(words, op_execution_mode, {id_main, execution_mode_local_size, 1, 1, 1});
    append_spirv_inst(words, op_decorate,
                      {id_global_invocation_id, decoration_built_in, built_in_global_invocation_id});
    append_spirv_inst(words, op_decorate, {id_runtime_array, decoration_array_stride, 4});
    append_spirv_inst(words, op_member_decorate, {id_buffer_struct, 0, decoration_offset, 0});
    append_spirv_inst(words, op_decorate, {id_buffer_struct, decoration_buffer_block});
    append_spirv_inst(words, op_decorate, {id_a, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_a, decoration_binding, 0});
    append_spirv_inst(words, op_decorate, {id_b, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_b, decoration_binding, 1});
    append_spirv_inst(words, op_decorate, {id_c, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_c, decoration_binding, 2});
    append_spirv_inst(words, op_type_void, {id_void});
    append_spirv_inst(words, op_type_function, {id_fn, id_void});
    append_spirv_inst(words, op_type_float, {id_f32, 32});
    append_spirv_inst(words, op_type_int, {id_u32, 32, 0});
    append_spirv_inst(words, op_type_vector, {id_v3u32, id_u32, 3});
    append_spirv_inst(words, op_type_pointer, {id_ptr_input_v3u32, storage_class_input, id_v3u32});
    for (std::uint32_t value = 0; value <= max_constant; ++value) {
        append_spirv_inst(words, op_constant, {id_u32, constants[value], value});
    }
    append_spirv_inst(words, op_type_runtime_array, {id_runtime_array, id_f32});
    append_spirv_inst(words, op_type_struct, {id_buffer_struct, id_runtime_array});
    append_spirv_inst(words, op_type_pointer, {id_ptr_buffer, storage_class_uniform, id_buffer_struct});
    append_spirv_inst(words, op_type_pointer, {id_ptr_f32, storage_class_uniform, id_f32});
    append_spirv_inst(words, op_variable, {id_ptr_input_v3u32, id_global_invocation_id, storage_class_input});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_a, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_b, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_c, storage_class_uniform});
    append_spirv_inst(words, op_function, {id_void, id_main, 0, id_fn});
    append_spirv_inst(words, op_label, {id_label});
    words.insert(words.end(), body.begin(), body.end());
    append_spirv_inst(words, op_return, {});
    append_spirv_inst(words, op_function_end, {});
    return words;
}

std::vector<std::uint32_t> i32_scaled_matmul_spirv(std::size_t k,
                                                  std::size_t n,
                                                  float scale_a,
                                                  float scale_b) {
    constexpr std::uint16_t op_capability = 17;
    constexpr std::uint16_t op_memory_model = 14;
    constexpr std::uint16_t op_entry_point = 15;
    constexpr std::uint16_t op_execution_mode = 16;
    constexpr std::uint16_t op_decorate = 71;
    constexpr std::uint16_t op_member_decorate = 72;
    constexpr std::uint16_t op_type_void = 19;
    constexpr std::uint16_t op_type_function = 33;
    constexpr std::uint16_t op_type_float = 22;
    constexpr std::uint16_t op_type_int = 21;
    constexpr std::uint16_t op_type_vector = 23;
    constexpr std::uint16_t op_constant = 43;
    constexpr std::uint16_t op_type_runtime_array = 29;
    constexpr std::uint16_t op_type_struct = 30;
    constexpr std::uint16_t op_type_pointer = 32;
    constexpr std::uint16_t op_variable = 59;
    constexpr std::uint16_t op_function = 54;
    constexpr std::uint16_t op_label = 248;
    constexpr std::uint16_t op_load = 61;
    constexpr std::uint16_t op_composite_extract = 81;
    constexpr std::uint16_t op_convert_s_to_f = 111;
    constexpr std::uint16_t op_iadd = 128;
    constexpr std::uint16_t op_fadd = 129;
    constexpr std::uint16_t op_imul = 132;
    constexpr std::uint16_t op_fmul = 133;
    constexpr std::uint16_t op_access_chain = 65;
    constexpr std::uint16_t op_store = 62;
    constexpr std::uint16_t op_return = 253;
    constexpr std::uint16_t op_function_end = 56;

    constexpr std::uint32_t capability_shader = 1;
    constexpr std::uint32_t addressing_model_logical = 0;
    constexpr std::uint32_t memory_model_glsl450 = 1;
    constexpr std::uint32_t execution_model_gl_compute = 5;
    constexpr std::uint32_t execution_mode_local_size = 17;
    constexpr std::uint32_t decoration_array_stride = 6;
    constexpr std::uint32_t decoration_offset = 35;
    constexpr std::uint32_t decoration_buffer_block = 3;
    constexpr std::uint32_t decoration_descriptor_set = 34;
    constexpr std::uint32_t decoration_binding = 33;
    constexpr std::uint32_t decoration_built_in = 11;
    constexpr std::uint32_t built_in_global_invocation_id = 28;
    constexpr std::uint32_t storage_class_input = 1;
    constexpr std::uint32_t storage_class_uniform = 2;

    const auto max_constant = std::max(k, n);

    const std::uint32_t id_void = 1;
    const std::uint32_t id_fn = 2;
    const std::uint32_t id_main = 3;
    const std::uint32_t id_label = 4;
    const std::uint32_t id_f32 = 5;
    const std::uint32_t id_u32 = 6;
    const std::uint32_t id_i32 = 7;
    const std::uint32_t id_v3u32 = 8;
    const std::uint32_t id_ptr_input_v3u32 = 9;
    const std::uint32_t id_global_invocation_id = 10;
    std::uint32_t next_id = 11;
    std::vector<std::uint32_t> constants(max_constant + 1);
    for (std::uint32_t value = 0; value <= max_constant; ++value) constants[value] = next_id++;
    const std::uint32_t id_scale_a = next_id++;
    const std::uint32_t id_scale_b = next_id++;
    const std::uint32_t id_i32_runtime_array = next_id++;
    const std::uint32_t id_f32_runtime_array = next_id++;
    const std::uint32_t id_i32_buffer_struct = next_id++;
    const std::uint32_t id_f32_buffer_struct = next_id++;
    const std::uint32_t id_ptr_i32_buffer = next_id++;
    const std::uint32_t id_ptr_f32_buffer = next_id++;
    const std::uint32_t id_ptr_i32 = next_id++;
    const std::uint32_t id_ptr_f32 = next_id++;
    const std::uint32_t id_a = next_id++;
    const std::uint32_t id_b = next_id++;
    const std::uint32_t id_c = next_id++;
    auto new_id = [&]() { return next_id++; };

    std::vector<std::uint32_t> body;
    auto emit_body = [&](std::uint16_t opcode, const std::vector<std::uint32_t>& operands) {
        append_spirv_inst(body, opcode, operands);
    };

    const auto gid = new_id();
    const auto col = new_id();
    const auto row = new_id();
    emit_body(op_load, {id_v3u32, gid, id_global_invocation_id});
    emit_body(op_composite_extract, {id_u32, col, gid, 0});
    emit_body(op_composite_extract, {id_u32, row, gid, 1});

    std::uint32_t acc = 0;
    for (std::uint32_t kk = 0; kk < static_cast<std::uint32_t>(k); ++kk) {
        const auto a_row_base = new_id();
        const auto a_index = new_id();
        const auto b_row_base = new_id();
        const auto b_index = new_id();
        const auto a_ptr = new_id();
        const auto b_ptr = new_id();
        const auto a_i = new_id();
        const auto b_i = new_id();
        const auto a_f = new_id();
        const auto b_f = new_id();
        const auto product = new_id();
        emit_body(op_imul, {id_u32, a_row_base, row, constants[static_cast<std::uint32_t>(k)]});
        emit_body(op_iadd, {id_u32, a_index, a_row_base, constants[kk]});
        emit_body(op_imul, {id_u32, b_row_base, constants[kk], constants[static_cast<std::uint32_t>(n)]});
        emit_body(op_iadd, {id_u32, b_index, b_row_base, col});
        emit_body(op_access_chain, {id_ptr_i32, a_ptr, id_a, constants[0], a_index});
        emit_body(op_access_chain, {id_ptr_i32, b_ptr, id_b, constants[0], b_index});
        emit_body(op_load, {id_i32, a_i, a_ptr});
        emit_body(op_load, {id_i32, b_i, b_ptr});
        emit_body(op_convert_s_to_f, {id_f32, a_f, a_i});
        emit_body(op_convert_s_to_f, {id_f32, b_f, b_i});
        emit_body(op_fmul, {id_f32, product, a_f, b_f});
        if (acc == 0) {
            acc = product;
        } else {
            const auto sum = new_id();
            emit_body(op_fadd, {id_f32, sum, acc, product});
            acc = sum;
        }
    }
    const auto scaled_a = new_id();
    const auto scaled_ab = new_id();
    const auto c_row_base = new_id();
    const auto c_index = new_id();
    const auto c_ptr = new_id();
    emit_body(op_fmul, {id_f32, scaled_a, acc, id_scale_a});
    emit_body(op_fmul, {id_f32, scaled_ab, scaled_a, id_scale_b});
    emit_body(op_imul, {id_u32, c_row_base, row, constants[static_cast<std::uint32_t>(n)]});
    emit_body(op_iadd, {id_u32, c_index, c_row_base, col});
    emit_body(op_access_chain, {id_ptr_f32, c_ptr, id_c, constants[0], c_index});
    emit_body(op_store, {c_ptr, scaled_ab});

    std::vector<std::uint32_t> words;
    words.reserve(220 + body.size());
    words.push_back(0x07230203);
    words.push_back(0x00010000);
    words.push_back(0);
    words.push_back(next_id);
    words.push_back(0);
    append_spirv_inst(words, op_capability, {capability_shader});
    append_spirv_inst(words, op_memory_model, {addressing_model_logical, memory_model_glsl450});
    words.push_back((6u << 16u) | op_entry_point);
    words.push_back(execution_model_gl_compute);
    words.push_back(id_main);
    append_spirv_string(words, "main");
    words.push_back(id_global_invocation_id);
    append_spirv_inst(words, op_execution_mode, {id_main, execution_mode_local_size, 1, 1, 1});
    append_spirv_inst(words, op_decorate,
                      {id_global_invocation_id, decoration_built_in, built_in_global_invocation_id});
    append_spirv_inst(words, op_decorate, {id_i32_runtime_array, decoration_array_stride, 4});
    append_spirv_inst(words, op_decorate, {id_f32_runtime_array, decoration_array_stride, 4});
    append_spirv_inst(words, op_member_decorate, {id_i32_buffer_struct, 0, decoration_offset, 0});
    append_spirv_inst(words, op_member_decorate, {id_f32_buffer_struct, 0, decoration_offset, 0});
    append_spirv_inst(words, op_decorate, {id_i32_buffer_struct, decoration_buffer_block});
    append_spirv_inst(words, op_decorate, {id_f32_buffer_struct, decoration_buffer_block});
    append_spirv_inst(words, op_decorate, {id_a, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_a, decoration_binding, 0});
    append_spirv_inst(words, op_decorate, {id_b, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_b, decoration_binding, 1});
    append_spirv_inst(words, op_decorate, {id_c, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_c, decoration_binding, 2});
    append_spirv_inst(words, op_type_void, {id_void});
    append_spirv_inst(words, op_type_function, {id_fn, id_void});
    append_spirv_inst(words, op_type_float, {id_f32, 32});
    append_spirv_inst(words, op_type_int, {id_u32, 32, 0});
    append_spirv_inst(words, op_type_int, {id_i32, 32, 1});
    append_spirv_inst(words, op_type_vector, {id_v3u32, id_u32, 3});
    append_spirv_inst(words, op_type_pointer, {id_ptr_input_v3u32, storage_class_input, id_v3u32});
    for (std::uint32_t value = 0; value <= max_constant; ++value) {
        append_spirv_inst(words, op_constant, {id_u32, constants[value], value});
    }
    append_spirv_inst(words, op_constant, {id_f32, id_scale_a, f32_bits(scale_a)});
    append_spirv_inst(words, op_constant, {id_f32, id_scale_b, f32_bits(scale_b)});
    append_spirv_inst(words, op_type_runtime_array, {id_i32_runtime_array, id_i32});
    append_spirv_inst(words, op_type_runtime_array, {id_f32_runtime_array, id_f32});
    append_spirv_inst(words, op_type_struct, {id_i32_buffer_struct, id_i32_runtime_array});
    append_spirv_inst(words, op_type_struct, {id_f32_buffer_struct, id_f32_runtime_array});
    append_spirv_inst(words, op_type_pointer, {id_ptr_i32_buffer, storage_class_uniform, id_i32_buffer_struct});
    append_spirv_inst(words, op_type_pointer, {id_ptr_f32_buffer, storage_class_uniform, id_f32_buffer_struct});
    append_spirv_inst(words, op_type_pointer, {id_ptr_i32, storage_class_uniform, id_i32});
    append_spirv_inst(words, op_type_pointer, {id_ptr_f32, storage_class_uniform, id_f32});
    append_spirv_inst(words, op_variable, {id_ptr_input_v3u32, id_global_invocation_id, storage_class_input});
    append_spirv_inst(words, op_variable, {id_ptr_i32_buffer, id_a, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_i32_buffer, id_b, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_f32_buffer, id_c, storage_class_uniform});
    append_spirv_inst(words, op_function, {id_void, id_main, 0, id_fn});
    append_spirv_inst(words, op_label, {id_label});
    words.insert(words.end(), body.begin(), body.end());
    append_spirv_inst(words, op_return, {});
    append_spirv_inst(words, op_function_end, {});
    return words;
}

std::vector<std::uint32_t> i8_scaled_matmul_spirv(std::size_t k,
                                                 std::size_t n,
                                                 float scale_a,
                                                 float scale_b) {
    constexpr std::uint16_t op_extension = 10;
    constexpr std::uint16_t op_capability = 17;
    constexpr std::uint16_t op_memory_model = 14;
    constexpr std::uint16_t op_entry_point = 15;
    constexpr std::uint16_t op_execution_mode = 16;
    constexpr std::uint16_t op_decorate = 71;
    constexpr std::uint16_t op_member_decorate = 72;
    constexpr std::uint16_t op_type_void = 19;
    constexpr std::uint16_t op_type_function = 33;
    constexpr std::uint16_t op_type_float = 22;
    constexpr std::uint16_t op_type_int = 21;
    constexpr std::uint16_t op_type_vector = 23;
    constexpr std::uint16_t op_constant = 43;
    constexpr std::uint16_t op_type_runtime_array = 29;
    constexpr std::uint16_t op_type_struct = 30;
    constexpr std::uint16_t op_type_pointer = 32;
    constexpr std::uint16_t op_variable = 59;
    constexpr std::uint16_t op_function = 54;
    constexpr std::uint16_t op_label = 248;
    constexpr std::uint16_t op_load = 61;
    constexpr std::uint16_t op_composite_extract = 81;
    constexpr std::uint16_t op_convert_s_to_f = 111;
    constexpr std::uint16_t op_iadd = 128;
    constexpr std::uint16_t op_fadd = 129;
    constexpr std::uint16_t op_imul = 132;
    constexpr std::uint16_t op_fmul = 133;
    constexpr std::uint16_t op_access_chain = 65;
    constexpr std::uint16_t op_store = 62;
    constexpr std::uint16_t op_return = 253;
    constexpr std::uint16_t op_function_end = 56;

    constexpr std::uint32_t capability_shader = 1;
    constexpr std::uint32_t capability_int8 = 39;
    constexpr std::uint32_t capability_uniform_and_storage_buffer_8bit_access = 4449;
    constexpr std::uint32_t addressing_model_logical = 0;
    constexpr std::uint32_t memory_model_glsl450 = 1;
    constexpr std::uint32_t execution_model_gl_compute = 5;
    constexpr std::uint32_t execution_mode_local_size = 17;
    constexpr std::uint32_t decoration_array_stride = 6;
    constexpr std::uint32_t decoration_offset = 35;
    constexpr std::uint32_t decoration_buffer_block = 3;
    constexpr std::uint32_t decoration_descriptor_set = 34;
    constexpr std::uint32_t decoration_binding = 33;
    constexpr std::uint32_t decoration_built_in = 11;
    constexpr std::uint32_t built_in_global_invocation_id = 28;
    constexpr std::uint32_t storage_class_input = 1;
    constexpr std::uint32_t storage_class_uniform = 2;

    const auto max_constant = std::max(k, n);

    const std::uint32_t id_void = 1;
    const std::uint32_t id_fn = 2;
    const std::uint32_t id_main = 3;
    const std::uint32_t id_label = 4;
    const std::uint32_t id_f32 = 5;
    const std::uint32_t id_u32 = 6;
    const std::uint32_t id_i8 = 7;
    const std::uint32_t id_v3u32 = 8;
    const std::uint32_t id_ptr_input_v3u32 = 9;
    const std::uint32_t id_global_invocation_id = 10;
    std::uint32_t next_id = 11;
    std::vector<std::uint32_t> constants(max_constant + 1);
    for (std::uint32_t value = 0; value <= max_constant; ++value) constants[value] = next_id++;
    const std::uint32_t id_scale_a = next_id++;
    const std::uint32_t id_scale_b = next_id++;
    const std::uint32_t id_i8_runtime_array = next_id++;
    const std::uint32_t id_f32_runtime_array = next_id++;
    const std::uint32_t id_i8_buffer_struct = next_id++;
    const std::uint32_t id_f32_buffer_struct = next_id++;
    const std::uint32_t id_ptr_i8_buffer = next_id++;
    const std::uint32_t id_ptr_f32_buffer = next_id++;
    const std::uint32_t id_ptr_i8 = next_id++;
    const std::uint32_t id_ptr_f32 = next_id++;
    const std::uint32_t id_a = next_id++;
    const std::uint32_t id_b = next_id++;
    const std::uint32_t id_c = next_id++;
    auto new_id = [&]() { return next_id++; };

    std::vector<std::uint32_t> body;
    auto emit_body = [&](std::uint16_t opcode, const std::vector<std::uint32_t>& operands) {
        append_spirv_inst(body, opcode, operands);
    };

    const auto gid = new_id();
    const auto col = new_id();
    const auto row = new_id();
    emit_body(op_load, {id_v3u32, gid, id_global_invocation_id});
    emit_body(op_composite_extract, {id_u32, col, gid, 0});
    emit_body(op_composite_extract, {id_u32, row, gid, 1});

    std::uint32_t acc = 0;
    for (std::uint32_t kk = 0; kk < static_cast<std::uint32_t>(k); ++kk) {
        const auto a_row_base = new_id();
        const auto a_index = new_id();
        const auto b_row_base = new_id();
        const auto b_index = new_id();
        const auto a_ptr = new_id();
        const auto b_ptr = new_id();
        const auto a_i = new_id();
        const auto b_i = new_id();
        const auto a_f = new_id();
        const auto b_f = new_id();
        const auto product = new_id();
        emit_body(op_imul, {id_u32, a_row_base, row, constants[static_cast<std::uint32_t>(k)]});
        emit_body(op_iadd, {id_u32, a_index, a_row_base, constants[kk]});
        emit_body(op_imul, {id_u32, b_row_base, constants[kk], constants[static_cast<std::uint32_t>(n)]});
        emit_body(op_iadd, {id_u32, b_index, b_row_base, col});
        emit_body(op_access_chain, {id_ptr_i8, a_ptr, id_a, constants[0], a_index});
        emit_body(op_access_chain, {id_ptr_i8, b_ptr, id_b, constants[0], b_index});
        emit_body(op_load, {id_i8, a_i, a_ptr});
        emit_body(op_load, {id_i8, b_i, b_ptr});
        emit_body(op_convert_s_to_f, {id_f32, a_f, a_i});
        emit_body(op_convert_s_to_f, {id_f32, b_f, b_i});
        emit_body(op_fmul, {id_f32, product, a_f, b_f});
        if (acc == 0) {
            acc = product;
        } else {
            const auto sum = new_id();
            emit_body(op_fadd, {id_f32, sum, acc, product});
            acc = sum;
        }
    }
    const auto scaled_a = new_id();
    const auto scaled_ab = new_id();
    const auto c_row_base = new_id();
    const auto c_index = new_id();
    const auto c_ptr = new_id();
    emit_body(op_fmul, {id_f32, scaled_a, acc, id_scale_a});
    emit_body(op_fmul, {id_f32, scaled_ab, scaled_a, id_scale_b});
    emit_body(op_imul, {id_u32, c_row_base, row, constants[static_cast<std::uint32_t>(n)]});
    emit_body(op_iadd, {id_u32, c_index, c_row_base, col});
    emit_body(op_access_chain, {id_ptr_f32, c_ptr, id_c, constants[0], c_index});
    emit_body(op_store, {c_ptr, scaled_ab});

    std::vector<std::uint32_t> words;
    words.reserve(230 + body.size());
    words.push_back(0x07230203);
    words.push_back(0x00010000);
    words.push_back(0);
    words.push_back(next_id);
    words.push_back(0);
    append_spirv_inst(words, op_capability, {capability_shader});
    append_spirv_inst(words, op_capability, {capability_int8});
    append_spirv_inst(words, op_capability, {capability_uniform_and_storage_buffer_8bit_access});
    append_spirv_string_inst(words, op_extension, "SPV_KHR_8bit_storage");
    append_spirv_inst(words, op_memory_model, {addressing_model_logical, memory_model_glsl450});
    words.push_back((6u << 16u) | op_entry_point);
    words.push_back(execution_model_gl_compute);
    words.push_back(id_main);
    append_spirv_string(words, "main");
    words.push_back(id_global_invocation_id);
    append_spirv_inst(words, op_execution_mode, {id_main, execution_mode_local_size, 1, 1, 1});
    append_spirv_inst(words, op_decorate,
                      {id_global_invocation_id, decoration_built_in, built_in_global_invocation_id});
    append_spirv_inst(words, op_decorate, {id_i8_runtime_array, decoration_array_stride, 1});
    append_spirv_inst(words, op_decorate, {id_f32_runtime_array, decoration_array_stride, 4});
    append_spirv_inst(words, op_member_decorate, {id_i8_buffer_struct, 0, decoration_offset, 0});
    append_spirv_inst(words, op_member_decorate, {id_f32_buffer_struct, 0, decoration_offset, 0});
    append_spirv_inst(words, op_decorate, {id_i8_buffer_struct, decoration_buffer_block});
    append_spirv_inst(words, op_decorate, {id_f32_buffer_struct, decoration_buffer_block});
    append_spirv_inst(words, op_decorate, {id_a, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_a, decoration_binding, 0});
    append_spirv_inst(words, op_decorate, {id_b, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_b, decoration_binding, 1});
    append_spirv_inst(words, op_decorate, {id_c, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_c, decoration_binding, 2});
    append_spirv_inst(words, op_type_void, {id_void});
    append_spirv_inst(words, op_type_function, {id_fn, id_void});
    append_spirv_inst(words, op_type_float, {id_f32, 32});
    append_spirv_inst(words, op_type_int, {id_u32, 32, 0});
    append_spirv_inst(words, op_type_int, {id_i8, 8, 1});
    append_spirv_inst(words, op_type_vector, {id_v3u32, id_u32, 3});
    append_spirv_inst(words, op_type_pointer, {id_ptr_input_v3u32, storage_class_input, id_v3u32});
    for (std::uint32_t value = 0; value <= max_constant; ++value) {
        append_spirv_inst(words, op_constant, {id_u32, constants[value], value});
    }
    append_spirv_inst(words, op_constant, {id_f32, id_scale_a, f32_bits(scale_a)});
    append_spirv_inst(words, op_constant, {id_f32, id_scale_b, f32_bits(scale_b)});
    append_spirv_inst(words, op_type_runtime_array, {id_i8_runtime_array, id_i8});
    append_spirv_inst(words, op_type_runtime_array, {id_f32_runtime_array, id_f32});
    append_spirv_inst(words, op_type_struct, {id_i8_buffer_struct, id_i8_runtime_array});
    append_spirv_inst(words, op_type_struct, {id_f32_buffer_struct, id_f32_runtime_array});
    append_spirv_inst(words, op_type_pointer, {id_ptr_i8_buffer, storage_class_uniform, id_i8_buffer_struct});
    append_spirv_inst(words, op_type_pointer, {id_ptr_f32_buffer, storage_class_uniform, id_f32_buffer_struct});
    append_spirv_inst(words, op_type_pointer, {id_ptr_i8, storage_class_uniform, id_i8});
    append_spirv_inst(words, op_type_pointer, {id_ptr_f32, storage_class_uniform, id_f32});
    append_spirv_inst(words, op_variable, {id_ptr_input_v3u32, id_global_invocation_id, storage_class_input});
    append_spirv_inst(words, op_variable, {id_ptr_i8_buffer, id_a, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_i8_buffer, id_b, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_f32_buffer, id_c, storage_class_uniform});
    append_spirv_inst(words, op_function, {id_void, id_main, 0, id_fn});
    append_spirv_inst(words, op_label, {id_label});
    words.insert(words.end(), body.begin(), body.end());
    append_spirv_inst(words, op_return, {});
    append_spirv_inst(words, op_function_end, {});
    return words;
}

std::vector<std::uint32_t> rmsnorm_spirv(std::size_t cols, float eps) {
    constexpr std::uint16_t op_ext_inst_import = 11;
    constexpr std::uint16_t op_ext_inst = 12;
    constexpr std::uint16_t op_capability = 17;
    constexpr std::uint16_t op_memory_model = 14;
    constexpr std::uint16_t op_entry_point = 15;
    constexpr std::uint16_t op_execution_mode = 16;
    constexpr std::uint16_t op_decorate = 71;
    constexpr std::uint16_t op_member_decorate = 72;
    constexpr std::uint16_t op_type_void = 19;
    constexpr std::uint16_t op_type_function = 33;
    constexpr std::uint16_t op_type_float = 22;
    constexpr std::uint16_t op_type_int = 21;
    constexpr std::uint16_t op_type_vector = 23;
    constexpr std::uint16_t op_constant = 43;
    constexpr std::uint16_t op_type_runtime_array = 29;
    constexpr std::uint16_t op_type_struct = 30;
    constexpr std::uint16_t op_type_pointer = 32;
    constexpr std::uint16_t op_variable = 59;
    constexpr std::uint16_t op_function = 54;
    constexpr std::uint16_t op_label = 248;
    constexpr std::uint16_t op_load = 61;
    constexpr std::uint16_t op_composite_extract = 81;
    constexpr std::uint16_t op_iadd = 128;
    constexpr std::uint16_t op_imul = 132;
    constexpr std::uint16_t op_fmul = 133;
    constexpr std::uint16_t op_fadd = 129;
    constexpr std::uint16_t op_access_chain = 65;
    constexpr std::uint16_t op_store = 62;
    constexpr std::uint16_t op_return = 253;
    constexpr std::uint16_t op_function_end = 56;

    constexpr std::uint32_t capability_shader = 1;
    constexpr std::uint32_t addressing_model_logical = 0;
    constexpr std::uint32_t memory_model_glsl450 = 1;
    constexpr std::uint32_t execution_model_gl_compute = 5;
    constexpr std::uint32_t execution_mode_local_size = 17;
    constexpr std::uint32_t decoration_array_stride = 6;
    constexpr std::uint32_t decoration_offset = 35;
    constexpr std::uint32_t decoration_buffer_block = 3;
    constexpr std::uint32_t decoration_descriptor_set = 34;
    constexpr std::uint32_t decoration_binding = 33;
    constexpr std::uint32_t decoration_built_in = 11;
    constexpr std::uint32_t built_in_global_invocation_id = 28;
    constexpr std::uint32_t storage_class_input = 1;
    constexpr std::uint32_t storage_class_uniform = 2;
    constexpr std::uint32_t glsl_inverse_sqrt = 32;

    const auto max_constant = cols;

    const std::uint32_t id_void = 1;
    const std::uint32_t id_fn = 2;
    const std::uint32_t id_main = 3;
    const std::uint32_t id_label = 4;
    const std::uint32_t id_glsl = 5;
    const std::uint32_t id_f32 = 6;
    const std::uint32_t id_u32 = 7;
    const std::uint32_t id_v3u32 = 8;
    const std::uint32_t id_ptr_input_v3u32 = 9;
    const std::uint32_t id_global_invocation_id = 10;
    std::uint32_t next_id = 11;
    std::vector<std::uint32_t> constants(max_constant + 1);
    for (std::uint32_t value = 0; value <= max_constant; ++value) constants[value] = next_id++;
    const std::uint32_t id_inv_cols = next_id++;
    const std::uint32_t id_eps = next_id++;
    const std::uint32_t id_runtime_array = next_id++;
    const std::uint32_t id_buffer_struct = next_id++;
    const std::uint32_t id_ptr_buffer = next_id++;
    const std::uint32_t id_ptr_f32 = next_id++;
    const std::uint32_t id_x = next_id++;
    const std::uint32_t id_weight = next_id++;
    const std::uint32_t id_out = next_id++;
    auto new_id = [&]() {
        return next_id++;
    };

    std::vector<std::uint32_t> body;
    auto emit_body = [&](std::uint16_t opcode, const std::vector<std::uint32_t>& operands) {
        append_spirv_inst(body, opcode, operands);
    };

    const auto gid = new_id();
    const auto row = new_id();
    const auto row_base = new_id();
    emit_body(op_load, {id_v3u32, gid, id_global_invocation_id});
    emit_body(op_composite_extract, {id_u32, row, gid, 0});
    emit_body(op_imul, {id_u32, row_base, row, constants[static_cast<std::uint32_t>(cols)]});

    std::uint32_t sumsq = 0;
    for (std::uint32_t col = 0; col < static_cast<std::uint32_t>(cols); ++col) {
        const auto index = new_id();
        const auto ptr = new_id();
        const auto value = new_id();
        const auto square = new_id();
        emit_body(op_iadd, {id_u32, index, row_base, constants[col]});
        emit_body(op_access_chain, {id_ptr_f32, ptr, id_x, constants[0], index});
        emit_body(op_load, {id_f32, value, ptr});
        emit_body(op_fmul, {id_f32, square, value, value});
        if (sumsq == 0) {
            sumsq = square;
        } else {
            const auto next_sum = new_id();
            emit_body(op_fadd, {id_f32, next_sum, sumsq, square});
            sumsq = next_sum;
        }
    }
    const auto mean_square = new_id();
    const auto variance = new_id();
    const auto inv_rms = new_id();
    emit_body(op_fmul, {id_f32, mean_square, sumsq, id_inv_cols});
    emit_body(op_fadd, {id_f32, variance, mean_square, id_eps});
    emit_body(op_ext_inst, {id_f32, inv_rms, id_glsl, glsl_inverse_sqrt, variance});

    for (std::uint32_t col = 0; col < static_cast<std::uint32_t>(cols); ++col) {
        const auto index = new_id();
        const auto x_ptr = new_id();
        const auto w_ptr = new_id();
        const auto out_ptr = new_id();
        const auto x_val = new_id();
        const auto w_val = new_id();
        const auto normed = new_id();
        const auto weighted = new_id();
        emit_body(op_iadd, {id_u32, index, row_base, constants[col]});
        emit_body(op_access_chain, {id_ptr_f32, x_ptr, id_x, constants[0], index});
        emit_body(op_access_chain, {id_ptr_f32, w_ptr, id_weight, constants[0], constants[col]});
        emit_body(op_access_chain, {id_ptr_f32, out_ptr, id_out, constants[0], index});
        emit_body(op_load, {id_f32, x_val, x_ptr});
        emit_body(op_load, {id_f32, w_val, w_ptr});
        emit_body(op_fmul, {id_f32, normed, x_val, inv_rms});
        emit_body(op_fmul, {id_f32, weighted, normed, w_val});
        emit_body(op_store, {out_ptr, weighted});
    }

    std::vector<std::uint32_t> words;
    words.reserve(220 + body.size());
    words.push_back(0x07230203);
    words.push_back(0x00010000);
    words.push_back(0);
    words.push_back(next_id);
    words.push_back(0);
    append_spirv_inst(words, op_capability, {capability_shader});
    words.push_back((6u << 16u) | op_ext_inst_import);
    words.push_back(id_glsl);
    append_spirv_string(words, "GLSL.std.450");
    append_spirv_inst(words, op_memory_model, {addressing_model_logical, memory_model_glsl450});
    words.push_back((6u << 16u) | op_entry_point);
    words.push_back(execution_model_gl_compute);
    words.push_back(id_main);
    append_spirv_string(words, "main");
    words.push_back(id_global_invocation_id);
    append_spirv_inst(words, op_execution_mode, {id_main, execution_mode_local_size, 1, 1, 1});
    append_spirv_inst(words, op_decorate,
                      {id_global_invocation_id, decoration_built_in, built_in_global_invocation_id});
    append_spirv_inst(words, op_decorate, {id_runtime_array, decoration_array_stride, 4});
    append_spirv_inst(words, op_member_decorate, {id_buffer_struct, 0, decoration_offset, 0});
    append_spirv_inst(words, op_decorate, {id_buffer_struct, decoration_buffer_block});
    append_spirv_inst(words, op_decorate, {id_x, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_x, decoration_binding, 0});
    append_spirv_inst(words, op_decorate, {id_weight, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_weight, decoration_binding, 1});
    append_spirv_inst(words, op_decorate, {id_out, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_out, decoration_binding, 2});
    append_spirv_inst(words, op_type_void, {id_void});
    append_spirv_inst(words, op_type_function, {id_fn, id_void});
    append_spirv_inst(words, op_type_float, {id_f32, 32});
    append_spirv_inst(words, op_type_int, {id_u32, 32, 0});
    append_spirv_inst(words, op_type_vector, {id_v3u32, id_u32, 3});
    append_spirv_inst(words, op_type_pointer, {id_ptr_input_v3u32, storage_class_input, id_v3u32});
    for (std::uint32_t value = 0; value <= max_constant; ++value) {
        append_spirv_inst(words, op_constant, {id_u32, constants[value], value});
    }
    append_spirv_inst(words, op_constant, {id_f32, id_inv_cols, f32_bits(1.0f / static_cast<float>(cols))});
    append_spirv_inst(words, op_constant, {id_f32, id_eps, f32_bits(eps)});
    append_spirv_inst(words, op_type_runtime_array, {id_runtime_array, id_f32});
    append_spirv_inst(words, op_type_struct, {id_buffer_struct, id_runtime_array});
    append_spirv_inst(words, op_type_pointer, {id_ptr_buffer, storage_class_uniform, id_buffer_struct});
    append_spirv_inst(words, op_type_pointer, {id_ptr_f32, storage_class_uniform, id_f32});
    append_spirv_inst(words, op_variable, {id_ptr_input_v3u32, id_global_invocation_id, storage_class_input});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_x, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_weight, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_out, storage_class_uniform});
    append_spirv_inst(words, op_function, {id_void, id_main, 0, id_fn});
    append_spirv_inst(words, op_label, {id_label});
    words.insert(words.end(), body.begin(), body.end());
    append_spirv_inst(words, op_return, {});
    append_spirv_inst(words, op_function_end, {});
    return words;
}

std::vector<std::uint32_t> swiglu_spirv(std::size_t hidden) {
    constexpr std::uint16_t op_ext_inst_import = 11;
    constexpr std::uint16_t op_ext_inst = 12;
    constexpr std::uint16_t op_capability = 17;
    constexpr std::uint16_t op_memory_model = 14;
    constexpr std::uint16_t op_entry_point = 15;
    constexpr std::uint16_t op_execution_mode = 16;
    constexpr std::uint16_t op_decorate = 71;
    constexpr std::uint16_t op_member_decorate = 72;
    constexpr std::uint16_t op_type_void = 19;
    constexpr std::uint16_t op_type_function = 33;
    constexpr std::uint16_t op_type_float = 22;
    constexpr std::uint16_t op_type_int = 21;
    constexpr std::uint16_t op_type_vector = 23;
    constexpr std::uint16_t op_constant = 43;
    constexpr std::uint16_t op_type_runtime_array = 29;
    constexpr std::uint16_t op_type_struct = 30;
    constexpr std::uint16_t op_type_pointer = 32;
    constexpr std::uint16_t op_variable = 59;
    constexpr std::uint16_t op_function = 54;
    constexpr std::uint16_t op_label = 248;
    constexpr std::uint16_t op_load = 61;
    constexpr std::uint16_t op_composite_extract = 81;
    constexpr std::uint16_t op_iadd = 128;
    constexpr std::uint16_t op_imul = 132;
    constexpr std::uint16_t op_fadd = 129;
    constexpr std::uint16_t op_fsub = 131;
    constexpr std::uint16_t op_fmul = 133;
    constexpr std::uint16_t op_fdiv = 136;
    constexpr std::uint16_t op_access_chain = 65;
    constexpr std::uint16_t op_store = 62;
    constexpr std::uint16_t op_return = 253;
    constexpr std::uint16_t op_function_end = 56;

    constexpr std::uint32_t capability_shader = 1;
    constexpr std::uint32_t addressing_model_logical = 0;
    constexpr std::uint32_t memory_model_glsl450 = 1;
    constexpr std::uint32_t execution_model_gl_compute = 5;
    constexpr std::uint32_t execution_mode_local_size = 17;
    constexpr std::uint32_t decoration_array_stride = 6;
    constexpr std::uint32_t decoration_offset = 35;
    constexpr std::uint32_t decoration_buffer_block = 3;
    constexpr std::uint32_t decoration_descriptor_set = 34;
    constexpr std::uint32_t decoration_binding = 33;
    constexpr std::uint32_t decoration_built_in = 11;
    constexpr std::uint32_t built_in_global_invocation_id = 28;
    constexpr std::uint32_t storage_class_input = 1;
    constexpr std::uint32_t storage_class_uniform = 2;
    constexpr std::uint32_t glsl_exp = 27;

    const std::uint32_t id_void = 1;
    const std::uint32_t id_fn = 2;
    const std::uint32_t id_main = 3;
    const std::uint32_t id_label = 4;
    const std::uint32_t id_glsl = 5;
    const std::uint32_t id_f32 = 6;
    const std::uint32_t id_u32 = 7;
    const std::uint32_t id_v3u32 = 8;
    const std::uint32_t id_ptr_input_v3u32 = 9;
    const std::uint32_t id_global_invocation_id = 10;
    const std::uint32_t id_const_u32_0 = 11;
    const std::uint32_t id_const_u32_1 = 12;
    const std::uint32_t id_const_hidden = 13;
    const std::uint32_t id_const_packed_stride = 14;
    const std::uint32_t id_const_f32_0 = 15;
    const std::uint32_t id_const_f32_1 = 16;
    const std::uint32_t id_runtime_array = 17;
    const std::uint32_t id_buffer_struct = 18;
    const std::uint32_t id_ptr_buffer = 19;
    const std::uint32_t id_ptr_f32 = 20;
    const std::uint32_t id_packed = 21;
    const std::uint32_t id_out = 22;
    std::uint32_t next_id = 23;
    auto new_id = [&]() {
        return next_id++;
    };

    std::vector<std::uint32_t> body;
    auto emit_body = [&](std::uint16_t opcode, const std::vector<std::uint32_t>& operands) {
        append_spirv_inst(body, opcode, operands);
    };

    const auto gid = new_id();
    const auto col = new_id();
    const auto row = new_id();
    const auto packed_base = new_id();
    const auto out_base = new_id();
    const auto gate_index = new_id();
    const auto up_offset = new_id();
    const auto up_index = new_id();
    const auto out_index = new_id();
    const auto gate_ptr = new_id();
    const auto up_ptr = new_id();
    const auto out_ptr = new_id();
    const auto gate = new_id();
    const auto up = new_id();
    const auto neg_gate = new_id();
    const auto exp_neg_gate = new_id();
    const auto denom = new_id();
    const auto silu = new_id();
    const auto value = new_id();
    emit_body(op_load, {id_v3u32, gid, id_global_invocation_id});
    emit_body(op_composite_extract, {id_u32, col, gid, 0});
    emit_body(op_composite_extract, {id_u32, row, gid, 1});
    emit_body(op_imul, {id_u32, packed_base, row, id_const_packed_stride});
    emit_body(op_imul, {id_u32, out_base, row, id_const_hidden});
    emit_body(op_iadd, {id_u32, gate_index, packed_base, col});
    emit_body(op_iadd, {id_u32, up_offset, col, id_const_hidden});
    emit_body(op_iadd, {id_u32, up_index, packed_base, up_offset});
    emit_body(op_iadd, {id_u32, out_index, out_base, col});
    emit_body(op_access_chain, {id_ptr_f32, gate_ptr, id_packed, id_const_u32_0, gate_index});
    emit_body(op_access_chain, {id_ptr_f32, up_ptr, id_packed, id_const_u32_0, up_index});
    emit_body(op_access_chain, {id_ptr_f32, out_ptr, id_out, id_const_u32_0, out_index});
    emit_body(op_load, {id_f32, gate, gate_ptr});
    emit_body(op_load, {id_f32, up, up_ptr});
    emit_body(op_fsub, {id_f32, neg_gate, id_const_f32_0, gate});
    emit_body(op_ext_inst, {id_f32, exp_neg_gate, id_glsl, glsl_exp, neg_gate});
    emit_body(op_fadd, {id_f32, denom, id_const_f32_1, exp_neg_gate});
    emit_body(op_fdiv, {id_f32, silu, gate, denom});
    emit_body(op_fmul, {id_f32, value, silu, up});
    emit_body(op_store, {out_ptr, value});

    std::vector<std::uint32_t> words;
    words.reserve(160 + body.size());
    words.push_back(0x07230203);
    words.push_back(0x00010000);
    words.push_back(0);
    words.push_back(next_id);
    words.push_back(0);
    append_spirv_inst(words, op_capability, {capability_shader});
    words.push_back((6u << 16u) | op_ext_inst_import);
    words.push_back(id_glsl);
    append_spirv_string(words, "GLSL.std.450");
    append_spirv_inst(words, op_memory_model, {addressing_model_logical, memory_model_glsl450});
    words.push_back((6u << 16u) | op_entry_point);
    words.push_back(execution_model_gl_compute);
    words.push_back(id_main);
    append_spirv_string(words, "main");
    words.push_back(id_global_invocation_id);
    append_spirv_inst(words, op_execution_mode, {id_main, execution_mode_local_size, 1, 1, 1});
    append_spirv_inst(words, op_decorate,
                      {id_global_invocation_id, decoration_built_in, built_in_global_invocation_id});
    append_spirv_inst(words, op_decorate, {id_runtime_array, decoration_array_stride, 4});
    append_spirv_inst(words, op_member_decorate, {id_buffer_struct, 0, decoration_offset, 0});
    append_spirv_inst(words, op_decorate, {id_buffer_struct, decoration_buffer_block});
    append_spirv_inst(words, op_decorate, {id_packed, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_packed, decoration_binding, 0});
    append_spirv_inst(words, op_decorate, {id_out, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_out, decoration_binding, 1});
    append_spirv_inst(words, op_type_void, {id_void});
    append_spirv_inst(words, op_type_function, {id_fn, id_void});
    append_spirv_inst(words, op_type_float, {id_f32, 32});
    append_spirv_inst(words, op_type_int, {id_u32, 32, 0});
    append_spirv_inst(words, op_type_vector, {id_v3u32, id_u32, 3});
    append_spirv_inst(words, op_type_pointer, {id_ptr_input_v3u32, storage_class_input, id_v3u32});
    append_spirv_inst(words, op_constant, {id_u32, id_const_u32_0, 0});
    append_spirv_inst(words, op_constant, {id_u32, id_const_u32_1, 1});
    append_spirv_inst(words, op_constant, {id_u32, id_const_hidden, static_cast<std::uint32_t>(hidden)});
    append_spirv_inst(words, op_constant, {id_u32, id_const_packed_stride,
                                           static_cast<std::uint32_t>(hidden * 2)});
    append_spirv_inst(words, op_constant, {id_f32, id_const_f32_0, f32_bits(0.0f)});
    append_spirv_inst(words, op_constant, {id_f32, id_const_f32_1, f32_bits(1.0f)});
    append_spirv_inst(words, op_type_runtime_array, {id_runtime_array, id_f32});
    append_spirv_inst(words, op_type_struct, {id_buffer_struct, id_runtime_array});
    append_spirv_inst(words, op_type_pointer, {id_ptr_buffer, storage_class_uniform, id_buffer_struct});
    append_spirv_inst(words, op_type_pointer, {id_ptr_f32, storage_class_uniform, id_f32});
    append_spirv_inst(words, op_variable, {id_ptr_input_v3u32, id_global_invocation_id, storage_class_input});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_packed, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_out, storage_class_uniform});
    append_spirv_inst(words, op_function, {id_void, id_main, 0, id_fn});
    append_spirv_inst(words, op_label, {id_label});
    words.insert(words.end(), body.begin(), body.end());
    append_spirv_inst(words, op_return, {});
    append_spirv_inst(words, op_function_end, {});
    return words;
}

std::vector<std::uint32_t> add_spirv() {
    constexpr std::uint16_t op_capability = 17;
    constexpr std::uint16_t op_memory_model = 14;
    constexpr std::uint16_t op_entry_point = 15;
    constexpr std::uint16_t op_execution_mode = 16;
    constexpr std::uint16_t op_decorate = 71;
    constexpr std::uint16_t op_member_decorate = 72;
    constexpr std::uint16_t op_type_void = 19;
    constexpr std::uint16_t op_type_function = 33;
    constexpr std::uint16_t op_type_float = 22;
    constexpr std::uint16_t op_type_int = 21;
    constexpr std::uint16_t op_type_vector = 23;
    constexpr std::uint16_t op_constant = 43;
    constexpr std::uint16_t op_type_runtime_array = 29;
    constexpr std::uint16_t op_type_struct = 30;
    constexpr std::uint16_t op_type_pointer = 32;
    constexpr std::uint16_t op_variable = 59;
    constexpr std::uint16_t op_function = 54;
    constexpr std::uint16_t op_label = 248;
    constexpr std::uint16_t op_load = 61;
    constexpr std::uint16_t op_composite_extract = 81;
    constexpr std::uint16_t op_fadd = 129;
    constexpr std::uint16_t op_access_chain = 65;
    constexpr std::uint16_t op_store = 62;
    constexpr std::uint16_t op_return = 253;
    constexpr std::uint16_t op_function_end = 56;

    constexpr std::uint32_t capability_shader = 1;
    constexpr std::uint32_t addressing_model_logical = 0;
    constexpr std::uint32_t memory_model_glsl450 = 1;
    constexpr std::uint32_t execution_model_gl_compute = 5;
    constexpr std::uint32_t execution_mode_local_size = 17;
    constexpr std::uint32_t decoration_array_stride = 6;
    constexpr std::uint32_t decoration_offset = 35;
    constexpr std::uint32_t decoration_buffer_block = 3;
    constexpr std::uint32_t decoration_descriptor_set = 34;
    constexpr std::uint32_t decoration_binding = 33;
    constexpr std::uint32_t decoration_built_in = 11;
    constexpr std::uint32_t built_in_global_invocation_id = 28;
    constexpr std::uint32_t storage_class_input = 1;
    constexpr std::uint32_t storage_class_uniform = 2;

    const std::uint32_t id_void = 1;
    const std::uint32_t id_fn = 2;
    const std::uint32_t id_main = 3;
    const std::uint32_t id_label = 4;
    const std::uint32_t id_f32 = 5;
    const std::uint32_t id_u32 = 6;
    const std::uint32_t id_v3u32 = 7;
    const std::uint32_t id_ptr_input_v3u32 = 8;
    const std::uint32_t id_global_invocation_id = 9;
    const std::uint32_t id_const_u32_0 = 10;
    const std::uint32_t id_runtime_array = 11;
    const std::uint32_t id_buffer_struct = 12;
    const std::uint32_t id_ptr_buffer = 13;
    const std::uint32_t id_ptr_f32 = 14;
    const std::uint32_t id_a = 15;
    const std::uint32_t id_b = 16;
    const std::uint32_t id_out = 17;
    const std::uint32_t id_gid = 18;
    const std::uint32_t id_index = 19;
    const std::uint32_t id_a_ptr = 20;
    const std::uint32_t id_b_ptr = 21;
    const std::uint32_t id_out_ptr = 22;
    const std::uint32_t id_a_val = 23;
    const std::uint32_t id_b_val = 24;
    const std::uint32_t id_sum = 25;
    const std::uint32_t bound = 26;

    std::vector<std::uint32_t> words;
    words.reserve(140);
    words.push_back(0x07230203);
    words.push_back(0x00010000);
    words.push_back(0);
    words.push_back(bound);
    words.push_back(0);
    append_spirv_inst(words, op_capability, {capability_shader});
    append_spirv_inst(words, op_memory_model, {addressing_model_logical, memory_model_glsl450});
    words.push_back((6u << 16u) | op_entry_point);
    words.push_back(execution_model_gl_compute);
    words.push_back(id_main);
    append_spirv_string(words, "main");
    words.push_back(id_global_invocation_id);
    append_spirv_inst(words, op_execution_mode, {id_main, execution_mode_local_size, 1, 1, 1});
    append_spirv_inst(words, op_decorate,
                      {id_global_invocation_id, decoration_built_in, built_in_global_invocation_id});
    append_spirv_inst(words, op_decorate, {id_runtime_array, decoration_array_stride, 4});
    append_spirv_inst(words, op_member_decorate, {id_buffer_struct, 0, decoration_offset, 0});
    append_spirv_inst(words, op_decorate, {id_buffer_struct, decoration_buffer_block});
    append_spirv_inst(words, op_decorate, {id_a, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_a, decoration_binding, 0});
    append_spirv_inst(words, op_decorate, {id_b, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_b, decoration_binding, 1});
    append_spirv_inst(words, op_decorate, {id_out, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_out, decoration_binding, 2});
    append_spirv_inst(words, op_type_void, {id_void});
    append_spirv_inst(words, op_type_function, {id_fn, id_void});
    append_spirv_inst(words, op_type_float, {id_f32, 32});
    append_spirv_inst(words, op_type_int, {id_u32, 32, 0});
    append_spirv_inst(words, op_type_vector, {id_v3u32, id_u32, 3});
    append_spirv_inst(words, op_type_pointer, {id_ptr_input_v3u32, storage_class_input, id_v3u32});
    append_spirv_inst(words, op_constant, {id_u32, id_const_u32_0, 0});
    append_spirv_inst(words, op_type_runtime_array, {id_runtime_array, id_f32});
    append_spirv_inst(words, op_type_struct, {id_buffer_struct, id_runtime_array});
    append_spirv_inst(words, op_type_pointer, {id_ptr_buffer, storage_class_uniform, id_buffer_struct});
    append_spirv_inst(words, op_type_pointer, {id_ptr_f32, storage_class_uniform, id_f32});
    append_spirv_inst(words, op_variable, {id_ptr_input_v3u32, id_global_invocation_id, storage_class_input});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_a, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_b, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_out, storage_class_uniform});
    append_spirv_inst(words, op_function, {id_void, id_main, 0, id_fn});
    append_spirv_inst(words, op_label, {id_label});
    append_spirv_inst(words, op_load, {id_v3u32, id_gid, id_global_invocation_id});
    append_spirv_inst(words, op_composite_extract, {id_u32, id_index, id_gid, 0});
    append_spirv_inst(words, op_access_chain, {id_ptr_f32, id_a_ptr, id_a, id_const_u32_0, id_index});
    append_spirv_inst(words, op_access_chain, {id_ptr_f32, id_b_ptr, id_b, id_const_u32_0, id_index});
    append_spirv_inst(words, op_access_chain, {id_ptr_f32, id_out_ptr, id_out, id_const_u32_0, id_index});
    append_spirv_inst(words, op_load, {id_f32, id_a_val, id_a_ptr});
    append_spirv_inst(words, op_load, {id_f32, id_b_val, id_b_ptr});
    append_spirv_inst(words, op_fadd, {id_f32, id_sum, id_a_val, id_b_val});
    append_spirv_inst(words, op_store, {id_out_ptr, id_sum});
    append_spirv_inst(words, op_return, {});
    append_spirv_inst(words, op_function_end, {});
    return words;
}

std::vector<std::uint32_t> sgd_update_spirv(float lr) {
    constexpr std::uint16_t op_capability = 17;
    constexpr std::uint16_t op_memory_model = 14;
    constexpr std::uint16_t op_entry_point = 15;
    constexpr std::uint16_t op_execution_mode = 16;
    constexpr std::uint16_t op_decorate = 71;
    constexpr std::uint16_t op_member_decorate = 72;
    constexpr std::uint16_t op_type_void = 19;
    constexpr std::uint16_t op_type_function = 33;
    constexpr std::uint16_t op_type_float = 22;
    constexpr std::uint16_t op_type_int = 21;
    constexpr std::uint16_t op_type_vector = 23;
    constexpr std::uint16_t op_constant = 43;
    constexpr std::uint16_t op_type_runtime_array = 29;
    constexpr std::uint16_t op_type_struct = 30;
    constexpr std::uint16_t op_type_pointer = 32;
    constexpr std::uint16_t op_variable = 59;
    constexpr std::uint16_t op_function = 54;
    constexpr std::uint16_t op_label = 248;
    constexpr std::uint16_t op_load = 61;
    constexpr std::uint16_t op_composite_extract = 81;
    constexpr std::uint16_t op_fsub = 131;
    constexpr std::uint16_t op_fmul = 133;
    constexpr std::uint16_t op_access_chain = 65;
    constexpr std::uint16_t op_store = 62;
    constexpr std::uint16_t op_return = 253;
    constexpr std::uint16_t op_function_end = 56;

    constexpr std::uint32_t capability_shader = 1;
    constexpr std::uint32_t addressing_model_logical = 0;
    constexpr std::uint32_t memory_model_glsl450 = 1;
    constexpr std::uint32_t execution_model_gl_compute = 5;
    constexpr std::uint32_t execution_mode_local_size = 17;
    constexpr std::uint32_t decoration_array_stride = 6;
    constexpr std::uint32_t decoration_offset = 35;
    constexpr std::uint32_t decoration_buffer_block = 3;
    constexpr std::uint32_t decoration_descriptor_set = 34;
    constexpr std::uint32_t decoration_binding = 33;
    constexpr std::uint32_t decoration_built_in = 11;
    constexpr std::uint32_t built_in_global_invocation_id = 28;
    constexpr std::uint32_t storage_class_input = 1;
    constexpr std::uint32_t storage_class_uniform = 2;

    const std::uint32_t id_void = 1;
    const std::uint32_t id_fn = 2;
    const std::uint32_t id_main = 3;
    const std::uint32_t id_label = 4;
    const std::uint32_t id_f32 = 5;
    const std::uint32_t id_u32 = 6;
    const std::uint32_t id_v3u32 = 7;
    const std::uint32_t id_ptr_input_v3u32 = 8;
    const std::uint32_t id_global_invocation_id = 9;
    const std::uint32_t id_const_u32_0 = 10;
    const std::uint32_t id_lr = 11;
    const std::uint32_t id_runtime_array = 12;
    const std::uint32_t id_buffer_struct = 13;
    const std::uint32_t id_ptr_buffer = 14;
    const std::uint32_t id_ptr_f32 = 15;
    const std::uint32_t id_param = 16;
    const std::uint32_t id_grad = 17;
    const std::uint32_t id_out = 18;
    const std::uint32_t id_gid = 19;
    const std::uint32_t id_index = 20;
    const std::uint32_t id_param_ptr = 21;
    const std::uint32_t id_grad_ptr = 22;
    const std::uint32_t id_out_ptr = 23;
    const std::uint32_t id_param_val = 24;
    const std::uint32_t id_grad_val = 25;
    const std::uint32_t id_step = 26;
    const std::uint32_t id_new_value = 27;
    const std::uint32_t bound = 28;

    std::vector<std::uint32_t> words;
    words.reserve(150);
    words.push_back(0x07230203);
    words.push_back(0x00010000);
    words.push_back(0);
    words.push_back(bound);
    words.push_back(0);
    append_spirv_inst(words, op_capability, {capability_shader});
    append_spirv_inst(words, op_memory_model, {addressing_model_logical, memory_model_glsl450});
    words.push_back((6u << 16u) | op_entry_point);
    words.push_back(execution_model_gl_compute);
    words.push_back(id_main);
    append_spirv_string(words, "main");
    words.push_back(id_global_invocation_id);
    append_spirv_inst(words, op_execution_mode, {id_main, execution_mode_local_size, 1, 1, 1});
    append_spirv_inst(words, op_decorate,
                      {id_global_invocation_id, decoration_built_in, built_in_global_invocation_id});
    append_spirv_inst(words, op_decorate, {id_runtime_array, decoration_array_stride, 4});
    append_spirv_inst(words, op_member_decorate, {id_buffer_struct, 0, decoration_offset, 0});
    append_spirv_inst(words, op_decorate, {id_buffer_struct, decoration_buffer_block});
    append_spirv_inst(words, op_decorate, {id_param, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_param, decoration_binding, 0});
    append_spirv_inst(words, op_decorate, {id_grad, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_grad, decoration_binding, 1});
    append_spirv_inst(words, op_decorate, {id_out, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_out, decoration_binding, 2});
    append_spirv_inst(words, op_type_void, {id_void});
    append_spirv_inst(words, op_type_function, {id_fn, id_void});
    append_spirv_inst(words, op_type_float, {id_f32, 32});
    append_spirv_inst(words, op_type_int, {id_u32, 32, 0});
    append_spirv_inst(words, op_type_vector, {id_v3u32, id_u32, 3});
    append_spirv_inst(words, op_type_pointer, {id_ptr_input_v3u32, storage_class_input, id_v3u32});
    append_spirv_inst(words, op_constant, {id_u32, id_const_u32_0, 0});
    append_spirv_inst(words, op_constant, {id_f32, id_lr, f32_bits(lr)});
    append_spirv_inst(words, op_type_runtime_array, {id_runtime_array, id_f32});
    append_spirv_inst(words, op_type_struct, {id_buffer_struct, id_runtime_array});
    append_spirv_inst(words, op_type_pointer, {id_ptr_buffer, storage_class_uniform, id_buffer_struct});
    append_spirv_inst(words, op_type_pointer, {id_ptr_f32, storage_class_uniform, id_f32});
    append_spirv_inst(words, op_variable, {id_ptr_input_v3u32, id_global_invocation_id, storage_class_input});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_param, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_grad, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_out, storage_class_uniform});
    append_spirv_inst(words, op_function, {id_void, id_main, 0, id_fn});
    append_spirv_inst(words, op_label, {id_label});
    append_spirv_inst(words, op_load, {id_v3u32, id_gid, id_global_invocation_id});
    append_spirv_inst(words, op_composite_extract, {id_u32, id_index, id_gid, 0});
    append_spirv_inst(words, op_access_chain, {id_ptr_f32, id_param_ptr, id_param, id_const_u32_0, id_index});
    append_spirv_inst(words, op_access_chain, {id_ptr_f32, id_grad_ptr, id_grad, id_const_u32_0, id_index});
    append_spirv_inst(words, op_access_chain, {id_ptr_f32, id_out_ptr, id_out, id_const_u32_0, id_index});
    append_spirv_inst(words, op_load, {id_f32, id_param_val, id_param_ptr});
    append_spirv_inst(words, op_load, {id_f32, id_grad_val, id_grad_ptr});
    append_spirv_inst(words, op_fmul, {id_f32, id_step, id_grad_val, id_lr});
    append_spirv_inst(words, op_fsub, {id_f32, id_new_value, id_param_val, id_step});
    append_spirv_inst(words, op_store, {id_out_ptr, id_new_value});
    append_spirv_inst(words, op_return, {});
    append_spirv_inst(words, op_function_end, {});
    return words;
}

std::vector<std::uint32_t> softmax_rows_spirv(std::size_t cols) {
    constexpr std::uint16_t op_ext_inst_import = 11;
    constexpr std::uint16_t op_ext_inst = 12;
    constexpr std::uint16_t op_capability = 17;
    constexpr std::uint16_t op_memory_model = 14;
    constexpr std::uint16_t op_entry_point = 15;
    constexpr std::uint16_t op_execution_mode = 16;
    constexpr std::uint16_t op_decorate = 71;
    constexpr std::uint16_t op_member_decorate = 72;
    constexpr std::uint16_t op_type_void = 19;
    constexpr std::uint16_t op_type_function = 33;
    constexpr std::uint16_t op_type_float = 22;
    constexpr std::uint16_t op_type_int = 21;
    constexpr std::uint16_t op_type_vector = 23;
    constexpr std::uint16_t op_constant = 43;
    constexpr std::uint16_t op_type_runtime_array = 29;
    constexpr std::uint16_t op_type_struct = 30;
    constexpr std::uint16_t op_type_pointer = 32;
    constexpr std::uint16_t op_variable = 59;
    constexpr std::uint16_t op_function = 54;
    constexpr std::uint16_t op_label = 248;
    constexpr std::uint16_t op_load = 61;
    constexpr std::uint16_t op_composite_extract = 81;
    constexpr std::uint16_t op_iadd = 128;
    constexpr std::uint16_t op_imul = 132;
    constexpr std::uint16_t op_fadd = 129;
    constexpr std::uint16_t op_fsub = 131;
    constexpr std::uint16_t op_fdiv = 136;
    constexpr std::uint16_t op_access_chain = 65;
    constexpr std::uint16_t op_store = 62;
    constexpr std::uint16_t op_return = 253;
    constexpr std::uint16_t op_function_end = 56;

    constexpr std::uint32_t capability_shader = 1;
    constexpr std::uint32_t addressing_model_logical = 0;
    constexpr std::uint32_t memory_model_glsl450 = 1;
    constexpr std::uint32_t execution_model_gl_compute = 5;
    constexpr std::uint32_t execution_mode_local_size = 17;
    constexpr std::uint32_t decoration_array_stride = 6;
    constexpr std::uint32_t decoration_offset = 35;
    constexpr std::uint32_t decoration_buffer_block = 3;
    constexpr std::uint32_t decoration_descriptor_set = 34;
    constexpr std::uint32_t decoration_binding = 33;
    constexpr std::uint32_t decoration_built_in = 11;
    constexpr std::uint32_t built_in_global_invocation_id = 28;
    constexpr std::uint32_t storage_class_input = 1;
    constexpr std::uint32_t storage_class_uniform = 2;
    constexpr std::uint32_t glsl_exp = 27;
    constexpr std::uint32_t glsl_fmax = 40;

    const auto max_constant = cols;

    const std::uint32_t id_void = 1;
    const std::uint32_t id_fn = 2;
    const std::uint32_t id_main = 3;
    const std::uint32_t id_label = 4;
    const std::uint32_t id_glsl = 5;
    const std::uint32_t id_f32 = 6;
    const std::uint32_t id_u32 = 7;
    const std::uint32_t id_v3u32 = 8;
    const std::uint32_t id_ptr_input_v3u32 = 9;
    const std::uint32_t id_global_invocation_id = 10;
    std::uint32_t next_id = 11;
    std::vector<std::uint32_t> constants(max_constant + 1);
    for (std::uint32_t value = 0; value <= max_constant; ++value) constants[value] = next_id++;
    const std::uint32_t id_runtime_array = next_id++;
    const std::uint32_t id_buffer_struct = next_id++;
    const std::uint32_t id_ptr_buffer = next_id++;
    const std::uint32_t id_ptr_f32 = next_id++;
    const std::uint32_t id_x = next_id++;
    const std::uint32_t id_out = next_id++;
    auto new_id = [&]() {
        return next_id++;
    };

    std::vector<std::uint32_t> body;
    auto emit_body = [&](std::uint16_t opcode, const std::vector<std::uint32_t>& operands) {
        append_spirv_inst(body, opcode, operands);
    };

    const auto gid = new_id();
    const auto row = new_id();
    const auto row_base = new_id();
    emit_body(op_load, {id_v3u32, gid, id_global_invocation_id});
    emit_body(op_composite_extract, {id_u32, row, gid, 0});
    emit_body(op_imul, {id_u32, row_base, row, constants[static_cast<std::uint32_t>(cols)]});

    const auto first_index = new_id();
    const auto first_ptr = new_id();
    std::uint32_t max_value = new_id();
    emit_body(op_iadd, {id_u32, first_index, row_base, constants[0]});
    emit_body(op_access_chain, {id_ptr_f32, first_ptr, id_x, constants[0], first_index});
    emit_body(op_load, {id_f32, max_value, first_ptr});
    for (std::uint32_t col = 1; col < static_cast<std::uint32_t>(cols); ++col) {
        const auto index = new_id();
        const auto ptr = new_id();
        const auto value = new_id();
        const auto next_max = new_id();
        emit_body(op_iadd, {id_u32, index, row_base, constants[col]});
        emit_body(op_access_chain, {id_ptr_f32, ptr, id_x, constants[0], index});
        emit_body(op_load, {id_f32, value, ptr});
        emit_body(op_ext_inst, {id_f32, next_max, id_glsl, glsl_fmax, max_value, value});
        max_value = next_max;
    }

    std::uint32_t denom = 0;
    for (std::uint32_t col = 0; col < static_cast<std::uint32_t>(cols); ++col) {
        const auto index = new_id();
        const auto ptr = new_id();
        const auto value = new_id();
        const auto shifted = new_id();
        const auto e = new_id();
        emit_body(op_iadd, {id_u32, index, row_base, constants[col]});
        emit_body(op_access_chain, {id_ptr_f32, ptr, id_x, constants[0], index});
        emit_body(op_load, {id_f32, value, ptr});
        emit_body(op_fsub, {id_f32, shifted, value, max_value});
        emit_body(op_ext_inst, {id_f32, e, id_glsl, glsl_exp, shifted});
        if (denom == 0) {
            denom = e;
        } else {
            const auto sum = new_id();
            emit_body(op_fadd, {id_f32, sum, denom, e});
            denom = sum;
        }
    }

    for (std::uint32_t col = 0; col < static_cast<std::uint32_t>(cols); ++col) {
        const auto index = new_id();
        const auto in_ptr = new_id();
        const auto value = new_id();
        const auto shifted = new_id();
        const auto e = new_id();
        const auto prob = new_id();
        const auto out_ptr = new_id();
        emit_body(op_iadd, {id_u32, index, row_base, constants[col]});
        emit_body(op_access_chain, {id_ptr_f32, in_ptr, id_x, constants[0], index});
        emit_body(op_load, {id_f32, value, in_ptr});
        emit_body(op_fsub, {id_f32, shifted, value, max_value});
        emit_body(op_ext_inst, {id_f32, e, id_glsl, glsl_exp, shifted});
        emit_body(op_fdiv, {id_f32, prob, e, denom});
        emit_body(op_access_chain, {id_ptr_f32, out_ptr, id_out, constants[0], index});
        emit_body(op_store, {out_ptr, prob});
    }

    std::vector<std::uint32_t> words;
    words.reserve(220 + body.size());
    words.push_back(0x07230203);
    words.push_back(0x00010000);
    words.push_back(0);
    words.push_back(next_id);
    words.push_back(0);
    append_spirv_inst(words, op_capability, {capability_shader});
    words.push_back((6u << 16u) | op_ext_inst_import);
    words.push_back(id_glsl);
    append_spirv_string(words, "GLSL.std.450");
    append_spirv_inst(words, op_memory_model, {addressing_model_logical, memory_model_glsl450});
    words.push_back((6u << 16u) | op_entry_point);
    words.push_back(execution_model_gl_compute);
    words.push_back(id_main);
    append_spirv_string(words, "main");
    words.push_back(id_global_invocation_id);
    append_spirv_inst(words, op_execution_mode, {id_main, execution_mode_local_size, 1, 1, 1});
    append_spirv_inst(words, op_decorate,
                      {id_global_invocation_id, decoration_built_in, built_in_global_invocation_id});
    append_spirv_inst(words, op_decorate, {id_runtime_array, decoration_array_stride, 4});
    append_spirv_inst(words, op_member_decorate, {id_buffer_struct, 0, decoration_offset, 0});
    append_spirv_inst(words, op_decorate, {id_buffer_struct, decoration_buffer_block});
    append_spirv_inst(words, op_decorate, {id_x, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_x, decoration_binding, 0});
    append_spirv_inst(words, op_decorate, {id_out, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_out, decoration_binding, 1});
    append_spirv_inst(words, op_type_void, {id_void});
    append_spirv_inst(words, op_type_function, {id_fn, id_void});
    append_spirv_inst(words, op_type_float, {id_f32, 32});
    append_spirv_inst(words, op_type_int, {id_u32, 32, 0});
    append_spirv_inst(words, op_type_vector, {id_v3u32, id_u32, 3});
    append_spirv_inst(words, op_type_pointer, {id_ptr_input_v3u32, storage_class_input, id_v3u32});
    for (std::uint32_t value = 0; value <= max_constant; ++value) {
        append_spirv_inst(words, op_constant, {id_u32, constants[value], value});
    }
    append_spirv_inst(words, op_type_runtime_array, {id_runtime_array, id_f32});
    append_spirv_inst(words, op_type_struct, {id_buffer_struct, id_runtime_array});
    append_spirv_inst(words, op_type_pointer, {id_ptr_buffer, storage_class_uniform, id_buffer_struct});
    append_spirv_inst(words, op_type_pointer, {id_ptr_f32, storage_class_uniform, id_f32});
    append_spirv_inst(words, op_variable, {id_ptr_input_v3u32, id_global_invocation_id, storage_class_input});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_x, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_out, storage_class_uniform});
    append_spirv_inst(words, op_function, {id_void, id_main, 0, id_fn});
    append_spirv_inst(words, op_label, {id_label});
    words.insert(words.end(), body.begin(), body.end());
    append_spirv_inst(words, op_return, {});
    append_spirv_inst(words, op_function_end, {});
    return words;
}

std::vector<std::uint32_t> gqa_forward_spirv(std::size_t query_tokens,
                                             std::size_t key_tokens,
                                             std::size_t n_head,
                                             std::size_t n_kv_head,
                                             std::size_t head_dim,
                                             float scale) {
    constexpr std::uint16_t op_ext_inst_import = 11;
    constexpr std::uint16_t op_ext_inst = 12;
    constexpr std::uint16_t op_capability = 17;
    constexpr std::uint16_t op_memory_model = 14;
    constexpr std::uint16_t op_entry_point = 15;
    constexpr std::uint16_t op_execution_mode = 16;
    constexpr std::uint16_t op_decorate = 71;
    constexpr std::uint16_t op_member_decorate = 72;
    constexpr std::uint16_t op_type_void = 19;
    constexpr std::uint16_t op_type_function = 33;
    constexpr std::uint16_t op_type_float = 22;
    constexpr std::uint16_t op_type_int = 21;
    constexpr std::uint16_t op_type_vector = 23;
    constexpr std::uint16_t op_constant = 43;
    constexpr std::uint16_t op_type_runtime_array = 29;
    constexpr std::uint16_t op_type_struct = 30;
    constexpr std::uint16_t op_type_pointer = 32;
    constexpr std::uint16_t op_variable = 59;
    constexpr std::uint16_t op_function = 54;
    constexpr std::uint16_t op_label = 248;
    constexpr std::uint16_t op_load = 61;
    constexpr std::uint16_t op_composite_extract = 81;
    constexpr std::uint16_t op_iadd = 128;
    constexpr std::uint16_t op_fadd = 129;
    constexpr std::uint16_t op_fsub = 131;
    constexpr std::uint16_t op_imul = 132;
    constexpr std::uint16_t op_fmul = 133;
    constexpr std::uint16_t op_udiv = 134;
    constexpr std::uint16_t op_fdiv = 136;
    constexpr std::uint16_t op_access_chain = 65;
    constexpr std::uint16_t op_store = 62;
    constexpr std::uint16_t op_return = 253;
    constexpr std::uint16_t op_function_end = 56;

    constexpr std::uint32_t capability_shader = 1;
    constexpr std::uint32_t addressing_model_logical = 0;
    constexpr std::uint32_t memory_model_glsl450 = 1;
    constexpr std::uint32_t execution_model_gl_compute = 5;
    constexpr std::uint32_t execution_mode_local_size = 17;
    constexpr std::uint32_t decoration_array_stride = 6;
    constexpr std::uint32_t decoration_offset = 35;
    constexpr std::uint32_t decoration_buffer_block = 3;
    constexpr std::uint32_t decoration_descriptor_set = 34;
    constexpr std::uint32_t decoration_binding = 33;
    constexpr std::uint32_t decoration_built_in = 11;
    constexpr std::uint32_t built_in_global_invocation_id = 28;
    constexpr std::uint32_t storage_class_input = 1;
    constexpr std::uint32_t storage_class_uniform = 2;
    constexpr std::uint32_t glsl_exp = 27;
    constexpr std::uint32_t glsl_fmax = 40;

    const auto group_size = static_cast<std::uint32_t>(n_head / n_kv_head);
    const auto q_stride = static_cast<std::uint32_t>(n_head * head_dim);
    const auto kv_stride = static_cast<std::uint32_t>(n_kv_head * head_dim);
    std::uint32_t max_constant = static_cast<std::uint32_t>(query_tokens);
    max_constant = std::max(max_constant, static_cast<std::uint32_t>(key_tokens));
    max_constant = std::max(max_constant, static_cast<std::uint32_t>(n_head));
    max_constant = std::max(max_constant, static_cast<std::uint32_t>(n_kv_head));
    max_constant = std::max(max_constant, static_cast<std::uint32_t>(head_dim));
    max_constant = std::max(max_constant, group_size);
    max_constant = std::max(max_constant, q_stride);
    max_constant = std::max(max_constant, kv_stride);

    const std::uint32_t id_void = 1;
    const std::uint32_t id_fn = 2;
    const std::uint32_t id_main = 3;
    const std::uint32_t id_label = 4;
    const std::uint32_t id_glsl = 5;
    const std::uint32_t id_f32 = 6;
    const std::uint32_t id_u32 = 7;
    const std::uint32_t id_v3u32 = 8;
    const std::uint32_t id_ptr_input_v3u32 = 9;
    const std::uint32_t id_global_invocation_id = 10;
    std::uint32_t next_id = 11;
    std::vector<std::uint32_t> constants(max_constant + 1);
    for (std::uint32_t value = 0; value <= max_constant; ++value) constants[value] = next_id++;
    const std::uint32_t id_scale = next_id++;
    const std::uint32_t id_runtime_array = next_id++;
    const std::uint32_t id_buffer_struct = next_id++;
    const std::uint32_t id_ptr_buffer = next_id++;
    const std::uint32_t id_ptr_f32 = next_id++;
    const std::uint32_t id_q = next_id++;
    const std::uint32_t id_k = next_id++;
    const std::uint32_t id_v = next_id++;
    const std::uint32_t id_out = next_id++;
    auto new_id = [&]() { return next_id++; };

    std::vector<std::uint32_t> body;
    auto emit = [&](std::uint16_t opcode, const std::vector<std::uint32_t>& operands) {
        append_spirv_inst(body, opcode, operands);
    };

    const auto gid = new_id();
    const auto d = new_id();
    const auto h = new_id();
    const auto tq = new_id();
    const auto kv_head = new_id();
    emit(op_load, {id_v3u32, gid, id_global_invocation_id});
    emit(op_composite_extract, {id_u32, d, gid, 0});
    emit(op_composite_extract, {id_u32, h, gid, 1});
    emit(op_composite_extract, {id_u32, tq, gid, 2});
    emit(op_udiv, {id_u32, kv_head, h, constants[group_size]});

    auto emit_score = [&](std::uint32_t tk) -> std::uint32_t {
        std::uint32_t dot = 0;
        for (std::uint32_t kd = 0; kd < static_cast<std::uint32_t>(head_dim); ++kd) {
            const auto q_tok_base = new_id();
            const auto q_head_base = new_id();
            const auto q_base = new_id();
            const auto q_index = new_id();
            const auto q_ptr = new_id();
            const auto q_val = new_id();
            const auto k_tok_base = new_id();
            const auto k_head_base = new_id();
            const auto k_base = new_id();
            const auto k_index = new_id();
            const auto k_ptr = new_id();
            const auto k_val = new_id();
            const auto prod = new_id();
            emit(op_imul, {id_u32, q_tok_base, tq, constants[q_stride]});
            emit(op_imul, {id_u32, q_head_base, h, constants[static_cast<std::uint32_t>(head_dim)]});
            emit(op_iadd, {id_u32, q_base, q_tok_base, q_head_base});
            emit(op_iadd, {id_u32, q_index, q_base, constants[kd]});
            emit(op_access_chain, {id_ptr_f32, q_ptr, id_q, constants[0], q_index});
            emit(op_load, {id_f32, q_val, q_ptr});

            emit(op_imul, {id_u32, k_tok_base, constants[tk], constants[kv_stride]});
            emit(op_imul, {id_u32, k_head_base, kv_head, constants[static_cast<std::uint32_t>(head_dim)]});
            emit(op_iadd, {id_u32, k_base, k_tok_base, k_head_base});
            emit(op_iadd, {id_u32, k_index, k_base, constants[kd]});
            emit(op_access_chain, {id_ptr_f32, k_ptr, id_k, constants[0], k_index});
            emit(op_load, {id_f32, k_val, k_ptr});
            emit(op_fmul, {id_f32, prod, q_val, k_val});
            if (dot == 0) {
                dot = prod;
            } else {
                const auto sum = new_id();
                emit(op_fadd, {id_f32, sum, dot, prod});
                dot = sum;
            }
        }
        const auto scaled = new_id();
        emit(op_fmul, {id_f32, scaled, dot, id_scale});
        return scaled;
    };

    std::uint32_t max_score = emit_score(0);
    for (std::uint32_t tk = 1; tk < static_cast<std::uint32_t>(key_tokens); ++tk) {
        const auto score = emit_score(tk);
        const auto next_max = new_id();
        emit(op_ext_inst, {id_f32, next_max, id_glsl, glsl_fmax, max_score, score});
        max_score = next_max;
    }

    std::uint32_t denom = 0;
    for (std::uint32_t tk = 0; tk < static_cast<std::uint32_t>(key_tokens); ++tk) {
        const auto score = emit_score(tk);
        const auto shifted = new_id();
        const auto e = new_id();
        emit(op_fsub, {id_f32, shifted, score, max_score});
        emit(op_ext_inst, {id_f32, e, id_glsl, glsl_exp, shifted});
        if (denom == 0) {
            denom = e;
        } else {
            const auto sum = new_id();
            emit(op_fadd, {id_f32, sum, denom, e});
            denom = sum;
        }
    }

    std::uint32_t acc_out = 0;
    for (std::uint32_t tk = 0; tk < static_cast<std::uint32_t>(key_tokens); ++tk) {
        const auto score = emit_score(tk);
        const auto shifted = new_id();
        const auto e = new_id();
        const auto prob = new_id();
        emit(op_fsub, {id_f32, shifted, score, max_score});
        emit(op_ext_inst, {id_f32, e, id_glsl, glsl_exp, shifted});
        emit(op_fdiv, {id_f32, prob, e, denom});

        const auto v_tok_base = new_id();
        const auto v_head_base = new_id();
        const auto v_base = new_id();
        const auto v_index = new_id();
        const auto v_ptr = new_id();
        const auto v_val = new_id();
        const auto weighted = new_id();
        emit(op_imul, {id_u32, v_tok_base, constants[tk], constants[kv_stride]});
        emit(op_imul, {id_u32, v_head_base, kv_head, constants[static_cast<std::uint32_t>(head_dim)]});
        emit(op_iadd, {id_u32, v_base, v_tok_base, v_head_base});
        emit(op_iadd, {id_u32, v_index, v_base, d});
        emit(op_access_chain, {id_ptr_f32, v_ptr, id_v, constants[0], v_index});
        emit(op_load, {id_f32, v_val, v_ptr});
        emit(op_fmul, {id_f32, weighted, prob, v_val});
        if (acc_out == 0) {
            acc_out = weighted;
        } else {
            const auto sum = new_id();
            emit(op_fadd, {id_f32, sum, acc_out, weighted});
            acc_out = sum;
        }
    }

    const auto out_tok_base = new_id();
    const auto out_head_base = new_id();
    const auto out_base = new_id();
    const auto out_index = new_id();
    const auto out_ptr = new_id();
    emit(op_imul, {id_u32, out_tok_base, tq, constants[q_stride]});
    emit(op_imul, {id_u32, out_head_base, h, constants[static_cast<std::uint32_t>(head_dim)]});
    emit(op_iadd, {id_u32, out_base, out_tok_base, out_head_base});
    emit(op_iadd, {id_u32, out_index, out_base, d});
    emit(op_access_chain, {id_ptr_f32, out_ptr, id_out, constants[0], out_index});
    emit(op_store, {out_ptr, acc_out});

    std::vector<std::uint32_t> words;
    words.reserve(300 + body.size());
    words.push_back(0x07230203);
    words.push_back(0x00010000);
    words.push_back(0);
    words.push_back(next_id);
    words.push_back(0);
    append_spirv_inst(words, op_capability, {capability_shader});
    words.push_back((6u << 16u) | op_ext_inst_import);
    words.push_back(id_glsl);
    append_spirv_string(words, "GLSL.std.450");
    append_spirv_inst(words, op_memory_model, {addressing_model_logical, memory_model_glsl450});
    words.push_back((6u << 16u) | op_entry_point);
    words.push_back(execution_model_gl_compute);
    words.push_back(id_main);
    append_spirv_string(words, "main");
    words.push_back(id_global_invocation_id);
    append_spirv_inst(words, op_execution_mode, {id_main, execution_mode_local_size, 1, 1, 1});
    append_spirv_inst(words, op_decorate,
                      {id_global_invocation_id, decoration_built_in, built_in_global_invocation_id});
    append_spirv_inst(words, op_decorate, {id_runtime_array, decoration_array_stride, 4});
    append_spirv_inst(words, op_member_decorate, {id_buffer_struct, 0, decoration_offset, 0});
    append_spirv_inst(words, op_decorate, {id_buffer_struct, decoration_buffer_block});
    append_spirv_inst(words, op_decorate, {id_q, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_q, decoration_binding, 0});
    append_spirv_inst(words, op_decorate, {id_k, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_k, decoration_binding, 1});
    append_spirv_inst(words, op_decorate, {id_v, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_v, decoration_binding, 2});
    append_spirv_inst(words, op_decorate, {id_out, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_out, decoration_binding, 3});
    append_spirv_inst(words, op_type_void, {id_void});
    append_spirv_inst(words, op_type_function, {id_fn, id_void});
    append_spirv_inst(words, op_type_float, {id_f32, 32});
    append_spirv_inst(words, op_type_int, {id_u32, 32, 0});
    append_spirv_inst(words, op_type_vector, {id_v3u32, id_u32, 3});
    append_spirv_inst(words, op_type_pointer, {id_ptr_input_v3u32, storage_class_input, id_v3u32});
    for (std::uint32_t value = 0; value <= max_constant; ++value) {
        append_spirv_inst(words, op_constant, {id_u32, constants[value], value});
    }
    append_spirv_inst(words, op_constant, {id_f32, id_scale, f32_bits(scale)});
    append_spirv_inst(words, op_type_runtime_array, {id_runtime_array, id_f32});
    append_spirv_inst(words, op_type_struct, {id_buffer_struct, id_runtime_array});
    append_spirv_inst(words, op_type_pointer, {id_ptr_buffer, storage_class_uniform, id_buffer_struct});
    append_spirv_inst(words, op_type_pointer, {id_ptr_f32, storage_class_uniform, id_f32});
    append_spirv_inst(words, op_variable, {id_ptr_input_v3u32, id_global_invocation_id, storage_class_input});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_q, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_k, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_v, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_out, storage_class_uniform});
    append_spirv_inst(words, op_function, {id_void, id_main, 0, id_fn});
    append_spirv_inst(words, op_label, {id_label});
    words.insert(words.end(), body.begin(), body.end());
    append_spirv_inst(words, op_return, {});
    append_spirv_inst(words, op_function_end, {});
    return words;
}

std::vector<std::uint32_t> counter_increment_spirv() {
    constexpr std::uint16_t op_capability = 17;
    constexpr std::uint16_t op_memory_model = 14;
    constexpr std::uint16_t op_entry_point = 15;
    constexpr std::uint16_t op_execution_mode = 16;
    constexpr std::uint16_t op_decorate = 71;
    constexpr std::uint16_t op_member_decorate = 72;
    constexpr std::uint16_t op_type_void = 19;
    constexpr std::uint16_t op_type_function = 33;
    constexpr std::uint16_t op_type_int = 21;
    constexpr std::uint16_t op_type_vector = 23;
    constexpr std::uint16_t op_constant = 43;
    constexpr std::uint16_t op_type_runtime_array = 29;
    constexpr std::uint16_t op_type_struct = 30;
    constexpr std::uint16_t op_type_pointer = 32;
    constexpr std::uint16_t op_variable = 59;
    constexpr std::uint16_t op_function = 54;
    constexpr std::uint16_t op_label = 248;
    constexpr std::uint16_t op_load = 61;
    constexpr std::uint16_t op_composite_extract = 81;
    constexpr std::uint16_t op_iadd = 128;
    constexpr std::uint16_t op_access_chain = 65;
    constexpr std::uint16_t op_store = 62;
    constexpr std::uint16_t op_shift_right_logical = 194;
    constexpr std::uint16_t op_shift_left_logical = 196;
    constexpr std::uint16_t op_bitwise_or = 197;
    constexpr std::uint16_t op_bitwise_and = 199;
    constexpr std::uint16_t op_return = 253;
    constexpr std::uint16_t op_function_end = 56;

    constexpr std::uint32_t capability_shader = 1;
    constexpr std::uint32_t addressing_model_logical = 0;
    constexpr std::uint32_t memory_model_glsl450 = 1;
    constexpr std::uint32_t execution_model_gl_compute = 5;
    constexpr std::uint32_t execution_mode_local_size = 17;
    constexpr std::uint32_t decoration_array_stride = 6;
    constexpr std::uint32_t decoration_offset = 35;
    constexpr std::uint32_t decoration_buffer_block = 3;
    constexpr std::uint32_t decoration_descriptor_set = 34;
    constexpr std::uint32_t decoration_binding = 33;
    constexpr std::uint32_t decoration_built_in = 11;
    constexpr std::uint32_t built_in_global_invocation_id = 28;
    constexpr std::uint32_t storage_class_input = 1;
    constexpr std::uint32_t storage_class_uniform = 2;

    const std::uint32_t id_void = 1;
    const std::uint32_t id_fn = 2;
    const std::uint32_t id_main = 3;
    const std::uint32_t id_label = 4;
    const std::uint32_t id_u32 = 5;
    const std::uint32_t id_v3u32 = 6;
    const std::uint32_t id_ptr_input_v3u32 = 7;
    const std::uint32_t id_global_invocation_id = 8;
    std::uint32_t next_id = 9;
    std::array<std::uint32_t, 64> constants{};
    for (std::uint32_t value = 0; value < constants.size(); ++value) constants[value] = next_id++;
    const std::uint32_t id_runtime_array = next_id++;
    const std::uint32_t id_buffer_struct = next_id++;
    const std::uint32_t id_ptr_buffer = next_id++;
    const std::uint32_t id_ptr_u32 = next_id++;
    const std::uint32_t id_state = next_id++;
    const std::uint32_t id_inc = next_id++;
    const std::uint32_t id_out = next_id++;
    auto new_id = [&]() { return next_id++; };

    std::vector<std::uint32_t> body;
    auto emit = [&](std::uint16_t opcode, const std::vector<std::uint32_t>& operands) {
        append_spirv_inst(body, opcode, operands);
    };

    const auto gid = new_id();
    const auto index = new_id();
    const auto state_ptr = new_id();
    const auto inc_ptr = new_id();
    const auto out_ptr = new_id();
    const auto word = new_id();
    const auto inc_word = new_id();
    emit(op_load, {id_v3u32, gid, id_global_invocation_id});
    emit(op_composite_extract, {id_u32, index, gid, 0});
    emit(op_access_chain, {id_ptr_u32, state_ptr, id_state, constants[0], index});
    emit(op_access_chain, {id_ptr_u32, inc_ptr, id_inc, constants[0], index});
    emit(op_access_chain, {id_ptr_u32, out_ptr, id_out, constants[0], index});
    emit(op_load, {id_u32, word, state_ptr});
    emit(op_load, {id_u32, inc_word, inc_ptr});
    std::uint32_t packed = constants[0];
    for (std::uint32_t lane = 0; lane < 4; ++lane) {
        const std::uint32_t shift = lane * 6;
        const auto shifted_word = new_id();
        const auto code = new_id();
        const auto shifted_inc = new_id();
        const auto inc = new_id();
        const auto updated = new_id();
        const auto packed_lane = new_id();
        const auto next_packed = new_id();
        emit(op_shift_right_logical, {id_u32, shifted_word, word, constants[shift]});
        emit(op_bitwise_and, {id_u32, code, shifted_word, constants[63]});
        emit(op_shift_right_logical, {id_u32, shifted_inc, inc_word, constants[shift]});
        emit(op_bitwise_and, {id_u32, inc, shifted_inc, constants[1]});
        emit(op_iadd, {id_u32, updated, code, inc});
        emit(op_shift_left_logical, {id_u32, packed_lane, updated, constants[shift]});
        emit(op_bitwise_or, {id_u32, next_packed, packed, packed_lane});
        packed = next_packed;
    }
    emit(op_store, {out_ptr, packed});

    std::vector<std::uint32_t> words;
    words.reserve(180 + body.size());
    words.push_back(0x07230203);
    words.push_back(0x00010000);
    words.push_back(0);
    words.push_back(next_id);
    words.push_back(0);
    append_spirv_inst(words, op_capability, {capability_shader});
    append_spirv_inst(words, op_memory_model, {addressing_model_logical, memory_model_glsl450});
    words.push_back((6u << 16u) | op_entry_point);
    words.push_back(execution_model_gl_compute);
    words.push_back(id_main);
    append_spirv_string(words, "main");
    words.push_back(id_global_invocation_id);
    append_spirv_inst(words, op_execution_mode, {id_main, execution_mode_local_size, 1, 1, 1});
    append_spirv_inst(words, op_decorate,
                      {id_global_invocation_id, decoration_built_in, built_in_global_invocation_id});
    append_spirv_inst(words, op_decorate, {id_runtime_array, decoration_array_stride, 4});
    append_spirv_inst(words, op_member_decorate, {id_buffer_struct, 0, decoration_offset, 0});
    append_spirv_inst(words, op_decorate, {id_buffer_struct, decoration_buffer_block});
    append_spirv_inst(words, op_decorate, {id_state, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_state, decoration_binding, 0});
    append_spirv_inst(words, op_decorate, {id_inc, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_inc, decoration_binding, 1});
    append_spirv_inst(words, op_decorate, {id_out, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_out, decoration_binding, 2});
    append_spirv_inst(words, op_type_void, {id_void});
    append_spirv_inst(words, op_type_function, {id_fn, id_void});
    append_spirv_inst(words, op_type_int, {id_u32, 32, 0});
    append_spirv_inst(words, op_type_vector, {id_v3u32, id_u32, 3});
    append_spirv_inst(words, op_type_pointer, {id_ptr_input_v3u32, storage_class_input, id_v3u32});
    for (std::uint32_t value = 0; value < constants.size(); ++value) {
        append_spirv_inst(words, op_constant, {id_u32, constants[value], value});
    }
    append_spirv_inst(words, op_type_runtime_array, {id_runtime_array, id_u32});
    append_spirv_inst(words, op_type_struct, {id_buffer_struct, id_runtime_array});
    append_spirv_inst(words, op_type_pointer, {id_ptr_buffer, storage_class_uniform, id_buffer_struct});
    append_spirv_inst(words, op_type_pointer, {id_ptr_u32, storage_class_uniform, id_u32});
    append_spirv_inst(words, op_variable, {id_ptr_input_v3u32, id_global_invocation_id, storage_class_input});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_state, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_inc, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_buffer, id_out, storage_class_uniform});
    append_spirv_inst(words, op_function, {id_void, id_main, 0, id_fn});
    append_spirv_inst(words, op_label, {id_label});
    words.insert(words.end(), body.begin(), body.end());
    append_spirv_inst(words, op_return, {});
    append_spirv_inst(words, op_function_end, {});
    return words;
}

std::vector<std::uint32_t> counter_backward_input_fused_spirv(std::size_t in_features,
                                                              std::size_t out_features,
                                                              std::size_t C) {
    constexpr std::uint16_t op_capability = 17;
    constexpr std::uint16_t op_memory_model = 14;
    constexpr std::uint16_t op_entry_point = 15;
    constexpr std::uint16_t op_execution_mode = 16;
    constexpr std::uint16_t op_decorate = 71;
    constexpr std::uint16_t op_member_decorate = 72;
    constexpr std::uint16_t op_type_void = 19;
    constexpr std::uint16_t op_type_function = 33;
    constexpr std::uint16_t op_type_float = 22;
    constexpr std::uint16_t op_type_int = 21;
    constexpr std::uint16_t op_type_vector = 23;
    constexpr std::uint16_t op_constant = 43;
    constexpr std::uint16_t op_type_runtime_array = 29;
    constexpr std::uint16_t op_type_struct = 30;
    constexpr std::uint16_t op_type_pointer = 32;
    constexpr std::uint16_t op_variable = 59;
    constexpr std::uint16_t op_function = 54;
    constexpr std::uint16_t op_label = 248;
    constexpr std::uint16_t op_load = 61;
    constexpr std::uint16_t op_composite_extract = 81;
    constexpr std::uint16_t op_convert_u_to_f = 112;
    constexpr std::uint16_t op_iadd = 128;
    constexpr std::uint16_t op_fadd = 129;
    constexpr std::uint16_t op_fsub = 131;
    constexpr std::uint16_t op_imul = 132;
    constexpr std::uint16_t op_fmul = 133;
    constexpr std::uint16_t op_udiv = 134;
    constexpr std::uint16_t op_umod = 137;
    constexpr std::uint16_t op_access_chain = 65;
    constexpr std::uint16_t op_store = 62;
    constexpr std::uint16_t op_shift_right_logical = 194;
    constexpr std::uint16_t op_bitwise_and = 199;
    constexpr std::uint16_t op_return = 253;
    constexpr std::uint16_t op_function_end = 56;

    constexpr std::uint32_t capability_shader = 1;
    constexpr std::uint32_t addressing_model_logical = 0;
    constexpr std::uint32_t memory_model_glsl450 = 1;
    constexpr std::uint32_t execution_model_gl_compute = 5;
    constexpr std::uint32_t execution_mode_local_size = 17;
    constexpr std::uint32_t decoration_array_stride = 6;
    constexpr std::uint32_t decoration_offset = 35;
    constexpr std::uint32_t decoration_buffer_block = 3;
    constexpr std::uint32_t decoration_descriptor_set = 34;
    constexpr std::uint32_t decoration_binding = 33;
    constexpr std::uint32_t decoration_built_in = 11;
    constexpr std::uint32_t built_in_global_invocation_id = 28;
    constexpr std::uint32_t storage_class_input = 1;
    constexpr std::uint32_t storage_class_uniform = 2;

    const auto gpr = static_cast<std::uint32_t>(in_features / 4);
    const auto lv = static_cast<std::uint32_t>(2 * C - 1);
    std::uint32_t max_constant = static_cast<std::uint32_t>(in_features);
    max_constant = std::max(max_constant, static_cast<std::uint32_t>(out_features));
    max_constant = std::max(max_constant, gpr);
    max_constant = std::max(max_constant, lv);
    max_constant = std::max(max_constant, 63u);

    const std::uint32_t id_void = 1;
    const std::uint32_t id_fn = 2;
    const std::uint32_t id_main = 3;
    const std::uint32_t id_label = 4;
    const std::uint32_t id_f32 = 5;
    const std::uint32_t id_u32 = 6;
    const std::uint32_t id_v3u32 = 7;
    const std::uint32_t id_ptr_input_v3u32 = 8;
    const std::uint32_t id_global_invocation_id = 9;
    std::uint32_t next_id = 10;
    std::vector<std::uint32_t> constants(max_constant + 1);
    for (std::uint32_t value = 0; value <= max_constant; ++value) constants[value] = next_id++;
    const std::uint32_t id_one_f = next_id++;
    const std::uint32_t id_u32_runtime_array = next_id++;
    const std::uint32_t id_f32_runtime_array = next_id++;
    const std::uint32_t id_u32_buffer_struct = next_id++;
    const std::uint32_t id_f32_buffer_struct = next_id++;
    const std::uint32_t id_ptr_u32_buffer = next_id++;
    const std::uint32_t id_ptr_f32_buffer = next_id++;
    const std::uint32_t id_ptr_u32 = next_id++;
    const std::uint32_t id_ptr_f32 = next_id++;
    const std::uint32_t id_grad_out = next_id++;
    const std::uint32_t id_state = next_id++;
    const std::uint32_t id_scale = next_id++;
    const std::uint32_t id_grad_x = next_id++;
    auto new_id = [&]() { return next_id++; };

    std::vector<std::uint32_t> body;
    auto emit = [&](std::uint16_t opcode, const std::vector<std::uint32_t>& operands) {
        append_spirv_inst(body, opcode, operands);
    };

    const auto gid = new_id();
    const auto elem = new_id();
    const auto r = new_id();
    const auto i = new_id();
    const auto group = new_id();
    const auto lane = new_id();
    const auto shift = new_id();
    emit(op_load, {id_v3u32, gid, id_global_invocation_id});
    emit(op_composite_extract, {id_u32, elem, gid, 0});
    emit(op_udiv, {id_u32, r, elem, constants[static_cast<std::uint32_t>(in_features)]});
    emit(op_umod, {id_u32, i, elem, constants[static_cast<std::uint32_t>(in_features)]});
    emit(op_udiv, {id_u32, group, i, constants[4]});
    emit(op_umod, {id_u32, lane, i, constants[4]});
    emit(op_imul, {id_u32, shift, lane, constants[6]});

    std::uint32_t acc = 0;
    for (std::uint32_t o = 0; o < static_cast<std::uint32_t>(out_features); ++o) {
        const auto state_row_base = new_id();
        const auto state_index = new_id();
        const auto state_ptr = new_id();
        const auto word = new_id();
        const auto shifted = new_id();
        const auto code = new_id();
        const auto bucket = new_id();
        const auto bucket_f = new_id();
        const auto t_f = new_id();
        const auto grad_row_base = new_id();
        const auto grad_index = new_id();
        const auto grad_ptr = new_id();
        const auto grad_val = new_id();
        const auto scale_ptr = new_id();
        const auto scale_val = new_id();
        const auto grad_scale = new_id();
        const auto term = new_id();

        emit(op_imul, {id_u32, state_row_base, constants[o], constants[gpr]});
        emit(op_iadd, {id_u32, state_index, state_row_base, group});
        emit(op_access_chain, {id_ptr_u32, state_ptr, id_state, constants[0], state_index});
        emit(op_load, {id_u32, word, state_ptr});
        emit(op_shift_right_logical, {id_u32, shifted, word, shift});
        emit(op_bitwise_and, {id_u32, code, shifted, constants[63]});
        emit(op_udiv, {id_u32, bucket, code, constants[lv]});
        emit(op_convert_u_to_f, {id_f32, bucket_f, bucket});
        emit(op_fsub, {id_f32, t_f, bucket_f, id_one_f});

        emit(op_imul, {id_u32, grad_row_base, r, constants[static_cast<std::uint32_t>(out_features)]});
        emit(op_iadd, {id_u32, grad_index, grad_row_base, constants[o]});
        emit(op_access_chain, {id_ptr_f32, grad_ptr, id_grad_out, constants[0], grad_index});
        emit(op_load, {id_f32, grad_val, grad_ptr});
        emit(op_access_chain, {id_ptr_f32, scale_ptr, id_scale, constants[0], constants[o]});
        emit(op_load, {id_f32, scale_val, scale_ptr});
        emit(op_fmul, {id_f32, grad_scale, grad_val, scale_val});
        emit(op_fmul, {id_f32, term, grad_scale, t_f});
        if (acc == 0) {
            acc = term;
        } else {
            const auto sum = new_id();
            emit(op_fadd, {id_f32, sum, acc, term});
            acc = sum;
        }
    }

    const auto out_ptr = new_id();
    emit(op_access_chain, {id_ptr_f32, out_ptr, id_grad_x, constants[0], elem});
    emit(op_store, {out_ptr, acc});

    std::vector<std::uint32_t> words;
    words.reserve(260 + body.size());
    words.push_back(0x07230203);
    words.push_back(0x00010000);
    words.push_back(0);
    words.push_back(next_id);
    words.push_back(0);
    append_spirv_inst(words, op_capability, {capability_shader});
    append_spirv_inst(words, op_memory_model, {addressing_model_logical, memory_model_glsl450});
    words.push_back((6u << 16u) | op_entry_point);
    words.push_back(execution_model_gl_compute);
    words.push_back(id_main);
    append_spirv_string(words, "main");
    words.push_back(id_global_invocation_id);
    append_spirv_inst(words, op_execution_mode, {id_main, execution_mode_local_size, 1, 1, 1});
    append_spirv_inst(words, op_decorate,
                      {id_global_invocation_id, decoration_built_in, built_in_global_invocation_id});
    append_spirv_inst(words, op_decorate, {id_u32_runtime_array, decoration_array_stride, 4});
    append_spirv_inst(words, op_decorate, {id_f32_runtime_array, decoration_array_stride, 4});
    append_spirv_inst(words, op_member_decorate, {id_u32_buffer_struct, 0, decoration_offset, 0});
    append_spirv_inst(words, op_member_decorate, {id_f32_buffer_struct, 0, decoration_offset, 0});
    append_spirv_inst(words, op_decorate, {id_u32_buffer_struct, decoration_buffer_block});
    append_spirv_inst(words, op_decorate, {id_f32_buffer_struct, decoration_buffer_block});
    append_spirv_inst(words, op_decorate, {id_grad_out, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_grad_out, decoration_binding, 0});
    append_spirv_inst(words, op_decorate, {id_state, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_state, decoration_binding, 1});
    append_spirv_inst(words, op_decorate, {id_scale, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_scale, decoration_binding, 2});
    append_spirv_inst(words, op_decorate, {id_grad_x, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_grad_x, decoration_binding, 3});
    append_spirv_inst(words, op_type_void, {id_void});
    append_spirv_inst(words, op_type_function, {id_fn, id_void});
    append_spirv_inst(words, op_type_float, {id_f32, 32});
    append_spirv_inst(words, op_type_int, {id_u32, 32, 0});
    append_spirv_inst(words, op_type_vector, {id_v3u32, id_u32, 3});
    append_spirv_inst(words, op_type_pointer, {id_ptr_input_v3u32, storage_class_input, id_v3u32});
    for (std::uint32_t value = 0; value <= max_constant; ++value) {
        append_spirv_inst(words, op_constant, {id_u32, constants[value], value});
    }
    append_spirv_inst(words, op_constant, {id_f32, id_one_f, f32_bits(1.0f)});
    append_spirv_inst(words, op_type_runtime_array, {id_u32_runtime_array, id_u32});
    append_spirv_inst(words, op_type_runtime_array, {id_f32_runtime_array, id_f32});
    append_spirv_inst(words, op_type_struct, {id_u32_buffer_struct, id_u32_runtime_array});
    append_spirv_inst(words, op_type_struct, {id_f32_buffer_struct, id_f32_runtime_array});
    append_spirv_inst(words, op_type_pointer, {id_ptr_u32_buffer, storage_class_uniform, id_u32_buffer_struct});
    append_spirv_inst(words, op_type_pointer, {id_ptr_f32_buffer, storage_class_uniform, id_f32_buffer_struct});
    append_spirv_inst(words, op_type_pointer, {id_ptr_u32, storage_class_uniform, id_u32});
    append_spirv_inst(words, op_type_pointer, {id_ptr_f32, storage_class_uniform, id_f32});
    append_spirv_inst(words, op_variable, {id_ptr_input_v3u32, id_global_invocation_id, storage_class_input});
    append_spirv_inst(words, op_variable, {id_ptr_f32_buffer, id_grad_out, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_u32_buffer, id_state, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_f32_buffer, id_scale, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_f32_buffer, id_grad_x, storage_class_uniform});
    append_spirv_inst(words, op_function, {id_void, id_main, 0, id_fn});
    append_spirv_inst(words, op_label, {id_label});
    words.insert(words.end(), body.begin(), body.end());
    append_spirv_inst(words, op_return, {});
    append_spirv_inst(words, op_function_end, {});
    return words;
}

[[maybe_unused]] std::vector<std::uint32_t> counter_decode_weight_u8_spirv(std::size_t in_features,
                                                          std::size_t out_features,
                                                          std::size_t C) {
    constexpr std::uint16_t op_extension = 10;
    constexpr std::uint16_t op_capability = 17;
    constexpr std::uint16_t op_memory_model = 14;
    constexpr std::uint16_t op_entry_point = 15;
    constexpr std::uint16_t op_execution_mode = 16;
    constexpr std::uint16_t op_decorate = 71;
    constexpr std::uint16_t op_member_decorate = 72;
    constexpr std::uint16_t op_type_void = 19;
    constexpr std::uint16_t op_type_function = 33;
    constexpr std::uint16_t op_type_float = 22;
    constexpr std::uint16_t op_type_int = 21;
    constexpr std::uint16_t op_type_vector = 23;
    constexpr std::uint16_t op_constant = 43;
    constexpr std::uint16_t op_type_runtime_array = 29;
    constexpr std::uint16_t op_type_struct = 30;
    constexpr std::uint16_t op_type_pointer = 32;
    constexpr std::uint16_t op_variable = 59;
    constexpr std::uint16_t op_function = 54;
    constexpr std::uint16_t op_label = 248;
    constexpr std::uint16_t op_load = 61;
    constexpr std::uint16_t op_store = 62;
    constexpr std::uint16_t op_access_chain = 65;
    constexpr std::uint16_t op_composite_extract = 81;
    constexpr std::uint16_t op_convert_u_to_f = 112;
    constexpr std::uint16_t op_u_convert = 113;
    constexpr std::uint16_t op_iadd = 128;
    constexpr std::uint16_t op_fsub = 131;
    constexpr std::uint16_t op_imul = 132;
    constexpr std::uint16_t op_fmul = 133;
    constexpr std::uint16_t op_udiv = 134;
    constexpr std::uint16_t op_umod = 137;
    constexpr std::uint16_t op_shift_right_logical = 194;
    constexpr std::uint16_t op_shift_left_logical = 196;
    constexpr std::uint16_t op_bitwise_or = 197;
    constexpr std::uint16_t op_bitwise_and = 199;
    constexpr std::uint16_t op_return = 253;
    constexpr std::uint16_t op_function_end = 56;

    constexpr std::uint32_t capability_shader = 1;
    constexpr std::uint32_t capability_int8 = 39;
    constexpr std::uint32_t capability_uniform_and_storage_buffer_8bit_access = 4449;
    constexpr std::uint32_t addressing_model_logical = 0;
    constexpr std::uint32_t memory_model_glsl450 = 1;
    constexpr std::uint32_t execution_model_gl_compute = 5;
    constexpr std::uint32_t execution_mode_local_size = 17;
    constexpr std::uint32_t decoration_array_stride = 6;
    constexpr std::uint32_t decoration_offset = 35;
    constexpr std::uint32_t decoration_buffer_block = 3;
    constexpr std::uint32_t decoration_descriptor_set = 34;
    constexpr std::uint32_t decoration_binding = 33;
    constexpr std::uint32_t decoration_built_in = 11;
    constexpr std::uint32_t built_in_global_invocation_id = 28;
    constexpr std::uint32_t storage_class_input = 1;
    constexpr std::uint32_t storage_class_uniform = 2;

    const auto gpr = static_cast<std::uint32_t>(in_features / 4);
    const auto lv = static_cast<std::uint32_t>(2 * C - 1);
    std::uint32_t max_constant = static_cast<std::uint32_t>(in_features);
    max_constant = std::max(max_constant, static_cast<std::uint32_t>(out_features));
    max_constant = std::max(max_constant, gpr);
    max_constant = std::max(max_constant, lv);
    max_constant = std::max(max_constant, 63u);

    const std::uint32_t id_void = 1;
    const std::uint32_t id_fn = 2;
    const std::uint32_t id_main = 3;
    const std::uint32_t id_label = 4;
    const std::uint32_t id_f32 = 5;
    const std::uint32_t id_u32 = 6;
    const std::uint32_t id_u8 = 7;
    const std::uint32_t id_v3u32 = 8;
    const std::uint32_t id_ptr_input_v3u32 = 9;
    const std::uint32_t id_global_invocation_id = 10;
    std::uint32_t next_id = 11;
    std::vector<std::uint32_t> constants(max_constant + 1);
    for (std::uint32_t value = 0; value <= max_constant; ++value) constants[value] = next_id++;
    const std::uint32_t id_one_f = next_id++;
    const std::uint32_t id_u8_runtime_array = next_id++;
    const std::uint32_t id_f32_runtime_array = next_id++;
    const std::uint32_t id_u8_buffer_struct = next_id++;
    const std::uint32_t id_f32_buffer_struct = next_id++;
    const std::uint32_t id_ptr_u8_buffer = next_id++;
    const std::uint32_t id_ptr_f32_buffer = next_id++;
    const std::uint32_t id_ptr_u8 = next_id++;
    const std::uint32_t id_ptr_f32 = next_id++;
    const std::uint32_t id_state = next_id++;
    const std::uint32_t id_scale = next_id++;
    const std::uint32_t id_weight = next_id++;
    auto new_id = [&]() { return next_id++; };

    std::vector<std::uint32_t> body;
    auto emit = [&](std::uint16_t opcode, const std::vector<std::uint32_t>& operands) {
        append_spirv_inst(body, opcode, operands);
    };

    const auto gid = new_id();
    const auto elem = new_id();
    const auto row = new_id();
    const auto i = new_id();
    const auto group = new_id();
    const auto lane = new_id();
    const auto shift = new_id();
    const auto row_base = new_id();
    const auto group_index = new_id();
    const auto byte_index = new_id();
    const auto byte_index_1 = new_id();
    const auto byte_index_2 = new_id();
    const auto b0_ptr = new_id();
    const auto b1_ptr = new_id();
    const auto b2_ptr = new_id();
    const auto b0 = new_id();
    const auto b1 = new_id();
    const auto b2 = new_id();
    const auto b0u = new_id();
    const auto b1u = new_id();
    const auto b2u = new_id();
    const auto b1s = new_id();
    const auto b2s = new_id();
    const auto w01 = new_id();
    const auto word = new_id();
    const auto shifted = new_id();
    const auto code = new_id();
    const auto bucket = new_id();
    const auto bucket_f = new_id();
    const auto t_f = new_id();
    const auto scale_ptr = new_id();
    const auto scale_val = new_id();
    const auto out_val = new_id();
    const auto out_ptr = new_id();

    emit(op_load, {id_v3u32, gid, id_global_invocation_id});
    emit(op_composite_extract, {id_u32, elem, gid, 0});
    emit(op_udiv, {id_u32, row, elem, constants[static_cast<std::uint32_t>(in_features)]});
    emit(op_umod, {id_u32, i, elem, constants[static_cast<std::uint32_t>(in_features)]});
    emit(op_udiv, {id_u32, group, i, constants[4]});
    emit(op_umod, {id_u32, lane, i, constants[4]});
    emit(op_imul, {id_u32, shift, lane, constants[6]});
    emit(op_imul, {id_u32, row_base, row, constants[gpr]});
    emit(op_iadd, {id_u32, group_index, row_base, group});
    emit(op_imul, {id_u32, byte_index, group_index, constants[3]});
    emit(op_iadd, {id_u32, byte_index_1, byte_index, constants[1]});
    emit(op_iadd, {id_u32, byte_index_2, byte_index, constants[2]});
    emit(op_access_chain, {id_ptr_u8, b0_ptr, id_state, constants[0], byte_index});
    emit(op_access_chain, {id_ptr_u8, b1_ptr, id_state, constants[0], byte_index_1});
    emit(op_access_chain, {id_ptr_u8, b2_ptr, id_state, constants[0], byte_index_2});
    emit(op_load, {id_u8, b0, b0_ptr});
    emit(op_load, {id_u8, b1, b1_ptr});
    emit(op_load, {id_u8, b2, b2_ptr});
    emit(op_u_convert, {id_u32, b0u, b0});
    emit(op_u_convert, {id_u32, b1u, b1});
    emit(op_u_convert, {id_u32, b2u, b2});
    emit(op_shift_left_logical, {id_u32, b1s, b1u, constants[8]});
    emit(op_shift_left_logical, {id_u32, b2s, b2u, constants[16]});
    emit(op_bitwise_or, {id_u32, w01, b0u, b1s});
    emit(op_bitwise_or, {id_u32, word, w01, b2s});
    emit(op_shift_right_logical, {id_u32, shifted, word, shift});
    emit(op_bitwise_and, {id_u32, code, shifted, constants[63]});
    emit(op_udiv, {id_u32, bucket, code, constants[lv]});
    emit(op_convert_u_to_f, {id_f32, bucket_f, bucket});
    emit(op_fsub, {id_f32, t_f, bucket_f, id_one_f});
    emit(op_access_chain, {id_ptr_f32, scale_ptr, id_scale, constants[0], row});
    emit(op_load, {id_f32, scale_val, scale_ptr});
    emit(op_fmul, {id_f32, out_val, scale_val, t_f});
    emit(op_access_chain, {id_ptr_f32, out_ptr, id_weight, constants[0], elem});
    emit(op_store, {out_ptr, out_val});

    std::vector<std::uint32_t> words;
    words.reserve(260 + body.size());
    words.push_back(0x07230203);
    words.push_back(0x00010000);
    words.push_back(0);
    words.push_back(next_id);
    words.push_back(0);
    append_spirv_inst(words, op_capability, {capability_shader});
    append_spirv_inst(words, op_capability, {capability_int8});
    append_spirv_inst(words, op_capability, {capability_uniform_and_storage_buffer_8bit_access});
    append_spirv_string_inst(words, op_extension, "SPV_KHR_8bit_storage");
    append_spirv_inst(words, op_memory_model, {addressing_model_logical, memory_model_glsl450});
    words.push_back((6u << 16u) | op_entry_point);
    words.push_back(execution_model_gl_compute);
    words.push_back(id_main);
    append_spirv_string(words, "main");
    words.push_back(id_global_invocation_id);
    append_spirv_inst(words, op_execution_mode, {id_main, execution_mode_local_size, 1, 1, 1});
    append_spirv_inst(words, op_decorate,
                      {id_global_invocation_id, decoration_built_in, built_in_global_invocation_id});
    append_spirv_inst(words, op_decorate, {id_u8_runtime_array, decoration_array_stride, 1});
    append_spirv_inst(words, op_decorate, {id_f32_runtime_array, decoration_array_stride, 4});
    append_spirv_inst(words, op_member_decorate, {id_u8_buffer_struct, 0, decoration_offset, 0});
    append_spirv_inst(words, op_member_decorate, {id_f32_buffer_struct, 0, decoration_offset, 0});
    append_spirv_inst(words, op_decorate, {id_u8_buffer_struct, decoration_buffer_block});
    append_spirv_inst(words, op_decorate, {id_f32_buffer_struct, decoration_buffer_block});
    append_spirv_inst(words, op_decorate, {id_state, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_state, decoration_binding, 0});
    append_spirv_inst(words, op_decorate, {id_scale, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_scale, decoration_binding, 1});
    append_spirv_inst(words, op_decorate, {id_weight, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_weight, decoration_binding, 2});
    append_spirv_inst(words, op_type_void, {id_void});
    append_spirv_inst(words, op_type_function, {id_fn, id_void});
    append_spirv_inst(words, op_type_float, {id_f32, 32});
    append_spirv_inst(words, op_type_int, {id_u32, 32, 0});
    append_spirv_inst(words, op_type_int, {id_u8, 8, 0});
    append_spirv_inst(words, op_type_vector, {id_v3u32, id_u32, 3});
    append_spirv_inst(words, op_type_pointer, {id_ptr_input_v3u32, storage_class_input, id_v3u32});
    for (std::uint32_t value = 0; value <= max_constant; ++value) {
        append_spirv_inst(words, op_constant, {id_u32, constants[value], value});
    }
    append_spirv_inst(words, op_constant, {id_f32, id_one_f, f32_bits(1.0f)});
    append_spirv_inst(words, op_type_runtime_array, {id_u8_runtime_array, id_u8});
    append_spirv_inst(words, op_type_runtime_array, {id_f32_runtime_array, id_f32});
    append_spirv_inst(words, op_type_struct, {id_u8_buffer_struct, id_u8_runtime_array});
    append_spirv_inst(words, op_type_struct, {id_f32_buffer_struct, id_f32_runtime_array});
    append_spirv_inst(words, op_type_pointer, {id_ptr_u8_buffer, storage_class_uniform, id_u8_buffer_struct});
    append_spirv_inst(words, op_type_pointer, {id_ptr_f32_buffer, storage_class_uniform, id_f32_buffer_struct});
    append_spirv_inst(words, op_type_pointer, {id_ptr_u8, storage_class_uniform, id_u8});
    append_spirv_inst(words, op_type_pointer, {id_ptr_f32, storage_class_uniform, id_f32});
    append_spirv_inst(words, op_variable, {id_ptr_input_v3u32, id_global_invocation_id, storage_class_input});
    append_spirv_inst(words, op_variable, {id_ptr_u8_buffer, id_state, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_f32_buffer, id_scale, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_f32_buffer, id_weight, storage_class_uniform});
    append_spirv_inst(words, op_function, {id_void, id_main, 0, id_fn});
    append_spirv_inst(words, op_label, {id_label});
    words.insert(words.end(), body.begin(), body.end());
    append_spirv_inst(words, op_return, {});
    append_spirv_inst(words, op_function_end, {});
    return words;
}

[[maybe_unused]] std::vector<std::uint32_t> counter_backward_input_u8_spirv(std::size_t in_features,
                                                           std::size_t out_features,
                                                           std::size_t C) {
    constexpr std::uint16_t op_extension = 10;
    constexpr std::uint16_t op_capability = 17;
    constexpr std::uint16_t op_memory_model = 14;
    constexpr std::uint16_t op_entry_point = 15;
    constexpr std::uint16_t op_execution_mode = 16;
    constexpr std::uint16_t op_decorate = 71;
    constexpr std::uint16_t op_member_decorate = 72;
    constexpr std::uint16_t op_type_void = 19;
    constexpr std::uint16_t op_type_function = 33;
    constexpr std::uint16_t op_type_float = 22;
    constexpr std::uint16_t op_type_int = 21;
    constexpr std::uint16_t op_type_vector = 23;
    constexpr std::uint16_t op_constant = 43;
    constexpr std::uint16_t op_type_runtime_array = 29;
    constexpr std::uint16_t op_type_struct = 30;
    constexpr std::uint16_t op_type_pointer = 32;
    constexpr std::uint16_t op_variable = 59;
    constexpr std::uint16_t op_function = 54;
    constexpr std::uint16_t op_label = 248;
    constexpr std::uint16_t op_load = 61;
    constexpr std::uint16_t op_store = 62;
    constexpr std::uint16_t op_access_chain = 65;
    constexpr std::uint16_t op_composite_extract = 81;
    constexpr std::uint16_t op_convert_u_to_f = 112;
    constexpr std::uint16_t op_u_convert = 113;
    constexpr std::uint16_t op_iadd = 128;
    constexpr std::uint16_t op_fadd = 129;
    constexpr std::uint16_t op_fsub = 131;
    constexpr std::uint16_t op_imul = 132;
    constexpr std::uint16_t op_fmul = 133;
    constexpr std::uint16_t op_udiv = 134;
    constexpr std::uint16_t op_umod = 137;
    constexpr std::uint16_t op_shift_right_logical = 194;
    constexpr std::uint16_t op_shift_left_logical = 196;
    constexpr std::uint16_t op_bitwise_or = 197;
    constexpr std::uint16_t op_bitwise_and = 199;
    constexpr std::uint16_t op_return = 253;
    constexpr std::uint16_t op_function_end = 56;

    constexpr std::uint32_t capability_shader = 1;
    constexpr std::uint32_t capability_int8 = 39;
    constexpr std::uint32_t capability_uniform_and_storage_buffer_8bit_access = 4449;
    constexpr std::uint32_t addressing_model_logical = 0;
    constexpr std::uint32_t memory_model_glsl450 = 1;
    constexpr std::uint32_t execution_model_gl_compute = 5;
    constexpr std::uint32_t execution_mode_local_size = 17;
    constexpr std::uint32_t decoration_array_stride = 6;
    constexpr std::uint32_t decoration_offset = 35;
    constexpr std::uint32_t decoration_buffer_block = 3;
    constexpr std::uint32_t decoration_descriptor_set = 34;
    constexpr std::uint32_t decoration_binding = 33;
    constexpr std::uint32_t decoration_built_in = 11;
    constexpr std::uint32_t built_in_global_invocation_id = 28;
    constexpr std::uint32_t storage_class_input = 1;
    constexpr std::uint32_t storage_class_uniform = 2;

    const auto gpr = static_cast<std::uint32_t>(in_features / 4);
    const auto lv = static_cast<std::uint32_t>(2 * C - 1);
    std::uint32_t max_constant = static_cast<std::uint32_t>(in_features);
    max_constant = std::max(max_constant, static_cast<std::uint32_t>(out_features));
    max_constant = std::max(max_constant, gpr);
    max_constant = std::max(max_constant, lv);
    max_constant = std::max(max_constant, 63u);

    const std::uint32_t id_void = 1;
    const std::uint32_t id_fn = 2;
    const std::uint32_t id_main = 3;
    const std::uint32_t id_label = 4;
    const std::uint32_t id_f32 = 5;
    const std::uint32_t id_u32 = 6;
    const std::uint32_t id_u8 = 7;
    const std::uint32_t id_v3u32 = 8;
    const std::uint32_t id_ptr_input_v3u32 = 9;
    const std::uint32_t id_global_invocation_id = 10;
    std::uint32_t next_id = 11;
    std::vector<std::uint32_t> constants(max_constant + 1);
    for (std::uint32_t value = 0; value <= max_constant; ++value) constants[value] = next_id++;
    const std::uint32_t id_one_f = next_id++;
    const std::uint32_t id_u8_runtime_array = next_id++;
    const std::uint32_t id_f32_runtime_array = next_id++;
    const std::uint32_t id_u8_buffer_struct = next_id++;
    const std::uint32_t id_f32_buffer_struct = next_id++;
    const std::uint32_t id_ptr_u8_buffer = next_id++;
    const std::uint32_t id_ptr_f32_buffer = next_id++;
    const std::uint32_t id_ptr_u8 = next_id++;
    const std::uint32_t id_ptr_f32 = next_id++;
    const std::uint32_t id_grad_out = next_id++;
    const std::uint32_t id_state = next_id++;
    const std::uint32_t id_scale = next_id++;
    const std::uint32_t id_grad_x = next_id++;
    auto new_id = [&]() { return next_id++; };

    std::vector<std::uint32_t> body;
    auto emit = [&](std::uint16_t opcode, const std::vector<std::uint32_t>& operands) {
        append_spirv_inst(body, opcode, operands);
    };

    const auto gid = new_id();
    const auto elem = new_id();
    const auto r = new_id();
    const auto i = new_id();
    const auto group = new_id();
    const auto lane = new_id();
    const auto shift = new_id();
    emit(op_load, {id_v3u32, gid, id_global_invocation_id});
    emit(op_composite_extract, {id_u32, elem, gid, 0});
    emit(op_udiv, {id_u32, r, elem, constants[static_cast<std::uint32_t>(in_features)]});
    emit(op_umod, {id_u32, i, elem, constants[static_cast<std::uint32_t>(in_features)]});
    emit(op_udiv, {id_u32, group, i, constants[4]});
    emit(op_umod, {id_u32, lane, i, constants[4]});
    emit(op_imul, {id_u32, shift, lane, constants[6]});

    std::uint32_t acc = 0;
    for (std::uint32_t o = 0; o < static_cast<std::uint32_t>(out_features); ++o) {
        const auto state_row_base = new_id();
        const auto state_group_index = new_id();
        const auto byte_index = new_id();
        const auto byte_index_1 = new_id();
        const auto byte_index_2 = new_id();
        const auto b0_ptr = new_id();
        const auto b1_ptr = new_id();
        const auto b2_ptr = new_id();
        const auto b0 = new_id();
        const auto b1 = new_id();
        const auto b2 = new_id();
        const auto b0u = new_id();
        const auto b1u = new_id();
        const auto b2u = new_id();
        const auto b1s = new_id();
        const auto b2s = new_id();
        const auto w01 = new_id();
        const auto word = new_id();
        const auto shifted = new_id();
        const auto code = new_id();
        const auto bucket = new_id();
        const auto bucket_f = new_id();
        const auto t_f = new_id();
        const auto grad_row_base = new_id();
        const auto grad_index = new_id();
        const auto grad_ptr = new_id();
        const auto grad_val = new_id();
        const auto scale_ptr = new_id();
        const auto scale_val = new_id();
        const auto grad_scale = new_id();
        const auto term = new_id();

        emit(op_imul, {id_u32, state_row_base, constants[o], constants[gpr]});
        emit(op_iadd, {id_u32, state_group_index, state_row_base, group});
        emit(op_imul, {id_u32, byte_index, state_group_index, constants[3]});
        emit(op_iadd, {id_u32, byte_index_1, byte_index, constants[1]});
        emit(op_iadd, {id_u32, byte_index_2, byte_index, constants[2]});
        emit(op_access_chain, {id_ptr_u8, b0_ptr, id_state, constants[0], byte_index});
        emit(op_access_chain, {id_ptr_u8, b1_ptr, id_state, constants[0], byte_index_1});
        emit(op_access_chain, {id_ptr_u8, b2_ptr, id_state, constants[0], byte_index_2});
        emit(op_load, {id_u8, b0, b0_ptr});
        emit(op_load, {id_u8, b1, b1_ptr});
        emit(op_load, {id_u8, b2, b2_ptr});
        emit(op_u_convert, {id_u32, b0u, b0});
        emit(op_u_convert, {id_u32, b1u, b1});
        emit(op_u_convert, {id_u32, b2u, b2});
        emit(op_shift_left_logical, {id_u32, b1s, b1u, constants[8]});
        emit(op_shift_left_logical, {id_u32, b2s, b2u, constants[16]});
        emit(op_bitwise_or, {id_u32, w01, b0u, b1s});
        emit(op_bitwise_or, {id_u32, word, w01, b2s});
        emit(op_shift_right_logical, {id_u32, shifted, word, shift});
        emit(op_bitwise_and, {id_u32, code, shifted, constants[63]});
        emit(op_udiv, {id_u32, bucket, code, constants[lv]});
        emit(op_convert_u_to_f, {id_f32, bucket_f, bucket});
        emit(op_fsub, {id_f32, t_f, bucket_f, id_one_f});

        emit(op_imul, {id_u32, grad_row_base, r, constants[static_cast<std::uint32_t>(out_features)]});
        emit(op_iadd, {id_u32, grad_index, grad_row_base, constants[o]});
        emit(op_access_chain, {id_ptr_f32, grad_ptr, id_grad_out, constants[0], grad_index});
        emit(op_load, {id_f32, grad_val, grad_ptr});
        emit(op_access_chain, {id_ptr_f32, scale_ptr, id_scale, constants[0], constants[o]});
        emit(op_load, {id_f32, scale_val, scale_ptr});
        emit(op_fmul, {id_f32, grad_scale, grad_val, scale_val});
        emit(op_fmul, {id_f32, term, grad_scale, t_f});
        if (acc == 0) {
            acc = term;
        } else {
            const auto sum = new_id();
            emit(op_fadd, {id_f32, sum, acc, term});
            acc = sum;
        }
    }

    const auto out_ptr = new_id();
    emit(op_access_chain, {id_ptr_f32, out_ptr, id_grad_x, constants[0], elem});
    emit(op_store, {out_ptr, acc});

    std::vector<std::uint32_t> words;
    words.reserve(300 + body.size());
    words.push_back(0x07230203);
    words.push_back(0x00010000);
    words.push_back(0);
    words.push_back(next_id);
    words.push_back(0);
    append_spirv_inst(words, op_capability, {capability_shader});
    append_spirv_inst(words, op_capability, {capability_int8});
    append_spirv_inst(words, op_capability, {capability_uniform_and_storage_buffer_8bit_access});
    append_spirv_string_inst(words, op_extension, "SPV_KHR_8bit_storage");
    append_spirv_inst(words, op_memory_model, {addressing_model_logical, memory_model_glsl450});
    words.push_back((6u << 16u) | op_entry_point);
    words.push_back(execution_model_gl_compute);
    words.push_back(id_main);
    append_spirv_string(words, "main");
    words.push_back(id_global_invocation_id);
    append_spirv_inst(words, op_execution_mode, {id_main, execution_mode_local_size, 1, 1, 1});
    append_spirv_inst(words, op_decorate,
                      {id_global_invocation_id, decoration_built_in, built_in_global_invocation_id});
    append_spirv_inst(words, op_decorate, {id_u8_runtime_array, decoration_array_stride, 1});
    append_spirv_inst(words, op_decorate, {id_f32_runtime_array, decoration_array_stride, 4});
    append_spirv_inst(words, op_member_decorate, {id_u8_buffer_struct, 0, decoration_offset, 0});
    append_spirv_inst(words, op_member_decorate, {id_f32_buffer_struct, 0, decoration_offset, 0});
    append_spirv_inst(words, op_decorate, {id_u8_buffer_struct, decoration_buffer_block});
    append_spirv_inst(words, op_decorate, {id_f32_buffer_struct, decoration_buffer_block});
    append_spirv_inst(words, op_decorate, {id_grad_out, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_grad_out, decoration_binding, 0});
    append_spirv_inst(words, op_decorate, {id_state, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_state, decoration_binding, 1});
    append_spirv_inst(words, op_decorate, {id_scale, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_scale, decoration_binding, 2});
    append_spirv_inst(words, op_decorate, {id_grad_x, decoration_descriptor_set, 0});
    append_spirv_inst(words, op_decorate, {id_grad_x, decoration_binding, 3});
    append_spirv_inst(words, op_type_void, {id_void});
    append_spirv_inst(words, op_type_function, {id_fn, id_void});
    append_spirv_inst(words, op_type_float, {id_f32, 32});
    append_spirv_inst(words, op_type_int, {id_u32, 32, 0});
    append_spirv_inst(words, op_type_int, {id_u8, 8, 0});
    append_spirv_inst(words, op_type_vector, {id_v3u32, id_u32, 3});
    append_spirv_inst(words, op_type_pointer, {id_ptr_input_v3u32, storage_class_input, id_v3u32});
    for (std::uint32_t value = 0; value <= max_constant; ++value) {
        append_spirv_inst(words, op_constant, {id_u32, constants[value], value});
    }
    append_spirv_inst(words, op_constant, {id_f32, id_one_f, f32_bits(1.0f)});
    append_spirv_inst(words, op_type_runtime_array, {id_u8_runtime_array, id_u8});
    append_spirv_inst(words, op_type_runtime_array, {id_f32_runtime_array, id_f32});
    append_spirv_inst(words, op_type_struct, {id_u8_buffer_struct, id_u8_runtime_array});
    append_spirv_inst(words, op_type_struct, {id_f32_buffer_struct, id_f32_runtime_array});
    append_spirv_inst(words, op_type_pointer, {id_ptr_u8_buffer, storage_class_uniform, id_u8_buffer_struct});
    append_spirv_inst(words, op_type_pointer, {id_ptr_f32_buffer, storage_class_uniform, id_f32_buffer_struct});
    append_spirv_inst(words, op_type_pointer, {id_ptr_u8, storage_class_uniform, id_u8});
    append_spirv_inst(words, op_type_pointer, {id_ptr_f32, storage_class_uniform, id_f32});
    append_spirv_inst(words, op_variable, {id_ptr_input_v3u32, id_global_invocation_id, storage_class_input});
    append_spirv_inst(words, op_variable, {id_ptr_f32_buffer, id_grad_out, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_u8_buffer, id_state, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_f32_buffer, id_scale, storage_class_uniform});
    append_spirv_inst(words, op_variable, {id_ptr_f32_buffer, id_grad_x, storage_class_uniform});
    append_spirv_inst(words, op_function, {id_void, id_main, 0, id_fn});
    append_spirv_inst(words, op_label, {id_label});
    words.insert(words.end(), body.begin(), body.end());
    append_spirv_inst(words, op_return, {});
    append_spirv_inst(words, op_function_end, {});
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

VulkanOpResult run_vulkan_embedding_gather(VulkanRuntime& runtime,
                                           const VulkanBuffer& weight,
                                           const VulkanBuffer& indices,
                                           VulkanBuffer& out,
                                           std::size_t vocab_size,
                                           std::size_t embed_dim,
                                           std::size_t token_count) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (vocab_size == 0 || embed_dim == 0 || token_count == 0)
        return fail("Vulkan embedding gather requires non-zero shapes");
    // Push-constant `n = token_count * embed_dim` is int32; reject overflow.
    constexpr std::size_t kMaxInt32 = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
    if (token_count > kMaxInt32 / embed_dim) return fail("Vulkan embedding gather token_count*embed_dim overflows int32 push constant");
    const auto weight_bytes = vocab_size * embed_dim * sizeof(float);
    const auto idx_bytes = token_count * sizeof(std::int32_t);
    const auto out_bytes = token_count * embed_dim * sizeof(float);
    if (weight.nbytes() < weight_bytes) return fail("Vulkan embedding gather weight buffer too small");
    if (indices.nbytes() < idx_bytes) return fail("Vulkan embedding gather indices buffer too small");
    if (out.nbytes() < out_bytes) return fail("Vulkan embedding gather output buffer too small");

    const struct {
        std::int32_t vocab_size;
        std::int32_t embed_dim;
        std::int32_t n;
        std::int32_t _pad;
    } push{static_cast<std::int32_t>(vocab_size),
           static_cast<std::int32_t>(embed_dim),
           static_cast<std::int32_t>(token_count * embed_dim),
           0};
    const std::vector<const VulkanBuffer*> buffers = {&weight, &indices, &out};
    const std::size_t total = token_count * embed_dim;
    return runtime.dispatch_cached(vkspirv::k_embedding_gather_f32_i32,
                                   vkspirv::k_embedding_gather_f32_i32_words,
                                   buffers, &push, sizeof(push),
                                   static_cast<std::uint32_t>((total + 63) / 64), 1, 1);
}

VulkanOpResult run_vulkan_embedding_weight_backward(VulkanRuntime& runtime,
                                                    const VulkanBuffer& indices,
                                                    const VulkanBuffer& grad_out,
                                                    VulkanBuffer& grad_weight,
                                                    std::size_t vocab_size,
                                                    std::size_t embed_dim,
                                                    std::size_t token_count) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (vocab_size == 0 || embed_dim == 0 || token_count == 0)
        return fail("Vulkan embedding weight backward requires non-zero shapes");
    // Push-constant `n = vocab_size * embed_dim` is int32; reject overflow.
    constexpr std::size_t kMaxInt32 = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
    if (vocab_size > kMaxInt32 / embed_dim) return fail("Vulkan embedding weight backward vocab_size*embed_dim overflows int32 push constant");
    const auto idx_bytes = token_count * sizeof(std::int32_t);
    const auto grad_bytes = token_count * embed_dim * sizeof(float);
    const auto out_bytes = vocab_size * embed_dim * sizeof(float);
    if (indices.nbytes() < idx_bytes) return fail("Vulkan embedding weight backward indices buffer too small");
    if (grad_out.nbytes() < grad_bytes) return fail("Vulkan embedding weight backward grad buffer too small");
    if (grad_weight.nbytes() < out_bytes) return fail("Vulkan embedding weight backward output buffer too small");

    const struct {
        std::int32_t vocab_size;
        std::int32_t embed_dim;
        std::int32_t token_count;
        std::int32_t n;
    } push{static_cast<std::int32_t>(vocab_size),
           static_cast<std::int32_t>(embed_dim),
           static_cast<std::int32_t>(token_count),
           static_cast<std::int32_t>(vocab_size * embed_dim)};
    const std::vector<const VulkanBuffer*> buffers = {&indices, &grad_out, &grad_weight};
    const std::size_t total = vocab_size * embed_dim;
    return runtime.dispatch_cached(vkspirv::k_embedding_weight_backward_f32_i32,
                                   vkspirv::k_embedding_weight_backward_f32_i32_words,
                                   buffers, &push, sizeof(push),
                                   static_cast<std::uint32_t>((total + 63) / 64), 1, 1);
}

VulkanOpResult run_vulkan_zero_f32(VulkanRuntime& runtime,
                                   VulkanBuffer& out,
                                   std::size_t elements) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (elements == 0) return fail("Vulkan zero_f32 requires non-zero element count");
    const auto nbytes = elements * sizeof(float);
    if (out.nbytes() < nbytes) return fail("Vulkan zero_f32 output buffer too small");
    const struct {
        std::uint32_t n;
    } push{static_cast<std::uint32_t>(elements)};
    const std::vector<const VulkanBuffer*> buffers = {&out};
    return runtime.dispatch_cached(vkspirv::k_zero_f32, vkspirv::k_zero_f32_words, buffers, &push,
                                   sizeof(push), static_cast<std::uint32_t>((elements + 63) / 64), 1, 1);
}

VulkanOpResult run_vulkan_embedding_weight_backward_scatter(VulkanRuntime& runtime,
                                                            const VulkanBuffer& indices,
                                                            const VulkanBuffer& grad_out,
                                                            VulkanBuffer& grad_weight,
                                                            std::size_t vocab_size,
                                                            std::size_t embed_dim,
                                                            std::size_t token_count) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (vocab_size == 0 || embed_dim == 0 || token_count == 0)
        return fail("Vulkan embedding weight backward scatter requires non-zero shapes");
    const auto idx_bytes = token_count * sizeof(std::int32_t);
    const auto grad_bytes = token_count * embed_dim * sizeof(float);
    const auto out_bytes = vocab_size * embed_dim * sizeof(float);
    if (indices.nbytes() < idx_bytes) return fail("Vulkan embedding weight backward scatter indices buffer too small");
    if (grad_out.nbytes() < grad_bytes) return fail("Vulkan embedding weight backward scatter grad buffer too small");
    if (grad_weight.nbytes() < out_bytes) return fail("Vulkan embedding weight backward scatter output buffer too small");
    // Caller MUST zero-fill grad_weight before calling (atomicAdd accumulates).
    constexpr std::size_t kMaxInt32 = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
    if (token_count > kMaxInt32 / embed_dim) return fail("Vulkan embedding weight backward scatter token_count*embed_dim overflows int32 push constant");
    const struct {
        std::int32_t vocab_size;
        std::int32_t embed_dim;
        std::int32_t token_count;
    } push{static_cast<std::int32_t>(vocab_size),
           static_cast<std::int32_t>(embed_dim),
           static_cast<std::int32_t>(token_count)};
    const std::vector<const VulkanBuffer*> buffers = {&indices, &grad_out, &grad_weight};
    const std::size_t total = token_count * embed_dim;
    return runtime.dispatch_cached(vkspirv::k_embedding_weight_backward_f32_i32_scatter,
                                   vkspirv::k_embedding_weight_backward_f32_i32_scatter_words,
                                   buffers, &push, sizeof(push),
                                   static_cast<std::uint32_t>((total + 63) / 64), 1, 1);
}

VulkanOpResult run_vulkan_embedding_weight_backward_scatter_cas(VulkanRuntime& runtime,
                                                                const VulkanBuffer& indices,
                                                                const VulkanBuffer& grad_out,
                                                                VulkanBuffer& grad_weight,
                                                                std::size_t vocab_size,
                                                                std::size_t embed_dim,
                                                                std::size_t token_count) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (vocab_size == 0 || embed_dim == 0 || token_count == 0)
        return fail("Vulkan embedding weight backward scatter cas requires non-zero shapes");
    const auto idx_bytes = token_count * sizeof(std::int32_t);
    const auto grad_bytes = token_count * embed_dim * sizeof(float);
    const auto out_bytes = vocab_size * embed_dim * sizeof(float);
    if (indices.nbytes() < idx_bytes) return fail("Vulkan embedding weight backward scatter cas indices buffer too small");
    if (grad_out.nbytes() < grad_bytes) return fail("Vulkan embedding weight backward scatter cas grad buffer too small");
    if (grad_weight.nbytes() < out_bytes) return fail("Vulkan embedding weight backward scatter cas output buffer too small");
    // Caller MUST zero-fill grad_weight before calling (the CAS accumulates).
    constexpr std::size_t kMaxInt32 = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
    if (token_count > kMaxInt32 / embed_dim) return fail("Vulkan embedding weight backward scatter cas token_count*embed_dim overflows int32 push constant");
    const struct {
        std::int32_t vocab_size;
        std::int32_t embed_dim;
        std::int32_t token_count;
    } push{static_cast<std::int32_t>(vocab_size),
           static_cast<std::int32_t>(embed_dim),
           static_cast<std::int32_t>(token_count)};
    const std::vector<const VulkanBuffer*> buffers = {&indices, &grad_out, &grad_weight};
    const std::size_t total = token_count * embed_dim;
    return runtime.dispatch_cached(vkspirv::k_embedding_weight_backward_f32_i32_scatter_cas,
                                   vkspirv::k_embedding_weight_backward_f32_i32_scatter_cas_words,
                                   buffers, &push, sizeof(push),
                                   static_cast<std::uint32_t>((total + 63) / 64), 1, 1);
}

VulkanOpResult run_vulkan_token_position_embedding(VulkanRuntime& runtime,
                                                    const VulkanBuffer& token_weight,
                                                    const VulkanBuffer& pos_weight,
                                                    const VulkanBuffer& token_ids,
                                                    VulkanBuffer& out,
                                                    std::size_t vocab_size,
                                                    std::size_t seq_len,
                                                    std::size_t embed_dim) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (vocab_size == 0 || seq_len == 0 || embed_dim == 0)
        return fail("Vulkan token+position embedding requires non-zero shapes");
    // The token_ids buffer length is implied by out.nbytes / embed_dim, matching
    // the OpenCL host contract.
    const auto token_count = out.nbytes() / (embed_dim * sizeof(float));
    if (token_count == 0) return fail("Vulkan token+position embedding output buffer too small");
    // Push-constant `n = token_count * embed_dim` is sent as int32; reject
    // shapes whose product would overflow the kernel's `if (gid >= p.n)` guard.
    constexpr std::size_t kMaxInt32 = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
    if (token_count > kMaxInt32 / embed_dim) return fail("Vulkan token+position embedding token_count*embed_dim overflows int32 push constant");
    if (token_weight.nbytes() < vocab_size * embed_dim * sizeof(float))
        return fail("Vulkan token+position embedding token weight buffer too small");
    if (pos_weight.nbytes() < seq_len * embed_dim * sizeof(float))
        return fail("Vulkan token+position embedding position weight buffer too small");
    if (token_ids.nbytes() < token_count * sizeof(std::int32_t))
        return fail("Vulkan token+position embedding token_ids buffer too small");

    const struct {
        std::int32_t vocab_size;
        std::int32_t seq_len;
        std::int32_t embed_dim;
        std::int32_t n;
    } push{static_cast<std::int32_t>(vocab_size),
           static_cast<std::int32_t>(seq_len),
           static_cast<std::int32_t>(embed_dim),
           static_cast<std::int32_t>(token_count * embed_dim)};
    const std::vector<const VulkanBuffer*> buffers = {&token_weight, &pos_weight, &token_ids, &out};
    const std::size_t total = token_count * embed_dim;
    return runtime.dispatch_cached(vkspirv::k_token_position_embedding_f32_i32,
                                   vkspirv::k_token_position_embedding_f32_i32_words,
                                   buffers, &push, sizeof(push),
                                   static_cast<std::uint32_t>((total + 63) / 64), 1, 1);
}

VulkanOpResult run_vulkan_position_embedding_backward(VulkanRuntime& runtime,
                                                       const VulkanBuffer& grad_out,
                                                       VulkanBuffer& grad_position,
                                                       std::size_t batch,
                                                       std::size_t seq_len,
                                                       std::size_t embed_dim) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (batch == 0 || seq_len == 0 || embed_dim == 0)
        return fail("Vulkan position embedding backward requires non-zero shapes");
    const auto grad_bytes = batch * seq_len * embed_dim * sizeof(float);
    const auto out_bytes = grad_position.nbytes();
    if (grad_out.nbytes() < grad_bytes) return fail("Vulkan position embedding backward grad buffer too small");
    if (out_bytes < seq_len * embed_dim * sizeof(float))
        return fail("Vulkan position embedding backward output buffer too small");
    // Output table size (rows) is the table length, which the host zeroed.
    const auto out_rows = out_bytes / (embed_dim * sizeof(float));

    const struct {
        std::int32_t batch;
        std::int32_t seq_len;
        std::int32_t embed_dim;
        std::int32_t n;
    } push{static_cast<std::int32_t>(batch),
           static_cast<std::int32_t>(seq_len),
           static_cast<std::int32_t>(embed_dim),
           static_cast<std::int32_t>(out_rows * embed_dim)};
    const std::vector<const VulkanBuffer*> buffers = {&grad_out, &grad_position};
    const std::size_t total = out_rows * embed_dim;
    return runtime.dispatch_cached(vkspirv::k_position_embedding_backward_f32_i32,
                                   vkspirv::k_position_embedding_backward_f32_i32_words,
                                   buffers, &push, sizeof(push),
                                   static_cast<std::uint32_t>((total + 63) / 64), 1, 1);
}

VulkanOpResult run_vulkan_rope(VulkanRuntime& runtime,
                               const VulkanBuffer& x,
                               VulkanBuffer& out,
                               std::size_t batch,
                               std::size_t tokens,
                               std::size_t channels,
                               std::size_t n_head,
                               std::size_t head_dim,
                               std::size_t rotary_dim,
                               std::size_t token_offset,
                               float theta,
                               bool inverse) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (batch == 0 || tokens == 0 || channels == 0 || n_head == 0 || head_dim == 0)
        return fail("Vulkan rope requires non-zero shapes");
    if (channels % n_head != 0) return fail("Vulkan rope channels must divide n_head");
    if (head_dim * n_head != channels) return fail("Vulkan rope head_dim*n_head must equal channels");
    if (!std::isfinite(theta) || theta <= 0.0f) return fail("Vulkan rope theta must be positive finite");
    // Guard the dispatch size: group_count_x = (total+63)/64 is narrowed to
    // uint32 below; reject shapes whose product would overflow.
    constexpr std::size_t kMaxRopeElements = static_cast<std::size_t>(std::uint32_t(-1)) * 64u;
    if (batch > kMaxRopeElements / tokens / channels) return fail("Vulkan rope shape product overflows dispatch range");
    const auto total = batch * tokens * channels;
    const auto bytes = total * sizeof(float);
    if (x.nbytes() < bytes) return fail("Vulkan rope x buffer too small");
    if (out.nbytes() < bytes) return fail("Vulkan rope output buffer too small");

    const struct {
        std::int32_t batch;
        std::int32_t tokens;
        std::int32_t channels;
        std::int32_t n_head;
        std::int32_t head_dim;
        std::int32_t rotary_dim;
        std::int32_t token_offset;
        float theta;
        std::int32_t inverse;
    } push{static_cast<std::int32_t>(batch),
           static_cast<std::int32_t>(tokens),
           static_cast<std::int32_t>(channels),
           static_cast<std::int32_t>(n_head),
           static_cast<std::int32_t>(head_dim),
           static_cast<std::int32_t>(rotary_dim),
           static_cast<std::int32_t>(token_offset),
           theta,
           inverse ? 1 : 0};
    const std::vector<const VulkanBuffer*> buffers = {&x, &out};
    return runtime.dispatch_cached(vkspirv::k_rope_f32, vkspirv::k_rope_f32_words,
                                   buffers, &push, sizeof(push),
                                   static_cast<std::uint32_t>((total + 63) / 64), 1, 1);
}

VulkanOpResult run_vulkan_rope_positions(VulkanRuntime& runtime,
                                         const VulkanBuffer& x,
                                         const VulkanBuffer& positions,
                                         VulkanBuffer& out,
                                         std::size_t batch,
                                         std::size_t tokens,
                                         std::size_t channels,
                                         std::size_t n_head,
                                         std::size_t head_dim,
                                         std::size_t rotary_dim,
                                         float theta) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (batch == 0 || tokens == 0 || channels == 0 || n_head == 0 || head_dim == 0)
        return fail("Vulkan rope_positions requires non-zero shapes");
    if (channels % n_head != 0) return fail("Vulkan rope_positions channels must divide n_head");
    if (head_dim * n_head != channels) return fail("Vulkan rope_positions head_dim*n_head must equal channels");
    if (!std::isfinite(theta) || theta <= 0.0f) return fail("Vulkan rope_positions theta must be positive finite");
    const auto total = batch * tokens * channels;
    const auto bytes = total * sizeof(float);
    if (x.nbytes() < bytes) return fail("Vulkan rope_positions x buffer too small");
    if (positions.nbytes() < batch * tokens * sizeof(std::int32_t))
        return fail("Vulkan rope_positions positions buffer too small");
    if (out.nbytes() < bytes) return fail("Vulkan rope_positions output buffer too small");

    const struct {
        std::int32_t batch;
        std::int32_t tokens;
        std::int32_t channels;
        std::int32_t n_head;
        std::int32_t head_dim;
        std::int32_t rotary_dim;
        float theta;
    } push{static_cast<std::int32_t>(batch),
           static_cast<std::int32_t>(tokens),
           static_cast<std::int32_t>(channels),
           static_cast<std::int32_t>(n_head),
           static_cast<std::int32_t>(head_dim),
           static_cast<std::int32_t>(rotary_dim),
           theta};
    const std::vector<const VulkanBuffer*> buffers = {&x, &positions, &out};
    return runtime.dispatch_cached(vkspirv::k_rope_positions_f32, vkspirv::k_rope_positions_f32_words,
                                   buffers, &push, sizeof(push),
                                   static_cast<std::uint32_t>((total + 63) / 64), 1, 1);
}

VulkanOpResult run_vulkan_rope_split_half(VulkanRuntime& runtime,
                                          const VulkanBuffer& x,
                                          VulkanBuffer& out,
                                          std::size_t batch,
                                          std::size_t tokens,
                                          std::size_t channels,
                                          std::size_t n_head,
                                          std::size_t head_dim,
                                          std::size_t rotary_dim,
                                          std::size_t token_offset,
                                          float theta,
                                          bool inverse) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (batch == 0 || tokens == 0 || channels == 0 || n_head == 0 || head_dim == 0)
        return fail("Vulkan rope_split_half requires non-zero shapes");
    if (channels % n_head != 0) return fail("Vulkan rope_split_half channels must divide n_head");
    if (head_dim * n_head != channels) return fail("Vulkan rope_split_half head_dim*n_head must equal channels");
    if (!std::isfinite(theta) || theta <= 0.0f) return fail("Vulkan rope_split_half theta must be positive finite");
    const auto total = batch * tokens * channels;
    const auto bytes = total * sizeof(float);
    if (x.nbytes() < bytes) return fail("Vulkan rope_split_half x buffer too small");
    if (out.nbytes() < bytes) return fail("Vulkan rope_split_half output buffer too small");

    const struct {
        std::int32_t batch;
        std::int32_t tokens;
        std::int32_t channels;
        std::int32_t n_head;
        std::int32_t head_dim;
        std::int32_t rotary_dim;
        std::int32_t token_offset;
        float theta;
        std::int32_t inverse;
    } push{static_cast<std::int32_t>(batch),
           static_cast<std::int32_t>(tokens),
           static_cast<std::int32_t>(channels),
           static_cast<std::int32_t>(n_head),
           static_cast<std::int32_t>(head_dim),
           static_cast<std::int32_t>(rotary_dim),
           static_cast<std::int32_t>(token_offset),
           theta,
           inverse ? 1 : 0};
    const std::vector<const VulkanBuffer*> buffers = {&x, &out};
    return runtime.dispatch_cached(vkspirv::k_rope_split_half_f32, vkspirv::k_rope_split_half_f32_words,
                                   buffers, &push, sizeof(push),
                                   static_cast<std::uint32_t>((total + 63) / 64), 1, 1);
}

VulkanOpResult run_vulkan_rope_positions_split_half(VulkanRuntime& runtime,
                                                    const VulkanBuffer& x,
                                                    const VulkanBuffer& positions,
                                                    VulkanBuffer& out,
                                                    std::size_t batch,
                                                    std::size_t tokens,
                                                    std::size_t channels,
                                                    std::size_t n_head,
                                                    std::size_t head_dim,
                                                    std::size_t rotary_dim,
                                                    float theta) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (batch == 0 || tokens == 0 || channels == 0 || n_head == 0 || head_dim == 0)
        return fail("Vulkan rope_positions_split_half requires non-zero shapes");
    if (channels % n_head != 0) return fail("Vulkan rope_positions_split_half channels must divide n_head");
    if (head_dim * n_head != channels) return fail("Vulkan rope_positions_split_half head_dim*n_head must equal channels");
    if (!std::isfinite(theta) || theta <= 0.0f) return fail("Vulkan rope_positions_split_half theta must be positive finite");
    const auto total = batch * tokens * channels;
    const auto bytes = total * sizeof(float);
    if (x.nbytes() < bytes) return fail("Vulkan rope_positions_split_half x buffer too small");
    if (positions.nbytes() < batch * tokens * sizeof(std::int32_t))
        return fail("Vulkan rope_positions_split_half positions buffer too small");
    if (out.nbytes() < bytes) return fail("Vulkan rope_positions_split_half output buffer too small");

    const struct {
        std::int32_t batch;
        std::int32_t tokens;
        std::int32_t channels;
        std::int32_t n_head;
        std::int32_t head_dim;
        std::int32_t rotary_dim;
        float theta;
    } push{static_cast<std::int32_t>(batch),
           static_cast<std::int32_t>(tokens),
           static_cast<std::int32_t>(channels),
           static_cast<std::int32_t>(n_head),
           static_cast<std::int32_t>(head_dim),
           static_cast<std::int32_t>(rotary_dim),
           theta};
    const std::vector<const VulkanBuffer*> buffers = {&x, &positions, &out};
    return runtime.dispatch_cached(vkspirv::k_rope_positions_split_half_f32,
                                   vkspirv::k_rope_positions_split_half_f32_words,
                                   buffers, &push, sizeof(push),
                                   static_cast<std::uint32_t>((total + 63) / 64), 1, 1);
}

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


namespace {

using VulkanStorageBufferComputeResult = VulkanStorageBufferDispatchResult;

VulkanStorageBufferComputeResult run_vulkan_storage_buffer_compute(
    const std::uint32_t* spirv,
    std::size_t spirv_word_count,
    const std::vector<VulkanStorageBufferSpec>& buffer_specs,
    const std::vector<std::size_t>& output_buffer_indices,
    std::uint32_t group_count_x = 1,
    std::uint32_t group_count_y = 1,
    std::uint32_t group_count_z = 1) {
    auto runtime = VulkanRuntime::create();
    return runtime.dispatch_storage_buffers(spirv, spirv_word_count, buffer_specs, output_buffer_indices,
                                            group_count_x, group_count_y, group_count_z);
}

} // namespace

struct VulkanRuntime::Impl {
    DynamicLibrary library;
    PFN_vkGetInstanceProcAddr get_proc = nullptr;
    PFN_vkDestroyInstance destroy_instance = nullptr;
    PFN_vkGetDeviceProcAddr get_device_proc = nullptr;

    VkInstance instance = nullptr;
    VkPhysicalDevice physical_device = nullptr;
    VkPhysicalDeviceMemoryProperties memory_properties{};
    VkDevice device = nullptr;
    VkQueue queue = nullptr;
    std::uint32_t queue_family = 0;

    PFN_vkDestroyDevice destroy_device = nullptr;
    PFN_vkGetDeviceQueue get_device_queue = nullptr;
    PFN_vkCreateBuffer create_buffer = nullptr;
    PFN_vkDestroyBuffer destroy_buffer = nullptr;
    PFN_vkGetBufferMemoryRequirements get_buffer_memory_requirements = nullptr;
    PFN_vkAllocateMemory allocate_memory = nullptr;
    PFN_vkFreeMemory free_memory = nullptr;
    PFN_vkBindBufferMemory bind_buffer_memory = nullptr;
    PFN_vkMapMemory map_memory = nullptr;
    PFN_vkUnmapMemory unmap_memory = nullptr;
    PFN_vkCreateDescriptorSetLayout create_descriptor_set_layout = nullptr;
    PFN_vkDestroyDescriptorSetLayout destroy_descriptor_set_layout = nullptr;
    PFN_vkCreateDescriptorPool create_descriptor_pool = nullptr;
    PFN_vkDestroyDescriptorPool destroy_descriptor_pool = nullptr;
    PFN_vkAllocateDescriptorSets allocate_descriptor_sets = nullptr;
    PFN_vkUpdateDescriptorSets update_descriptor_sets = nullptr;
    PFN_vkCreatePipelineLayout create_pipeline_layout = nullptr;
    PFN_vkDestroyPipelineLayout destroy_pipeline_layout = nullptr;
    PFN_vkCreateShaderModule create_shader_module = nullptr;
    PFN_vkDestroyShaderModule destroy_shader_module = nullptr;
    PFN_vkCreateComputePipelines create_compute_pipelines = nullptr;
    PFN_vkDestroyPipeline destroy_pipeline = nullptr;
    PFN_vkCreatePipelineCache create_pipeline_cache = nullptr;
    PFN_vkDestroyPipelineCache destroy_pipeline_cache = nullptr;
    PFN_vkGetPipelineCacheData get_pipeline_cache_data = nullptr;
    VkPipelineCache vk_pipeline_cache = nullptr;
    std::string vk_pipeline_cache_path;
    PFN_vkCreateCommandPool create_command_pool = nullptr;
    PFN_vkDestroyCommandPool destroy_command_pool = nullptr;
    PFN_vkAllocateCommandBuffers allocate_command_buffers = nullptr;
    PFN_vkBeginCommandBuffer begin_command_buffer = nullptr;
    PFN_vkEndCommandBuffer end_command_buffer = nullptr;
    PFN_vkCmdBindPipeline cmd_bind_pipeline = nullptr;
    PFN_vkCmdBindDescriptorSets cmd_bind_descriptor_sets = nullptr;
    PFN_vkCmdDispatch cmd_dispatch = nullptr;
    PFN_vkQueueSubmit queue_submit = nullptr;
    PFN_vkQueueWaitIdle queue_wait_idle = nullptr;

    // fast-dispatch path functions (all core 1.0)
    PFN_vkCreateFence create_fence = nullptr;
    PFN_vkDestroyFence destroy_fence = nullptr;
    PFN_vkResetFences reset_fences = nullptr;
    PFN_vkWaitForFences wait_for_fences = nullptr;
    PFN_vkResetCommandPool reset_command_pool = nullptr;
    PFN_vkCmdPushConstants cmd_push_constants = nullptr;
    PFN_vkCmdPipelineBarrier cmd_pipeline_barrier = nullptr;
    PFN_vkCmdCopyBuffer cmd_copy_buffer = nullptr;
    PFN_vkCreateQueryPool create_query_pool = nullptr;
    PFN_vkDestroyQueryPool destroy_query_pool = nullptr;
    PFN_vkCmdResetQueryPool cmd_reset_query_pool = nullptr;
    PFN_vkCmdWriteTimestamp cmd_write_timestamp = nullptr;
    PFN_vkGetQueryPoolResults get_query_pool_results = nullptr;

    bool ready = false;
    bool storage_buffer_i8 = false;
    bool caps_supports_atomic_float = false;
    bool atomic_float_smoke_pending = false;  // run smoke check lazily in VulkanRuntime::create
    std::string device_name;
    std::string error;

    // --- cached fast-dispatch state (lazily initialized) ---
    VulkanDeviceCaps caps{};
    std::uint32_t timestamp_valid_bits = 0;

    struct LayoutEntry {
        VkDescriptorSetLayout set_layout = nullptr;
        VkPipelineLayout pipeline_layout = nullptr;
    };
    // (binding_count, push_constant_bytes) -> layouts
    std::map<std::pair<std::uint32_t, std::uint32_t>, LayoutEntry> layout_cache;
    // (spirv pointer, word count, binding_count, push_constant_bytes) -> pipeline.
    // Fast-path SPIR-V lives in static storage (embedded arrays), so pointer
    // identity is a stable cache key.
    std::map<std::tuple<const std::uint32_t*, std::size_t, std::uint32_t, std::uint32_t>, VkPipeline> pipeline_cache;

    std::vector<VkDescriptorPool> descriptor_pools;
    std::map<std::uint32_t, std::vector<VkDescriptorSet>> free_descriptor_sets;
    std::vector<std::pair<std::uint32_t, VkDescriptorSet>> inflight_descriptor_sets;

    VkCommandPool fast_command_pool = nullptr;
    VkCommandBuffer fast_command_buffer = nullptr;
    VkFence fast_fence = nullptr;
    VkCommandPool transfer_command_pool = nullptr;
    VkCommandBuffer transfer_command_buffer = nullptr;
    VkFence transfer_fence = nullptr;

    VkBuffer staging_buffer = nullptr;
    VkDeviceMemory staging_memory = nullptr;
    void* staging_mapped = nullptr;
    std::size_t staging_capacity = 0;

    VkQueryPool query_pool = nullptr;
    bool timing_enabled = false;
    bool batch_timed = false;
    double last_gpu_us = -1.0;

    bool batch_open = false;
    std::uint32_t batch_dispatch_count = 0;
    std::string batch_error;
    // While a batch is open, every referenced buffer allocation is kept alive
    // here until the submission's fence signals: op-internal temporaries
    // (e.g. loss partials, backward scratch) legally go out of scope before
    // batch_end, and destroying their VkBuffer mid-recording would leave the
    // command buffer referencing freed memory.
    std::vector<std::shared_ptr<VulkanBuffer::Impl>> batch_keepalive;

    bool fast_ready = false;
    std::string fast_error;

    // Buffer pool: Tensor-churny paths (autograd temporaries) allocate and
    // free device buffers every step, and vkAllocateMemory is far too slow
    // for that. Freed allocations are pooled by power-of-two capacity and
    // host-visibility class and reused by create_buffer.
    struct PooledBuffer {
        VkBuffer buffer = nullptr;
        VkDeviceMemory memory = nullptr;
    };
    std::map<std::pair<std::size_t, bool>, std::vector<PooledBuffer>> buffer_pool;
    std::size_t buffer_pool_bytes = 0;

    static constexpr std::size_t kBufferPoolCapBytes = 1024u * 1024u * 1024u;

    void pool_release(VkBuffer buffer, VkDeviceMemory memory, std::size_t capacity, bool host_visible) {
        if (!device || !ready || capacity == 0 ||
            buffer_pool_bytes + capacity > kBufferPoolCapBytes) {
            if (buffer && destroy_buffer) destroy_buffer(device, buffer, nullptr);
            if (memory && free_memory) free_memory(device, memory, nullptr);
            return;
        }
        buffer_pool[{capacity, host_visible}].push_back(PooledBuffer{buffer, memory});
        buffer_pool_bytes += capacity;
    }

    bool pool_acquire(std::size_t capacity, bool host_visible, VkBuffer& buffer, VkDeviceMemory& memory) {
        auto it = buffer_pool.find({capacity, host_visible});
        if (it == buffer_pool.end() || it->second.empty()) return false;
        buffer = it->second.back().buffer;
        memory = it->second.back().memory;
        it->second.pop_back();
        buffer_pool_bytes -= capacity;
        return true;
    }

    void destroy_buffer_pool() {
        for (auto& bucket : buffer_pool) {
            for (auto& entry : bucket.second) {
                if (entry.buffer && destroy_buffer) destroy_buffer(device, entry.buffer, nullptr);
                if (entry.memory && free_memory) free_memory(device, entry.memory, nullptr);
            }
        }
        buffer_pool.clear();
        buffer_pool_bytes = 0;
    }

    // Dispatch capture state (see VulkanDispatchRecording).
    struct CapturedDispatch {
        const std::uint32_t* spirv = nullptr;
        std::size_t spirv_words = 0;
        std::vector<std::shared_ptr<VulkanBuffer::Impl>> keepalive;
        std::vector<VkBuffer> raw_buffers;
        std::vector<VkDeviceSize> raw_sizes;
        std::vector<std::uint8_t> push;
        std::uint32_t gx = 1, gy = 1, gz = 1;
    };
    bool capturing = false;
    std::vector<CapturedDispatch> capture_list;

    void destroy_fast_path() {
        if (!device) return;
        // Serialize the driver-side pipeline cache to disk so the next process
        // start skips SPIR-V -> ISA compilation for already-seen kernels. Best
        // effort: failures are silently ignored (cache is advisory, not required
        // for correctness).
        if (vk_pipeline_cache && get_pipeline_cache_data && !vk_pipeline_cache_path.empty()) {
            std::size_t sz = 0;
            if (get_pipeline_cache_data(device, vk_pipeline_cache, &sz, nullptr) == VK_SUCCESS && sz > 0) {
                std::vector<std::uint8_t> blob(sz);
                if (get_pipeline_cache_data(device, vk_pipeline_cache, &sz, blob.data()) == VK_SUCCESS) {
                    std::ofstream out_file(vk_pipeline_cache_path, std::ios::binary | std::ios::trunc);
                    if (out_file) out_file.write(reinterpret_cast<const char*>(blob.data()), static_cast<std::streamsize>(sz));
                }
            }
        }
        if (vk_pipeline_cache && destroy_pipeline_cache) destroy_pipeline_cache(device, vk_pipeline_cache, nullptr);
        vk_pipeline_cache = nullptr;
        destroy_buffer_pool();
        for (auto& entry : pipeline_cache) {
            if (entry.second && destroy_pipeline) destroy_pipeline(device, entry.second, nullptr);
        }
        pipeline_cache.clear();
        for (auto& entry : layout_cache) {
            if (entry.second.pipeline_layout && destroy_pipeline_layout) {
                destroy_pipeline_layout(device, entry.second.pipeline_layout, nullptr);
            }
            if (entry.second.set_layout && destroy_descriptor_set_layout) {
                destroy_descriptor_set_layout(device, entry.second.set_layout, nullptr);
            }
        }
        layout_cache.clear();
        for (auto pool : descriptor_pools) {
            if (pool && destroy_descriptor_pool) destroy_descriptor_pool(device, pool, nullptr);
        }
        descriptor_pools.clear();
        free_descriptor_sets.clear();
        inflight_descriptor_sets.clear();
        if (query_pool && destroy_query_pool) destroy_query_pool(device, query_pool, nullptr);
        query_pool = nullptr;
        if (fast_fence && destroy_fence) destroy_fence(device, fast_fence, nullptr);
        fast_fence = nullptr;
        if (transfer_fence && destroy_fence) destroy_fence(device, transfer_fence, nullptr);
        transfer_fence = nullptr;
        if (fast_command_pool && destroy_command_pool) destroy_command_pool(device, fast_command_pool, nullptr);
        fast_command_pool = nullptr;
        fast_command_buffer = nullptr;
        if (transfer_command_pool && destroy_command_pool) {
            destroy_command_pool(device, transfer_command_pool, nullptr);
        }
        transfer_command_pool = nullptr;
        transfer_command_buffer = nullptr;
        if (staging_mapped && unmap_memory && staging_memory) unmap_memory(device, staging_memory);
        staging_mapped = nullptr;
        if (staging_buffer && destroy_buffer) destroy_buffer(device, staging_buffer, nullptr);
        staging_buffer = nullptr;
        if (staging_memory && free_memory) free_memory(device, staging_memory, nullptr);
        staging_memory = nullptr;
        staging_capacity = 0;
        fast_ready = false;
    }

    ~Impl() {
        destroy_fast_path();
        if (device && destroy_device) {
            destroy_device(device, nullptr);
            device = nullptr;
        }
        if (instance && destroy_instance) {
            destroy_instance(instance, nullptr);
            instance = nullptr;
        }
    }

    // ---- cached fast-dispatch machinery -----------------------------------

    bool fast_functions_present() const {
        return create_fence && destroy_fence && reset_fences && wait_for_fences && reset_command_pool &&
               cmd_push_constants && cmd_pipeline_barrier && cmd_copy_buffer && create_command_pool &&
               allocate_command_buffers && begin_command_buffer && end_command_buffer && queue_submit;
    }

    bool ensure_fast_path() {
        if (fast_ready) return true;
        if (!ready || !device) {
            fast_error = "Vulkan runtime is not available";
            return false;
        }
        if (!fast_functions_present()) {
            fast_error = "Vulkan device is missing core functions for the cached dispatch path";
            return false;
        }
        const VkCommandPoolCreateInfo pool_info{
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            nullptr,
            0,
            queue_family,
        };
        if (create_command_pool(device, &pool_info, nullptr, &fast_command_pool) != VK_SUCCESS ||
            !fast_command_pool) {
            fast_error = "vkCreateCommandPool failed for cached dispatch";
            return false;
        }
        if (create_command_pool(device, &pool_info, nullptr, &transfer_command_pool) != VK_SUCCESS ||
            !transfer_command_pool) {
            fast_error = "vkCreateCommandPool failed for staging transfers";
            return false;
        }

        // Pipeline cache: load persisted blob from disk (if present) so the
        // driver skips SPIR-V -> ISA compilation for kernels seen in any prior
        // run. The path defaults to "motifcl_vulkan_pipeline_cache.bin" in the
        // cwd, override via MOTIFCL_VULKAN_PIPELINE_CACHE env var. Set the env
        // var to "off" (or empty path) to disable disk persistence entirely.
        std::vector<std::uint8_t> cache_seed;
        const char* cache_env = std::getenv("MOTIFCL_VULKAN_PIPELINE_CACHE");
        std::string cache_path;
        if (cache_env && cache_env[0] != '\0' && std::string(cache_env) != "off") {
            cache_path = cache_env;
        } else if (!cache_env) {
            cache_path = "motifcl_vulkan_pipeline_cache.bin";
        }
        if (!cache_path.empty()) {
            std::ifstream cache_file(cache_path, std::ios::binary);
            if (cache_file) {
                cache_file.seekg(0, std::ios::end);
                const auto sz = cache_file.tellg();
                cache_file.seekg(0, std::ios::beg);
                if (sz > 0) {
                    cache_seed.resize(static_cast<std::size_t>(sz));
                    cache_file.read(reinterpret_cast<char*>(cache_seed.data()), sz);
                }
            }
            vk_pipeline_cache_path = cache_path;
        }
        const VkPipelineCacheCreateInfo cache_info{
            VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            nullptr,
            0,
            cache_seed.size(),
            cache_seed.empty() ? nullptr : cache_seed.data(),
        };
        if (create_pipeline_cache(device, &cache_info, nullptr, &vk_pipeline_cache) != VK_SUCCESS) {
            // Non-fatal: vk_pipeline_cache stays null and create_compute_pipelines
            // falls back to per-call compilation. Log to fast_error for debug.
            vk_pipeline_cache = nullptr;
        }
        const VkCommandBufferAllocateInfo fast_alloc{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            nullptr,
            fast_command_pool,
            VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            1,
        };
        if (allocate_command_buffers(device, &fast_alloc, &fast_command_buffer) != VK_SUCCESS ||
            !fast_command_buffer) {
            fast_error = "vkAllocateCommandBuffers failed for cached dispatch";
            return false;
        }
        const VkCommandBufferAllocateInfo transfer_alloc{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            nullptr,
            transfer_command_pool,
            VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            1,
        };
        if (allocate_command_buffers(device, &transfer_alloc, &transfer_command_buffer) != VK_SUCCESS ||
            !transfer_command_buffer) {
            fast_error = "vkAllocateCommandBuffers failed for staging transfers";
            return false;
        }
        const VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
        if (create_fence(device, &fence_info, nullptr, &fast_fence) != VK_SUCCESS || !fast_fence) {
            fast_error = "vkCreateFence failed for cached dispatch";
            return false;
        }
        if (create_fence(device, &fence_info, nullptr, &transfer_fence) != VK_SUCCESS || !transfer_fence) {
            fast_error = "vkCreateFence failed for staging transfers";
            return false;
        }
        if (caps.timestamps && create_query_pool && cmd_reset_query_pool && cmd_write_timestamp &&
            get_query_pool_results) {
            const VkQueryPoolCreateInfo query_info{
                VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                nullptr,
                0,
                VK_QUERY_TYPE_TIMESTAMP,
                2,
                0,
            };
            if (create_query_pool(device, &query_info, nullptr, &query_pool) != VK_SUCCESS) {
                query_pool = nullptr;  // timing degrades, dispatch still works
            }
        }
        fast_ready = true;
        return true;
    }

    LayoutEntry* get_layout(std::uint32_t binding_count, std::uint32_t push_bytes) {
        const auto key = std::make_pair(binding_count, push_bytes);
        auto it = layout_cache.find(key);
        if (it != layout_cache.end()) return &it->second;

        std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.reserve(binding_count);
        for (std::uint32_t i = 0; i < binding_count; ++i) {
            bindings.push_back(VkDescriptorSetLayoutBinding{
                i,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                1,
                VK_SHADER_STAGE_COMPUTE_BIT,
                nullptr,
            });
        }
        const VkDescriptorSetLayoutCreateInfo set_layout_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            nullptr,
            0,
            binding_count,
            bindings.data(),
        };
        LayoutEntry entry;
        if (create_descriptor_set_layout(device, &set_layout_info, nullptr, &entry.set_layout) != VK_SUCCESS ||
            !entry.set_layout) {
            fast_error = "vkCreateDescriptorSetLayout failed for cached dispatch";
            return nullptr;
        }
        const VkPushConstantRange push_range{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes};
        const VkPipelineLayoutCreateInfo layout_info{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            nullptr,
            0,
            1,
            &entry.set_layout,
            push_bytes > 0 ? 1u : 0u,
            push_bytes > 0 ? &push_range : nullptr,
        };
        if (create_pipeline_layout(device, &layout_info, nullptr, &entry.pipeline_layout) != VK_SUCCESS ||
            !entry.pipeline_layout) {
            destroy_descriptor_set_layout(device, entry.set_layout, nullptr);
            fast_error = "vkCreatePipelineLayout failed for cached dispatch";
            return nullptr;
        }
        auto emplaced = layout_cache.emplace(key, entry);
        return &emplaced.first->second;
    }

    VkPipeline get_pipeline(const std::uint32_t* spirv,
                            std::size_t spirv_word_count,
                            std::uint32_t binding_count,
                            std::uint32_t push_bytes) {
        const auto key = std::make_tuple(spirv, spirv_word_count, binding_count, push_bytes);
        auto it = pipeline_cache.find(key);
        if (it != pipeline_cache.end()) return it->second;

        LayoutEntry* layouts = get_layout(binding_count, push_bytes);
        if (!layouts) return nullptr;

        const VkShaderModuleCreateInfo module_info{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            nullptr,
            0,
            spirv_word_count * sizeof(std::uint32_t),
            spirv,
        };
        VkShaderModule module = nullptr;
        if (create_shader_module(device, &module_info, nullptr, &module) != VK_SUCCESS || !module) {
            fast_error = "vkCreateShaderModule failed for cached dispatch";
            return nullptr;
        }
        const VkPipelineShaderStageCreateInfo stage_info{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_COMPUTE_BIT,
            module,
            "main",
            nullptr,
        };
        const VkComputePipelineCreateInfo pipeline_info{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            nullptr,
            0,
            stage_info,
            layouts->pipeline_layout,
            nullptr,
            -1,
        };
        VkPipeline pipeline = nullptr;
        const auto created = create_compute_pipelines(device, vk_pipeline_cache, 1, &pipeline_info, nullptr, &pipeline);
        destroy_shader_module(device, module, nullptr);
        if (created != VK_SUCCESS || !pipeline) {
            fast_error = "vkCreateComputePipelines failed for cached dispatch";
            return nullptr;
        }
        pipeline_cache.emplace(key, pipeline);
        return pipeline;
    }

    bool grow_descriptor_pool() {
        const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2048};
        const VkDescriptorPoolCreateInfo pool_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            nullptr,
            0,
            256,
            1,
            &pool_size,
        };
        VkDescriptorPool pool = nullptr;
        if (create_descriptor_pool(device, &pool_info, nullptr, &pool) != VK_SUCCESS || !pool) {
            fast_error = "vkCreateDescriptorPool failed for cached dispatch";
            return false;
        }
        descriptor_pools.push_back(pool);
        return true;
    }

    VkDescriptorSet acquire_descriptor_set(std::uint32_t binding_count, VkDescriptorSetLayout layout) {
        auto& free_list = free_descriptor_sets[binding_count];
        if (!free_list.empty()) {
            VkDescriptorSet set = free_list.back();
            free_list.pop_back();
            return set;
        }
        for (int attempt = 0; attempt < 2; ++attempt) {
            if (!descriptor_pools.empty()) {
                const VkDescriptorSetAllocateInfo alloc_info{
                    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    nullptr,
                    descriptor_pools.back(),
                    1,
                    &layout,
                };
                VkDescriptorSet set = nullptr;
                if (allocate_descriptor_sets(device, &alloc_info, &set) == VK_SUCCESS && set) {
                    return set;
                }
            }
            if (!grow_descriptor_pool()) return nullptr;
        }
        fast_error = "descriptor set allocation failed after pool growth";
        return nullptr;
    }

    void recycle_inflight_sets() {
        for (auto& entry : inflight_descriptor_sets) {
            free_descriptor_sets[entry.first].push_back(entry.second);
        }
        inflight_descriptor_sets.clear();
        batch_keepalive.clear();
    }

    bool begin_fast_commands(bool timed) {
        if (reset_command_pool(device, fast_command_pool, 0) != VK_SUCCESS) {
            fast_error = "vkResetCommandPool failed for cached dispatch";
            return false;
        }
        const VkCommandBufferBeginInfo begin_info{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            nullptr,
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            nullptr,
        };
        if (begin_command_buffer(fast_command_buffer, &begin_info) != VK_SUCCESS) {
            fast_error = "vkBeginCommandBuffer failed for cached dispatch";
            return false;
        }
        // Make writes from earlier submissions (compute or staging transfers)
        // visible to this submission regardless of driver flush behaviour at
        // submit boundaries: one global barrier per command buffer.
        const VkMemoryBarrier acquire{
            VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        };
        cmd_pipeline_barrier(fast_command_buffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &acquire, 0, nullptr, 0, nullptr);
        batch_timed = timed && query_pool != nullptr;
        if (batch_timed) {
            cmd_reset_query_pool(fast_command_buffer, query_pool, 0, 2);
            cmd_write_timestamp(fast_command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, 0);
        }
        return true;
    }

    VulkanOpResult submit_fast_commands() {
        VulkanOpResult result;
        result.device_name = device_name;
        auto fail_submit = [&](const std::string& message) {
            recycle_inflight_sets();
            result.error = message;
            result.success = false;
            return result;
        };
        if (batch_timed) {
            cmd_write_timestamp(fast_command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, 1);
        }
        if (end_command_buffer(fast_command_buffer) != VK_SUCCESS) {
            return fail_submit("vkEndCommandBuffer failed for cached dispatch");
        }
        const VkSubmitInfo submit_info{
            VK_STRUCTURE_TYPE_SUBMIT_INFO,
            nullptr,
            0,
            nullptr,
            nullptr,
            1,
            &fast_command_buffer,
            0,
            nullptr,
        };
        if (queue_submit(queue, 1, &submit_info, fast_fence) != VK_SUCCESS) {
            return fail_submit("vkQueueSubmit failed for cached dispatch");
        }
        if (wait_for_fences(device, 1, &fast_fence, 1, ~std::uint64_t{0}) != VK_SUCCESS) {
            return fail_submit("vkWaitForFences failed for cached dispatch");
        }
        if (reset_fences(device, 1, &fast_fence) != VK_SUCCESS) {
            return fail_submit("vkResetFences failed for cached dispatch");
        }
        last_gpu_us = -1.0;
        if (batch_timed) {
            std::uint64_t stamps[2] = {0, 0};
            if (get_query_pool_results(device, query_pool, 0, 2, sizeof(stamps), stamps, sizeof(std::uint64_t),
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
                std::uint64_t mask = ~std::uint64_t{0};
                if (timestamp_valid_bits > 0 && timestamp_valid_bits < 64) {
                    mask = (std::uint64_t{1} << timestamp_valid_bits) - 1;
                }
                const std::uint64_t begin = stamps[0] & mask;
                const std::uint64_t end = stamps[1] & mask;
                const std::uint64_t delta = end >= begin ? end - begin : (mask - begin) + end + 1;
                last_gpu_us = static_cast<double>(delta) * caps.timestamp_period_ns / 1000.0;
            }
        }
        recycle_inflight_sets();
        result.success = true;
        return result;
    }

    VulkanOpResult fast_dispatch(const std::uint32_t* spirv,
                                 std::size_t spirv_word_count,
                                 const VkBuffer* raw_buffers,
                                 const VkDeviceSize* raw_sizes,
                                 std::uint32_t buffer_count,
                                 const void* push_data,
                                 std::uint32_t push_bytes,
                                 std::uint32_t group_count_x,
                                 std::uint32_t group_count_y,
                                 std::uint32_t group_count_z) {
        VulkanOpResult result;
        result.device_name = device_name;
        auto fail_dispatch = [&](const std::string& message) {
            result.error = message;
            result.success = false;
            if (batch_open) batch_error = message;
            return result;
        };
        if (!ensure_fast_path()) return fail_dispatch(fast_error);
        if (batch_open && !batch_error.empty()) return fail_dispatch(batch_error);
        if (push_bytes > caps.max_push_constant_bytes) {
            return fail_dispatch("push constant payload exceeds device limit");
        }

        VkPipeline pipeline = get_pipeline(spirv, spirv_word_count, buffer_count, push_bytes);
        if (!pipeline) return fail_dispatch(fast_error);
        LayoutEntry* layouts = get_layout(buffer_count, push_bytes);
        if (!layouts) return fail_dispatch(fast_error);
        VkDescriptorSet set = acquire_descriptor_set(buffer_count, layouts->set_layout);
        if (!set) return fail_dispatch(fast_error);
        inflight_descriptor_sets.emplace_back(buffer_count, set);

        std::array<VkDescriptorBufferInfo, 16> buffer_infos{};
        std::array<VkWriteDescriptorSet, 16> writes{};
        for (std::uint32_t i = 0; i < buffer_count; ++i) {
            buffer_infos[i] = VkDescriptorBufferInfo{raw_buffers[i], 0, raw_sizes[i]};
            writes[i] = VkWriteDescriptorSet{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                set,
                i,
                0,
                1,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                nullptr,
                &buffer_infos[i],
                nullptr,
            };
        }
        update_descriptor_sets(device, buffer_count, writes.data(), 0, nullptr);

        if (!batch_open) {
            if (!begin_fast_commands(timing_enabled)) return fail_dispatch(fast_error);
        } else if (batch_dispatch_count > 0) {
            const VkMemoryBarrier barrier{
                VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            };
            cmd_pipeline_barrier(fast_command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        }

        cmd_bind_pipeline(fast_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        cmd_bind_descriptor_sets(fast_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, layouts->pipeline_layout, 0,
                                 1, &set, 0, nullptr);
        if (push_bytes > 0) {
            cmd_push_constants(fast_command_buffer, layouts->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               push_bytes, push_data);
        }
        cmd_dispatch(fast_command_buffer, group_count_x, group_count_y, group_count_z);

        if (batch_open) {
            ++batch_dispatch_count;
            result.success = true;
            return result;
        }
        return submit_fast_commands();
    }

    bool do_batch_begin() {
        if (batch_open) {
            fast_error = "Vulkan batch is already open";
            return false;
        }
        if (!ensure_fast_path()) return false;
        if (!begin_fast_commands(timing_enabled)) return false;
        batch_open = true;
        batch_dispatch_count = 0;
        batch_error.clear();
        return true;
    }

    VulkanOpResult do_batch_end() {
        VulkanOpResult result;
        result.device_name = device_name;
        if (!batch_open) {
            result.error = "Vulkan batch is not open";
            return result;
        }
        batch_open = false;
        if (!batch_error.empty()) {
            reset_command_pool(device, fast_command_pool, 0);
            recycle_inflight_sets();
            result.error = batch_error;
            batch_error.clear();
            return result;
        }
        if (batch_dispatch_count == 0) {
            // Nothing recorded; discard the empty command buffer.
            end_command_buffer(fast_command_buffer);
            reset_command_pool(device, fast_command_pool, 0);
            recycle_inflight_sets();
            result.success = true;
            return result;
        }
        return submit_fast_commands();
    }

    // ---- staging transfers for device-local buffers -----------------------

    bool ensure_staging(std::size_t bytes) {
        constexpr std::size_t kMaxStaging = 64u * 1024u * 1024u;
        const std::size_t wanted = std::min(kMaxStaging, std::max<std::size_t>(bytes, 1u << 20));
        if (staging_capacity >= std::min(bytes, kMaxStaging) && staging_buffer) return true;
        std::size_t capacity = staging_capacity ? staging_capacity : (1u << 20);
        while (capacity < wanted) capacity *= 2;
        capacity = std::min(capacity, kMaxStaging);

        if (staging_mapped) {
            unmap_memory(device, staging_memory);
            staging_mapped = nullptr;
        }
        if (staging_buffer) {
            destroy_buffer(device, staging_buffer, nullptr);
            staging_buffer = nullptr;
        }
        if (staging_memory) {
            free_memory(device, staging_memory, nullptr);
            staging_memory = nullptr;
        }
        staging_capacity = 0;

        const VkBufferCreateInfo buffer_info{
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            nullptr,
            0,
            static_cast<VkDeviceSize>(capacity),
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_SHARING_MODE_EXCLUSIVE,
            0,
            nullptr,
        };
        if (create_buffer(device, &buffer_info, nullptr, &staging_buffer) != VK_SUCCESS || !staging_buffer) {
            fast_error = "vkCreateBuffer failed for staging";
            return false;
        }
        VkMemoryRequirements requirements{};
        get_buffer_memory_requirements(device, staging_buffer, &requirements);
        std::uint32_t type = 0xffffffffu;
        for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
            const bool allowed = (requirements.memoryTypeBits & (1u << i)) != 0;
            const auto flags = memory_properties.memoryTypes[i].propertyFlags;
            if (allowed && (flags & kMemHostVisibleBit) && (flags & kMemHostCoherentBit)) {
                type = i;
                break;
            }
        }
        if (type == 0xffffffffu) {
            destroy_buffer(device, staging_buffer, nullptr);
            staging_buffer = nullptr;
            fast_error = "no host-visible coherent memory type for staging";
            return false;
        }
        const VkMemoryAllocateInfo alloc_info{
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            nullptr,
            requirements.size,
            type,
        };
        if (allocate_memory(device, &alloc_info, nullptr, &staging_memory) != VK_SUCCESS || !staging_memory) {
            destroy_buffer(device, staging_buffer, nullptr);
            staging_buffer = nullptr;
            fast_error = "vkAllocateMemory failed for staging";
            return false;
        }
        if (bind_buffer_memory(device, staging_buffer, staging_memory, 0) != VK_SUCCESS) {
            destroy_buffer(device, staging_buffer, nullptr);
            free_memory(device, staging_memory, nullptr);
            staging_buffer = nullptr;
            staging_memory = nullptr;
            fast_error = "vkBindBufferMemory failed for staging";
            return false;
        }
        if (map_memory(device, staging_memory, 0, static_cast<VkDeviceSize>(capacity), 0, &staging_mapped) !=
                VK_SUCCESS ||
            !staging_mapped) {
            destroy_buffer(device, staging_buffer, nullptr);
            free_memory(device, staging_memory, nullptr);
            staging_buffer = nullptr;
            staging_memory = nullptr;
            staging_mapped = nullptr;
            fast_error = "vkMapMemory failed for staging";
            return false;
        }
        staging_capacity = capacity;
        return true;
    }

    bool run_transfer(VkBuffer src, VkBuffer dst, VkDeviceSize src_offset, VkDeviceSize dst_offset,
                      VkDeviceSize size, std::string& error_out) {
        if (reset_command_pool(device, transfer_command_pool, 0) != VK_SUCCESS) {
            error_out = "vkResetCommandPool failed for staging transfer";
            return false;
        }
        const VkCommandBufferBeginInfo begin_info{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            nullptr,
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            nullptr,
        };
        if (begin_command_buffer(transfer_command_buffer, &begin_info) != VK_SUCCESS) {
            error_out = "vkBeginCommandBuffer failed for staging transfer";
            return false;
        }
        const VkBufferCopy region{src_offset, dst_offset, size};
        cmd_copy_buffer(transfer_command_buffer, src, dst, 1, &region);
        if (end_command_buffer(transfer_command_buffer) != VK_SUCCESS) {
            error_out = "vkEndCommandBuffer failed for staging transfer";
            return false;
        }
        const VkSubmitInfo submit_info{
            VK_STRUCTURE_TYPE_SUBMIT_INFO,
            nullptr,
            0,
            nullptr,
            nullptr,
            1,
            &transfer_command_buffer,
            0,
            nullptr,
        };
        if (queue_submit(queue, 1, &submit_info, transfer_fence) != VK_SUCCESS) {
            error_out = "vkQueueSubmit failed for staging transfer";
            return false;
        }
        if (wait_for_fences(device, 1, &transfer_fence, 1, ~std::uint64_t{0}) != VK_SUCCESS) {
            error_out = "vkWaitForFences failed for staging transfer";
            return false;
        }
        if (reset_fences(device, 1, &transfer_fence) != VK_SUCCESS) {
            error_out = "vkResetFences failed for staging transfer";
            return false;
        }
        return true;
    }

    bool staging_write(VkBuffer dst, const void* data, std::size_t bytes, std::size_t offset,
                       std::string& error_out) {
        if (!ensure_fast_path() || !ensure_staging(bytes)) {
            error_out = fast_error;
            return false;
        }
        const auto* src = static_cast<const std::uint8_t*>(data);
        std::size_t done = 0;
        while (done < bytes) {
            const std::size_t chunk = std::min(bytes - done, staging_capacity);
            std::memcpy(staging_mapped, src + done, chunk);
            if (!run_transfer(staging_buffer, dst, 0, static_cast<VkDeviceSize>(offset + done),
                              static_cast<VkDeviceSize>(chunk), error_out)) {
                return false;
            }
            done += chunk;
        }
        return true;
    }

    bool staging_read(VkBuffer src, void* data, std::size_t bytes, std::size_t offset, std::string& error_out) {
        if (!ensure_fast_path() || !ensure_staging(bytes)) {
            error_out = fast_error;
            return false;
        }
        auto* dst = static_cast<std::uint8_t*>(data);
        std::size_t done = 0;
        while (done < bytes) {
            const std::size_t chunk = std::min(bytes - done, staging_capacity);
            if (!run_transfer(src, staging_buffer, static_cast<VkDeviceSize>(offset + done), 0,
                              static_cast<VkDeviceSize>(chunk), error_out)) {
                return false;
            }
            std::memcpy(dst + done, staging_mapped, chunk);
            done += chunk;
        }
        return true;
    }

    void fail(const std::string& message) {
        error = message;
        ready = false;
    }

    bool initialize() {
        if (!open_vulkan_loader(library)) {
            fail("Vulkan loader not found");
            return false;
        }

        get_proc = load_symbol<PFN_vkGetInstanceProcAddr>(library, "vkGetInstanceProcAddr");
        if (!get_proc) {
            fail("vkGetInstanceProcAddr not found");
            return false;
        }

        auto create_instance = load_instance_function<PFN_vkCreateInstance>(get_proc, nullptr, "vkCreateInstance");
        if (!create_instance) create_instance = load_symbol<PFN_vkCreateInstance>(library, "vkCreateInstance");
        if (!create_instance) {
            fail("Vulkan loader is missing vkCreateInstance");
            return false;
        }

        const VkApplicationInfo app_info{
            VK_STRUCTURE_TYPE_APPLICATION_INFO,
            nullptr,
            "MotifCL persistent Vulkan runtime",
            1,
            "MotifCL",
            1,
            vk_make_api_version(0, 1, 1, 0),
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
        if (create_instance(&create_info, nullptr, &instance) != VK_SUCCESS || !instance) {
            fail("vkCreateInstance failed");
            return false;
        }

        destroy_instance = load_instance_function<PFN_vkDestroyInstance>(get_proc, instance, "vkDestroyInstance");
        if (!destroy_instance) destroy_instance = load_symbol<PFN_vkDestroyInstance>(library, "vkDestroyInstance");
        if (!destroy_instance) {
            fail("Vulkan loader is missing vkDestroyInstance");
            return false;
        }

        auto enumerate_physical_devices =
            load_instance_function<PFN_vkEnumeratePhysicalDevices>(get_proc, instance, "vkEnumeratePhysicalDevices");
        auto get_physical_device_properties =
            load_instance_function<PFN_vkGetPhysicalDeviceProperties>(get_proc, instance, "vkGetPhysicalDeviceProperties");
        auto get_queue_family_properties = load_instance_function<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
            get_proc, instance, "vkGetPhysicalDeviceQueueFamilyProperties");
        auto get_memory_properties = load_instance_function<PFN_vkGetPhysicalDeviceMemoryProperties>(
            get_proc, instance, "vkGetPhysicalDeviceMemoryProperties");
        auto enumerate_device_extension_properties =
            load_instance_function<PFN_vkEnumerateDeviceExtensionProperties>(
                get_proc, instance, "vkEnumerateDeviceExtensionProperties");
        auto get_physical_device_features2 =
            load_instance_function<PFN_vkGetPhysicalDeviceFeatures2>(
                get_proc, instance, "vkGetPhysicalDeviceFeatures2");
        auto get_physical_device_properties2 =
            load_instance_function<PFN_vkGetPhysicalDeviceProperties2>(
                get_proc, instance, "vkGetPhysicalDeviceProperties2");
        if (!get_physical_device_properties2) {
            // On Vulkan 1.0 instances the function is exposed as a KHR
            // extension entry point; try the suffixed name too.
            get_physical_device_properties2 =
                load_instance_function<PFN_vkGetPhysicalDeviceProperties2>(
                    get_proc, instance, "vkGetPhysicalDeviceProperties2KHR");
        }
        auto create_device = load_instance_function<PFN_vkCreateDevice>(get_proc, instance, "vkCreateDevice");
        get_device_proc = load_instance_function<PFN_vkGetDeviceProcAddr>(get_proc, instance, "vkGetDeviceProcAddr");
        if (!enumerate_physical_devices || !get_queue_family_properties || !get_memory_properties ||
            !create_device || !get_device_proc) {
            fail("Vulkan instance is missing required compute setup functions");
            return false;
        }

        std::uint32_t physical_device_count = 0;
        if (enumerate_physical_devices(instance, &physical_device_count, nullptr) != VK_SUCCESS ||
            physical_device_count == 0) {
            fail("No Vulkan physical devices found");
            return false;
        }
        std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
        if (enumerate_physical_devices(instance, &physical_device_count, physical_devices.data()) != VK_SUCCESS) {
            fail("Failed to enumerate Vulkan physical devices");
            return false;
        }

        for (auto candidate : physical_devices) {
            std::uint32_t queue_family_count = 0;
            get_queue_family_properties(candidate, &queue_family_count, nullptr);
            std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
            if (queue_family_count > 0) {
                get_queue_family_properties(candidate, &queue_family_count, queue_families.data());
            }
            for (std::uint32_t i = 0; i < queue_family_count; ++i) {
                if (queue_families[i].queueCount > 0 &&
                    (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                    physical_device = candidate;
                    queue_family = i;
                    timestamp_valid_bits = queue_families[i].timestampValidBits;
                    break;
                }
            }
            if (physical_device) break;
        }
        if (!physical_device) {
            fail("No Vulkan compute queue family found");
            return false;
        }

        if (get_physical_device_properties) {
            struct alignas(8) PropertiesStorage {
                std::array<std::uint8_t, 4096> bytes;
            };
            PropertiesStorage storage{};
            get_physical_device_properties(physical_device, storage.bytes.data());
            const auto* prefix = reinterpret_cast<const VkPhysicalDevicePropertiesPrefix*>(storage.bytes.data());
            device_name = bounded_string(prefix->deviceName, sizeof(prefix->deviceName));

            const auto* full = reinterpret_cast<const VkPhysicalDevicePropertiesFull*>(storage.bytes.data());
            const float period = full->limits.timestampPeriod;
            if (period > 0.0f && period < 1.0e6f) {
                caps.timestamp_period_ns = static_cast<double>(period);
            }
            const auto shared_bytes = full->limits.maxComputeSharedMemorySize;
            if (shared_bytes >= 1024u && shared_bytes <= (1u << 20)) {
                caps.max_shared_memory_bytes = shared_bytes;
            }
            const auto invocations = full->limits.maxComputeWorkGroupInvocations;
            if (invocations >= 64u && invocations <= 4096u) {
                caps.max_workgroup_invocations = invocations;
            }
            const auto push_bytes = full->limits.maxPushConstantsSize;
            if (push_bytes >= 128u && push_bytes <= 4096u) {
                caps.max_push_constant_bytes = push_bytes;
            }
        }
        caps.timestamps = timestamp_valid_bits > 0 && caps.timestamp_period_ns > 0.0;

        // Subgroup capability detection (vulkan1.1+): query
        // VkPhysicalDeviceSubgroupProperties through the properties2 chain.
        // caps.subgroup_arithmetic_compute gates whether reduction kernels
        // may use subgroupAdd / subgroupBroadcast instead of the manual
        // shared-memory tree reduction.
        if (get_physical_device_properties2) {
            if (std::getenv("MOTIFCL_VULKAN_DEBUG_SUBGROUP")) {
                std::fprintf(stderr, "get_physical_device_properties2 loaded, querying subgroup\n");
            }
            VkPhysicalDeviceSubgroupProperties subgroup{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
                nullptr,
                0, 0, 0, 0,
            };
            // VkPhysicalDeviceProperties2 with the real inline-struct layout
            // (sType + pNext + ~2KB inline VkPhysicalDeviceProperties). The
            // driver writes both the inline properties and any chained ext
            // structs (subgroup here) in one call.
            VkPhysicalDeviceProperties2 props2{};
            constexpr std::int32_t VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 = 1000059000;
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props2.pNext = &subgroup;
            get_physical_device_properties2(physical_device, &props2);
            if (std::getenv("MOTIFCL_VULKAN_DEBUG_SUBGROUP")) {
                std::fprintf(stderr, "subgroup: size=%u stages=0x%x ops=0x%x\n",
                             subgroup.subgroupSize, subgroup.supportedStages, subgroup.supportedOperations);
            }
            // VK_SHADER_STAGE_COMPUTE_BIT = 0x10, VK_SUBGROUP_FEATURE_ARITHMETIC_BIT = 0x4.
            if ((subgroup.supportedStages & 0x10u) && (subgroup.supportedOperations & 0x4u) &&
                subgroup.subgroupSize > 0) {
                caps.subgroup_arithmetic_compute = true;
                caps.subgroup_size = subgroup.subgroupSize;
            }
        }

        get_memory_properties(physical_device, &memory_properties);

        bool has_i8_storage_extension = false;
        bool has_atomic_float_extension = false;
        if (enumerate_device_extension_properties) {
            std::uint32_t extension_count = 0;
            if (enumerate_device_extension_properties(physical_device, nullptr, &extension_count, nullptr) ==
                    VK_SUCCESS &&
                extension_count > 0) {
                std::vector<VkExtensionProperties> extensions(extension_count);
                if (enumerate_device_extension_properties(physical_device, nullptr, &extension_count,
                                                          extensions.data()) == VK_SUCCESS) {
                    for (const auto& extension : extensions) {
                        if (std::strcmp(extension.extensionName, "VK_KHR_8bit_storage") == 0) {
                            has_i8_storage_extension = true;
                        }
                        if (std::strcmp(extension.extensionName, "VK_EXT_shader_atomic_float") == 0) {
                            has_atomic_float_extension = true;
                        }
                    }
                }
            }
        }

        VkPhysicalDevice8BitStorageFeatures queried_i8_features{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
            nullptr,
            0,
            0,
            0,
        };
        if (has_i8_storage_extension && get_physical_device_features2) {
            VkPhysicalDeviceFeatures2 features2{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                &queried_i8_features,
                {},
            };
            get_physical_device_features2(physical_device, &features2);
        }
        VkPhysicalDevice8BitStorageFeatures enabled_i8_features{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
            nullptr,
            0,
            queried_i8_features.uniformAndStorageBuffer8BitAccess ? 1u : 0u,
            0,
        };
        const bool enable_i8_storage =
            has_i8_storage_extension && enabled_i8_features.uniformAndStorageBuffer8BitAccess != 0;
        const char* i8_storage_extension = "VK_KHR_8bit_storage";
        const char* atomic_float_extension = "VK_EXT_shader_atomic_float";
        // Enable VK_EXT_shader_atomic_float when present so the
        // embedding_weight_backward scatter kernel can use float atomicAdd.
        // GCN4 (RX 580) does NOT expose this extension (no hardware float
        // atomics); the scatter path stays dormant and the brute-force
        // embedding weight backward is used instead.
        const bool enable_atomic_float = has_atomic_float_extension;
        std::vector<const char*> enabled_device_extensions;
        if (enable_i8_storage) enabled_device_extensions.push_back(i8_storage_extension);
        if (enable_atomic_float) enabled_device_extensions.push_back(atomic_float_extension);

        const float priority = 1.0f;
        const VkDeviceQueueCreateInfo queue_create_info{
            VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            nullptr,
            0,
            queue_family,
            1,
            &priority,
        };
        const VkDeviceCreateInfo device_create_info{
            VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            enable_i8_storage ? &enabled_i8_features : nullptr,
            0,
            1,
            &queue_create_info,
            0,
            nullptr,
            static_cast<std::uint32_t>(enabled_device_extensions.size()),
            enabled_device_extensions.empty() ? nullptr : enabled_device_extensions.data(),
            nullptr,
        };
        if (create_device(physical_device, &device_create_info, nullptr, &device) != VK_SUCCESS || !device) {
            fail("vkCreateDevice failed");
            return false;
        }
        storage_buffer_i8 = enable_i8_storage;
        // Surface atomic-float capability through VulkanDeviceCaps so callers
        // can pick the scatter path only where it actually works. The smoke
        // validation runs later (VulkanRuntime::create) once the buffer/
        // dispatch machinery is fully wired up.
        caps_supports_atomic_float = enable_atomic_float;
        caps.supports_atomic_float = enable_atomic_float;
        atomic_float_smoke_pending = enable_atomic_float;

        auto load_device = [&](auto tag, const char* name) {
            using Fn = decltype(tag);
            return load_device_function<Fn>(get_device_proc, device, name);
        };
        destroy_device = load_device(PFN_vkDestroyDevice{}, "vkDestroyDevice");
        get_device_queue = load_device(PFN_vkGetDeviceQueue{}, "vkGetDeviceQueue");
        create_buffer = load_device(PFN_vkCreateBuffer{}, "vkCreateBuffer");
        destroy_buffer = load_device(PFN_vkDestroyBuffer{}, "vkDestroyBuffer");
        get_buffer_memory_requirements =
            load_device(PFN_vkGetBufferMemoryRequirements{}, "vkGetBufferMemoryRequirements");
        allocate_memory = load_device(PFN_vkAllocateMemory{}, "vkAllocateMemory");
        free_memory = load_device(PFN_vkFreeMemory{}, "vkFreeMemory");
        bind_buffer_memory = load_device(PFN_vkBindBufferMemory{}, "vkBindBufferMemory");
        map_memory = load_device(PFN_vkMapMemory{}, "vkMapMemory");
        unmap_memory = load_device(PFN_vkUnmapMemory{}, "vkUnmapMemory");
        create_descriptor_set_layout =
            load_device(PFN_vkCreateDescriptorSetLayout{}, "vkCreateDescriptorSetLayout");
        destroy_descriptor_set_layout =
            load_device(PFN_vkDestroyDescriptorSetLayout{}, "vkDestroyDescriptorSetLayout");
        create_descriptor_pool = load_device(PFN_vkCreateDescriptorPool{}, "vkCreateDescriptorPool");
        destroy_descriptor_pool = load_device(PFN_vkDestroyDescriptorPool{}, "vkDestroyDescriptorPool");
        allocate_descriptor_sets = load_device(PFN_vkAllocateDescriptorSets{}, "vkAllocateDescriptorSets");
        update_descriptor_sets = load_device(PFN_vkUpdateDescriptorSets{}, "vkUpdateDescriptorSets");
        create_pipeline_layout = load_device(PFN_vkCreatePipelineLayout{}, "vkCreatePipelineLayout");
        destroy_pipeline_layout = load_device(PFN_vkDestroyPipelineLayout{}, "vkDestroyPipelineLayout");
        create_shader_module = load_device(PFN_vkCreateShaderModule{}, "vkCreateShaderModule");
        destroy_shader_module = load_device(PFN_vkDestroyShaderModule{}, "vkDestroyShaderModule");
        create_compute_pipelines = load_device(PFN_vkCreateComputePipelines{}, "vkCreateComputePipelines");
        destroy_pipeline = load_device(PFN_vkDestroyPipeline{}, "vkDestroyPipeline");
        create_pipeline_cache = load_device(PFN_vkCreatePipelineCache{}, "vkCreatePipelineCache");
        destroy_pipeline_cache = load_device(PFN_vkDestroyPipelineCache{}, "vkDestroyPipelineCache");
        get_pipeline_cache_data = load_device(PFN_vkGetPipelineCacheData{}, "vkGetPipelineCacheData");
        create_command_pool = load_device(PFN_vkCreateCommandPool{}, "vkCreateCommandPool");
        destroy_command_pool = load_device(PFN_vkDestroyCommandPool{}, "vkDestroyCommandPool");
        allocate_command_buffers = load_device(PFN_vkAllocateCommandBuffers{}, "vkAllocateCommandBuffers");
        begin_command_buffer = load_device(PFN_vkBeginCommandBuffer{}, "vkBeginCommandBuffer");
        end_command_buffer = load_device(PFN_vkEndCommandBuffer{}, "vkEndCommandBuffer");
        cmd_bind_pipeline = load_device(PFN_vkCmdBindPipeline{}, "vkCmdBindPipeline");
        cmd_bind_descriptor_sets = load_device(PFN_vkCmdBindDescriptorSets{}, "vkCmdBindDescriptorSets");
        cmd_dispatch = load_device(PFN_vkCmdDispatch{}, "vkCmdDispatch");
        queue_submit = load_device(PFN_vkQueueSubmit{}, "vkQueueSubmit");
        queue_wait_idle = load_device(PFN_vkQueueWaitIdle{}, "vkQueueWaitIdle");
        create_fence = load_device(PFN_vkCreateFence{}, "vkCreateFence");
        destroy_fence = load_device(PFN_vkDestroyFence{}, "vkDestroyFence");
        reset_fences = load_device(PFN_vkResetFences{}, "vkResetFences");
        wait_for_fences = load_device(PFN_vkWaitForFences{}, "vkWaitForFences");
        reset_command_pool = load_device(PFN_vkResetCommandPool{}, "vkResetCommandPool");
        cmd_push_constants = load_device(PFN_vkCmdPushConstants{}, "vkCmdPushConstants");
        cmd_pipeline_barrier = load_device(PFN_vkCmdPipelineBarrier{}, "vkCmdPipelineBarrier");
        cmd_copy_buffer = load_device(PFN_vkCmdCopyBuffer{}, "vkCmdCopyBuffer");
        create_query_pool = load_device(PFN_vkCreateQueryPool{}, "vkCreateQueryPool");
        destroy_query_pool = load_device(PFN_vkDestroyQueryPool{}, "vkDestroyQueryPool");
        cmd_reset_query_pool = load_device(PFN_vkCmdResetQueryPool{}, "vkCmdResetQueryPool");
        cmd_write_timestamp = load_device(PFN_vkCmdWriteTimestamp{}, "vkCmdWriteTimestamp");
        get_query_pool_results = load_device(PFN_vkGetQueryPoolResults{}, "vkGetQueryPoolResults");

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
            fail("Vulkan device is missing required compute functions");
            return false;
        }

        get_device_queue(device, queue_family, 0, &queue);
        if (!queue) {
            fail("vkGetDeviceQueue returned null");
            return false;
        }

        error.clear();
        ready = true;
        return true;
    }

    VulkanStorageBufferDispatchResult dispatch_storage_buffers(
        const std::uint32_t* spirv,
        std::size_t spirv_word_count,
        const std::vector<VulkanStorageBufferSpec>& buffer_specs,
        const std::vector<std::size_t>& output_buffer_indices,
        std::uint32_t group_count_x,
        std::uint32_t group_count_y,
        std::uint32_t group_count_z) {
        VulkanStorageBufferDispatchResult result;
        result.device_name = device_name;

        auto fail_result = [&](const std::string& message) {
            result.error = message;
            result.success = false;
            return result;
        };
        if (!ready) return fail_result(error.empty() ? "Vulkan runtime is not available" : error);
        if (!spirv || spirv_word_count == 0) return fail_result("Vulkan compute shader is empty");
        if (buffer_specs.empty()) return fail_result("Vulkan compute requires at least one storage buffer");
        if (buffer_specs.size() > 16) return fail_result("Vulkan compute smoke supports at most 16 storage buffers");
        if (group_count_x == 0 || group_count_y == 0 || group_count_z == 0) {
            return fail_result("Vulkan compute dispatch dimensions must be non-zero");
        }
        for (std::size_t i = 0; i < buffer_specs.size(); ++i) {
            if (buffer_specs[i].nbytes == 0) return fail_result("Vulkan compute storage buffer must be non-empty");
        }
        for (const auto output_index : output_buffer_indices) {
            if (output_index >= buffer_specs.size()) return fail_result("Vulkan compute output index is out of range");
        }

        const std::size_t buffer_count = buffer_specs.size();
        std::vector<VkBuffer> buffers(buffer_count, nullptr);
        std::vector<VkDeviceMemory> memories(buffer_count, nullptr);
        VkDescriptorSetLayout descriptor_set_layout = nullptr;
        VkDescriptorPool descriptor_pool = nullptr;
        VkPipelineLayout pipeline_layout = nullptr;
        VkShaderModule shader_module = nullptr;
        VkPipeline pipeline = nullptr;
        VkCommandPool command_pool = nullptr;

        auto cleanup = [&]() {
            if (destroy_command_pool && command_pool) destroy_command_pool(device, command_pool, nullptr);
            if (destroy_pipeline && pipeline) destroy_pipeline(device, pipeline, nullptr);
            if (destroy_shader_module && shader_module) destroy_shader_module(device, shader_module, nullptr);
            if (destroy_pipeline_layout && pipeline_layout) destroy_pipeline_layout(device, pipeline_layout, nullptr);
            if (destroy_descriptor_pool && descriptor_pool) destroy_descriptor_pool(device, descriptor_pool, nullptr);
            if (destroy_descriptor_set_layout && descriptor_set_layout) {
                destroy_descriptor_set_layout(device, descriptor_set_layout, nullptr);
            }
            if (destroy_buffer) {
                for (auto buffer : buffers) {
                    if (buffer) destroy_buffer(device, buffer, nullptr);
                }
            }
            if (free_memory) {
                for (auto memory : memories) {
                    if (memory) free_memory(device, memory, nullptr);
                }
            }
        };

        auto fail_with_cleanup = [&](const std::string& message) {
            cleanup();
            return fail_result(message);
        };

        for (std::size_t i = 0; i < buffer_count; ++i) {
            const VkBufferCreateInfo buffer_create_info{
                VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                nullptr,
                0,
                static_cast<VkDeviceSize>(buffer_specs[i].nbytes),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_SHARING_MODE_EXCLUSIVE,
                0,
                nullptr,
            };
            if (create_buffer(device, &buffer_create_info, nullptr, &buffers[i]) != VK_SUCCESS || !buffers[i]) {
                return fail_with_cleanup("vkCreateBuffer failed");
            }

            VkMemoryRequirements memory_requirements{};
            get_buffer_memory_requirements(device, buffers[i], &memory_requirements);
            const std::uint32_t memory_type =
                find_host_visible_coherent_memory_type(memory_properties, memory_requirements.memoryTypeBits);
            if (memory_type == 0xffffffffu) {
                return fail_with_cleanup("No host-visible coherent Vulkan memory type for storage buffer");
            }
            const VkMemoryAllocateInfo allocate_info{
                VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                nullptr,
                memory_requirements.size,
                memory_type,
            };
            if (allocate_memory(device, &allocate_info, nullptr, &memories[i]) != VK_SUCCESS || !memories[i]) {
                return fail_with_cleanup("vkAllocateMemory failed");
            }
            if (bind_buffer_memory(device, buffers[i], memories[i], 0) != VK_SUCCESS) {
                return fail_with_cleanup("vkBindBufferMemory failed");
            }

            void* mapped = nullptr;
            if (map_memory(device, memories[i], 0, static_cast<VkDeviceSize>(buffer_specs[i].nbytes), 0, &mapped) !=
                    VK_SUCCESS ||
                !mapped) {
                return fail_with_cleanup("vkMapMemory failed while initializing storage buffer");
            }
            if (buffer_specs[i].initial_data) {
                std::memcpy(mapped, buffer_specs[i].initial_data, buffer_specs[i].nbytes);
            } else {
                std::memset(mapped, 0, buffer_specs[i].nbytes);
            }
            unmap_memory(device, memories[i]);
        }

        std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.reserve(buffer_count);
        for (std::size_t i = 0; i < buffer_count; ++i) {
            bindings.push_back(VkDescriptorSetLayoutBinding{
                static_cast<std::uint32_t>(i),
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                1,
                VK_SHADER_STAGE_COMPUTE_BIT,
                nullptr,
            });
        }
        const VkDescriptorSetLayoutCreateInfo descriptor_set_layout_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            nullptr,
            0,
            static_cast<std::uint32_t>(bindings.size()),
            bindings.data(),
        };
        if (create_descriptor_set_layout(device, &descriptor_set_layout_info, nullptr, &descriptor_set_layout) !=
                VK_SUCCESS ||
            !descriptor_set_layout) {
            return fail_with_cleanup("vkCreateDescriptorSetLayout failed");
        }

        const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, static_cast<std::uint32_t>(buffer_count)};
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

        std::vector<VkDescriptorBufferInfo> descriptor_buffer_infos;
        std::vector<VkWriteDescriptorSet> write_descriptors;
        descriptor_buffer_infos.reserve(buffer_count);
        write_descriptors.reserve(buffer_count);
        for (std::size_t i = 0; i < buffer_count; ++i) {
            descriptor_buffer_infos.push_back(VkDescriptorBufferInfo{
                buffers[i],
                0,
                static_cast<VkDeviceSize>(buffer_specs[i].nbytes),
            });
            write_descriptors.push_back(VkWriteDescriptorSet{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                descriptor_set,
                static_cast<std::uint32_t>(i),
                0,
                1,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                nullptr,
                &descriptor_buffer_infos.back(),
                nullptr,
            });
        }
        update_descriptor_sets(device, static_cast<std::uint32_t>(write_descriptors.size()), write_descriptors.data(),
                               0, nullptr);

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

        const VkShaderModuleCreateInfo shader_module_info{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            nullptr,
            0,
            spirv_word_count * sizeof(std::uint32_t),
            spirv,
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
        if (create_compute_pipelines(device, vk_pipeline_cache, 1, &compute_pipeline_info, nullptr, &pipeline) != VK_SUCCESS ||
            !pipeline) {
            return fail_with_cleanup("vkCreateComputePipelines failed");
        }

        const VkCommandPoolCreateInfo command_pool_info{
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            nullptr,
            0,
            queue_family,
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
        cmd_dispatch(command_buffer, group_count_x, group_count_y, group_count_z);
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

        result.outputs.clear();
        result.outputs.reserve(output_buffer_indices.size());
        for (const auto output_index : output_buffer_indices) {
            const auto nbytes = buffer_specs[output_index].nbytes;
            std::vector<std::uint8_t> output(nbytes);
            void* mapped = nullptr;
            if (map_memory(device, memories[output_index], 0, static_cast<VkDeviceSize>(nbytes), 0, &mapped) !=
                    VK_SUCCESS ||
                !mapped) {
                return fail_with_cleanup("vkMapMemory failed while reading storage buffer");
            }
            std::memcpy(output.data(), mapped, nbytes);
            unmap_memory(device, memories[output_index]);
            result.outputs.push_back(std::move(output));
        }

        result.success = true;
        cleanup();
        return result;
    }
};

namespace {

const std::string& empty_vulkan_runtime_string() {
    static const std::string empty;
    return empty;
}

} // namespace

VulkanRuntime::VulkanRuntime() = default;
VulkanRuntime::~VulkanRuntime() = default;
VulkanRuntime::VulkanRuntime(VulkanRuntime&&) noexcept = default;
VulkanRuntime& VulkanRuntime::operator=(VulkanRuntime&&) noexcept = default;

VulkanRuntime::VulkanRuntime(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

VulkanRuntime VulkanRuntime::create() {
    auto impl = std::make_shared<Impl>();
    impl->initialize();
    VulkanRuntime rt(std::move(impl));
    // Run the atomic-float smoke check now that the buffer/dispatch machinery
    // is fully wired up. Some drivers (GCN4 / RX 580 proprietary Windows)
    // report VK_EXT_shader_atomic_float but float atomicAdd silently produces
    // wrong results; detect that here and clear the capability so callers
    // fall back to the brute-force path.
    if (rt.impl_ && rt.impl_->atomic_float_smoke_pending) {
        // Multi-address parity smoke. The probe MUST scatter into several
        // distinct output rows (not a single address): on GCN4 (Radeon RX 580)
        // the proprietary Windows driver exposes VK_EXT_shader_atomic_float but
        // its float atomicAdd is catastrophically broken — every scattered add
        // collapses onto one address while all other rows stay zero. A
        // single-address probe (all indices == 0) passes spuriously because the
        // collapse target coincides with the sole real address; distinct rows
        // expose the corruption and clear the capability so callers fall back to
        // the correct brute-force path. Mixed indices exercise both contended
        // (row 0, twice) and single-write (rows 2, 3) accumulation.
        constexpr std::size_t kV = 4, kD = 2, kT = 4;
        const std::vector<std::int32_t> idx = {0, 2, 0, 3};
        std::vector<float> grad(kT * kD);
        for (std::size_t i = 0; i < grad.size(); ++i) grad[i] = static_cast<float>(i + 1);
        std::vector<float> expected(kV * kD, 0.0f);
        for (std::size_t t = 0; t < kT; ++t)
            for (std::size_t d = 0; d < kD; ++d)
                expected[static_cast<std::size_t>(idx[t]) * kD + d] += grad[t * kD + d];
        auto acc_buf = rt.create_buffer(kV * kD * sizeof(float));
        auto grad_buf = rt.create_buffer(grad.size() * sizeof(float), grad.data());
        auto idx_buf = rt.create_buffer(idx.size() * sizeof(std::int32_t), idx.data());
        const auto zr = run_vulkan_zero_f32(rt, acc_buf, kV * kD);
        const auto sr = zr.success
            ? run_vulkan_embedding_weight_backward_scatter(rt, idx_buf, grad_buf, acc_buf,
                                                            /*vocab=*/kV, /*embed=*/kD, /*tokens=*/kT)
            : zr;
        bool ok = sr.success;
        if (ok) {
            std::vector<float> got(kV * kD, 0.0f);
            acc_buf.download(got.data(), got.size() * sizeof(float));
            for (std::size_t i = 0; i < got.size(); ++i) {
                if (std::fabs(got[i] - expected[i]) > 1e-3f) { ok = false; break; }
            }
        }
        if (!ok) {
            rt.impl_->caps.supports_atomic_float = false;
            rt.impl_->caps_supports_atomic_float = false;
        }
        rt.impl_->atomic_float_smoke_pending = false;
    }
    // One-time matmul micro-benchmark: decide whether the register-block variant
    // (mm_f32_nn_rb4) actually beats the base 16x16 tile on this device. rb4 wins
    // on newer GPUs but loses ~1.5x on GCN4 (RX 580) where the base tile is best;
    // timing it here replaces the previous (incorrect) subgroup-support proxy so
    // matmul always runs the faster kernel per device.
    if (rt.impl_ && rt.impl_->ready) {
        constexpr std::size_t kMB = 256;
        const std::vector<float> hb(kMB * kMB, 0.01f);
        auto ba = rt.create_buffer(kMB * kMB * sizeof(float), hb.data());
        auto bb = rt.create_buffer(kMB * kMB * sizeof(float), hb.data());
        auto bc = rt.create_buffer(kMB * kMB * sizeof(float));
        const struct { std::uint32_t m, k, n; } mpush{static_cast<std::uint32_t>(kMB),
                                                       static_cast<std::uint32_t>(kMB),
                                                       static_cast<std::uint32_t>(kMB)};
        const std::vector<const VulkanBuffer*> mbufs = {&ba, &bb, &bc};
        const std::uint32_t g16 = static_cast<std::uint32_t>((kMB + 15) / 16);
        const std::uint32_t g32 = static_cast<std::uint32_t>((kMB + 31) / 32);
        auto run_base = [&]() {
            return rt.dispatch_cached(vkspirv::k_mm_f32_nn, vkspirv::k_mm_f32_nn_words, mbufs, &mpush,
                                      sizeof(mpush), g16, g16, 1).success;
        };
        auto run_rb4 = [&]() {
            return rt.dispatch_cached(vkspirv::k_mm_f32_nn_rb4, vkspirv::k_mm_f32_nn_rb4_words, mbufs, &mpush,
                                      sizeof(mpush), g32, g32, 1).success;
        };
        const bool ok = run_base() && run_rb4();  // warmup + validity (pipeline compile)
        if (ok) {
            constexpr int kIters = 8;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < kIters; ++i) run_base();
            const auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < kIters; ++i) run_rb4();
            const auto t2 = std::chrono::steady_clock::now();
            const double base_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
            const double rb4_ns = std::chrono::duration<double, std::nano>(t2 - t1).count();
            rt.impl_->caps.prefer_rb4_matmul = rb4_ns < base_ns;
        }
    }
    return rt;
}

bool VulkanRuntime::available() const {
    return impl_ && impl_->ready;
}

const std::string& VulkanRuntime::device_name() const {
    return impl_ ? impl_->device_name : empty_vulkan_runtime_string();
}

const std::string& VulkanRuntime::error() const {
    return impl_ ? impl_->error : empty_vulkan_runtime_string();
}

bool VulkanRuntime::supports_storage_buffer_i8() const {
    return impl_ && impl_->ready && impl_->storage_buffer_i8;
}

const VulkanDeviceCaps& VulkanRuntime::caps() const {
    static const VulkanDeviceCaps kEmptyCaps{};
    return impl_ ? impl_->caps : kEmptyCaps;
}

bool VulkanRuntime::batch_begin() {
    return impl_ && impl_->ready && impl_->do_batch_begin();
}

VulkanOpResult VulkanRuntime::batch_end() {
    VulkanOpResult result;
    if (!impl_) {
        result.error = "Vulkan runtime is not initialized";
        return result;
    }
    return impl_->do_batch_end();
}

bool VulkanRuntime::batch_active() const {
    return impl_ && impl_->batch_open;
}

void VulkanRuntime::set_gpu_timing_enabled(bool enabled) {
    if (impl_) impl_->timing_enabled = enabled;
}

double VulkanRuntime::last_gpu_time_us() const {
    return impl_ ? impl_->last_gpu_us : -1.0;
}

struct VulkanBuffer::Impl {
    std::shared_ptr<VulkanRuntime::Impl> runtime;
    VkBuffer buffer = nullptr;
    VkDeviceMemory memory = nullptr;
    std::size_t nbytes = 0;
    // Allocation bucket size (>= nbytes); pooled allocations are keyed by it.
    // 0 means "not poolable" (legacy exact-size allocation).
    std::size_t capacity = 0;
    // Host-visible-coherent memory maps directly; device-local memory goes
    // through the runtime staging path in upload()/download().
    bool host_visible = true;

    Impl(std::shared_ptr<VulkanRuntime::Impl> owner, VkBuffer vk_buffer, VkDeviceMemory vk_memory, std::size_t bytes,
         bool mappable = true, std::size_t bucket = 0)
        : runtime(std::move(owner)),
          buffer(vk_buffer),
          memory(vk_memory),
          nbytes(bytes),
          capacity(bucket),
          host_visible(mappable) {}

    ~Impl() {
        if (runtime && runtime->device) {
            if (capacity > 0) {
                runtime->pool_release(buffer, memory, capacity, host_visible);
                buffer = nullptr;
                memory = nullptr;
                return;
            }
            if (buffer && runtime->destroy_buffer) {
                runtime->destroy_buffer(runtime->device, buffer, nullptr);
                buffer = nullptr;
            }
            if (memory && runtime->free_memory) {
                runtime->free_memory(runtime->device, memory, nullptr);
                memory = nullptr;
            }
        }
    }
};

VulkanBuffer::VulkanBuffer() = default;
VulkanBuffer::~VulkanBuffer() = default;
VulkanBuffer::VulkanBuffer(VulkanBuffer&&) noexcept = default;
VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&&) noexcept = default;

VulkanBuffer::VulkanBuffer(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

bool VulkanBuffer::valid() const {
    return impl_ && impl_->buffer && impl_->memory && impl_->runtime && impl_->runtime->ready;
}

std::size_t VulkanBuffer::nbytes() const {
    return impl_ ? impl_->nbytes : 0;
}

void VulkanBuffer::upload(const void* data, std::size_t bytes, std::size_t offset) {
    if (!valid()) throw std::runtime_error("Vulkan buffer is not valid");
    if (!data && bytes > 0) throw std::runtime_error("Vulkan buffer upload source is null");
    if (offset > impl_->nbytes || bytes > impl_->nbytes - offset) {
        throw std::runtime_error("Vulkan buffer upload exceeds allocation size");
    }
    if (bytes == 0) return;
    if (impl_->runtime->batch_open) {
        for (const auto& kept : impl_->runtime->batch_keepalive) {
            if (kept.get() == impl_.get()) {
                throw std::runtime_error(
                    "Vulkan buffer upload forbidden while an open batch is writing to this buffer — "
                    "upload executes immediately, but the batch dispatches are deferred to batch_end");
            }
        }
    }
    if (!impl_->host_visible) {
        std::string error;
        if (!impl_->runtime->staging_write(impl_->buffer, data, bytes, offset, error)) {
            throw std::runtime_error("Vulkan staging upload failed: " + error);
        }
        return;
    }
    void* mapped = nullptr;
    if (impl_->runtime->map_memory(impl_->runtime->device, impl_->memory, static_cast<VkDeviceSize>(offset),
                                   static_cast<VkDeviceSize>(bytes), 0, &mapped) != VK_SUCCESS ||
        !mapped) {
        throw std::runtime_error("vkMapMemory failed while uploading Vulkan buffer");
    }
    if (bytes > 0) std::memcpy(mapped, data, bytes);
    impl_->runtime->unmap_memory(impl_->runtime->device, impl_->memory);
}

void VulkanBuffer::download(void* data, std::size_t bytes, std::size_t offset) const {
    if (!valid()) throw std::runtime_error("Vulkan buffer is not valid");
    if (!data && bytes > 0) throw std::runtime_error("Vulkan buffer download destination is null");
    if (offset > impl_->nbytes || bytes > impl_->nbytes - offset) {
        throw std::runtime_error("Vulkan buffer download exceeds allocation size");
    }
    if (bytes == 0) return;
    if (impl_->runtime->batch_open) {
        for (const auto& kept : impl_->runtime->batch_keepalive) {
            if (kept.get() == impl_.get()) {
                throw std::runtime_error(
                    "Vulkan buffer download forbidden while an open batch is writing to this buffer — "
                    "download reads immediately, but the batch dispatches are deferred to batch_end");
            }
        }
    }
    if (!impl_->host_visible) {
        std::string error;
        if (!impl_->runtime->staging_read(impl_->buffer, data, bytes, offset, error)) {
            throw std::runtime_error("Vulkan staging download failed: " + error);
        }
        return;
    }
    void* mapped = nullptr;
    if (impl_->runtime->map_memory(impl_->runtime->device, impl_->memory, static_cast<VkDeviceSize>(offset),
                                   static_cast<VkDeviceSize>(bytes), 0, &mapped) != VK_SUCCESS ||
        !mapped) {
        throw std::runtime_error("vkMapMemory failed while downloading Vulkan buffer");
    }
    if (bytes > 0) std::memcpy(data, mapped, bytes);
    impl_->runtime->unmap_memory(impl_->runtime->device, impl_->memory);
}

VulkanBuffer VulkanRuntime::create_buffer(std::size_t nbytes, const void* initial_data) {
    if (!available()) {
        throw std::runtime_error(error().empty() ? "Vulkan runtime is not available" : error());
    }
    if (nbytes == 0) throw std::runtime_error("Vulkan buffer allocation requires non-zero size");

    // Storage buffers prefer device-local VRAM (uploads/downloads go through
    // the staging path); MOTIFCL_VK_HOST_VISIBLE=1 forces the legacy
    // host-visible mapping behaviour.
    static const bool force_host_visible = []() {
        const char* env = std::getenv("MOTIFCL_VK_HOST_VISIBLE");
        return env && *env && *env != '0';
    }();

    // Allocations are bucketed to the next power of two and recycled through
    // the runtime buffer pool; vkAllocateMemory per Tensor temporary is far
    // too slow for autograd churn.
    std::size_t bucket = 256;
    while (bucket < nbytes) bucket <<= 1;

    {
        VkBuffer pooled_buffer = nullptr;
        VkDeviceMemory pooled_memory = nullptr;
        bool served = false;
        bool served_host_visible = false;
        if (force_host_visible) {
            // A forced-host-visible caller must get mappable memory.
            served = impl_->pool_acquire(bucket, true, pooled_buffer, pooled_memory);
            served_host_visible = true;
        } else {
            // Either class works (host-visible maps directly, device-local
            // goes through staging); prefer device-local VRAM.
            if (impl_->pool_acquire(bucket, false, pooled_buffer, pooled_memory)) {
                served = true;
                served_host_visible = false;
            } else if (impl_->pool_acquire(bucket, true, pooled_buffer, pooled_memory)) {
                served = true;
                served_host_visible = true;
            }
        }
        if (served) {
            auto buffer_impl = std::make_shared<VulkanBuffer::Impl>(impl_, pooled_buffer, pooled_memory, nbytes,
                                                                    served_host_visible, bucket);
            VulkanBuffer out(std::move(buffer_impl));
            if (initial_data) out.upload(initial_data, nbytes);
            return out;
        }
    }

    const VkBufferCreateInfo buffer_create_info{
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        nullptr,
        0,
        static_cast<VkDeviceSize>(bucket),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_SHARING_MODE_EXCLUSIVE,
        0,
        nullptr,
    };
    VkBuffer buffer = nullptr;
    if (impl_->create_buffer(impl_->device, &buffer_create_info, nullptr, &buffer) != VK_SUCCESS || !buffer) {
        throw std::runtime_error("vkCreateBuffer failed");
    }

    VkMemoryRequirements memory_requirements{};
    impl_->get_buffer_memory_requirements(impl_->device, buffer, &memory_requirements);

    const auto& memory_properties = impl_->memory_properties;
    auto pick_type = [&](VkMemoryPropertyFlags required, VkMemoryPropertyFlags forbidden) -> std::uint32_t {
        for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
            if ((memory_requirements.memoryTypeBits & (1u << i)) == 0) continue;
            const auto flags = memory_properties.memoryTypes[i].propertyFlags;
            if ((flags & required) != required) continue;
            if ((flags & forbidden) != 0) continue;
            return i;
        }
        return 0xffffffffu;
    };

    std::uint32_t memory_type = 0xffffffffu;
    bool host_visible = false;
    if (!force_host_visible) {
        memory_type = pick_type(kMemDeviceLocalBit, kMemHostVisibleBit);
    }
    if (memory_type == 0xffffffffu) {
        memory_type = pick_type(kMemHostVisibleBit | kMemHostCoherentBit, 0);
        host_visible = memory_type != 0xffffffffu;
    }
    if (memory_type == 0xffffffffu) {
        impl_->destroy_buffer(impl_->device, buffer, nullptr);
        throw std::runtime_error("No suitable Vulkan memory type for storage buffer");
    }
    const VkMemoryAllocateInfo allocate_info{
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        nullptr,
        memory_requirements.size,
        memory_type,
    };
    VkDeviceMemory memory = nullptr;
    if (impl_->allocate_memory(impl_->device, &allocate_info, nullptr, &memory) != VK_SUCCESS || !memory) {
        // Device-local heap may be exhausted; retry in host-visible memory.
        if (!host_visible) {
            memory_type = pick_type(kMemHostVisibleBit | kMemHostCoherentBit, 0);
            if (memory_type != 0xffffffffu) {
                const VkMemoryAllocateInfo retry_info{
                    VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    nullptr,
                    memory_requirements.size,
                    memory_type,
                };
                if (impl_->allocate_memory(impl_->device, &retry_info, nullptr, &memory) == VK_SUCCESS && memory) {
                    host_visible = true;
                }
            }
        }
        if (!memory) {
            impl_->destroy_buffer(impl_->device, buffer, nullptr);
            throw std::runtime_error("vkAllocateMemory failed");
        }
    }
    if (impl_->bind_buffer_memory(impl_->device, buffer, memory, 0) != VK_SUCCESS) {
        impl_->free_memory(impl_->device, memory, nullptr);
        impl_->destroy_buffer(impl_->device, buffer, nullptr);
        throw std::runtime_error("vkBindBufferMemory failed");
    }

    auto buffer_impl = std::make_shared<VulkanBuffer::Impl>(impl_, buffer, memory, nbytes, host_visible, bucket);
    VulkanBuffer out(std::move(buffer_impl));
    if (initial_data) out.upload(initial_data, nbytes);
    return out;
}

VulkanOpResult VulkanRuntime::dispatch_cached(const std::uint32_t* spirv,
                                              std::size_t spirv_word_count,
                                              const std::vector<const VulkanBuffer*>& buffers,
                                              const void* push_constants,
                                              std::uint32_t push_constant_bytes,
                                              std::uint32_t group_count_x,
                                              std::uint32_t group_count_y,
                                              std::uint32_t group_count_z) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (!impl_ || !impl_->ready) {
        return fail(error().empty() ? "Vulkan runtime is not available" : error());
    }
    result.device_name = device_name();
    if (!spirv || spirv_word_count == 0) return fail("Vulkan compute shader is empty");
    if (buffers.empty() || buffers.size() > 16) {
        return fail("Vulkan cached dispatch supports 1..16 storage buffers");
    }
    if (group_count_x == 0 || group_count_y == 0 || group_count_z == 0) {
        return fail("Vulkan compute dispatch dimensions must be non-zero");
    }
    if (push_constant_bytes > 0 && !push_constants) {
        return fail("Vulkan cached dispatch push constant payload is null");
    }
    std::array<VkBuffer, 16> raw_buffers{};
    std::array<VkDeviceSize, 16> raw_sizes{};
    for (std::size_t i = 0; i < buffers.size(); ++i) {
        if (!buffers[i] || !buffers[i]->valid()) return fail("Vulkan cached dispatch buffer must be valid");
        if (!buffers[i]->impl_ || buffers[i]->impl_->runtime.get() != impl_.get()) {
            return fail("Vulkan cached dispatch buffers must belong to the same VulkanRuntime");
        }
        raw_buffers[i] = buffers[i]->impl_->buffer;
        raw_sizes[i] = static_cast<VkDeviceSize>(buffers[i]->impl_->nbytes);
    }
    if (impl_->batch_open) {
        for (const auto* buffer : buffers) impl_->batch_keepalive.push_back(buffer->impl_);
    }
    if (impl_->capturing) {
        Impl::CapturedDispatch entry;
        entry.spirv = spirv;
        entry.spirv_words = spirv_word_count;
        entry.keepalive.reserve(buffers.size());
        entry.raw_buffers.assign(raw_buffers.begin(), raw_buffers.begin() + buffers.size());
        entry.raw_sizes.assign(raw_sizes.begin(), raw_sizes.begin() + buffers.size());
        for (const auto* buffer : buffers) entry.keepalive.push_back(buffer->impl_);
        if (push_constant_bytes > 0) {
            const auto* bytes = static_cast<const std::uint8_t*>(push_constants);
            entry.push.assign(bytes, bytes + push_constant_bytes);
        }
        entry.gx = group_count_x;
        entry.gy = group_count_y;
        entry.gz = group_count_z;
        impl_->capture_list.push_back(std::move(entry));
    }
    return impl_->fast_dispatch(spirv, spirv_word_count, raw_buffers.data(), raw_sizes.data(),
                                static_cast<std::uint32_t>(buffers.size()), push_constants, push_constant_bytes,
                                group_count_x, group_count_y, group_count_z);
}

struct VulkanDispatchRecording::Impl {
    std::vector<VulkanRuntime::Impl::CapturedDispatch> dispatches;
};

VulkanDispatchRecording::VulkanDispatchRecording() = default;
VulkanDispatchRecording::~VulkanDispatchRecording() = default;
VulkanDispatchRecording::VulkanDispatchRecording(VulkanDispatchRecording&&) noexcept = default;
VulkanDispatchRecording& VulkanDispatchRecording::operator=(VulkanDispatchRecording&&) noexcept = default;

bool VulkanDispatchRecording::empty() const {
    return !impl_ || impl_->dispatches.empty();
}

std::size_t VulkanDispatchRecording::size() const {
    return impl_ ? impl_->dispatches.size() : 0;
}

bool VulkanRuntime::capture_begin() {
    if (!impl_ || !impl_->ready || impl_->capturing || impl_->batch_open) return false;
    impl_->capturing = true;
    impl_->capture_list.clear();
    return true;
}

VulkanDispatchRecording VulkanRuntime::capture_end() {
    VulkanDispatchRecording recording;
    if (!impl_ || !impl_->capturing) return recording;
    impl_->capturing = false;
    recording.impl_ = std::make_shared<VulkanDispatchRecording::Impl>();
    recording.impl_->dispatches = std::move(impl_->capture_list);
    impl_->capture_list.clear();
    return recording;
}

bool VulkanRuntime::capture_active() const {
    return impl_ && impl_->capturing;
}

VulkanOpResult VulkanRuntime::replay(const VulkanDispatchRecording& recording) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (!impl_ || !impl_->ready) {
        return fail(error().empty() ? "Vulkan runtime is not available" : error());
    }
    result.device_name = device_name();
    if (impl_->capturing) return fail("Vulkan replay is not allowed while capturing");
    if (impl_->batch_open) return fail("Vulkan replay is not allowed inside an open batch");
    if (recording.empty()) {
        result.success = true;
        return result;
    }
    if (!impl_->do_batch_begin()) return fail(impl_->fast_error);
    for (const auto& entry : recording.impl_->dispatches) {
        auto dispatched = impl_->fast_dispatch(entry.spirv, entry.spirv_words, entry.raw_buffers.data(),
                                               entry.raw_sizes.data(),
                                               static_cast<std::uint32_t>(entry.raw_buffers.size()),
                                               entry.push.empty() ? nullptr : entry.push.data(),
                                               static_cast<std::uint32_t>(entry.push.size()), entry.gx, entry.gy,
                                               entry.gz);
        if (!dispatched.success) {
            impl_->do_batch_end();
            return dispatched;
        }
    }
    return impl_->do_batch_end();
}

VulkanStorageBufferDispatchResult VulkanRuntime::dispatch_storage_buffers(
    const std::uint32_t* spirv,
    std::size_t spirv_word_count,
    const std::vector<VulkanStorageBufferSpec>& buffer_specs,
    const std::vector<std::size_t>& output_buffer_indices,
    std::uint32_t group_count_x,
    std::uint32_t group_count_y,
    std::uint32_t group_count_z) {
    if (!impl_) {
        VulkanStorageBufferDispatchResult result;
        result.error = "Vulkan runtime is not initialized";
        return result;
    }
    return impl_->dispatch_storage_buffers(spirv, spirv_word_count, buffer_specs, output_buffer_indices,
                                           group_count_x, group_count_y, group_count_z);
}

VulkanStorageBufferDispatchResult VulkanRuntime::dispatch_storage_buffers(
    const std::uint32_t* spirv,
    std::size_t spirv_word_count,
    const std::vector<const VulkanBuffer*>& buffers,
    const std::vector<std::size_t>& output_buffer_indices,
    std::uint32_t group_count_x,
    std::uint32_t group_count_y,
    std::uint32_t group_count_z) {
    VulkanStorageBufferDispatchResult result;
    result.device_name = device_name();

    auto fail = [&](const std::string& message) {
        result.error = message;
        result.success = false;
        return result;
    };
    if (!impl_ || !impl_->ready) return fail(error().empty() ? "Vulkan runtime is not available" : error());
    if (!spirv || spirv_word_count == 0) return fail("Vulkan compute shader is empty");
    if (buffers.empty()) return fail("Vulkan compute requires at least one storage buffer");
    if (buffers.size() > 16) return fail("Vulkan compute smoke supports at most 16 storage buffers");
    if (group_count_x == 0 || group_count_y == 0 || group_count_z == 0) {
        return fail("Vulkan compute dispatch dimensions must be non-zero");
    }
    for (std::size_t i = 0; i < buffers.size(); ++i) {
        if (!buffers[i] || !buffers[i]->valid()) return fail("Vulkan compute storage buffer must be valid");
        if (buffers[i]->nbytes() == 0) return fail("Vulkan compute storage buffer must be non-empty");
        if (!buffers[i]->impl_ || buffers[i]->impl_->runtime.get() != impl_.get()) {
            return fail("Vulkan compute storage buffers must belong to the same VulkanRuntime");
        }
    }
    for (const auto output_index : output_buffer_indices) {
        if (output_index >= buffers.size()) return fail("Vulkan compute output index is out of range");
    }

    VkDescriptorSetLayout descriptor_set_layout = nullptr;
    VkDescriptorPool descriptor_pool = nullptr;
    VkPipelineLayout pipeline_layout = nullptr;
    VkShaderModule shader_module = nullptr;
    VkPipeline pipeline = nullptr;
    VkCommandPool command_pool = nullptr;

    auto cleanup = [&]() {
        if (impl_->destroy_command_pool && command_pool) impl_->destroy_command_pool(impl_->device, command_pool, nullptr);
        if (impl_->destroy_pipeline && pipeline) impl_->destroy_pipeline(impl_->device, pipeline, nullptr);
        if (impl_->destroy_shader_module && shader_module) impl_->destroy_shader_module(impl_->device, shader_module, nullptr);
        if (impl_->destroy_pipeline_layout && pipeline_layout) {
            impl_->destroy_pipeline_layout(impl_->device, pipeline_layout, nullptr);
        }
        if (impl_->destroy_descriptor_pool && descriptor_pool) {
            impl_->destroy_descriptor_pool(impl_->device, descriptor_pool, nullptr);
        }
        if (impl_->destroy_descriptor_set_layout && descriptor_set_layout) {
            impl_->destroy_descriptor_set_layout(impl_->device, descriptor_set_layout, nullptr);
        }
    };

    auto fail_with_cleanup = [&](const std::string& message) {
        cleanup();
        return fail(message);
    };

    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.reserve(buffers.size());
    for (std::size_t i = 0; i < buffers.size(); ++i) {
        bindings.push_back(VkDescriptorSetLayoutBinding{
            static_cast<std::uint32_t>(i),
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            1,
            VK_SHADER_STAGE_COMPUTE_BIT,
            nullptr,
        });
    }
    const VkDescriptorSetLayoutCreateInfo descriptor_set_layout_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        nullptr,
        0,
        static_cast<std::uint32_t>(bindings.size()),
        bindings.data(),
    };
    if (impl_->create_descriptor_set_layout(impl_->device, &descriptor_set_layout_info, nullptr,
                                            &descriptor_set_layout) != VK_SUCCESS ||
        !descriptor_set_layout) {
        return fail_with_cleanup("vkCreateDescriptorSetLayout failed");
    }

    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, static_cast<std::uint32_t>(buffers.size())};
    const VkDescriptorPoolCreateInfo descriptor_pool_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        nullptr,
        0,
        1,
        1,
        &pool_size,
    };
    if (impl_->create_descriptor_pool(impl_->device, &descriptor_pool_info, nullptr, &descriptor_pool) != VK_SUCCESS ||
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
    if (impl_->allocate_descriptor_sets(impl_->device, &descriptor_set_allocate_info, &descriptor_set) != VK_SUCCESS ||
        !descriptor_set) {
        return fail_with_cleanup("vkAllocateDescriptorSets failed");
    }

    std::vector<VkDescriptorBufferInfo> descriptor_buffer_infos;
    std::vector<VkWriteDescriptorSet> write_descriptors;
    descriptor_buffer_infos.reserve(buffers.size());
    write_descriptors.reserve(buffers.size());
    for (std::size_t i = 0; i < buffers.size(); ++i) {
        descriptor_buffer_infos.push_back(VkDescriptorBufferInfo{
            buffers[i]->impl_->buffer,
            0,
            static_cast<VkDeviceSize>(buffers[i]->impl_->nbytes),
        });
        write_descriptors.push_back(VkWriteDescriptorSet{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            descriptor_set,
            static_cast<std::uint32_t>(i),
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            &descriptor_buffer_infos.back(),
            nullptr,
        });
    }
    impl_->update_descriptor_sets(impl_->device, static_cast<std::uint32_t>(write_descriptors.size()),
                                  write_descriptors.data(), 0, nullptr);

    const VkPipelineLayoutCreateInfo pipeline_layout_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        nullptr,
        0,
        1,
        &descriptor_set_layout,
        0,
        nullptr,
    };
    if (impl_->create_pipeline_layout(impl_->device, &pipeline_layout_info, nullptr, &pipeline_layout) != VK_SUCCESS ||
        !pipeline_layout) {
        return fail_with_cleanup("vkCreatePipelineLayout failed");
    }

    const VkShaderModuleCreateInfo shader_module_info{
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        nullptr,
        0,
        spirv_word_count * sizeof(std::uint32_t),
        spirv,
    };
    if (impl_->create_shader_module(impl_->device, &shader_module_info, nullptr, &shader_module) != VK_SUCCESS ||
        !shader_module) {
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
    if (impl_->create_compute_pipelines(impl_->device, nullptr, 1, &compute_pipeline_info, nullptr, &pipeline) !=
            VK_SUCCESS ||
        !pipeline) {
        return fail_with_cleanup("vkCreateComputePipelines failed");
    }

    const VkCommandPoolCreateInfo command_pool_info{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        nullptr,
        0,
        impl_->queue_family,
    };
    if (impl_->create_command_pool(impl_->device, &command_pool_info, nullptr, &command_pool) != VK_SUCCESS ||
        !command_pool) {
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
    if (impl_->allocate_command_buffers(impl_->device, &command_buffer_allocate_info, &command_buffer) != VK_SUCCESS ||
        !command_buffer) {
        return fail_with_cleanup("vkAllocateCommandBuffers failed");
    }

    const VkCommandBufferBeginInfo begin_info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr,
        0,
        nullptr,
    };
    if (impl_->begin_command_buffer(command_buffer, &begin_info) != VK_SUCCESS) {
        return fail_with_cleanup("vkBeginCommandBuffer failed");
    }
    impl_->cmd_bind_pipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    impl_->cmd_bind_descriptor_sets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                                    &descriptor_set, 0, nullptr);
    impl_->cmd_dispatch(command_buffer, group_count_x, group_count_y, group_count_z);
    if (impl_->end_command_buffer(command_buffer) != VK_SUCCESS) {
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
    if (impl_->queue_submit(impl_->queue, 1, &submit_info, nullptr) != VK_SUCCESS) {
        return fail_with_cleanup("vkQueueSubmit failed");
    }
    if (impl_->queue_wait_idle(impl_->queue) != VK_SUCCESS) {
        return fail_with_cleanup("vkQueueWaitIdle failed");
    }

    result.outputs.clear();
    result.outputs.reserve(output_buffer_indices.size());
    for (const auto output_index : output_buffer_indices) {
        const auto nbytes = buffers[output_index]->nbytes();
        std::vector<std::uint8_t> output(nbytes);
        buffers[output_index]->download(output.data(), nbytes);
        result.outputs.push_back(std::move(output));
    }
    result.success = true;
    cleanup();
    return result;
}

VulkanSmokeComputeResult run_vulkan_smoke_compute() {
    auto runtime = VulkanRuntime::create();
    return run_vulkan_smoke_compute(runtime);
}

VulkanSmokeComputeResult run_vulkan_smoke_compute(VulkanRuntime& runtime) {
    VulkanSmokeComputeResult result;
    float output = 0.0f;
    const auto& shader = smoke_compute_spirv();
    const std::vector<VulkanStorageBufferSpec> buffers = {{nullptr, sizeof(output)}};
    const auto run = runtime.dispatch_storage_buffers(shader.data(), shader.size(), buffers, {0});
    result.success = run.success;
    result.device_name = run.device_name;
    result.error = run.error;
    if (run.success && run.outputs.size() == 1 && run.outputs[0].size() == sizeof(output)) {
        std::memcpy(&result.output, run.outputs[0].data(), sizeof(result.output));
        if (result.output != 42.0f) {
            result.success = false;
            result.error = "Vulkan smoke compute returned unexpected output";
        }
    } else if (run.success) {
        result.success = false;
        result.error = "Vulkan smoke compute returned malformed output";
    }
    return result;
}

VulkanOpResult run_vulkan_f32_matmul(VulkanRuntime& runtime,
                                     const VulkanBuffer& a,
                                     const VulkanBuffer& b,
                                     VulkanBuffer& c,
                                     std::size_t m,
                                     std::size_t k,
                                     std::size_t n) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (m == 0 || k == 0 || n == 0) return fail("Vulkan f32 matmul requires non-zero M, K, and N");
    constexpr std::size_t kMaxDim = 1u << 24;
    if (m > kMaxDim || n > kMaxDim || k > kMaxDim) {
        return fail("Vulkan f32 matmul dimensions exceed supported range");
    }
    if (a.nbytes() < m * k * sizeof(float)) return fail("Vulkan f32 matmul A buffer is too small");
    if (b.nbytes() < k * n * sizeof(float)) return fail("Vulkan f32 matmul B buffer is too small");
    if (c.nbytes() < m * n * sizeof(float)) return fail("Vulkan f32 matmul C buffer is too small");

    const std::vector<const VulkanBuffer*> buffers = {&a, &b, &c};
    if (m == 1) {
        const struct {
            std::uint32_t k;
            std::uint32_t n;
        } push{static_cast<std::uint32_t>(k), static_cast<std::uint32_t>(n)};
        return runtime.dispatch_cached(vkspirv::k_mm_f32_m1n, vkspirv::k_mm_f32_m1n_words, buffers, &push,
                                       sizeof(push), static_cast<std::uint32_t>((n + 63) / 64), 1, 1);
    }
    const struct {
        std::uint32_t m;
        std::uint32_t k;
        std::uint32_t n;
    } push{static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(k), static_cast<std::uint32_t>(n)};
    const bool use_rb4 = runtime.caps().prefer_rb4_matmul;
    if (use_rb4) {
        // Register-block 32x32 tile, 8x8 lanes (one wave64), 16 outputs/lane.
        const std::uint32_t groups_x = static_cast<std::uint32_t>((n + 31) / 32);
        const std::uint32_t groups_y = static_cast<std::uint32_t>((m + 31) / 32);
        return runtime.dispatch_cached(vkspirv::k_mm_f32_nn_rb4, vkspirv::k_mm_f32_nn_rb4_words, buffers,
                                       &push, sizeof(push), groups_x, groups_y, 1);
    }
    const std::uint32_t groups_x = static_cast<std::uint32_t>((n + 15) / 16);
    const std::uint32_t groups_y = static_cast<std::uint32_t>((m + 15) / 16);
    return runtime.dispatch_cached(vkspirv::k_mm_f32_nn, vkspirv::k_mm_f32_nn_words, buffers, &push,
                                   sizeof(push), groups_x, groups_y, 1);
}

VulkanOpResult run_vulkan_f32_matmul_transpose_b(VulkanRuntime& runtime,
                                                 const VulkanBuffer& a,
                                                 const VulkanBuffer& b,
                                                 VulkanBuffer& c,
                                                 std::size_t m,
                                                 std::size_t k,
                                                 std::size_t n) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (m == 0 || k == 0 || n == 0) return fail("Vulkan f32 transpose-B matmul requires non-zero M, K, and N");
    constexpr std::size_t kMaxDim = 1u << 24;
    if (m > kMaxDim || n > kMaxDim || k > kMaxDim) {
        return fail("Vulkan f32 transpose-B matmul dimensions exceed supported range");
    }
    if (a.nbytes() < m * k * sizeof(float)) return fail("Vulkan f32 transpose-B matmul A buffer is too small");
    if (b.nbytes() < n * k * sizeof(float)) return fail("Vulkan f32 transpose-B matmul B buffer is too small");
    if (c.nbytes() < m * n * sizeof(float)) return fail("Vulkan f32 transpose-B matmul C buffer is too small");

    const std::vector<const VulkanBuffer*> buffers = {&a, &b, &c};
    if (m == 1) {
        // Decode form: one wave64 workgroup per output element with a
        // shared-memory reduction over the contiguous B row.
        const struct {
            std::uint32_t k;
            std::uint32_t n;
        } push{static_cast<std::uint32_t>(k), static_cast<std::uint32_t>(n)};
        return runtime.dispatch_cached(vkspirv::k_mm_f32_m1nt, vkspirv::k_mm_f32_m1nt_words, buffers, &push,
                                       sizeof(push), static_cast<std::uint32_t>(n), 1, 1);
    }
    const struct {
        std::uint32_t m;
        std::uint32_t k;
        std::uint32_t n;
    } push{static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(k), static_cast<std::uint32_t>(n)};
    if (runtime.caps().prefer_rb4_matmul) {
        return runtime.dispatch_cached(vkspirv::k_mm_f32_nt_rb4, vkspirv::k_mm_f32_nt_rb4_words, buffers, &push,
                                       sizeof(push), static_cast<std::uint32_t>((n + 31) / 32),
                                       static_cast<std::uint32_t>((m + 31) / 32), 1);
    }
    return runtime.dispatch_cached(vkspirv::k_mm_f32_nt, vkspirv::k_mm_f32_nt_words, buffers, &push,
                                   sizeof(push), static_cast<std::uint32_t>((n + 15) / 16),
                                   static_cast<std::uint32_t>((m + 15) / 16), 1);
}

VulkanOpResult run_vulkan_f32_matmul_transpose_a(VulkanRuntime& runtime,
                                                 const VulkanBuffer& a,
                                                 const VulkanBuffer& b,
                                                 VulkanBuffer& c,
                                                 std::size_t m,
                                                 std::size_t k,
                                                 std::size_t n) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (m == 0 || k == 0 || n == 0) return fail("Vulkan f32 transpose-A matmul requires non-zero M, K, and N");
    constexpr std::size_t kMaxDim = 1u << 24;
    if (m > kMaxDim || n > kMaxDim || k > kMaxDim) {
        return fail("Vulkan f32 transpose-A matmul dimensions exceed supported range");
    }
    if (a.nbytes() < k * m * sizeof(float)) return fail("Vulkan f32 transpose-A matmul A buffer is too small");
    if (b.nbytes() < k * n * sizeof(float)) return fail("Vulkan f32 transpose-A matmul B buffer is too small");
    if (c.nbytes() < m * n * sizeof(float)) return fail("Vulkan f32 transpose-A matmul C buffer is too small");

    const struct {
        std::uint32_t m;
        std::uint32_t k;
        std::uint32_t n;
    } push{static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(k), static_cast<std::uint32_t>(n)};
    const std::vector<const VulkanBuffer*> buffers = {&a, &b, &c};
    if (runtime.caps().prefer_rb4_matmul) {
        return runtime.dispatch_cached(vkspirv::k_mm_f32_tn_rb4, vkspirv::k_mm_f32_tn_rb4_words, buffers, &push,
                                       sizeof(push), static_cast<std::uint32_t>((n + 31) / 32),
                                       static_cast<std::uint32_t>((m + 31) / 32), 1);
    }
    return runtime.dispatch_cached(vkspirv::k_mm_f32_tn, vkspirv::k_mm_f32_tn_words, buffers, &push,
                                   sizeof(push), static_cast<std::uint32_t>((n + 15) / 16),
                                   static_cast<std::uint32_t>((m + 15) / 16), 1);
}

VulkanOpResult run_vulkan_i8_scaled_matmul(VulkanRuntime& runtime,
                                           const VulkanBuffer& a,
                                           const VulkanBuffer& b,
                                           VulkanBuffer& c,
                                           std::size_t m,
                                           std::size_t k,
                                           std::size_t n,
                                           float scale_a,
                                           float scale_b) {
    VulkanOpResult result;
    constexpr std::size_t kMaxSpecializedK = 256;
    constexpr std::size_t kMaxDispatchDim = 4096;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    // dispatch_storage_buffers does its own vkQueueSubmit+vkQueueWaitIdle and
    // bypasses batch_begin/batch_end and capture/replay (which only
    // dispatch_cached participates in). Refuse to call it inside an active
    // batch/capture so we don't break the atomic-submit contract or drop the
    // dispatch from the recording.
    if (runtime.batch_active()) {
        return fail("Vulkan i8 scaled matmul cannot run inside an active batch "
                    "(dispatch_storage_buffers bypasses batch recording)");
    }
    if (runtime.capture_active()) {
        return fail("Vulkan i8 scaled matmul cannot run while capturing "
                    "(dispatch_storage_buffers bypasses capture recording)");
    }
    if (m == 0 || k == 0 || n == 0) return fail("Vulkan i8 scaled matmul requires non-zero M, K, and N");
    if (k > kMaxSpecializedK) return fail("Vulkan i8 scaled matmul currently supports K up to 256");
    if (m > kMaxDispatchDim || n > kMaxDispatchDim) {
        return fail("Vulkan i8 scaled matmul currently supports M,N up to 4096");
    }
    if (!std::isfinite(scale_a) || !std::isfinite(scale_b)) {
        return fail("Vulkan i8 scaled matmul requires finite scales");
    }
    if (!runtime.supports_storage_buffer_i8()) {
        return fail("Vulkan i8 scaled matmul requires VK_KHR_8bit_storage on persistent Tensor buffers");
    }
    if (m > std::numeric_limits<std::size_t>::max() / k) return fail("Vulkan i8 scaled matmul M*K overflows size_t");
    if (k > std::numeric_limits<std::size_t>::max() / n) return fail("Vulkan i8 scaled matmul K*N overflows size_t");
    if (a.nbytes() < m * k * sizeof(std::int8_t)) return fail("Vulkan i8 scaled matmul A buffer is too small");
    if (b.nbytes() < k * n * sizeof(std::int8_t)) return fail("Vulkan i8 scaled matmul B buffer is too small");
    if (c.nbytes() < m * n * sizeof(float)) return fail("Vulkan i8 scaled matmul C buffer is too small");

    const auto shader = i8_scaled_matmul_spirv(k, n, scale_a, scale_b);
    const std::vector<const VulkanBuffer*> buffers = {&a, &b, &c};
    const auto run = runtime.dispatch_storage_buffers(
        shader.data(), shader.size(), buffers, {},
        static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(m), 1);
    result.success = run.success;
    result.device_name = run.device_name;
    result.error = run.error;
    return result;
}

// === Quant core (Slice Q1) dispatchers ===

VulkanOpResult run_vulkan_quantize_q8_rowwise(VulkanRuntime& runtime,
                                              const VulkanBuffer& in_f32,
                                              VulkanBuffer& out_i8,
                                              VulkanBuffer& out_scales,
                                              std::size_t m,
                                              std::size_t k) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (m == 0 || k == 0) return fail("Vulkan quantize q8 rowwise requires non-zero M and K");
    if (m > std::numeric_limits<std::size_t>::max() / k)
        return fail("Vulkan quantize q8 rowwise M*K overflows size_t");
    const auto nbytes_in = m * k * sizeof(float);
    const auto nbytes_out = m * k * sizeof(std::int8_t);
    const auto nbytes_scale = m * sizeof(float);
    if (in_f32.nbytes() < nbytes_in) return fail("Vulkan quantize q8 rowwise input buffer is too small");
    if (out_i8.nbytes() < nbytes_out) return fail("Vulkan quantize q8 rowwise int8 buffer is too small");
    if (out_scales.nbytes() < nbytes_scale) return fail("Vulkan quantize q8 rowwise scales buffer is too small");
    constexpr std::size_t kMaxInt32 = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
    if (k > kMaxInt32) return fail("Vulkan quantize q8 rowwise K overflows int32 push constant");
    const struct {
        std::uint32_t M;
        std::uint32_t K;
        std::uint32_t N;
    } push{static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(k),
           static_cast<std::uint32_t>(m * k)};
    const std::vector<const VulkanBuffer*> buffers = {&in_f32, &out_i8, &out_scales};
    return runtime.dispatch_cached(vkspirv::k_quantize_q8_rowwise_f32,
                                   vkspirv::k_quantize_q8_rowwise_f32_words,
                                   buffers, &push, sizeof(push),
                                   static_cast<std::uint32_t>(m), 1, 1);
}

VulkanOpResult run_vulkan_dequantize_q8_scaled(VulkanRuntime& runtime,
                                               const VulkanBuffer& in_i8,
                                               const VulkanBuffer& scales,
                                               VulkanBuffer& out_f32,
                                               std::size_t count,
                                               std::uint32_t mode,
                                               std::size_t rows,
                                               std::size_t cols) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (count == 0) return fail("Vulkan dequantize q8 requires non-zero count");
    if (in_i8.nbytes() < count * sizeof(std::int8_t)) return fail("Vulkan dequantize q8 int8 buffer is too small");
    if (out_f32.nbytes() < count * sizeof(float)) return fail("Vulkan dequantize q8 output buffer is too small");
    const std::size_t min_scales = (mode == 0) ? 1 : (mode == 1) ? rows : cols;
    if (scales.nbytes() < min_scales * sizeof(float)) return fail("Vulkan dequantize q8 scales buffer is too small");
    constexpr std::size_t kMaxInt32 = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
    if (count > kMaxInt32) return fail("Vulkan dequantize q8 count overflows int32 push constant");
    const struct {
        std::uint32_t count;
        std::uint32_t mode;
        std::uint32_t rows;
        std::uint32_t cols;
    } push{static_cast<std::uint32_t>(count), mode,
           static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols)};
    const std::vector<const VulkanBuffer*> buffers = {&in_i8, &scales, &out_f32};
    return runtime.dispatch_cached(vkspirv::k_dequantize_q8_scaled_f32,
                                   vkspirv::k_dequantize_q8_scaled_f32_words,
                                   buffers, &push, sizeof(push),
                                   static_cast<std::uint32_t>((count + 63) / 64), 1, 1);
}

VulkanOpResult run_vulkan_dequantize_q4_scaled(VulkanRuntime& runtime,
                                               const VulkanBuffer& packed,
                                               const VulkanBuffer& scales,
                                               VulkanBuffer& out_f32,
                                               std::size_t count,
                                               std::uint32_t mode,
                                               std::size_t rows,
                                               std::size_t cols) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (count == 0) return fail("Vulkan dequantize q4 requires non-zero count");
    const auto packed_bytes = (count + 1) / 2;
    if (packed.nbytes() < packed_bytes) return fail("Vulkan dequantize q4 packed buffer is too small");
    if (out_f32.nbytes() < count * sizeof(float)) return fail("Vulkan dequantize q4 output buffer is too small");
    const std::size_t min_scales = (mode == 0) ? 1 : (mode == 1) ? rows : cols;
    if (scales.nbytes() < min_scales * sizeof(float)) return fail("Vulkan dequantize q4 scales buffer is too small");
    constexpr std::size_t kMaxInt32 = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
    if (count > kMaxInt32) return fail("Vulkan dequantize q4 count overflows int32 push constant");
    const struct {
        std::uint32_t count;
        std::uint32_t mode;
        std::uint32_t rows;
        std::uint32_t cols;
    } push{static_cast<std::uint32_t>(count), mode,
           static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols)};
    const std::vector<const VulkanBuffer*> buffers = {&packed, &scales, &out_f32};
    return runtime.dispatch_cached(vkspirv::k_dequantize_q4_scaled_f32,
                                   vkspirv::k_dequantize_q4_scaled_f32_words,
                                   buffers, &push, sizeof(push),
                                   static_cast<std::uint32_t>((count + 63) / 64), 1, 1);
}

VulkanOpResult run_vulkan_matmul_q8q8_scaled(VulkanRuntime& runtime,
                                             const VulkanBuffer& a_i8,
                                             const VulkanBuffer& a_scales,
                                             const VulkanBuffer& b_i8,
                                             const VulkanBuffer& b_scales,
                                             VulkanBuffer& c_f32,
                                             std::size_t m,
                                             std::size_t k,
                                             std::size_t n) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (m == 0 || k == 0 || n == 0) return fail("Vulkan q8q8 matmul requires non-zero M, K, and N");
    constexpr std::size_t kMaxDim = 1u << 24;
    if (m > kMaxDim || n > kMaxDim || k > kMaxDim) {
        return fail("Vulkan q8q8 matmul dimensions exceed supported range");
    }
    if (a_i8.nbytes() < m * k * sizeof(std::int8_t)) return fail("Vulkan q8q8 matmul A buffer is too small");
    if (b_i8.nbytes() < k * n * sizeof(std::int8_t)) return fail("Vulkan q8q8 matmul B buffer is too small");
    if (a_scales.nbytes() < m * sizeof(float)) return fail("Vulkan q8q8 matmul A scales buffer is too small");
    if (b_scales.nbytes() < n * sizeof(float)) return fail("Vulkan q8q8 matmul B scales buffer is too small");
    if (c_f32.nbytes() < m * n * sizeof(float)) return fail("Vulkan q8q8 matmul C buffer is too small");
    constexpr std::size_t kMaxInt32 = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
    if (m > kMaxInt32 || k > kMaxInt32 || n > kMaxInt32)
        return fail("Vulkan q8q8 matmul dimensions overflow push constants");
    const struct {
        std::uint32_t M;
        std::uint32_t K;
        std::uint32_t N;
    } push{static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(k),
           static_cast<std::uint32_t>(n)};
    const std::vector<const VulkanBuffer*> buffers = {&a_i8, &b_i8, &a_scales, &b_scales, &c_f32};
    const std::uint32_t groups_x = static_cast<std::uint32_t>((n + 15) / 16);
    const std::uint32_t groups_y = static_cast<std::uint32_t>((m + 15) / 16);
    return runtime.dispatch_cached(vkspirv::k_mm_q8q8_scaled_f32, vkspirv::k_mm_q8q8_scaled_f32_words,
                                   buffers, &push, sizeof(push), groups_x, groups_y, 1);
}

VulkanOpResult run_vulkan_matmul_q8q4_scaled(VulkanRuntime& runtime,
                                             const VulkanBuffer& a_i8,
                                             const VulkanBuffer& a_scales,
                                             const VulkanBuffer& b_q4,
                                             const VulkanBuffer& b_scales,
                                             VulkanBuffer& c_f32,
                                             std::size_t m,
                                             std::size_t k,
                                             std::size_t n) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (m == 0 || k == 0 || n == 0) return fail("Vulkan q8q4 matmul requires non-zero M, K, and N");
    constexpr std::size_t kMaxDim = 1u << 24;
    if (m > kMaxDim || n > kMaxDim || k > kMaxDim) {
        return fail("Vulkan q8q4 matmul dimensions exceed supported range");
    }
    if (a_i8.nbytes() < m * k * sizeof(std::int8_t)) return fail("Vulkan q8q4 matmul A buffer is too small");
    const auto b_packed_bytes = ((k * n) + 1) / 2;
    if (b_q4.nbytes() < b_packed_bytes) return fail("Vulkan q8q4 matmul B (q4) buffer is too small");
    if (a_scales.nbytes() < m * sizeof(float)) return fail("Vulkan q8q4 matmul A scales buffer is too small");
    if (b_scales.nbytes() < n * sizeof(float)) return fail("Vulkan q8q4 matmul B scales buffer is too small");
    if (c_f32.nbytes() < m * n * sizeof(float)) return fail("Vulkan q8q4 matmul C buffer is too small");
    constexpr std::size_t kMaxInt32 = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
    if (m > kMaxInt32 || k > kMaxInt32 || n > kMaxInt32)
        return fail("Vulkan q8q4 matmul dimensions overflow push constants");
    const struct {
        std::uint32_t M;
        std::uint32_t K;
        std::uint32_t N;
    } push{static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(k),
           static_cast<std::uint32_t>(n)};
    const std::vector<const VulkanBuffer*> buffers = {&a_i8, &b_q4, &a_scales, &b_scales, &c_f32};
    const std::uint32_t groups_x = static_cast<std::uint32_t>((n + 15) / 16);
    const std::uint32_t groups_y = static_cast<std::uint32_t>((m + 15) / 16);
    return runtime.dispatch_cached(vkspirv::k_mm_q8q4_scaled_f32, vkspirv::k_mm_q8q4_scaled_f32_words,
                                   buffers, &push, sizeof(push), groups_x, groups_y, 1);
}

VulkanOpResult run_vulkan_matmul_f32q4_m1(VulkanRuntime& runtime,
                                          const VulkanBuffer& a_f32,
                                          const VulkanBuffer& b_q4,
                                          const VulkanBuffer& b_scales,
                                          VulkanBuffer& c_f32,
                                          std::size_t k,
                                          std::size_t n) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (k == 0 || n == 0) return fail("Vulkan f32q4 M=1 matmul requires non-zero K and N");
    constexpr std::size_t kMaxDim = 1u << 24;
    if (k > kMaxDim || n > kMaxDim) return fail("Vulkan f32q4 M=1 matmul dimensions exceed supported range");
    if (a_f32.nbytes() < k * sizeof(float)) return fail("Vulkan f32q4 M=1 matmul A buffer is too small");
    const auto b_packed_bytes = ((k * n) + 1) / 2;
    if (b_q4.nbytes() < b_packed_bytes) return fail("Vulkan f32q4 M=1 matmul B (q4) buffer is too small");
    if (b_scales.nbytes() < n * sizeof(float)) return fail("Vulkan f32q4 M=1 matmul scales buffer is too small");
    if (c_f32.nbytes() < n * sizeof(float)) return fail("Vulkan f32q4 M=1 matmul C buffer is too small");
    constexpr std::size_t kMaxInt32 = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
    if (k > kMaxInt32 || n > kMaxInt32)
        return fail("Vulkan f32q4 M=1 matmul dimensions overflow push constants");
    const struct {
        std::uint32_t K;
        std::uint32_t N;
    } push{static_cast<std::uint32_t>(k), static_cast<std::uint32_t>(n)};
    const std::vector<const VulkanBuffer*> buffers = {&a_f32, &b_q4, &b_scales, &c_f32};
    const std::uint32_t groups = static_cast<std::uint32_t>((n + 3) / 4);
    return runtime.dispatch_cached(vkspirv::k_mm_f32q4_m1_wg64_f32, vkspirv::k_mm_f32q4_m1_wg64_f32_words,
                                   buffers, &push, sizeof(push), groups, 1, 1);
}

VulkanOpResult run_vulkan_softmax_rows(VulkanRuntime& runtime,
                                       const VulkanBuffer& x,
                                       VulkanBuffer& out,
                                       std::size_t rows,
                                       std::size_t cols) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (rows == 0 || cols == 0) return fail("Vulkan softmax rows requires non-zero rows and cols");
    constexpr std::size_t kMaxDim = 1u << 24;
    if (rows > kMaxDim || cols > kMaxDim) {
        return fail("Vulkan softmax rows dimensions exceed supported range");
    }
    const auto nbytes = rows * cols * sizeof(float);
    if (x.nbytes() < nbytes) return fail("Vulkan softmax input buffer is too small");
    if (out.nbytes() < nbytes) return fail("Vulkan softmax output buffer is too small");

    const struct {
        std::uint32_t rows;
        std::uint32_t cols;
    } push{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols)};
    const std::vector<const VulkanBuffer*> buffers = {&x, &out};
    const bool use_sg = runtime.caps().subgroup_arithmetic_compute;
    const auto* spirv = use_sg ? vkspirv::k_softmax_rows_f32_subgroup : vkspirv::k_softmax_rows_f32;
    const auto words = use_sg ? vkspirv::k_softmax_rows_f32_subgroup_words : vkspirv::k_softmax_rows_f32_words;
    return runtime.dispatch_cached(spirv, words, buffers,
                                   &push, sizeof(push), static_cast<std::uint32_t>(rows), 1, 1);
}

VulkanOpResult run_vulkan_softmax_rows_backward(VulkanRuntime& runtime,
                                                const VulkanBuffer& y,
                                                const VulkanBuffer& dy,
                                                VulkanBuffer& dx,
                                                std::size_t rows,
                                                std::size_t cols) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (rows == 0 || cols == 0) return fail("Vulkan softmax backward requires non-zero rows and cols");
    const auto nbytes = rows * cols * sizeof(float);
    if (y.nbytes() < nbytes || dy.nbytes() < nbytes || dx.nbytes() < nbytes) {
        return fail("Vulkan softmax backward buffer is too small");
    }
    const struct {
        std::uint32_t rows;
        std::uint32_t cols;
    } push{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols)};
    const std::vector<const VulkanBuffer*> buffers = {&y, &dy, &dx};
    return runtime.dispatch_cached(vkspirv::k_softmax_rows_bwd_f32, vkspirv::k_softmax_rows_bwd_f32_words,
                                   buffers, &push, sizeof(push), static_cast<std::uint32_t>(rows), 1, 1);
}

VulkanOpResult run_vulkan_rmsnorm(VulkanRuntime& runtime,
                                  const VulkanBuffer& x,
                                  const VulkanBuffer& weight,
                                  VulkanBuffer& out,
                                  std::size_t rows,
                                  std::size_t cols,
                                  float eps) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (rows == 0 || cols == 0) return fail("Vulkan RMSNorm requires non-zero rows and cols");
    constexpr std::size_t kMaxDim = 1u << 24;
    if (rows > kMaxDim || cols > kMaxDim) {
        return fail("Vulkan RMSNorm dimensions exceed supported range");
    }
    if (!std::isfinite(eps) || eps < 0.0f) return fail("Vulkan RMSNorm eps must be finite and non-negative");
    const auto x_bytes = rows * cols * sizeof(float);
    if (x.nbytes() < x_bytes) return fail("Vulkan RMSNorm input buffer is too small");
    if (out.nbytes() < x_bytes) return fail("Vulkan RMSNorm output buffer is too small");
    if (weight.nbytes() < cols * sizeof(float)) return fail("Vulkan RMSNorm weight buffer is too small");

    const struct {
        std::uint32_t rows;
        std::uint32_t cols;
        float eps;
    } push{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols), eps};
    const std::vector<const VulkanBuffer*> buffers = {&x, &weight, &out};
    const bool use_sg = runtime.caps().subgroup_arithmetic_compute;
    const auto* spirv = use_sg ? vkspirv::k_rmsnorm_f32_subgroup : vkspirv::k_rmsnorm_f32;
    const auto words = use_sg ? vkspirv::k_rmsnorm_f32_subgroup_words : vkspirv::k_rmsnorm_f32_words;
    return runtime.dispatch_cached(spirv, words, buffers, &push,
                                   sizeof(push), static_cast<std::uint32_t>(rows), 1, 1);
}

VulkanOpResult run_vulkan_rmsnorm_backward_x(VulkanRuntime& runtime,
                                             const VulkanBuffer& x,
                                             const VulkanBuffer& weight,
                                             const VulkanBuffer& grad_out,
                                             VulkanBuffer& grad_x,
                                             std::size_t rows,
                                             std::size_t cols,
                                             float eps) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (rows == 0 || cols == 0) return fail("Vulkan RMSNorm backward requires non-zero rows and cols");
    if (!std::isfinite(eps) || eps < 0.0f) return fail("Vulkan RMSNorm backward eps must be finite and non-negative");
    const auto nbytes = rows * cols * sizeof(float);
    if (x.nbytes() < nbytes || grad_out.nbytes() < nbytes || grad_x.nbytes() < nbytes) {
        return fail("Vulkan RMSNorm backward buffer is too small");
    }
    if (weight.nbytes() < cols * sizeof(float)) return fail("Vulkan RMSNorm backward weight buffer is too small");

    const struct {
        std::uint32_t rows;
        std::uint32_t cols;
        float eps;
    } push{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols), eps};
    const std::vector<const VulkanBuffer*> buffers = {&x, &weight, &grad_out, &grad_x};
    const bool use_sg = runtime.caps().subgroup_arithmetic_compute;
    const auto* spirv = use_sg ? vkspirv::k_rmsnorm_bwd_x_f32_subgroup : vkspirv::k_rmsnorm_bwd_x_f32;
    const auto words = use_sg ? vkspirv::k_rmsnorm_bwd_x_f32_subgroup_words : vkspirv::k_rmsnorm_bwd_x_f32_words;
    return runtime.dispatch_cached(spirv, words, buffers,
                                   &push, sizeof(push), static_cast<std::uint32_t>(rows), 1, 1);
}

VulkanOpResult run_vulkan_rmsnorm_backward_weight(VulkanRuntime& runtime,
                                                  const VulkanBuffer& x,
                                                  const VulkanBuffer& grad_out,
                                                  VulkanBuffer& row_inv_scratch,
                                                  VulkanBuffer& grad_weight,
                                                  std::size_t rows,
                                                  std::size_t cols,
                                                  float eps) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (rows == 0 || cols == 0) return fail("Vulkan RMSNorm weight backward requires non-zero rows and cols");
    if (!std::isfinite(eps) || eps < 0.0f) {
        return fail("Vulkan RMSNorm weight backward eps must be finite and non-negative");
    }
    const auto nbytes = rows * cols * sizeof(float);
    if (x.nbytes() < nbytes || grad_out.nbytes() < nbytes) {
        return fail("Vulkan RMSNorm weight backward input buffer is too small");
    }
    if (row_inv_scratch.nbytes() < rows * sizeof(float)) {
        return fail("Vulkan RMSNorm weight backward scratch buffer is too small");
    }
    if (grad_weight.nbytes() < cols * sizeof(float)) {
        return fail("Vulkan RMSNorm weight backward output buffer is too small");
    }

    // Stage 1: per-row inverse RMS. Stage 2: per-column reduction over rows.
    // In immediate mode each dispatch submits + waits; in batch mode the
    // cached path inserts compute->compute barriers between dispatches.
    const struct {
        std::uint32_t rows;
        std::uint32_t cols;
        float eps;
    } inv_push{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols), eps};
    const std::vector<const VulkanBuffer*> inv_buffers = {&x, &row_inv_scratch};
    auto inv_run = runtime.dispatch_cached(vkspirv::k_rmsnorm_row_inv_f32, vkspirv::k_rmsnorm_row_inv_f32_words,
                                           inv_buffers, &inv_push, sizeof(inv_push),
                                           static_cast<std::uint32_t>(rows), 1, 1);
    if (!inv_run.success) return inv_run;

    const struct {
        std::uint32_t rows;
        std::uint32_t cols;
    } w_push{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols)};
    const std::vector<const VulkanBuffer*> w_buffers = {&x, &grad_out, &row_inv_scratch, &grad_weight};
    return runtime.dispatch_cached(vkspirv::k_rmsnorm_bwd_w_f32, vkspirv::k_rmsnorm_bwd_w_f32_words, w_buffers,
                                   &w_push, sizeof(w_push), static_cast<std::uint32_t>((cols + 63) / 64), 1, 1);
}

VulkanOpResult run_vulkan_swiglu(VulkanRuntime& runtime,
                                 const VulkanBuffer& packed,
                                 VulkanBuffer& out,
                                 std::size_t rows,
                                 std::size_t hidden) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (rows == 0 || hidden == 0) return fail("Vulkan SwiGLU requires non-zero rows and hidden");
    constexpr std::size_t kMaxDim = 1u << 24;
    if (rows > kMaxDim || hidden > kMaxDim) {
        return fail("Vulkan SwiGLU dimensions exceed supported range");
    }
    if (packed.nbytes() < rows * hidden * 2 * sizeof(float)) return fail("Vulkan SwiGLU packed buffer is too small");
    if (out.nbytes() < rows * hidden * sizeof(float)) return fail("Vulkan SwiGLU output buffer is too small");

    const struct {
        std::uint32_t rows;
        std::uint32_t hidden;
    } push{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(hidden)};
    const std::vector<const VulkanBuffer*> buffers = {&packed, &out};
    const auto total = static_cast<std::uint32_t>((rows * hidden + 63) / 64);
    return runtime.dispatch_cached(vkspirv::k_swiglu_f32, vkspirv::k_swiglu_f32_words, buffers, &push,
                                   sizeof(push), total, 1, 1);
}

VulkanOpResult run_vulkan_swiglu_backward(VulkanRuntime& runtime,
                                          const VulkanBuffer& packed,
                                          const VulkanBuffer& grad_out,
                                          VulkanBuffer& grad_packed,
                                          std::size_t rows,
                                          std::size_t hidden) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (rows == 0 || hidden == 0) return fail("Vulkan SwiGLU backward requires non-zero rows and hidden");
    if (packed.nbytes() < rows * hidden * 2 * sizeof(float) ||
        grad_packed.nbytes() < rows * hidden * 2 * sizeof(float)) {
        return fail("Vulkan SwiGLU backward packed buffer is too small");
    }
    if (grad_out.nbytes() < rows * hidden * sizeof(float)) {
        return fail("Vulkan SwiGLU backward grad_out buffer is too small");
    }
    const struct {
        std::uint32_t rows;
        std::uint32_t hidden;
    } push{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(hidden)};
    const std::vector<const VulkanBuffer*> buffers = {&packed, &grad_out, &grad_packed};
    const auto total = static_cast<std::uint32_t>((rows * hidden * 2 + 63) / 64);
    return runtime.dispatch_cached(vkspirv::k_swiglu_bwd_f32, vkspirv::k_swiglu_bwd_f32_words, buffers, &push,
                                   sizeof(push), total, 1, 1);
}

VulkanOpResult run_vulkan_add(VulkanRuntime& runtime,
                              const VulkanBuffer& a,
                              const VulkanBuffer& b,
                              VulkanBuffer& out,
                              std::size_t elements) {
    VulkanOpResult result;
    constexpr std::size_t kMaxElements = 16 * 1024 * 1024;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (elements == 0) return fail("Vulkan add requires non-zero element count");
    if (elements > kMaxElements * 16) return fail("Vulkan add element count exceeds supported range");
    const auto nbytes = elements * sizeof(float);
    if (a.nbytes() < nbytes) return fail("Vulkan add A buffer is too small");
    if (b.nbytes() < nbytes) return fail("Vulkan add B buffer is too small");
    if (out.nbytes() < nbytes) return fail("Vulkan add output buffer is too small");

    const struct {
        std::uint32_t n;
    } push{static_cast<std::uint32_t>(elements)};
    const std::vector<const VulkanBuffer*> buffers = {&a, &b, &out};
    return runtime.dispatch_cached(vkspirv::k_add_f32, vkspirv::k_add_f32_words, buffers, &push, sizeof(push),
                                   static_cast<std::uint32_t>((elements + 63) / 64), 1, 1);
}

VulkanOpResult run_vulkan_mul_scalar(VulkanRuntime& runtime,
                                     const VulkanBuffer& x,
                                     VulkanBuffer& out,
                                     std::size_t elements,
                                     float alpha) {
    VulkanOpResult result;
    constexpr std::size_t kMaxElements = 16 * 1024 * 1024;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (elements == 0) return fail("Vulkan mul_scalar requires non-zero element count");
    if (elements > kMaxElements * 16) return fail("Vulkan mul_scalar element count exceeds supported range");
    if (!std::isfinite(alpha)) return fail("Vulkan mul_scalar requires finite alpha");
    const auto nbytes = elements * sizeof(float);
    if (x.nbytes() < nbytes) return fail("Vulkan mul_scalar x buffer is too small");
    if (out.nbytes() < nbytes) return fail("Vulkan mul_scalar output buffer is too small");

    const struct {
        float alpha;
        std::uint32_t n;
    } push{alpha, static_cast<std::uint32_t>(elements)};
    const std::vector<const VulkanBuffer*> buffers = {&x, &out};
    return runtime.dispatch_cached(vkspirv::k_mul_scalar_f32, vkspirv::k_mul_scalar_f32_words,
                                   buffers, &push, sizeof(push),
                                   static_cast<std::uint32_t>((elements + 63) / 64), 1, 1);
}

VulkanOpResult run_vulkan_add_scalar(VulkanRuntime& runtime,
                                     const VulkanBuffer& x,
                                     VulkanBuffer& out,
                                     std::size_t elements,
                                     float value) {
    VulkanOpResult result;
    constexpr std::size_t kMaxElements = 16 * 1024 * 1024;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (elements == 0) return fail("Vulkan add_scalar requires non-zero element count");
    if (elements > kMaxElements * 16) return fail("Vulkan add_scalar element count exceeds supported range");
    if (!std::isfinite(value)) return fail("Vulkan add_scalar requires finite value");
    const auto nbytes = elements * sizeof(float);
    if (x.nbytes() < nbytes) return fail("Vulkan add_scalar x buffer is too small");
    if (out.nbytes() < nbytes) return fail("Vulkan add_scalar output buffer is too small");

    const struct {
        float value;
        std::uint32_t n;
    } push{value, static_cast<std::uint32_t>(elements)};
    const std::vector<const VulkanBuffer*> buffers = {&x, &out};
    return runtime.dispatch_cached(vkspirv::k_add_scalar_f32, vkspirv::k_add_scalar_f32_words,
                                   buffers, &push, sizeof(push),
                                   static_cast<std::uint32_t>((elements + 63) / 64), 1, 1);
}

VulkanOpResult run_vulkan_sub(VulkanRuntime& runtime,
                              const VulkanBuffer& a,
                              const VulkanBuffer& b,
                              VulkanBuffer& out,
                              std::size_t elements) {
    VulkanOpResult result;
    constexpr std::size_t kMaxElements = 16 * 1024 * 1024;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (elements == 0) return fail("Vulkan sub requires non-zero element count");
    if (elements > kMaxElements * 16) return fail("Vulkan sub element count exceeds supported range");
    const auto nbytes = elements * sizeof(float);
    if (a.nbytes() < nbytes) return fail("Vulkan sub A buffer is too small");
    if (b.nbytes() < nbytes) return fail("Vulkan sub B buffer is too small");
    if (out.nbytes() < nbytes) return fail("Vulkan sub output buffer is too small");

    const struct {
        std::uint32_t n;
    } push{static_cast<std::uint32_t>(elements)};
    const std::vector<const VulkanBuffer*> buffers = {&a, &b, &out};
    return runtime.dispatch_cached(vkspirv::k_sub_f32, vkspirv::k_sub_f32_words, buffers, &push, sizeof(push),
                                   static_cast<std::uint32_t>((elements + 63) / 64), 1, 1);
}

VulkanOpResult run_vulkan_sgd_update(VulkanRuntime& runtime,
                                      const VulkanBuffer& param,
                                      const VulkanBuffer& grad,
                                      VulkanBuffer& out,
                                      std::size_t elements,
                                      float lr) {
    VulkanOpResult result;
    constexpr std::size_t kMaxElements = 16 * 1024 * 1024;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (elements == 0) return fail("Vulkan SGD update requires non-zero element count");
    if (elements > kMaxElements * 16) return fail("Vulkan SGD update element count exceeds supported range");
    if (!std::isfinite(lr)) return fail("Vulkan SGD update requires finite lr");
    const auto nbytes = elements * sizeof(float);
    if (param.nbytes() < nbytes) return fail("Vulkan SGD update param buffer is too small");
    if (grad.nbytes() < nbytes) return fail("Vulkan SGD update grad buffer is too small");
    if (out.nbytes() < nbytes) return fail("Vulkan SGD update output buffer is too small");

    const struct {
        std::uint32_t n;
        float lr;
    } push{static_cast<std::uint32_t>(elements), lr};
    const std::vector<const VulkanBuffer*> buffers = {&param, &grad, &out};
    return runtime.dispatch_cached(vkspirv::k_sgd_update_f32, vkspirv::k_sgd_update_f32_words, buffers, &push,
                                   sizeof(push), static_cast<std::uint32_t>((elements + 63) / 64), 1, 1);
}

VulkanOpResult run_vulkan_gelu(VulkanRuntime& runtime,
                               const VulkanBuffer& x,
                               VulkanBuffer& out,
                               std::size_t elements) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (elements == 0) return fail("Vulkan GELU requires non-zero element count");
    const auto nbytes = elements * sizeof(float);
    if (x.nbytes() < nbytes || out.nbytes() < nbytes) return fail("Vulkan GELU buffer is too small");
    const struct {
        std::uint32_t n;
    } push{static_cast<std::uint32_t>(elements)};
    const std::vector<const VulkanBuffer*> buffers = {&x, &out};
    return runtime.dispatch_cached(vkspirv::k_gelu_f32, vkspirv::k_gelu_f32_words, buffers, &push, sizeof(push),
                                   static_cast<std::uint32_t>((elements + 63) / 64), 1, 1);
}

VulkanOpResult run_vulkan_softmax_cross_entropy(VulkanRuntime& runtime,
                                                const VulkanBuffer& logits,
                                                const VulkanBuffer& targets,
                                                VulkanBuffer& partial,
                                                VulkanBuffer& out,
                                                std::size_t rows,
                                                std::size_t cols) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (rows == 0 || cols == 0) return fail("Vulkan cross-entropy requires non-zero rows and cols");
    if (logits.nbytes() < rows * cols * sizeof(float)) return fail("Vulkan cross-entropy logits buffer is too small");
    if (targets.nbytes() < rows * sizeof(std::int32_t)) return fail("Vulkan cross-entropy targets buffer is too small");
    if (partial.nbytes() < rows * sizeof(float)) return fail("Vulkan cross-entropy partial buffer is too small");
    if (out.nbytes() < sizeof(float)) return fail("Vulkan cross-entropy output buffer is too small");

    const struct {
        std::uint32_t rows;
        std::uint32_t cols;
    } row_push{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols)};
    const std::vector<const VulkanBuffer*> row_buffers = {&logits, &targets, &partial};
    const bool use_sg = runtime.caps().subgroup_arithmetic_compute;
    const auto* row_spirv = use_sg ? vkspirv::k_ce_fwd_rows_f32_subgroup : vkspirv::k_ce_fwd_rows_f32;
    const auto row_words = use_sg ? vkspirv::k_ce_fwd_rows_f32_subgroup_words : vkspirv::k_ce_fwd_rows_f32_words;
    auto stage = runtime.dispatch_cached(row_spirv, row_words,
                                         row_buffers, &row_push, sizeof(row_push),
                                         static_cast<std::uint32_t>(rows), 1, 1);
    if (!stage.success) return stage;

    const struct {
        std::uint32_t count;
        std::uint32_t denominator;
    } mean_push{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(rows)};
    const std::vector<const VulkanBuffer*> mean_buffers = {&partial, &out};
    const bool use_sg_mean = runtime.caps().subgroup_arithmetic_compute;
    const auto* mean_spirv = use_sg_mean ? vkspirv::k_mean_reduce_f32_subgroup : vkspirv::k_mean_reduce_f32;
    const auto mean_words = use_sg_mean ? vkspirv::k_mean_reduce_f32_subgroup_words : vkspirv::k_mean_reduce_f32_words;
    return runtime.dispatch_cached(mean_spirv, mean_words, mean_buffers,
                                   &mean_push, sizeof(mean_push), 1, 1, 1);
}

VulkanOpResult run_vulkan_softmax_cross_entropy_backward(VulkanRuntime& runtime,
                                                         const VulkanBuffer& logits,
                                                         const VulkanBuffer& targets,
                                                         const VulkanBuffer& grad_out,
                                                         VulkanBuffer& grad_logits,
                                                         std::size_t rows,
                                                         std::size_t cols) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (rows == 0 || cols == 0) return fail("Vulkan cross-entropy backward requires non-zero rows and cols");
    if (logits.nbytes() < rows * cols * sizeof(float) || grad_logits.nbytes() < rows * cols * sizeof(float)) {
        return fail("Vulkan cross-entropy backward buffer is too small");
    }
    if (targets.nbytes() < rows * sizeof(std::int32_t)) {
        return fail("Vulkan cross-entropy backward targets buffer is too small");
    }
    if (grad_out.nbytes() < sizeof(float)) return fail("Vulkan cross-entropy backward grad buffer is too small");

    const struct {
        std::uint32_t rows;
        std::uint32_t cols;
    } push{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols)};
    const std::vector<const VulkanBuffer*> buffers = {&logits, &targets, &grad_out, &grad_logits};
    return runtime.dispatch_cached(vkspirv::k_ce_bwd_f32, vkspirv::k_ce_bwd_f32_words, buffers, &push,
                                   sizeof(push), static_cast<std::uint32_t>(rows), 1, 1);
}

VulkanOpResult run_vulkan_compact_counter_apply_update_fused(VulkanRuntime& runtime,
                                                             VulkanBuffer& state,
                                                             VulkanBuffer& scale,
                                                             VulkanBuffer& v,
                                                             const VulkanBuffer& grad_out,
                                                             const VulkanBuffer& x,
                                                             VulkanBuffer& scale_new_scratch,
                                                             VulkanBuffer& denom_scratch,
                                                             std::size_t C,
                                                             std::size_t in_features,
                                                             std::size_t out_features,
                                                             std::size_t batch,
                                                             float lr,
                                                             float lr_scale,
                                                             float rms_beta,
                                                             float rms_eps,
                                                             std::uint32_t seed) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (C == 0 || in_features == 0 || out_features == 0 || batch == 0) {
        return fail("Vulkan counter update requires non-zero dimensions");
    }
    if (in_features % 4 != 0) return fail("Vulkan counter update requires in_features % 4 == 0");
    // Validate optimizer hyperparameters: NaN/Inf in lr/lr_scale/rms_beta/rms_eps
    // silently poisons the whole weight update (and a negative rms_eps would
    // bypass the max(sqrt(vv), eps) guard in row_stats). Mirror SGD/RMSNorm.
    if (!std::isfinite(lr)) return fail("Vulkan counter update requires finite lr");
    if (!std::isfinite(lr_scale)) return fail("Vulkan counter update requires finite lr_scale");
    if (!std::isfinite(rms_beta)) return fail("Vulkan counter update requires finite rms_beta");
    if (!std::isfinite(rms_eps) || rms_eps < 0.0f) return fail("Vulkan counter update requires non-negative finite rms_eps");
    if (!runtime.supports_storage_buffer_i8()) {
        return fail("Vulkan counter update requires VK_KHR_8bit_storage (uniformAndStorageBuffer8BitAccess)");
    }
    const std::size_t n_groups = out_features * (in_features / 4);
    if (state.nbytes() < n_groups * 3) return fail("Vulkan counter update state buffer is too small");
    if (scale.nbytes() < out_features * sizeof(float) || v.nbytes() < out_features * sizeof(float) ||
        scale_new_scratch.nbytes() < out_features * sizeof(float) ||
        denom_scratch.nbytes() < out_features * sizeof(float)) {
        return fail("Vulkan counter update per-row buffer is too small");
    }
    if (grad_out.nbytes() < batch * out_features * sizeof(float) ||
        x.nbytes() < batch * in_features * sizeof(float)) {
        return fail("Vulkan counter update activation buffer is too small");
    }

    const struct {
        std::int32_t C;
        std::int32_t in_features;
        std::int32_t out_features;
        std::int32_t N;
        float lr_scale;
        float rms_beta;
        float rms_eps;
    } stats_push{static_cast<std::int32_t>(C),        static_cast<std::int32_t>(in_features),
                 static_cast<std::int32_t>(out_features), static_cast<std::int32_t>(batch),
                 lr_scale,                             rms_beta,
                 rms_eps};
    const std::vector<const VulkanBuffer*> stats_buffers = {&state, &scale, &v, &grad_out, &x,
                                                            &scale_new_scratch, &denom_scratch};
    const bool use_sg_stats = runtime.caps().subgroup_arithmetic_compute;
    const auto* stats_spirv = use_sg_stats ? vkspirv::k_counter_row_stats_fused_f32_subgroup
                                            : vkspirv::k_counter_row_stats_fused_f32;
    const auto stats_words = use_sg_stats ? vkspirv::k_counter_row_stats_fused_f32_subgroup_words
                                          : vkspirv::k_counter_row_stats_fused_f32_words;
    auto stage = runtime.dispatch_cached(stats_spirv, stats_words, stats_buffers,
                                         &stats_push, sizeof(stats_push),
                                         static_cast<std::uint32_t>(out_features), 1, 1);
    if (!stage.success) return stage;

    const struct {
        std::int32_t C;
        std::int32_t in_features;
        std::int32_t out_features;
        std::int32_t N;
        std::int32_t n_groups;
        float lr;
        std::uint32_t seed;
    } apply_push{static_cast<std::int32_t>(C),        static_cast<std::int32_t>(in_features),
                 static_cast<std::int32_t>(out_features), static_cast<std::int32_t>(batch),
                 static_cast<std::int32_t>(n_groups),  lr,
                 seed};
    const std::vector<const VulkanBuffer*> apply_buffers = {&state, &scale, &scale_new_scratch,
                                                            &denom_scratch, &grad_out, &x};
    stage = runtime.dispatch_cached(vkspirv::k_counter_apply_update_fused_f32,
                                    vkspirv::k_counter_apply_update_fused_f32_words, apply_buffers,
                                    &apply_push, sizeof(apply_push),
                                    static_cast<std::uint32_t>((n_groups + 63) / 64), 1, 1);
    if (!stage.success) return stage;

    const struct {
        std::uint32_t n;
    } copy_push{static_cast<std::uint32_t>(out_features)};
    const std::vector<const VulkanBuffer*> copy_buffers = {&scale_new_scratch, &scale};
    return runtime.dispatch_cached(vkspirv::k_copy_f32, vkspirv::k_copy_f32_words, copy_buffers, &copy_push,
                                   sizeof(copy_push), static_cast<std::uint32_t>((out_features + 63) / 64), 1,
                                   1);
}

VulkanOpResult run_vulkan_gelu_backward(VulkanRuntime& runtime,
                                        const VulkanBuffer& x,
                                        const VulkanBuffer& grad_out,
                                        VulkanBuffer& grad_x,
                                        std::size_t elements) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (elements == 0) return fail("Vulkan GELU backward requires non-zero element count");
    const auto nbytes = elements * sizeof(float);
    if (x.nbytes() < nbytes || grad_out.nbytes() < nbytes || grad_x.nbytes() < nbytes) {
        return fail("Vulkan GELU backward buffer is too small");
    }
    const struct {
        std::uint32_t n;
    } push{static_cast<std::uint32_t>(elements)};
    const std::vector<const VulkanBuffer*> buffers = {&x, &grad_out, &grad_x};
    return runtime.dispatch_cached(vkspirv::k_gelu_bwd_f32, vkspirv::k_gelu_bwd_f32_words, buffers, &push,
                                   sizeof(push), static_cast<std::uint32_t>((elements + 63) / 64), 1, 1);
}

VulkanOpResult run_vulkan_compact_counter_decode_weight(VulkanRuntime& runtime,
                                                        const VulkanBuffer& state,
                                                        const VulkanBuffer& scale,
                                                        VulkanBuffer& weight,
                                                        std::size_t in_features,
                                                        std::size_t out_features,
                                                        std::size_t C) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (in_features == 0 || out_features == 0) {
        return fail("Vulkan compact-counter decode requires non-zero dimensions");
    }
    if (in_features % 4 != 0) {
        return fail("Vulkan compact-counter decode requires in_features divisible by 4");
    }
    if (C == 0 || 3 * (2 * C - 1) > 64) {
        return fail("Vulkan compact-counter decode C is out of 6-bit range");
    }
    if (!runtime.supports_storage_buffer_i8()) {
        return fail("Vulkan compact-counter decode requires VK_KHR_8bit_storage on persistent Tensor buffers");
    }
    if (in_features > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) ||
        out_features > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) ||
        C > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        return fail("Vulkan compact-counter decode dimensions exceed specialized SPIR-V limits");
    }
    if (out_features > std::numeric_limits<std::size_t>::max() / (in_features / 4)) {
        return fail("Vulkan compact-counter decode state size overflows size_t");
    }
    const std::size_t groups = out_features * (in_features / 4);
    if (groups > std::numeric_limits<std::size_t>::max() / 3) {
        return fail("Vulkan compact-counter decode state byte size overflows size_t");
    }
    if (out_features > std::numeric_limits<std::size_t>::max() / in_features) {
        return fail("Vulkan compact-counter decode output size overflows size_t");
    }
    const std::size_t elements = out_features * in_features;
    if (elements > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        return fail("Vulkan compact-counter decode dispatch exceeds uint32 workgroup count");
    }
    if (state.nbytes() < groups * 3) return fail("Vulkan compact-counter state buffer is too small");
    if (scale.nbytes() < out_features * sizeof(float)) return fail("Vulkan compact-counter scale buffer is too small");
    if (weight.nbytes() < elements * sizeof(float)) return fail("Vulkan compact-counter weight buffer is too small");

    const struct {
        std::int32_t C;
        std::int32_t in_features;
        std::int32_t n_groups;
    } push{static_cast<std::int32_t>(C), static_cast<std::int32_t>(in_features),
           static_cast<std::int32_t>(groups)};
    const std::vector<const VulkanBuffer*> buffers = {&state, &scale, &weight};
    return runtime.dispatch_cached(vkspirv::k_counter_decode_weight_f32,
                                   vkspirv::k_counter_decode_weight_f32_words, buffers, &push, sizeof(push),
                                   static_cast<std::uint32_t>((groups + 63) / 64), 1, 1);
}

VulkanOpResult run_vulkan_compact_counter_backward_input_u8(VulkanRuntime& runtime,
                                                            const VulkanBuffer& state,
                                                            const VulkanBuffer& scale,
                                                            const VulkanBuffer& grad_out,
                                                            VulkanBuffer& grad_x,
                                                            std::size_t batch,
                                                            std::size_t in_features,
                                                            std::size_t out_features,
                                                            std::size_t C) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (batch == 0 || in_features == 0 || out_features == 0) {
        return fail("Vulkan compact-counter backward-input requires non-zero dimensions");
    }
    if (in_features % 4 != 0) {
        return fail("Vulkan compact-counter backward-input requires in_features divisible by 4");
    }
    if (C == 0 || 3 * (2 * C - 1) > 64) {
        return fail("Vulkan compact-counter backward-input C is out of 6-bit range");
    }
    if (!runtime.supports_storage_buffer_i8()) {
        return fail("Vulkan compact-counter backward-input requires VK_KHR_8bit_storage on persistent Tensor buffers");
    }
    if (batch > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) ||
        in_features > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) ||
        out_features > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) ||
        C > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        return fail("Vulkan compact-counter backward-input dimensions exceed specialized SPIR-V limits");
    }
    if (out_features > std::numeric_limits<std::size_t>::max() / (in_features / 4)) {
        return fail("Vulkan compact-counter backward-input state size overflows size_t");
    }
    const std::size_t groups = out_features * (in_features / 4);
    if (groups > std::numeric_limits<std::size_t>::max() / 3) {
        return fail("Vulkan compact-counter backward-input state byte size overflows size_t");
    }
    if (batch > std::numeric_limits<std::size_t>::max() / out_features) {
        return fail("Vulkan compact-counter backward-input grad_out size overflows size_t");
    }
    if (batch > std::numeric_limits<std::size_t>::max() / in_features) {
        return fail("Vulkan compact-counter backward-input grad_x size overflows size_t");
    }
    const std::size_t grad_out_elements = batch * out_features;
    const std::size_t grad_x_elements = batch * in_features;
    if (grad_x_elements > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        return fail("Vulkan compact-counter backward-input dispatch exceeds uint32 workgroup count");
    }
    if (state.nbytes() < groups * 3) return fail("Vulkan compact-counter state buffer is too small");
    if (scale.nbytes() < out_features * sizeof(float)) return fail("Vulkan compact-counter scale buffer is too small");
    if (grad_out.nbytes() < grad_out_elements * sizeof(float)) {
        return fail("Vulkan compact-counter grad_out buffer is too small");
    }
    if (grad_x.nbytes() < grad_x_elements * sizeof(float)) {
        return fail("Vulkan compact-counter grad_x buffer is too small");
    }

    const struct {
        std::int32_t C;
        std::int32_t in_features;
        std::int32_t out_features;
        std::int32_t N;
    } push{static_cast<std::int32_t>(C), static_cast<std::int32_t>(in_features),
           static_cast<std::int32_t>(out_features), static_cast<std::int32_t>(batch)};
    const std::vector<const VulkanBuffer*> buffers = {&grad_out, &state, &scale, &grad_x};
    // One invocation per (row, group-of-4): batch * (in_features / 4) groups.
    const std::size_t dispatch_groups = batch * (in_features / 4);
    const auto run = runtime.dispatch_cached(vkspirv::k_counter_backward_input_f32,
                                             vkspirv::k_counter_backward_input_f32_words, buffers, &push,
                                             sizeof(push),
                                             static_cast<std::uint32_t>((dispatch_groups + 63) / 64), 1, 1);
    result.success = run.success;
    result.device_name = run.device_name;
    result.error = run.error;
    return result;
}

VulkanOpResult run_vulkan_f32_m1_matmul(VulkanRuntime& runtime,
                                        const VulkanBuffer& a,
                                        const VulkanBuffer& b,
                                        VulkanBuffer& c,
                                        std::size_t k,
                                        std::size_t n) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (k == 0 || n == 0) return fail("Vulkan f32 M=1 matmul requires non-zero K and N");
    constexpr std::size_t kMaxDim = 1u << 24;
    if (k > kMaxDim || n > kMaxDim) return fail("Vulkan f32 M=1 matmul dimensions exceed supported range");
    if (a.nbytes() < k * sizeof(float)) return fail("Vulkan f32 M=1 matmul A buffer is too small");
    if (b.nbytes() < k * n * sizeof(float)) return fail("Vulkan f32 M=1 matmul B buffer is too small");
    if (c.nbytes() < n * sizeof(float)) return fail("Vulkan f32 M=1 matmul C buffer is too small");

    const struct {
        std::uint32_t k;
        std::uint32_t n;
    } push{static_cast<std::uint32_t>(k), static_cast<std::uint32_t>(n)};
    const std::vector<const VulkanBuffer*> buffers = {&a, &b, &c};
    return runtime.dispatch_cached(vkspirv::k_mm_f32_m1n, vkspirv::k_mm_f32_m1n_words, buffers, &push,
                                   sizeof(push), static_cast<std::uint32_t>((n + 63) / 64), 1, 1);
}

VulkanF32MatmulSmokeResult run_vulkan_f32_matmul(const std::vector<float>& a,
                                                 const std::vector<float>& b,
                                                 std::size_t m,
                                                 std::size_t k,
                                                 std::size_t n) {
    VulkanF32MatmulSmokeResult result;
    constexpr std::size_t kMaxSpecializedK = 256;
    constexpr std::size_t kMaxDispatchDim = 4096;

    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (m == 0 || k == 0 || n == 0) return fail("Vulkan f32 matmul requires non-zero M, K, and N");
    if (k > kMaxSpecializedK) return fail("Vulkan f32 matmul currently supports K up to 256");
    if (m > kMaxDispatchDim || n > kMaxDispatchDim) {
        return fail("Vulkan f32 matmul dispatch currently supports M,N up to 4096");
    }
    if (m > std::numeric_limits<std::size_t>::max() / k) return fail("Vulkan f32 matmul M*K overflows size_t");
    if (k > std::numeric_limits<std::size_t>::max() / n) return fail("Vulkan f32 matmul K*N overflows size_t");
    if (m > std::numeric_limits<std::size_t>::max() / n) return fail("Vulkan f32 matmul M*N overflows size_t");
    if (a.size() != m * k) return fail("Vulkan f32 matmul A size must equal M*K");
    if (b.size() != k * n) return fail("Vulkan f32 matmul B size must equal K*N");

    std::vector<float> c(m * n, 0.0f);
    const auto shader = f32_matmul_spirv(k, n);
    const std::vector<VulkanStorageBufferSpec> buffers = {
        {a.data(), a.size() * sizeof(float)},
        {b.data(), b.size() * sizeof(float)},
        {c.data(), c.size() * sizeof(float)},
    };
    const auto run = run_vulkan_storage_buffer_compute(
        shader.data(), shader.size(), buffers, {2},
        static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(m), 1);
    result.success = run.success;
    result.device_name = run.device_name;
    result.error = run.error;
    if (!run.success) return result;
    if (run.outputs.size() != 1 || run.outputs[0].size() != c.size() * sizeof(float)) {
        result.success = false;
        result.error = "Vulkan f32 matmul returned malformed output";
        return result;
    }

    result.output.resize(c.size());
    std::memcpy(result.output.data(), run.outputs[0].data(), run.outputs[0].size());
    return result;
}

VulkanF32TensorResult run_vulkan_softmax_rows(const std::vector<float>& x,
                                              std::size_t rows,
                                              std::size_t cols) {
    VulkanF32TensorResult result;
    constexpr std::size_t kMaxRows = 4096;
    constexpr std::size_t kMaxCols = 256;

    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (rows == 0 || cols == 0) return fail("Vulkan softmax rows requires non-zero rows and cols");
    if (rows > kMaxRows || cols > kMaxCols) {
        return fail("Vulkan softmax rows currently supports rows up to 4096 and cols up to 256");
    }
    if (rows > std::numeric_limits<std::size_t>::max() / cols) {
        return fail("Vulkan softmax rows rows*cols overflows size_t");
    }
    if (x.size() != rows * cols) return fail("Vulkan softmax rows input size must equal rows*cols");

    std::vector<float> out(x.size(), 0.0f);
    const auto shader = softmax_rows_spirv(cols);
    const std::vector<VulkanStorageBufferSpec> buffers = {
        {x.data(), x.size() * sizeof(float)},
        {out.data(), out.size() * sizeof(float)},
    };
    const auto run = run_vulkan_storage_buffer_compute(
        shader.data(), shader.size(), buffers, {1}, static_cast<std::uint32_t>(rows), 1, 1);
    result.success = run.success;
    result.device_name = run.device_name;
    result.error = run.error;
    if (!run.success) return result;
    if (run.outputs.size() != 1 || run.outputs[0].size() != out.size() * sizeof(float)) {
        result.success = false;
        result.error = "Vulkan softmax rows returned malformed output";
        return result;
    }

    result.output.resize(out.size());
    std::memcpy(result.output.data(), run.outputs[0].data(), run.outputs[0].size());
    return result;
}

VulkanF32TensorResult run_vulkan_rmsnorm(const std::vector<float>& x,
                                         const std::vector<float>& weight,
                                         std::size_t rows,
                                         std::size_t cols,
                                         float eps) {
    VulkanF32TensorResult result;
    constexpr std::size_t kMaxRows = 4096;
    constexpr std::size_t kMaxCols = 256;

    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (rows == 0 || cols == 0) return fail("Vulkan RMSNorm requires non-zero rows and cols");
    if (rows > kMaxRows || cols > kMaxCols) {
        return fail("Vulkan RMSNorm currently supports rows up to 4096 and cols up to 256");
    }
    if (!std::isfinite(eps) || !(eps > 0.0f)) return fail("Vulkan RMSNorm requires finite positive eps");
    if (rows > std::numeric_limits<std::size_t>::max() / cols) {
        return fail("Vulkan RMSNorm rows*cols overflows size_t");
    }
    if (x.size() != rows * cols) return fail("Vulkan RMSNorm input size must equal rows*cols");
    if (weight.size() != cols) return fail("Vulkan RMSNorm weight size must equal cols");

    std::vector<float> out(x.size(), 0.0f);
    const auto shader = rmsnorm_spirv(cols, eps);
    const std::vector<VulkanStorageBufferSpec> buffers = {
        {x.data(), x.size() * sizeof(float)},
        {weight.data(), weight.size() * sizeof(float)},
        {out.data(), out.size() * sizeof(float)},
    };
    const auto run = run_vulkan_storage_buffer_compute(
        shader.data(), shader.size(), buffers, {2}, static_cast<std::uint32_t>(rows), 1, 1);
    result.success = run.success;
    result.device_name = run.device_name;
    result.error = run.error;
    if (!run.success) return result;
    if (run.outputs.size() != 1 || run.outputs[0].size() != out.size() * sizeof(float)) {
        result.success = false;
        result.error = "Vulkan RMSNorm returned malformed output";
        return result;
    }

    result.output.resize(out.size());
    std::memcpy(result.output.data(), run.outputs[0].data(), run.outputs[0].size());
    return result;
}

VulkanF32TensorResult run_vulkan_swiglu(const std::vector<float>& packed,
                                        std::size_t rows,
                                        std::size_t hidden) {
    VulkanF32TensorResult result;
    constexpr std::size_t kMaxRows = 4096;
    constexpr std::size_t kMaxHidden = 4096;

    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (rows == 0 || hidden == 0) return fail("Vulkan SwiGLU requires non-zero rows and hidden");
    if (rows > kMaxRows || hidden > kMaxHidden) {
        return fail("Vulkan SwiGLU currently supports rows and hidden up to 4096");
    }
    if (hidden > std::numeric_limits<std::size_t>::max() / 2) {
        return fail("Vulkan SwiGLU 2*hidden overflows size_t");
    }
    const auto packed_cols = hidden * 2;
    if (rows > std::numeric_limits<std::size_t>::max() / packed_cols) {
        return fail("Vulkan SwiGLU rows*2*hidden overflows size_t");
    }
    if (rows > std::numeric_limits<std::size_t>::max() / hidden) {
        return fail("Vulkan SwiGLU rows*hidden overflows size_t");
    }
    if (packed.size() != rows * packed_cols) {
        return fail("Vulkan SwiGLU packed input size must equal rows*2*hidden");
    }

    std::vector<float> out(rows * hidden, 0.0f);
    const auto shader = swiglu_spirv(hidden);
    const std::vector<VulkanStorageBufferSpec> buffers = {
        {packed.data(), packed.size() * sizeof(float)},
        {out.data(), out.size() * sizeof(float)},
    };
    const auto run = run_vulkan_storage_buffer_compute(
        shader.data(), shader.size(), buffers, {1},
        static_cast<std::uint32_t>(hidden), static_cast<std::uint32_t>(rows), 1);
    result.success = run.success;
    result.device_name = run.device_name;
    result.error = run.error;
    if (!run.success) return result;
    if (run.outputs.size() != 1 || run.outputs[0].size() != out.size() * sizeof(float)) {
        result.success = false;
        result.error = "Vulkan SwiGLU returned malformed output";
        return result;
    }

    result.output.resize(out.size());
    std::memcpy(result.output.data(), run.outputs[0].data(), run.outputs[0].size());
    return result;
}

VulkanF32TensorResult run_vulkan_add(const std::vector<float>& a,
                                     const std::vector<float>& b) {
    VulkanF32TensorResult result;
    constexpr std::size_t kMaxElements = 1u << 20u;

    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (a.empty()) return fail("Vulkan add requires non-empty inputs");
    if (a.size() != b.size()) return fail("Vulkan add input sizes must match");
    if (a.size() > kMaxElements) return fail("Vulkan add currently supports up to 1048576 elements");

    std::vector<float> out(a.size(), 0.0f);
    const auto shader = add_spirv();
    const std::vector<VulkanStorageBufferSpec> buffers = {
        {a.data(), a.size() * sizeof(float)},
        {b.data(), b.size() * sizeof(float)},
        {out.data(), out.size() * sizeof(float)},
    };
    const auto run = run_vulkan_storage_buffer_compute(
        shader.data(), shader.size(), buffers, {2}, static_cast<std::uint32_t>(a.size()), 1, 1);
    result.success = run.success;
    result.device_name = run.device_name;
    result.error = run.error;
    if (!run.success) return result;
    if (run.outputs.size() != 1 || run.outputs[0].size() != out.size() * sizeof(float)) {
        result.success = false;
        result.error = "Vulkan add returned malformed output";
        return result;
    }

    result.output.resize(out.size());
    std::memcpy(result.output.data(), run.outputs[0].data(), run.outputs[0].size());
    return result;
}

// Standalone vector-API overload for sub is intentionally omitted: the
// host-staging path is not on the memory-native training hot path (only
// device-resident dispatch is used). sub parity is covered through
// run_vulkan_sub(runtime, ...) in the standalone test and through motifcl::sub
// in test_vulkan_backend.

VulkanF32TensorResult run_vulkan_i8_scaled_matmul(const std::vector<std::int8_t>& a,
                                                  const std::vector<std::int8_t>& b,
                                                  std::size_t m,
                                                  std::size_t k,
                                                  std::size_t n,
                                                  float scale_a,
                                                  float scale_b) {
    VulkanF32TensorResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (m == 0 || k == 0 || n == 0) return fail("Vulkan i8 scaled matmul requires non-zero M, K, and N");
    if (!std::isfinite(scale_a) || !std::isfinite(scale_b)) {
        return fail("Vulkan i8 scaled matmul requires finite scales");
    }
    if (m > std::numeric_limits<std::size_t>::max() / k) return fail("Vulkan i8 scaled matmul M*K overflows size_t");
    if (k > std::numeric_limits<std::size_t>::max() / n) return fail("Vulkan i8 scaled matmul K*N overflows size_t");
    if (a.size() != m * k) return fail("Vulkan i8 scaled matmul A size must equal M*K");
    if (b.size() != k * n) return fail("Vulkan i8 scaled matmul B size must equal K*N");

    auto runtime = VulkanRuntime::create();
    if (runtime.available() && runtime.supports_storage_buffer_i8()) {
        auto a_buffer = runtime.create_buffer(a.size() * sizeof(std::int8_t), a.data());
        auto b_buffer = runtime.create_buffer(b.size() * sizeof(std::int8_t), b.data());
        auto out_buffer = runtime.create_buffer(m * n * sizeof(float));
        const auto shader = i8_scaled_matmul_spirv(k, n, scale_a, scale_b);
        const std::vector<const VulkanBuffer*> buffers = {&a_buffer, &b_buffer, &out_buffer};
        const auto run = runtime.dispatch_storage_buffers(
            shader.data(), shader.size(), buffers, {},
            static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(m), 1);
        result.success = run.success;
        result.device_name = run.device_name;
        result.error = run.error;
        if (!run.success) return result;
        result.output.resize(m * n);
        out_buffer.download(result.output.data(), result.output.size() * sizeof(float));
        return result;
    }

    // Portability fallback for Vulkan devices without VK_KHR_8bit_storage: still executes the
    // signed dot and scaling in the Vulkan shader, but widens the uploaded payload to int32.
    std::vector<std::int32_t> ai(a.size());
    std::vector<std::int32_t> bi(b.size());
    for (std::size_t i = 0; i < a.size(); ++i) ai[i] = static_cast<std::int32_t>(a[i]);
    for (std::size_t i = 0; i < b.size(); ++i) bi[i] = static_cast<std::int32_t>(b[i]);

    std::vector<float> out(m * n, 0.0f);
    const auto shader = i32_scaled_matmul_spirv(k, n, scale_a, scale_b);
    const std::vector<VulkanStorageBufferSpec> buffers = {
        {ai.data(), ai.size() * sizeof(std::int32_t)},
        {bi.data(), bi.size() * sizeof(std::int32_t)},
        {out.data(), out.size() * sizeof(float)},
    };
    const auto run = run_vulkan_storage_buffer_compute(
        shader.data(), shader.size(), buffers, {2},
        static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(m), 1);
    result.success = run.success;
    result.device_name = run.device_name;
    result.error = run.error;
    if (!run.success) return result;
    if (run.outputs.size() != 1 || run.outputs[0].size() != out.size() * sizeof(float)) {
        result.success = false;
        result.error = "Vulkan i8 scaled matmul returned malformed output";
        return result;
    }
    result.output.resize(out.size());
    std::memcpy(result.output.data(), run.outputs[0].data(), run.outputs[0].size());
    return result;
}

VulkanOpResult run_vulkan_grouped_query_attention(VulkanRuntime& runtime,
                                                  const VulkanBuffer& q,
                                                  const VulkanBuffer& k,
                                                  const VulkanBuffer& v,
                                                  VulkanBuffer& out,
                                                  std::size_t query_tokens,
                                                  std::size_t key_tokens,
                                                  std::size_t n_head,
                                                  std::size_t n_kv_head,
                                                  std::size_t head_dim,
                                                  float scale) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (query_tokens == 0 || key_tokens == 0 || n_head == 0 || n_kv_head == 0 || head_dim == 0) {
        return fail("Vulkan GQA requires non-zero query/key/head dimensions");
    }
    if (n_head % n_kv_head != 0) return fail("Vulkan GQA requires n_head % n_kv_head == 0");
    if (key_tokens > 1024) return fail("Vulkan GQA currently supports key_tokens <= 1024 (shared-memory softmax)");
    if (!std::isfinite(scale)) return fail("Vulkan GQA requires a finite scale");
    const std::size_t q_size = query_tokens * n_head * head_dim;
    const std::size_t kv_size = key_tokens * n_kv_head * head_dim;
    if (q.nbytes() < q_size * sizeof(float)) return fail("Vulkan GQA q buffer is too small");
    if (k.nbytes() < kv_size * sizeof(float)) return fail("Vulkan GQA k buffer is too small");
    if (v.nbytes() < kv_size * sizeof(float)) return fail("Vulkan GQA v buffer is too small");
    if (out.nbytes() < q_size * sizeof(float)) return fail("Vulkan GQA output buffer is too small");

    const struct {
        std::uint32_t query_tokens;
        std::uint32_t key_tokens;
        std::uint32_t n_head;
        std::uint32_t n_kv_head;
        std::uint32_t head_dim;
        float scale;
    } push{static_cast<std::uint32_t>(query_tokens), static_cast<std::uint32_t>(key_tokens),
           static_cast<std::uint32_t>(n_head),       static_cast<std::uint32_t>(n_kv_head),
           static_cast<std::uint32_t>(head_dim),     scale};
    const std::vector<const VulkanBuffer*> buffers = {&q, &k, &v, &out};
    const bool use_sg_gqa = runtime.caps().subgroup_arithmetic_compute;
    const auto* gqa_spirv = use_sg_gqa ? vkspirv::k_gqa_fwd_f32_subgroup : vkspirv::k_gqa_fwd_f32;
    const auto gqa_words = use_sg_gqa ? vkspirv::k_gqa_fwd_f32_subgroup_words : vkspirv::k_gqa_fwd_f32_words;
    return runtime.dispatch_cached(gqa_spirv, gqa_words, buffers, &push,
                                   sizeof(push), static_cast<std::uint32_t>(query_tokens),
                                   static_cast<std::uint32_t>(n_head), 1);
}

VulkanOpResult run_vulkan_grouped_query_attention_backward(VulkanRuntime& runtime,
                                                           const VulkanBuffer& q,
                                                           const VulkanBuffer& k,
                                                           const VulkanBuffer& v,
                                                           const VulkanBuffer& grad_out,
                                                           VulkanBuffer& probs_scratch,
                                                           VulkanBuffer& ds_scratch,
                                                           VulkanBuffer& grad_q,
                                                           VulkanBuffer& grad_k,
                                                           VulkanBuffer& grad_v,
                                                           std::size_t query_tokens,
                                                           std::size_t key_tokens,
                                                           std::size_t n_head,
                                                           std::size_t n_kv_head,
                                                           std::size_t head_dim,
                                                           float scale) {
    VulkanOpResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (query_tokens == 0 || key_tokens == 0 || n_head == 0 || n_kv_head == 0 || head_dim == 0) {
        return fail("Vulkan GQA backward requires non-zero query/key/head dimensions");
    }
    if (n_head % n_kv_head != 0) return fail("Vulkan GQA backward requires n_head % n_kv_head == 0");
    if (key_tokens > 1024) {
        return fail("Vulkan GQA backward currently supports key_tokens <= 1024 (shared-memory softmax)");
    }
    if (!std::isfinite(scale)) return fail("Vulkan GQA backward requires a finite scale");
    const std::size_t q_size = query_tokens * n_head * head_dim;
    const std::size_t kv_size = key_tokens * n_kv_head * head_dim;
    const std::size_t probs_size = n_head * query_tokens * key_tokens;
    if (q.nbytes() < q_size * sizeof(float) || grad_out.nbytes() < q_size * sizeof(float) ||
        grad_q.nbytes() < q_size * sizeof(float)) {
        return fail("Vulkan GQA backward q/grad buffer is too small");
    }
    if (k.nbytes() < kv_size * sizeof(float) || v.nbytes() < kv_size * sizeof(float) ||
        grad_k.nbytes() < kv_size * sizeof(float) || grad_v.nbytes() < kv_size * sizeof(float)) {
        return fail("Vulkan GQA backward k/v buffer is too small");
    }
    if (probs_scratch.nbytes() < probs_size * sizeof(float) || ds_scratch.nbytes() < probs_size * sizeof(float)) {
        return fail("Vulkan GQA backward scratch buffer is too small");
    }

    const struct {
        std::uint32_t query_tokens;
        std::uint32_t key_tokens;
        std::uint32_t n_head;
        std::uint32_t n_kv_head;
        std::uint32_t head_dim;
        float scale;
    } push{static_cast<std::uint32_t>(query_tokens), static_cast<std::uint32_t>(key_tokens),
           static_cast<std::uint32_t>(n_head),       static_cast<std::uint32_t>(n_kv_head),
           static_cast<std::uint32_t>(head_dim),     scale};

    // Stage 1: softmax probabilities + ds. Stage 2: dQ. Stage 3: dK/dV. In
    // immediate mode each cached dispatch submits and waits on its fence; in
    // batch mode the cached path emits compute->compute barriers.
    const std::vector<const VulkanBuffer*> probs_buffers = {&q, &k, &v, &grad_out, &probs_scratch, &ds_scratch};
    const bool use_sg_bwd = runtime.caps().subgroup_arithmetic_compute;
    const auto* bwd_spirv = use_sg_bwd ? vkspirv::k_gqa_bwd_probs_f32_subgroup : vkspirv::k_gqa_bwd_probs_f32;
    const auto bwd_words = use_sg_bwd ? vkspirv::k_gqa_bwd_probs_f32_subgroup_words : vkspirv::k_gqa_bwd_probs_f32_words;
    auto stage = runtime.dispatch_cached(bwd_spirv, bwd_words,
                                         probs_buffers, &push, sizeof(push),
                                         static_cast<std::uint32_t>(query_tokens),
                                         static_cast<std::uint32_t>(n_head), 1);
    if (!stage.success) return stage;

    const std::vector<const VulkanBuffer*> dq_buffers = {&k, &ds_scratch, &grad_q};
    stage = runtime.dispatch_cached(vkspirv::k_gqa_bwd_dq_f32, vkspirv::k_gqa_bwd_dq_f32_words, dq_buffers,
                                    &push, sizeof(push), static_cast<std::uint32_t>(query_tokens),
                                    static_cast<std::uint32_t>(n_head), 1);
    if (!stage.success) return stage;

    const std::vector<const VulkanBuffer*> dkv_buffers = {&q, &grad_out, &probs_scratch, &ds_scratch, &grad_k,
                                                          &grad_v};
    return runtime.dispatch_cached(vkspirv::k_gqa_bwd_dkv_f32, vkspirv::k_gqa_bwd_dkv_f32_words, dkv_buffers,
                                   &push, sizeof(push), static_cast<std::uint32_t>(key_tokens),
                                   static_cast<std::uint32_t>(n_kv_head), 1);
}

VulkanF32TensorResult run_vulkan_grouped_query_attention(const std::vector<float>& q,
                                                         const std::vector<float>& k,
                                                         const std::vector<float>& v,
                                                         std::size_t query_tokens,
                                                         std::size_t key_tokens,
                                                         std::size_t n_head,
                                                         std::size_t n_kv_head,
                                                         std::size_t head_dim,
                                                         float scale) {
    VulkanF32TensorResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (query_tokens == 0 || key_tokens == 0 || n_head == 0 || n_kv_head == 0 || head_dim == 0) {
        return fail("Vulkan GQA requires non-zero query/key/head dimensions");
    }
    if (n_head % n_kv_head != 0) return fail("Vulkan GQA requires n_head % n_kv_head == 0");
    if (head_dim > 64 || key_tokens > 64) return fail("Vulkan GQA staged path currently supports head_dim,key_tokens <= 64");
    if (!std::isfinite(scale)) return fail("Vulkan GQA requires a finite scale");
    const std::size_t q_size = query_tokens * n_head * head_dim;
    const std::size_t kv_size = key_tokens * n_kv_head * head_dim;
    if (q.size() != q_size) return fail("Vulkan GQA q size mismatch");
    if (k.size() != kv_size) return fail("Vulkan GQA k size mismatch");
    if (v.size() != kv_size) return fail("Vulkan GQA v size mismatch");

    std::vector<float> out(q_size, 0.0f);
    const auto shader = gqa_forward_spirv(query_tokens, key_tokens, n_head, n_kv_head, head_dim, scale);
    const std::vector<VulkanStorageBufferSpec> buffers = {
        {q.data(), q.size() * sizeof(float)},
        {k.data(), k.size() * sizeof(float)},
        {v.data(), v.size() * sizeof(float)},
        {out.data(), out.size() * sizeof(float)},
    };
    const auto run = run_vulkan_storage_buffer_compute(
        shader.data(), shader.size(), buffers, {3},
        static_cast<std::uint32_t>(head_dim),
        static_cast<std::uint32_t>(n_head),
        static_cast<std::uint32_t>(query_tokens));
    result.success = run.success;
    result.device_name = run.device_name;
    result.error = run.error;
    if (!run.success) return result;
    if (run.outputs.size() != 1 || run.outputs[0].size() != out.size() * sizeof(float)) {
        result.success = false;
        result.error = "Vulkan GQA returned malformed output";
        return result;
    }
    result.output.resize(out.size());
    std::memcpy(result.output.data(), run.outputs[0].data(), run.outputs[0].size());
    return result;
}

VulkanF32TensorResult run_vulkan_compact_counter_backward_input(
    const std::vector<std::uint32_t>& packed_state_words,
    const std::vector<float>& scale,
    const std::vector<float>& grad_out,
    std::size_t batch,
    std::size_t in_features,
    std::size_t out_features,
    std::size_t C) {
    VulkanF32TensorResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (batch == 0 || in_features == 0 || out_features == 0) {
        return fail("Vulkan compact-counter backward requires non-zero dimensions");
    }
    if (in_features % 4 != 0) return fail("Vulkan compact-counter backward requires in_features divisible by 4");
    if (C == 0 || 3 * (2 * C - 1) > 64) return fail("Vulkan compact-counter backward C is out of 6-bit range");
    const std::size_t gpr = in_features / 4;
    if (packed_state_words.size() != out_features * gpr) return fail("Vulkan compact-counter state word size mismatch");
    if (scale.size() != out_features) return fail("Vulkan compact-counter scale size mismatch");
    if (grad_out.size() != batch * out_features) return fail("Vulkan compact-counter grad_out size mismatch");

    auto runtime = VulkanRuntime::create();
    if (!runtime.available()) {
        return fail(runtime.error().empty() ? "Vulkan runtime is not available" : runtime.error());
    }
    auto state_buffer = runtime.create_buffer(packed_state_words.size() * sizeof(std::uint32_t),
                                              packed_state_words.data());
    auto scale_buffer = runtime.create_buffer(scale.size() * sizeof(float), scale.data());
    auto grad_out_buffer = runtime.create_buffer(grad_out.size() * sizeof(float), grad_out.data());
    auto grad_x_buffer = runtime.create_buffer(batch * in_features * sizeof(float));

    const auto shader = counter_backward_input_fused_spirv(in_features, out_features, C);
    const std::vector<const VulkanBuffer*> buffers = {&grad_out_buffer, &state_buffer, &scale_buffer, &grad_x_buffer};
    const auto run = runtime.dispatch_storage_buffers(
        shader.data(), shader.size(), buffers, {},
        static_cast<std::uint32_t>(batch * in_features), 1, 1);
    result.success = run.success;
    result.device_name = run.device_name;
    result.error = run.error;
    if (!run.success) return result;
    result.output.resize(batch * in_features);
    grad_x_buffer.download(result.output.data(), result.output.size() * sizeof(float));
    return result;
}

VulkanU32TensorResult run_vulkan_compact_counter_increment(
    const std::vector<std::uint32_t>& packed_state_words,
    const std::vector<std::uint32_t>& packed_increment_words) {
    VulkanU32TensorResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (packed_state_words.empty()) return fail("Vulkan compact-counter increment requires non-empty state");
    if (packed_state_words.size() != packed_increment_words.size()) {
        return fail("Vulkan compact-counter increment state/increment sizes must match");
    }
    std::vector<std::uint32_t> out(packed_state_words.size(), 0);
    const auto shader = counter_increment_spirv();
    const std::vector<VulkanStorageBufferSpec> buffers = {
        {packed_state_words.data(), packed_state_words.size() * sizeof(std::uint32_t)},
        {packed_increment_words.data(), packed_increment_words.size() * sizeof(std::uint32_t)},
        {out.data(), out.size() * sizeof(std::uint32_t)},
    };
    const auto run = run_vulkan_storage_buffer_compute(
        shader.data(), shader.size(), buffers, {2}, static_cast<std::uint32_t>(out.size()), 1, 1);
    result.success = run.success;
    result.device_name = run.device_name;
    result.error = run.error;
    if (!run.success) return result;
    if (run.outputs.size() != 1 || run.outputs[0].size() != out.size() * sizeof(std::uint32_t)) {
        result.success = false;
        result.error = "Vulkan compact-counter increment returned malformed output";
        return result;
    }
    result.output.resize(out.size());
    std::memcpy(result.output.data(), run.outputs[0].data(), run.outputs[0].size());
    return result;
}

VulkanF32TensorResult run_vulkan_sgd_update(const std::vector<float>& param,
                                            const std::vector<float>& grad,
                                            float lr) {
    VulkanF32TensorResult result;
    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (param.empty()) return fail("Vulkan SGD update requires non-empty parameters");
    if (param.size() != grad.size()) return fail("Vulkan SGD update param/grad sizes must match");
    if (!std::isfinite(lr)) return fail("Vulkan SGD update requires finite lr");
    std::vector<float> out(param.size(), 0.0f);
    const auto shader = sgd_update_spirv(lr);
    const std::vector<VulkanStorageBufferSpec> buffers = {
        {param.data(), param.size() * sizeof(float)},
        {grad.data(), grad.size() * sizeof(float)},
        {out.data(), out.size() * sizeof(float)},
    };
    const auto run = run_vulkan_storage_buffer_compute(
        shader.data(), shader.size(), buffers, {2}, static_cast<std::uint32_t>(out.size()), 1, 1);
    result.success = run.success;
    result.device_name = run.device_name;
    result.error = run.error;
    if (!run.success) return result;
    if (run.outputs.size() != 1 || run.outputs[0].size() != out.size() * sizeof(float)) {
        result.success = false;
        result.error = "Vulkan SGD update returned malformed output";
        return result;
    }
    result.output.resize(out.size());
    std::memcpy(result.output.data(), run.outputs[0].data(), run.outputs[0].size());
    return result;
}

VulkanF32MatmulSmokeResult run_vulkan_f32_m1_matmul(const std::vector<float>& a,
                                                    const std::vector<float>& b,
                                                    std::size_t k,
                                                    std::size_t n) {
    VulkanF32MatmulSmokeResult result;
    constexpr std::size_t kMaxSpecializedDim = 64;

    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };
    if (k == 0 || n == 0) return fail("Vulkan f32 M=1 matmul requires non-zero K and N");
    if (k > kMaxSpecializedDim || n > kMaxSpecializedDim) {
        return fail("Vulkan f32 M=1 matmul smoke supports K,N up to 64");
    }
    if (a.size() != k) return fail("Vulkan f32 M=1 matmul A size must equal K");
    if (k > std::numeric_limits<std::size_t>::max() / n) {
        return fail("Vulkan f32 M=1 matmul K*N overflows size_t");
    }
    if (b.size() != k * n) return fail("Vulkan f32 M=1 matmul B size must equal K*N");

    std::vector<float> c(n, 0.0f);
    const auto shader = f32_m1_matmul_spirv(k, n);
    const std::vector<VulkanStorageBufferSpec> buffers = {
        {a.data(), a.size() * sizeof(float)},
        {b.data(), b.size() * sizeof(float)},
        {c.data(), c.size() * sizeof(float)},
    };
    const auto run = run_vulkan_storage_buffer_compute(shader.data(), shader.size(), buffers, {2});
    result.success = run.success;
    result.device_name = run.device_name;
    result.error = run.error;
    if (!run.success) return result;
    if (run.outputs.size() != 1 || run.outputs[0].size() != c.size() * sizeof(float)) {
        result.success = false;
        result.error = "Vulkan f32 M=1 matmul returned malformed output";
        return result;
    }

    result.output.resize(c.size());
    std::memcpy(result.output.data(), run.outputs[0].data(), run.outputs[0].size());
    return result;
}

VulkanF32MatmulSmokeResult run_vulkan_f32_matmul_smoke() {
    VulkanF32MatmulSmokeResult result;
    const std::array<float, 4> a = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::array<float, 16> b = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f,
    };
    const std::array<float, 4> expected = {90.0f, 100.0f, 110.0f, 120.0f};
    std::array<float, 4> c = {0.0f, 0.0f, 0.0f, 0.0f};

    const auto& shader = f32_matmul_1x4x4_spirv();
    const std::vector<VulkanStorageBufferSpec> buffers = {
        {a.data(), a.size() * sizeof(float)},
        {b.data(), b.size() * sizeof(float)},
        {c.data(), c.size() * sizeof(float)},
    };
    const auto run = run_vulkan_storage_buffer_compute(shader.data(), shader.size(), buffers, {2});
    result.success = run.success;
    result.device_name = run.device_name;
    result.error = run.error;
    if (!run.success) return result;
    if (run.outputs.size() != 1 || run.outputs[0].size() != c.size() * sizeof(float)) {
        result.success = false;
        result.error = "Vulkan f32 matmul smoke returned malformed output";
        return result;
    }

    result.output.resize(c.size());
    std::memcpy(result.output.data(), run.outputs[0].data(), run.outputs[0].size());
    if (!std::equal(result.output.begin(), result.output.end(), expected.begin())) {
        result.success = false;
        result.error = "Vulkan f32 matmul smoke returned unexpected output";
    }
    return result;
}

} // namespace motifcl
