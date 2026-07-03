#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::fprintf(stderr, "Metal device unavailable\n");
            return 77;
        }

        std::printf("Metal device: %s\n", device.name.UTF8String);
        std::printf("low power: %s\n", device.lowPower ? "yes" : "no");
        std::printf("headless: %s\n", device.headless ? "yes" : "no");
        if (@available(macOS 10.15, iOS 13.0, *)) {
            std::printf("unified memory: %s\n", device.hasUnifiedMemory ? "yes" : "no");
        }
        std::printf("max buffer length: %llu\n",
                    static_cast<unsigned long long>(device.maxBufferLength));
    }
    return 0;
}
