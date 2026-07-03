#!/usr/bin/env bash
set -euo pipefail

mode="${1:-macos}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out_dir="$root/build/apple-metal-probe/$mode"
mkdir -p "$out_dir"

case "$mode" in
  macos)
    xcrun --sdk macosx clang++ -std=c++17 -fobjc-arc \
      -framework Foundation -framework Metal \
      "$root/ports/apple/metal_probe.mm" \
      -o "$out_dir/metal_probe"
    "$out_dir/metal_probe"
    ;;
  ios)
    xcrun --sdk iphoneos clang++ -std=c++17 -fobjc-arc \
      -arch arm64 -miphoneos-version-min=13.0 \
      -framework Foundation -framework Metal \
      "$root/ports/apple/metal_probe.mm" \
      -o "$out_dir/metal_probe_ios_arm64"
    echo "Built $out_dir/metal_probe_ios_arm64"
    echo "Deploy/sign it through Xcode or your iOS provisioning flow."
    ;;
  *)
    echo "usage: $0 [macos|ios]" >&2
    exit 2
    ;;
esac
