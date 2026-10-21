"""CPU-only known-answer probe of trainer timing snippets, with no GPU build.

Compile the actual accumulator, formatting and reset code extracted from each
trainer into a temporary standalone program. Synthetic samples make last-step
versus window-average errors observable without timing noise or a GPU context.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def block(source: str, start: int) -> str:
    begin = source.index("{", start)
    depth = 1
    end = begin + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def build_probe(name: str) -> str:
    source = (ROOT / "tools" / name).read_text(encoding="utf-8")
    moe = name == "train_moe15m.cpp"
    begin = source.index("double win_ms = 0.0;")
    declaration_end = source.index("auto ms_since" if moe else "for (int step = start_step;", begin)
    declarations = source[begin:declaration_end]
    accumulation = source[source.index("win_ms +=", declaration_end):]
    accumulation = accumulation[:accumulation.index("if ((step + 1) % log_every")]
    log_start = source.index("if ((step + 1) % log_every", declaration_end)
    print_start = source.index('std::printf("step %d loss', log_start)
    prefix = source[print_start:source.index(");", print_start) + 2]
    gpu = block(source, source.index("if (gpu_timing) {", print_start))
    reset_start = source.index("win_ms = 0.0;", source.index("std::fflush(stdout);", print_start))
    reset = source[reset_start:source.index("\n            }", reset_start)]
    phase_assignments = """
        phases.upload_ms = s[0]; phases.fwd_ms = s[1];
        phases.record_ms = s[2]; phases.submit_ms = s[3];
        phases.sync_ms = s[4]; phases.balance_ms = s[5];
        phases.optim_ms = s[6]; phases.total_ms = s[7];
    """ if moe else ""
    return r'''
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
struct MockRuntime {
    double buffer_bytes_live() const { return 0; }
    double buffer_bytes_peak() const { return 0; }
} runtime;
int main() {
    constexpr int N = 2048, steps_total = 200;
    constexpr bool gpu_timing = true;
    double lv = 1.0, lr_c = 0.004;
    double gpu_fb_ms = 0, gpu_opt_ms = 0, gpu_step_ms = 0, sync_ms = 0;
    @DECLARATIONS@
    const std::array<std::array<double,10>,5> samples{{
        {{1,2,4,5,6,7,8,40,9,1}},
        {{3,6,8,7,10,11,12,60,15,3}},
        {{5,10,12,9,14,15,16,80,20,4}},
        {{1,2,4,5,6,7,8,40,-1,-1}},
        {{3,6,8,7,10,11,12,60,15,3}},
    }};
    for (int step = 0; step < int(samples.size()); ++step) {
        const auto& s = samples[step];
        @ASSIGN_PHASES@
        gpu_fb_ms = s[8]; gpu_opt_ms = s[9]; gpu_step_ms = s[8]; sync_ms = s[4];
        const auto t0 = std::chrono::steady_clock::time_point{};
        const auto t1 = t0 + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double,std::milli>(s[7]));
        @ACCUMULATION@
        if (step == 1 || step == 2 || step == 4) {
            const double ms = win_ms / win_n;
            const double tok_s = double(N) * 1000.0 / ms;
            const double eta_h = 0;
            @PREFIX@
            @GPU@
            std::printf("\n");
            @RESET@
        }
    }
}
'''.replace("@DECLARATIONS@", declarations).replace("@ASSIGN_PHASES@", phase_assignments).replace(
        "@ACCUMULATION@", accumulation).replace("@PREFIX@", prefix).replace("@GPU@", gpu).replace("@RESET@", reset)


class TimingWindowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="motifcl-timing-window-")
        cls.output = {}
        compiler = os.environ.get("CXX") or shutil.which("c++") or r"C:\Strawberry\c\bin\c++.exe"
        for name in ("train_moe15m.cpp", "train_ternary15m.cpp"):
            source = Path(cls.temp.name) / name
            exe = source.with_suffix(".exe" if os.name == "nt" else ".bin")
            source.write_text(build_probe(name), encoding="utf-8")
            subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(exe)], check=True)
            cls.output[name] = subprocess.check_output([str(exe)], text=True)
            print(f"SOURCE-DERIVED {name}:\n{cls.output[name]}", flush=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_moe_average_reset_and_missing_gpu(self):
        pattern = (r"ms fwd\+bwd gpu ([-\d.]+) opt gpu ([-\d.]+) sync ([-\d.]+)"
                   r"\s+phases upload ([\d.]+) fwd ([\d.]+) record ([\d.]+) submit ([\d.]+)"
                   r" balance ([\d.]+) optim ([\d.]+) total ([\d.]+)")
        got = [tuple(map(float, m)) for m in re.findall(pattern, self.output["train_moe15m.cpp"])]
        self.assertEqual(got, [
            (12,2,8,2,4,6,6,9,10,50),
            (20,4,14,5,10,12,9,15,16,80),
            (-1,-1,8,2,4,6,6,9,10,50),
        ])

    def test_dense_average_reset_and_missing_gpu(self):
        got = [tuple(map(float, m)) for m in re.findall(
            r"ms gpu ([-\d.]+) sync ([-\d.]+)", self.output["train_ternary15m.cpp"])]
        self.assertEqual(got, [(12,8), (20,14), (-1,8)])

    def test_existing_parser_compatibility_and_step_time(self):
        spec = importlib.util.spec_from_file_location("moe_log_summary", ROOT / "tools/moe_log_summary.py")
        parser = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parser)
        for name, output in self.output.items():
            with self.subTest(trainer=name):
                path = Path(self.temp.name) / (name + ".log")
                path.write_text(output, encoding="utf-8")
                parsed = parser.parse(path)
                self.assertEqual(parsed["steps"], 3)
                self.assertEqual(parsed["tok_s"], [40960,25600,40960])
                lines = [line for line in output.splitlines() if line.startswith("step ")]
                self.assertTrue(all("ms " in line for line in lines), "timing must stay on the step line")


if __name__ == "__main__":
    unittest.main(verbosity=2)
