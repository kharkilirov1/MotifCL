"""Stdlib-only parser/environment/orchestration tests; never launch a GPU process."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("benchmark_vulkan_training", ROOT / "tools/benchmark_vulkan_training.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def log_text(rates=(99999, 1000, 2000), window=True, final=True):
    lines = [f"mn-moe15m pretrain: steps={len(rates)*20} B=8 T=256 start=0"]
    for index, rate in enumerate(rates, 1):
        suffix = "  phases total 1.0 window 20" if window else ""
        lines.append(f"step {index * 20} loss 4.2 ema 4.3 lr 0.004  {rate} tok/s  eta 0.0 h  devmem 100/200 MB{suffix}")
    if final:
        lines.append("FINAL VAL loss 4.25  ppl 70.105")
    return "\n".join(lines) + "\n"


class ParserTest(unittest.TestCase):
    def parse(self, text, exit_code=0):
        return bench.parse_log(text, exit_code, steps=60, warmup=20, log_every=20)

    def test_harmonic_excludes_warmup_and_preserves_window_labels(self):
        for new_labels in (False, True):
            with self.subTest(new_labels=new_labels):
                got = self.parse(log_text(window=new_labels))
                self.assertTrue(got["valid"], got["errors"])
                self.assertAlmostEqual(got["steady_tok_s"], 4000 / 3)
                self.assertEqual(got["steady_steps"], 40)
                self.assertEqual([w["steps"] for w in got["windows"]], [20,20,20])
                self.assertEqual(got["val_loss"], 4.25)

    def test_exit_nonzero_invalid_even_with_complete_log(self):
        got = self.parse(log_text(), 7)
        self.assertFalse(got["valid"])
        self.assertIsNone(got["steady_tok_s"])
        self.assertTrue(any("7" in error for error in got["errors"]))

    def test_missing_middle_or_final_window_invalid(self):
        for step in (40,60):
            text = "\n".join(line for line in log_text().splitlines() if not line.startswith(f"step {step} "))
            got = self.parse(text)
            self.assertFalse(got["valid"])
            self.assertIsNone(got["steady_tok_s"])

    def test_missing_or_nonfinite_final_val_invalid(self):
        for text in (log_text(final=False), log_text().replace("FINAL VAL loss 4.25", "FINAL VAL loss nan")):
            got = self.parse(text)
            self.assertFalse(got["valid"])
            self.assertIsNone(got["steady_tok_s"])

    def test_nonpositive_nonfinite_rate_or_loss_invalid(self):
        for text in (log_text(rates=(1,0,2)), log_text(rates=(1,"nan",2)),
                     log_text(rates=(1,"inf",2)), log_text().replace("loss 4.2", "loss nan")):
            got = self.parse(text)
            self.assertFalse(got["valid"])
            self.assertIsNone(got["steady_tok_s"])

    def test_duplicate_out_of_order_and_incorrect_window_count_invalid(self):
        text = log_text()
        for bad in (text + text.splitlines()[1] + "\n", text.replace("window 20", "window 19"),
                    text.replace("step 40 ", "step 60 ")):
            self.assertFalse(self.parse(bad)["valid"])

    def test_known_legacy_dense_timing_join_only(self):
        rates = (13836,15877,15879,15855,15834,15889)
        text = log_text(rates=rates, window=False)
        prefixes = ("  ms gpu 125.47 sync 0.02", "  ms gpu 125.90 sync 0.02",
                    "  ms gpu 125.71 sync 0.02", "  ms gpu 127.10 sync 0.02",
                    "  ms gpu 126.09 sync 0.03")
        for step, prefix in zip(range(40,121,20), prefixes):
            text = text.replace(f"\nstep {step} loss ", f"\n{prefix}step {step} loss ")
        got = bench.parse_log(text, 0, steps=120, warmup=20, log_every=20)
        self.assertTrue(got["valid"], got["errors"])
        self.assertEqual([w["step"] for w in got["windows"]], [20,40,60,80,100,120])
        self.assertAlmostEqual(got["steady_tok_s"], 5 / sum(1 / rate for rate in rates[1:]))
        for malformed in (text.replace(prefixes[0], "driver debug" + prefixes[0]),
                          text.replace(prefixes[0], "  ms gpu nan sync 0.02"),
                          text.replace(prefixes[0], "  ms gpu 125.47 sync BROKEN0.02"),
                          text + "step 40 loss 4.2 ema 4.3 lr 0.004  15877 tok/s\n"):
            self.assertFalse(bench.parse_log(malformed, 0, steps=120, warmup=20, log_every=20)["valid"])


class EnvironmentTest(unittest.TestCase):
    def test_allowlist_and_fixed_training_settings(self):
        env = bench.build_environment(
            {"PATH":"test-path", "SystemRoot":"C:/Windows", "TEMP":"test-temp",
             "MOTIFCL_VK_CE_CACHE":"0", "motifcl_vk_rms_dw_split":"0", "LR":"999",
             "BATCH":"777", "EVAL_ONLY":"1", "API_TOKEN":"do-not-forward", "UNRELATED":"1"},
            data_dir=Path("data"), run_dir=Path("run"), steps=120, log_every=20, profile=False)
        for key, value in {"BATCH":"8", "SEQ":"256", "SEED":"1337", "HIDDEN_EXP":"192",
                           "HIDDEN_SHARED":"192", "HIDDEN":"768", "LR":"0.004",
                           "RESUME":"0", "SAVE_CKPT":"0", "EVAL_ONLY":"0",
                           "MOTIFCL_VK_PROFILE":"0", "STEPS":"120", "LOG_EVERY":"20"}.items():
            self.assertEqual(env[key], value)
        self.assertNotIn("API_TOKEN", env)
        self.assertNotIn("UNRELATED", env)
        self.assertNotIn("MOTIFCL_VK_CE_CACHE", env)
        self.assertNotIn("motifcl_vk_rms_dw_split", env)
        self.assertEqual(env["SystemRoot"], "C:/Windows")

    def test_mixed_case_windows_path_is_not_lost_or_duplicated(self):
        env = bench.build_environment({"Path":"must-keep"}, data_dir=Path("data"),
            run_dir=Path("run"), steps=120, log_every=20, profile=False)
        self.assertIn("must-keep", env["PATH"])
        self.assertNotIn("Path", env)


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="motifcl-abba-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.baseline = self.root / "baseline.exe"
        self.candidate = self.root / "candidate.exe"
        self.baseline.write_bytes(b"baseline known binary")
        self.candidate.write_bytes(b"candidate known binary")
        self.data = self.root / "data"
        self.data.mkdir()
        for name in ("train.bin", "val.bin"):
            (self.data / name).write_bytes(b"synthetic test data")
        self.output = self.root / "report"

    def argv(self):
        return [str(self.baseline), str(self.candidate), "--data-dir", str(self.data),
                "--output-dir", str(self.output), "--steps", "60", "--pairs", "2"]

    def fake_run(self, command, *, stdout, stderr, cwd, env):
        self.assertEqual(stderr, subprocess.STDOUT)
        rate = 1000 if Path(command[0]) == self.baseline else 2000
        stdout.write(("raw driver diagnostic\n" + log_text(rates=(99999,rate,rate))).encode())
        return subprocess.CompletedProcess(command, 0)

    def test_abba_order_hash_settings_rawlogs_and_comparison(self):
        with mock.patch.object(bench.subprocess, "run", side_effect=self.fake_run) as run:
            self.assertEqual(bench.main(self.argv()), 0)
        calls = [Path(call.args[0][0]).stem for call in run.call_args_list]
        self.assertEqual(calls, ["baseline","candidate","candidate","baseline"] * 2)
        report = json.loads((self.output / "report.json").read_text())
        self.assertTrue(report["valid"])
        self.assertEqual(report["summary"]["speedup"], 2.0)
        self.assertEqual(len(report["runs"]), 8)
        for result in report["runs"]:
            self.assertEqual(result["sha256"], hashlib.sha256(Path(result["exe"]).read_bytes()).hexdigest())
            self.assertEqual(result["exit"], 0)
            self.assertEqual(result["settings"]["SEED"], "1337")
            raw = Path(result["log"]).read_bytes()
            self.assertTrue(raw.startswith(b"raw driver diagnostic\n"))
            self.assertEqual(result["log_sha256"], hashlib.sha256(raw).hexdigest())

    def test_nonzero_exit_retained_without_speedup(self):
        def fail(command, **kwargs):
            self.fake_run(command, **kwargs)
            return subprocess.CompletedProcess(command, 9)
        with mock.patch.object(bench.subprocess, "run", side_effect=fail):
            self.assertEqual(bench.main(self.argv()), 1)
        report = json.loads((self.output / "report.json").read_text())
        self.assertFalse(report["valid"])
        self.assertNotIn("speedup", report["summary"])
        self.assertEqual(report["runs"][0]["exit"], 9)
        self.assertTrue(Path(report["runs"][0]["log"]).exists())

    def test_missing_final_or_windows_cannot_produce_comparison(self):
        for text in (log_text(final=False), "FINAL VAL loss 4.25\n"):
            with self.subTest(text=text):
                self.output = self.root / ("report-" + str(len(list(self.root.glob("report-*")))))
                def invalid(command, **kwargs):
                    kwargs["stdout"].write(text.encode())
                    return subprocess.CompletedProcess(command, 0)
                with mock.patch.object(bench.subprocess, "run", side_effect=invalid):
                    self.assertEqual(bench.main(self.argv()), 1)
                report = json.loads((self.output / "report.json").read_text())
                self.assertFalse(report["valid"])
                self.assertNotIn("speedup", report["summary"])

    def test_profile_is_separate_noncomparison_mode(self):
        with mock.patch.object(bench.subprocess, "run", side_effect=self.fake_run) as run:
            self.assertEqual(bench.main(self.argv() + ["--profile"]), 0)
        report = json.loads((self.output / "report.json").read_text())
        self.assertEqual(report["mode"], "profile")
        self.assertEqual(run.call_count, 2)
        self.assertNotIn("speedup", report["summary"])
        self.assertTrue(all(call.kwargs["env"]["MOTIFCL_VK_PROFILE"] == "1" for call in run.call_args_list))

    def test_profile_can_run_warmup_only_without_throughput_claim(self):
        args = self.argv() + ["--profile", "--steps", "20"]
        def profile(command, **kwargs):
            kwargs["stdout"].write(log_text(rates=(1000,)).encode())
            return subprocess.CompletedProcess(command, 0)
        with mock.patch.object(bench.subprocess, "run", side_effect=profile):
            self.assertEqual(bench.main(args), 0)
        report = json.loads((self.output / "report.json").read_text())
        self.assertTrue(report["valid"])
        self.assertNotIn("speedup", report["summary"])
        self.assertTrue(all(run["steady_tok_s"] is None for run in report["runs"]))

    def test_binary_drift_and_launch_failure_invalidate_without_comparison(self):
        for drift in (True,False):
            with self.subTest(drift=drift):
                self.output = self.root / ("drift" if drift else "launch-error")
                def changed(command, **kwargs):
                    if not drift:
                        raise OSError("seeded launch failure")
                    self.fake_run(command, **kwargs)
                    Path(command[0]).write_bytes(b"changed during run")
                    return subprocess.CompletedProcess(command, 0)
                with mock.patch.object(bench.subprocess, "run", side_effect=changed):
                    self.assertEqual(bench.main(self.argv()), 1)
                report = json.loads((self.output / "report.json").read_text())
                self.assertFalse(report["valid"])
                self.assertNotIn("speedup", report["summary"])
                if drift:
                    self.assertNotEqual(report["runs"][0]["sha256"], report["runs"][0]["sha256_after"])
                else:
                    self.assertIsNone(report["runs"][0]["exit"])
                    self.assertIn("seeded launch failure", report["runs"][0]["launch_error"])

    def test_existing_output_is_not_overwritten(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("keep")
        with mock.patch.object(bench.subprocess, "run") as run:
            self.assertNotEqual(bench.main(self.argv()), 0)
        run.assert_not_called()
        self.assertEqual(marker.read_text(), "keep")

    def test_validation_bin_fallback_complete_mocked_cli(self):
        (self.data / "val.bin").unlink()
        validation = self.data / "validation.bin"
        validation.write_bytes(b"alternate validation fixture")
        with mock.patch.object(bench.subprocess, "run", side_effect=self.fake_run) as run:
            self.assertEqual(bench.main(self.argv()), 0)
        self.assertEqual(run.call_count, 8)
        report = json.loads((self.output / "report.json").read_text())
        self.assertTrue(report["valid"])
        self.assertEqual(report["dataset"]["validation"]["path"], str(validation.resolve()))
        self.assertEqual(report["dataset"]["validation"]["size_bytes"], validation.stat().st_size)
        self.assertEqual(report["dataset"]["train"]["path"], str((self.data / "train.bin").resolve()))
        self.assertEqual(report["dataset"]["train"]["size_bytes"], (self.data / "train.bin").stat().st_size)

    def test_val_bin_priority_when_both_names_exist(self):
        (self.data / "validation.bin").write_bytes(b"must not select this alternate fixture")
        with mock.patch.object(bench.subprocess, "run", side_effect=self.fake_run):
            self.assertEqual(bench.main(self.argv()), 0)
        report = json.loads((self.output / "report.json").read_text())
        self.assertEqual(report["dataset"]["validation"]["path"], str((self.data / "val.bin").resolve()))
        self.assertEqual(report["dataset"]["validation"]["size_bytes"], (self.data / "val.bin").stat().st_size)


if __name__ == "__main__":
    unittest.main(verbosity=2)
