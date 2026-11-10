"""End-to-end regression: all tests + master sim + Monte Carlo in one command."""

import subprocess
import sys
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def banner(title, char="="):
    print()
    print(char * 70)
    print(f"  {title}")
    print(char * 70)


def run_step(name, cmd, timeout=300):
    """Run a subprocess, return (success, duration, output)."""
    print(f"\n>>> {name}")
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        duration = time.time() - start
        success = result.returncode == 0
        output = result.stdout + result.stderr
        status = "PASS" if success else "FAIL"
        print(f"    [{status}] {duration:.1f}s")
        return success, duration, output
    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT after {timeout}s]")
        return False, timeout, "TIMEOUT"
    except Exception as e:
        print(f"    [ERROR] {e}")
        return False, 0, str(e)


def main():
    banner("END-TO-END REGRESSION SUITE")
    print(f"  Working directory: {ROOT}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = []

    # ----- STEP 1: Unit tests -----
    success, dur, out = run_step(
        "Unit test suite (pytest)",
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
        timeout=120,
    )
    results.append(("Unit Tests", success, dur))
    if not success:
        print("    Unit test failures detected. Aborting regression.")
        print(out[-2000:])
        finalize(results)
        return

    # ----- STEP 2: CasADi-generated C parity -----
    success, dur, out = run_step(
        "C-Python parity verification",
        [sys.executable, "scripts/verify_c_parity_nogcc.py"],
        timeout=60,
    )
    results.append(("C Parity", success, dur))

    # ----- STEP 3: Master simulation -----
    success, dur, out = run_step(
        "Master simulation (PID + gusts + corridor)",
        [sys.executable, "scripts/finalize.py"],
        timeout=180,
    )
    results.append(("Master Sim", success, dur))

    # ----- STEP 4: Quick Monte Carlo (reduced runs for speed) -----
    # NOTE: Full Monte Carlo (200 runs) takes 4+ minutes. We do 20 here.
    # To run full: python scripts/monte_carlo_gusts.py
    print("\n>>> Monte Carlo (quick: 20 runs via statistical_comparison)")
    # Create a lightweight 20-run version by patching the script in-place
    try:
        with open("scripts/statistical_comparison.py", "r", encoding="utf-8") as f:
            src = f.read()
        patched = src.replace("n_runs = 200", "n_runs = 20")
        with open("scripts/_quick_mc.py", "w", encoding="utf-8") as f:
            f.write(patched)
        success, dur, out = run_step(
            "Monte Carlo (quick 20 runs)",
            [sys.executable, "scripts/_quick_mc.py"],
            timeout=180,
        )
        results.append(("Monte Carlo (quick)", success, dur))
        os.remove("scripts/_quick_mc.py")  # clean up
    except Exception as e:
        print(f"    [SKIPPED] {e}")
        results.append(("Monte Carlo (quick)", False, 0))

    # ----- Final verdict -----
    finalize(results)


def finalize(results):
    banner("REGRESSION SUMMARY")
    print(f"{'Test':<30} {'Status':<10} {'Time':>8}")
    print("-" * 50)
    total_time = 0
    all_pass = True
    for name, success, dur in results:
        status = "PASS" if success else "FAIL"
        if not success:
            all_pass = False
        print(f"{name:<30} {status:<10} {dur:>7.1f}s")
        total_time += dur

    print("-" * 50)
    print(f"{'TOTAL':<30} {'':<10} {total_time:>7.1f}s")
    print()

    if all_pass:
        print("=" * 70)
        print("  VERDICT: ALL REGRESSION TESTS PASS")
        print("  The project is stable. Ready for final documentation.")
        print("=" * 70)
        sys.exit(0)
    else:
        print("=" * 70)
        print("  VERDICT: REGRESSION FAILURES DETECTED")
        print("  Fix the failing steps above before proceeding.")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
