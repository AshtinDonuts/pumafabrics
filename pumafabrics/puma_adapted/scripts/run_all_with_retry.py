#!/usr/bin/env python3
"""
Managed runner for schedule_all_datasets.sh.
- Runs all 15 jobs (5 datasets × 3 boundary-loss weights) sequentially.
- On failure: captures stderr, attempts auto-debug, retries once.
- After 2 consecutive failures for the same job: sends alert email and aborts.
"""

import os
import re
import smtplib
import socket
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ALERT_EMAIL = "a1865590@adelaide.edu.au"
MX_HOSTS = ["au-smtp-inbound-1.mimecast.com", "au-smtp-inbound-2.mimecast.com"]
SMTP_PORT = 25
SENDER = f"puma-runner@{socket.gethostname()}"

CONDA_ENV = "pumafabrics"
PUMA_ADAPTED = Path(__file__).resolve().parent.parent  # .../puma_adapted
RESULTS_BASE = str(PUMA_ADAPTED) + "/"

DATASETS = [
    "2nd_order_R3S3_sweeping_16may",
    "2nd_order_R3S3_simple_w_shape",
    "2nd_order_R3S3_simple_line",
    "2nd_order_R3S3_simple_wipe",
    "2nd_order_R3S3_table_wipe_puma0",
]
BLW_VARIANTS = [("0.01", "blw0p01"), ("1.0", "blw1p0"), ("10.0", "blw10p0")]

LOG_FILE = PUMA_ADAPTED / "scripts" / "run_all_with_retry.log"
ALERT_LOG = PUMA_ADAPTED / "scripts" / "alert.log"

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg, *, also_print=True):
    line = f"[{ts()}] {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
    if also_print:
        print(line, flush=True)


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------
def send_email(subject: str, body: str) -> bool:
    """Attempt to send email via direct SMTP to MX.  Returns True on success."""
    msg = MIMEMultipart()
    msg["From"] = SENDER
    msg["To"] = ALERT_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    raw = msg.as_string()

    for mx in MX_HOSTS:
        try:
            log(f"  Trying SMTP to {mx}:{SMTP_PORT} …")
            with smtplib.SMTP(mx, SMTP_PORT, timeout=15) as smtp:
                smtp.ehlo()
                smtp.sendmail(SENDER, [ALERT_EMAIL], raw)
            log(f"  Email sent via {mx}")
            return True
        except Exception as exc:
            log(f"  SMTP via {mx} failed: {exc}")

    # Fallback: write to alert.log
    with open(ALERT_LOG, "a") as f:
        f.write(f"\n{'='*70}\n[{ts()}] UNSENT ALERT\nSubject: {subject}\n\n{body}\n{'='*70}\n")
    log(f"  Email delivery failed – alert written to {ALERT_LOG}", also_print=True)
    return False


def alert_and_abort(job_tag: str, attempts: list[str]):
    subject = f"[PUMA RUNNER] Job FAILED after 2 attempts – {job_tag}"
    body = textwrap.dedent(f"""\
        Automated PUMA training runner on {socket.gethostname()} aborted.

        Job:     {job_tag}
        Host:    {socket.gethostname()}
        Time:    {ts()}
        Log:     {LOG_FILE}

        ── Attempt 1 error ──────────────────────────────────────
        {attempts[0] if len(attempts) > 0 else '(none)'}

        ── Attempt 2 error ──────────────────────────────────────
        {attempts[1] if len(attempts) > 1 else '(none)'}
    """)
    log(f"ALERT: {subject}")
    send_email(subject, body)
    log("Aborting all remaining jobs.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Auto-debug helpers
# ---------------------------------------------------------------------------
def diagnose(stderr: str, stdout: str, params: str, blw: str) -> list[str]:
    """
    Return a list of human-readable diagnosis lines.
    Tries to identify common failure modes from the combined output.
    """
    combined = stderr + "\n" + stdout
    hints = []

    if "ModuleNotFoundError" in combined or "ImportError" in combined:
        m = re.search(r"(ModuleNotFoundError|ImportError): (.+)", combined)
        hints.append(f"Import error: {m.group(2).strip() if m else 'unknown module'}")
        hints.append("  → Check that the 'pumafabrics' conda env has all dependencies installed.")

    if "No such file or directory" in combined:
        m = re.search(r"No such file or directory: '([^']+)'", combined)
        hints.append(f"Missing file/dir: {m.group(1) if m else 'unknown'}")
        hints.append("  → Verify that the data file / param module path exists.")

    if "No module named 'params." in combined:
        m = re.search(r"No module named 'params\.([^']+)'", combined)
        missing = m.group(1) if m else params
        hints.append(f"Missing params module: params/{missing}.py")
        hints.append(f"  → Expected at: {PUMA_ADAPTED}/params/{missing}.py")

    if "CUDA" in combined or "cuda" in combined:
        if "out of memory" in combined.lower():
            hints.append("CUDA out-of-memory.")
            hints.append("  → Try reducing batch size in the params file.")
        elif "device" in combined.lower():
            hints.append("CUDA device error – GPU may be unavailable or occupied.")

    if "KeyError" in combined or "AttributeError" in combined:
        m = re.search(r"(KeyError|AttributeError): (.+)", combined)
        hints.append(f"Data/config error: {m.group(0).strip() if m else 'unknown'}")

    if "FileNotFoundError" in combined:
        m = re.search(r"FileNotFoundError: \[Errno \d+\] .+: '([^']+)'", combined)
        hints.append(f"FileNotFoundError: {m.group(1) if m else 'unknown'}")

    if not hints:
        # Generic: grab last non-empty stderr line
        last_lines = [l.strip() for l in combined.splitlines() if l.strip()]
        if last_lines:
            hints.append(f"Last output line: {last_lines[-1]}")

    return hints


# ---------------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------------
def run_job(params: str, blw: str, tag: str, attempt: int) -> tuple[bool, str]:
    """
    Run one training job.  Returns (success, error_summary).
    """
    cmd = [
        "conda", "run", "--no-capture-output", "-n", CONDA_ENV,
        "python", "train.py",
        "--params", params,
        "--results-base-directory", RESULTS_BASE,
        "--results-path", f"results/{tag}/",
        "--boundary-loss-weight", blw,
    ]
    log(f"  [attempt {attempt}] Running: python train.py --params {params} --boundary-loss-weight {blw}")
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PUMA_ADAPTED),
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - t0
        if result.returncode == 0:
            log(f"  SUCCESS in {elapsed:.1f}s")
            return True, ""
        else:
            log(f"  FAILED (exit {result.returncode}) after {elapsed:.1f}s")
            # Write full output to log
            with open(LOG_FILE, "a") as f:
                f.write(f"\n--- stdout ---\n{result.stdout[-4000:]}\n--- stderr ---\n{result.stderr[-4000:]}\n")
            hints = diagnose(result.stderr, result.stdout, params, blw)
            summary = "\n".join(hints) if hints else "(no diagnosis)"
            log(f"  Diagnosis:\n    " + "\n    ".join(hints))
            error_detail = (
                f"Exit code: {result.returncode}\n"
                f"Diagnosis:\n{summary}\n\n"
                f"--- last stderr ---\n{result.stderr[-2000:]}"
            )
            return False, error_detail
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        log(f"  EXCEPTION after {elapsed:.1f}s: {exc}")
        return False, str(exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log(f"{'='*70}")
    log(f"PUMA training runner started – {len(DATASETS)} datasets × {len(BLW_VARIANTS)} weights = {len(DATASETS)*len(BLW_VARIANTS)} jobs")
    log(f"Working directory : {PUMA_ADAPTED}")
    log(f"Alert email        : {ALERT_EMAIL}")
    log(f"Log file           : {LOG_FILE}")
    log(f"{'='*70}")

    total = len(DATASETS) * len(BLW_VARIANTS)
    done = 0
    failed_jobs = []

    for ds in DATASETS:
        for blw_val, blw_suffix in BLW_VARIANTS:
            tag = f"{ds}_{blw_suffix}"
            params_name = tag  # e.g. 2nd_order_R3S3_sweeping_16may_blw0p01
            done += 1
            log(f"\n[{done}/{total}] ===== {tag} =====")

            errors = []
            success = False

            for attempt in range(1, 3):  # max 2 attempts
                ok, err = run_job(params_name, blw_val, tag, attempt)
                if ok:
                    success = True
                    break
                errors.append(err)
                if attempt == 1:
                    log(f"  Retrying after failure …")

            if not success:
                failed_jobs.append(tag)
                log(f"  *** Job {tag} FAILED after 2 attempts – alerting and aborting ***")
                alert_and_abort(tag, errors)

    # All jobs completed
    log(f"\n{'='*70}")
    log(f"All {total} jobs finished successfully.")
    subject = f"[PUMA RUNNER] All {total} jobs completed on {socket.gethostname()}"
    body = f"All {total} PUMA training jobs finished successfully.\n\nHost: {socket.gethostname()}\nTime: {ts()}\nLog: {LOG_FILE}\n"
    send_email(subject, body)


if __name__ == "__main__":
    main()
