"""
Autonomous DevOps runner for schedule_all_datasets.sh
- Runs all 15 jobs (skips already-complete ones)
- On failure: diagnoses, attempts up to 2 auto-fixes per job
- Logs everything to auto_fix.log
- Emails a failure report to robertwongkh@gmail.com if a job fails after 2 fix attempts
"""

import subprocess
import os
import sys
import time
import traceback
import smtplib
import socket
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

# ── Config ────────────────────────────────────────────────────────────────────
PUMA_ADAPTED  = Path(__file__).parent.resolve()
PYTHON        = "/home/khw/miniconda3/envs/pumafabrics/bin/python"
RESULTS_BASE  = str(PUMA_ADAPTED) + "/"
LOG_FILE      = PUMA_ADAPTED / "auto_fix.log"
EMAIL_TO      = "robertwongkh@gmail.com"
MAX_ATTEMPTS  = 2

DATASETS = [
    "2nd_order_R3S3_sweeping_16may",
    "2nd_order_R3S3_simple_w_shape",
    "2nd_order_R3S3_simple_line",
    "2nd_order_R3S3_simple_wipe",
    "2nd_order_R3S3_table_wipe_puma0",
]
BLW_VALUES = [("blw0p01", 0.01), ("blw1p0", 1.0), ("blw10p0", 10.0)]

# ── Logging ───────────────────────────────────────────────────────────────────
def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str):
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ── Job completion check ───────────────────────────────────────────────────────
def job_complete(tag: str) -> bool:
    """A job is considered complete if its results dir contains a trained model."""
    result_dir = PUMA_ADAPTED / "results" / tag
    for subdir in result_dir.glob("*/"):
        if (subdir / "model").exists():
            return True
    return False

# ── Diagnose & fix helpers ────────────────────────────────────────────────────
def diagnose(stderr: str) -> str:
    """Return a human-readable diagnosis from stderr/stdout."""
    lines = stderr.strip().splitlines()
    for line in reversed(lines):
        stripped = line.strip()
        if stripped:
            return stripped
    return "Unknown error (no output)"

def attempt_fix(attempt: int, stderr: str, params: str) -> Optional[str]:
    """
    Try to auto-fix common errors.
    Returns a description of the fix applied, or None if no fix known.
    """
    text = stderr.lower()

    # Fix 1: missing __pycache__ / stale .pyc
    if "importerror" in text or "modulenotfounderror" in text:
        log(f"  [fix] Clearing __pycache__ for stale imports")
        for p in PUMA_ADAPTED.rglob("__pycache__"):
            subprocess.run(["rm", "-rf", str(p)], check=False)
        return "Cleared __pycache__ to resolve stale import"

    # Fix 2: params file missing
    params_file = PUMA_ADAPTED / "params" / f"{params}.py"
    if "no module named" in text and not params_file.exists():
        log(f"  [fix] Params file not found: {params_file}")
        return None  # can't auto-fix a missing dataset config

    # Fix 3: CUDA out of memory — wait and retry
    if "out of memory" in text or "cuda" in text:
        wait = 30 * attempt
        log(f"  [fix] CUDA OOM detected, waiting {wait}s before retry")
        time.sleep(wait)
        return f"Waited {wait}s after CUDA OOM"

    # Fix 4: generic transient error — just retry
    if attempt == 1:
        log(f"  [fix] Transient failure, retrying immediately")
        return "Retry after transient failure"

    return None

# ── Email report ──────────────────────────────────────────────────────────────
def send_failure_email(failed_jobs: List[Dict]):
    log(f"Attempting to send failure report to {EMAIL_TO}")
    hostname = socket.gethostname()
    subject = f"[auto_fix] {len(failed_jobs)} job(s) failed on {hostname}"

    body_lines = [
        f"Host:    {hostname}",
        f"Time:    {ts()}",
        f"Log:     {LOG_FILE}",
        "",
        f"{len(failed_jobs)} job(s) failed after {MAX_ATTEMPTS} fix attempt(s) each:",
        "",
    ]
    for j in failed_jobs:
        body_lines += [
            f"  Tag:   {j['tag']}",
            f"  Cause: {j['diagnosis']}",
            f"  Last error snippet:",
        ]
        for line in j["stderr"].splitlines()[-20:]:
            body_lines.append(f"    {line}")
        body_lines.append("")

    body = "\n".join(body_lines)

    msg = MIMEMultipart()
    msg["From"]    = f"auto-fix@{hostname}"
    msg["To"]      = EMAIL_TO
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("localhost", timeout=10) as smtp:
            smtp.sendmail(msg["From"], [EMAIL_TO], msg.as_string())
        log(f"Email sent successfully to {EMAIL_TO}")
    except Exception as e:
        log(f"Could not send email (no local MTA?): {e}")
        report_path = PUMA_ADAPTED / "failure_report.txt"
        report_path.write_text(body)
        log(f"Failure report saved to {report_path} instead")

# ── Run one job ───────────────────────────────────────────────────────────────
def run_job(params: str, blw: float, tag: str):
    """Run a single train.py job. Returns (success, stderr)."""
    cmd = [
        PYTHON, str(PUMA_ADAPTED / "train.py"),
        "--params",                 params,
        "--results-base-directory", RESULTS_BASE,
        "--results-path",           f"results/{tag}/",
        "--boundary-loss-weight",   str(blw),
    ]
    log(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(PUMA_ADAPTED),
        capture_output=False,       # let stdout/stderr stream live to terminal
        text=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    combined = (result.stdout or "") + (result.stderr or "")
    # Also stream to log file
    with open(LOG_FILE, "a") as f:
        f.write(combined)
    return result.returncode == 0, combined

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log("=" * 70)
    log("Auto-fix runner starting")
    log(f"PUMA_ADAPTED = {PUMA_ADAPTED}")
    log(f"PYTHON       = {PYTHON}")
    log("=" * 70)

    jobs = [
        (f"{ds}_{blw_tag}", float(blw_val), f"{ds}_{blw_tag}")
        for ds in DATASETS
        for blw_tag, blw_val in BLW_VALUES
    ]

    failed_jobs   = []
    skipped       = 0
    succeeded     = 0

    for params, blw, tag in jobs:
        log(f"\n{'─'*60}")
        log(f"JOB: {tag}  (blw={blw})")

        if job_complete(tag):
            log(f"  SKIP — results already exist")
            skipped += 1
            continue

        success   = False
        last_err  = ""
        diagnosis = ""

        for attempt in range(1, MAX_ATTEMPTS + 1):
            log(f"  Attempt {attempt}/{MAX_ATTEMPTS}")
            success, last_err = run_job(params, blw, tag)

            if success:
                log(f"  SUCCESS on attempt {attempt}")
                succeeded += 1
                break

            diagnosis = diagnose(last_err)
            log(f"  FAILED — {diagnosis}")

            if attempt < MAX_ATTEMPTS:
                fix = attempt_fix(attempt, last_err, params)
                if fix:
                    log(f"  Applied fix: {fix}")
                else:
                    log(f"  No auto-fix available; retrying anyway")

        if not success:
            log(f"  GAVE UP after {MAX_ATTEMPTS} attempts: {tag}")
            failed_jobs.append({"tag": tag, "diagnosis": diagnosis, "stderr": last_err})

    log(f"\n{'='*70}")
    log(f"DONE — succeeded={succeeded}, skipped={skipped}, failed={len(failed_jobs)}")

    if failed_jobs:
        log(f"Failed jobs: {[j['tag'] for j in failed_jobs]}")
        send_failure_email(failed_jobs)
        sys.exit(1)
    else:
        log("All jobs completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(f"FATAL: {traceback.format_exc()}")
        sys.exit(2)
