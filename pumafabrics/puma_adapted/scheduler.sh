#!/bin/bash
# Runs schedule_train*.sh in order. Emails xyz@gmail.com on FAIL (any step nonzero exit) or END (all succeeded).

ALERT_EMAIL="a1865590@adelaide.edu.au"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}" || exit 1

send_alert() {
    local status="$1"
    local body="$2"
    local subject="[puma scheduler] ${status}"
    if command -v mail >/dev/null 2>&1; then
        printf '%s\n' "${body}" | mail -s "${subject}" "${ALERT_EMAIL}"
    elif command -v mailx >/dev/null 2>&1; then
        printf '%s\n' "${body}" | mailx -s "${subject}" "${ALERT_EMAIL}"
    else
        echo "WARNING: neither mail nor mailx found; alert not sent." >&2
        echo "--- ${subject} ---" >&2
        printf '%s\n' "${body}" >&2
    fi
}

SCHEDULES=(
    "${SCRIPT_DIR}/schedule_train.sh"
    "${SCRIPT_DIR}/schedule_train2.sh"
    "${SCRIPT_DIR}/schedule_train3.sh"
)

started_at="$(date -Iseconds)"
host="$(hostname)"

for sched in "${SCHEDULES[@]}"; do
    name="$(basename "${sched}")"
    echo "==== Running ${name} ===="
    if ! bash "${sched}"; then
        send_alert "FAIL" \
            "Schedule step from PUMAFABRICS failed.

Host: ${host}
Repo: ${REPO_ROOT}
Started: ${started_at}
Failed script: ${sched}
Status: nonzero exit from ${name}"
        exit 1
    fi
done

script_list=""
for s in "${SCHEDULES[@]}"; do
    script_list="${script_list}- $(basename "${s}")"$'\n'
done

send_alert "END" \
    "All schedule steps from PUMAFABRICS finished successfully.

Host: ${host}
Repo: ${REPO_ROOT}
Started: ${started_at}
Finished: $(date -Iseconds)
Scripts (in order):
${script_list}"
