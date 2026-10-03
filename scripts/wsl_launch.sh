#!/usr/bin/env bash
# The Linux half of run_linux.ps1. Runs inside WSL; the PowerShell half only frees
# the Windows port and calls this. Usable on its own from a WSL shell too:
#
#   bash scripts/wsl_launch.sh <mode> <windows-project-dir-as-wsl-path> [port]
#
#   serve      sync code in, check imports, start the API (port 8000, or
#              $CREDITSURV_API_PORT) and the dashboard as its client (the default)
#   api        sync code in, check imports, start the API only, hold until Ctrl+C
#   api-bg     the same, but leave the API running and return: it keeps serving
#              after this shell, this window and the terminal are gone
#   check      sync code in, check imports, stop
#   copy-back  copy finished results from WSL back to the Windows folder
#   run        run one script in the WSL copy:  run <win> -- scripts/x.py --args
#              (no sync: the Model registry page runs "check" first, "copy-back"
#              after, each as its own stage so its log shows which step failed)
#   run-bg     the same, detached, returning once it is running and logging:
#              run-bg <win> -- scripts/03d_explainer_validation.py --model-tag x
#
# The WSL copy lives in $CREDITSURV_WSL_DIR (default ~/creditsurv). The Windows
# folder is the source of truth for code; WSL is the source of truth for results
# produced there. Nothing here changes any Windows setting.
set -euo pipefail

MODE="${1:-serve}"
WIN="${2:?usage: wsl_launch.sh <serve|api|api-bg|check|copy-back|run|run-bg> <windows project dir> [port]}"
PORT="${3:-8501}"
DEST="${CREDITSURV_WSL_DIR:-$HOME/creditsurv}"
PY="$DEST/.venv/bin/python"

say()  { printf '%s\n' "$*"; }
fail() { printf 'FAILED: %s\n' "$*" >&2; exit 1; }

[ -d "$WIN/src/creditsurv" ] || fail "no creditsurv project at $WIN (expected $WIN/src/creditsurv)."
command -v rsync >/dev/null || fail "rsync is not installed in WSL. Fix: sudo apt install rsync"

# -r recursive, -t keep modification times (what makes "only when newer" work).
# Not -a: owners and permissions do not map cleanly between NTFS and ext4.
# --modify-window: NTFS and ext4 round times differently; a sub-2s difference is
# the same file.
# Prints each file it copies; directories are left out of the list.
rs() { rsync -rt --modify-window=2 --out-format='%n' "$@" | sed -n '/[^/]$/s/^/  copied /p'; }

sync_in() {
    say "Syncing code  $WIN  ->  $DEST"
    mkdir -p "$DEST"
    # Marks this folder as a synced copy. config/ here is replaced from Windows on
    # every sync, so the model registry refuses to record an approval in it.
    printf '%s\n' "$WIN" > "$DEST/.synced_from_windows"
    # Code: Windows is authoritative, so a file deleted there is deleted here too.
    # Caches and the editable-install metadata are left alone on both sides.
    for d in src app scripts config tests .streamlit; do
        [ -d "$WIN/$d" ] || continue
        rs --delete --exclude '__pycache__/' --exclude '.pytest_cache/' \
            --exclude '*.egg-info/' "$WIN/$d/" "$DEST/$d/"
    done
    for f in README.md pyproject.toml requirements-linux.txt; do
        [ -f "$WIN/$f" ] && rs "$WIN/$f" "$DEST/$f"
    done
    # FINDINGS.md is written by 05_report.py, so a Full run in WSL makes the WSL copy
    # the newer one. Never overwrite that with an older Windows copy.
    if [ -f "$DEST/FINDINGS.md" ] && [ "$DEST/FINDINGS.md" -nt "$WIN/FINDINGS.md" ] \
            && ! cmp -s "$DEST/FINDINGS.md" "$WIN/FINDINGS.md"; then
        say "  kept FINDINGS.md: the WSL copy is newer than the Windows one."
        say "  Bring it to Windows with: run_linux.ps1 -CopyBack"
    else
        rs "$WIN/FINDINGS.md" "$DEST/FINDINGS.md"
    fi

    # Models and data are large (hundreds of MB, and ~4 GB): copied only when the
    # Windows file is newer, and never deleted here.
    for d in outputs/models outputs/data; do
        [ -d "$WIN/$d" ] || continue
        say "Syncing $d (newer files only)"
        mkdir -p "$DEST/$d"
        rs --update "$WIN/$d/" "$DEST/$d/"
    done
}

check_env() {
    if [ ! -x "$PY" ]; then
        say "MISSING  the Linux virtual environment ($DEST/.venv)."
        say "  fix:   sudo apt install python3-venv libgomp1"
        say "         cd $DEST && python3 -m venv .venv && .venv/bin/python -m pip install -r requirements-linux.txt"
        exit 1
    fi
    say "Checking imports in $PY"
    # creditsurv.environment uses only the standard library, so it can report on a
    # venv that is missing everything else.
    if ! (cd "$DEST" && PYTHONPATH="$DEST/src" "$PY" -m creditsurv.environment); then
        say ""
        say "Run the fixes above in WSL (wsl -e bash -lc 'cd $DEST && ...'), then run this again."
        exit 1
    fi
}

API_PORT="${CREDITSURV_API_PORT:-8000}"

stop_old_server() {
    # A server started by an earlier launch still holds its port inside WSL. Stop it
    # if (and only if) it is ours -- Streamlit, or the creditsurv API under uvicorn;
    # anything else is named and left running.
    local port="$1" want="$2" pids pid cmd
    pids=$(ss -Hltnp "sport = :$port" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u || true)
    for pid in $pids; do
        cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
        if [[ "$cmd" == *"$want"* ]]; then
            say "Stopping the earlier $want on port $port (pid $pid)"
            kill "$pid" 2>/dev/null || true
            for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
        else
            fail "port $port in WSL is held by pid $pid ($cmd), which is not $want. Stop it or pick another port."
        fi
    done
}

# ------------------------------------------------------------- detaching ----
# Every job that has to outlive the shell that starts it is started through here,
# and nowhere else with a bare "&".
#
# Why a handshake and not just setsid: "wsl.exe -e bash ..." makes this script the
# session leader of a fresh pts, and a non-interactive shell has job control off,
# so "cmd &" leaves the job in *this* shell's process group -- which is that
# terminal's foreground group. The instant this script exits, the kernel hangs that
# group up and the job goes with it (measured: a plain "&" job stopped at the exact
# second the launcher returned, and its own trap reported SIGHUP).
#
# setsid moves the job out of that group -- but only once it has actually run, and
# "setsid cmd &" lets this shell exit in the same instant it forks. The hangup then
# overtakes the setsid() call and the job dies before its first line of output:
# nothing in ps, an empty log, immediately. That is the whole bug. "setsid --fork",
# "disown" and "nohup" do not change that ordering (measured: all three still lost
# the job when the launcher exited at once); a sleep only makes the race likelier
# to win -- 50 ms was enough here and 0 ms never was -- which is not winning it.
#
# So the job says when it is safe: it writes its pid from inside its new session,
# after setsid() has taken effect, and this shell does not go on until it has read
# it. No timing assumption and nothing to tune. Prints the pid.
detach() {
    local marker="$1" log="$2"; shift 2
    local pid="" sid i
    rm -f "$marker" "$marker.new"
    mkdir -p "$(dirname "$marker")" "$(dirname "$log")"
    # The pid is written and then moved into place, so this shell never reads half
    # of it. stdin, stdout and stderr are all repointed: a job still holding this
    # shell's stdout would keep the caller -- and wsl.exe -- from ever returning.
    setsid bash -c 'printf "%s\n" "$$" > "$1.new" && mv "$1.new" "$1"
                    shift
                    exec "$@"' detach-child "$marker" "$@" \
        </dev/null >> "$log" 2>&1 &
    for i in $(seq 1 300); do             # a fork and an exec; 30 s is far over
        [ -s "$marker" ] && { pid=$(tr -d '[:space:]' < "$marker"); break; }
        sleep 0.1
    done
    if [ -z "$pid" ]; then
        printf 'the job never reported a pid; last lines of %s:\n' "$log" >&2
        tail -20 "$log" >&2 2>/dev/null || true
        return 1
    fi
    # The marker was written from inside the new session, so the detach has already
    # happened; this confirms it while the job is still there. No session at all
    # means a job that has finished, which the marker says detached first.
    sid=$(ps -o sid= -p "$pid" 2>/dev/null | tr -d '[:space:]')
    if [ -n "$sid" ] && [ "$sid" != "$pid" ]; then
        printf 'pid %s is in session %s, not one of its own\n' "$pid" "$sid" >&2
        return 1
    fi
    printf '%s\n' "$pid"
}

start_api() {
    # The API that does the work: scoring, Phase 2, the registry, the history. The
    # dashboard is its client. Logged to outputs/logs/api.log. Detached, so the one
    # thing that stops it is the caller's trap (serve, api) or a kill (api-bg).
    stop_old_server "$API_PORT" "creditsurv.api.app"
    mkdir -p "$DEST/outputs/logs"
    cd "$DEST"
    API_PID=$(detach "$DEST/outputs/logs/api.pid" "$DEST/outputs/logs/api.log" \
        env PYTHONPATH="$DEST/src" "$PY" -m uvicorn creditsurv.api.app:app \
        --host 127.0.0.1 --port "$API_PORT") \
        || fail "the API did not start at all (see outputs/logs/api.log)"
    for _ in $(seq 1 60); do
        if "$PY" -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:$API_PORT/meta', timeout=2)" 2>/dev/null; then
            say "API ready on http://127.0.0.1:$API_PORT (pid $API_PID, log outputs/logs/api.log)"
            return 0
        fi
        kill -0 "$API_PID" 2>/dev/null || { tail -20 "$DEST/outputs/logs/api.log" >&2; fail "the API exited while starting (log above)"; }
        sleep 1
    done
    fail "the API did not answer on port $API_PORT within 60 s (see outputs/logs/api.log)"
}

serve() {
    stop_old_server "$PORT" "streamlit"
    start_api
    trap 'kill "$API_PID" 2>/dev/null || true' EXIT INT TERM
    local ip
    ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    say ""
    say "=============================================================================="
    say " creditsurv is starting in Linux (WSL) from $DEST"
    say ""
    say "   Dashboard:  http://localhost:$PORT"
    [ -n "$ip" ] && say "   (if localhost does not answer: http://$ip:$PORT)"
    say "   API:        http://localhost:$API_PORT/docs   (every endpoint, interactive)"
    say ""
    say " The header should read 'API on Linux (WSL)'. Ctrl+C stops both."
    say " Code edits on Windows are synced in and both restart on them by themselves"
    say " (within ~20 s, whenever no job is running and no run is open on a page)."
    say "=============================================================================="
    say ""
    cd "$DEST"
    CREDITSURV_API_URL="http://127.0.0.1:$API_PORT" "$PY" -m streamlit run app/app.py \
        --server.headless true --server.address 0.0.0.0 --server.port "$PORT"
}

api_only() {
    start_api
    say "API only: http://localhost:$API_PORT/docs  -- Ctrl+C stops it."
    trap 'kill "$API_PID" 2>/dev/null || true' EXIT INT TERM
    # Not "wait": the API has a session of its own now and is no longer a child of
    # this shell, so there is nothing here to wait on.
    while kill -0 "$API_PID" 2>/dev/null; do sleep 1; done
}

api_background() {
    start_api
    say ""
    say "The API runs in a session of its own: Ctrl+C here, this shell exiting and"
    say "this window closing all leave it serving."
    say "   Stop it:  kill $API_PID        Log:  outputs/logs/api.log"
    say "   Its pid is in outputs/logs/api.pid; starting it again replaces it."
}

copy_back() {
    say "Copying finished results  $DEST/outputs  ->  $WIN/outputs  (newer files only)"
    local n=0 skipped=0 d name
    # Score Applicants uploads: provenance.json is written last, so its presence
    # means the run finished. A run still in progress is left for next time.
    if [ -d "$DEST/outputs/runs" ]; then
        for d in "$DEST"/outputs/runs/*/; do
            [ -d "$d" ] || continue
            name=$(basename "$d")
            if [ -f "$d/provenance.json" ] && [ -f "$d/scored_applicants.csv" ]; then
                mkdir -p "$WIN/outputs/runs/$name"
                rs --update "$d" "$WIN/outputs/runs/$name/"
                n=$((n + 1))
            else
                say "  skipped outputs/runs/$name: not finished"
                skipped=$((skipped + 1))
            fi
        done
    fi
    # Pipeline runs started from the Run pipeline page: finished once the runner
    # has stamped a finish time (completed, failed or stopped alike).
    if [ -d "$DEST/outputs/logs/runs" ]; then
        for d in "$DEST"/outputs/logs/runs/*/; do
            [ -f "$d/status.json" ] || continue
            name=$(basename "$d")
            if grep -Eq '"finished": *null' "$d/status.json"; then
                say "  skipped outputs/logs/runs/$name: still running (or interrupted)"
                skipped=$((skipped + 1))
            else
                mkdir -p "$WIN/outputs/logs/runs/$name"
                rs --update "$d" "$WIN/outputs/logs/runs/$name/"
                n=$((n + 1))
            fi
        done
    fi
    # Stage outputs the pipeline writes. --update keeps anything newer on Windows.
    # outputs/data is not copied back: it is ~4 GB and rebuildable from the CSVs.
    for d in outputs/tables outputs/figures outputs/eda outputs/models; do
        [ -d "$DEST/$d" ] || continue
        mkdir -p "$WIN/$d"
        rs --update "$DEST/$d/" "$WIN/$d/"
    done
    if [ -d "$DEST/outputs/logs" ]; then
        rs --update --exclude 'runs/' --exclude '*.lock' \
            "$DEST/outputs/logs/" "$WIN/outputs/logs/"
    fi
    for f in outputs/provenance_baseline.json FINDINGS.md; do
        [ -f "$DEST/$f" ] && rs --update "$DEST/$f" "$WIN/$f"
    done
    say "Done: $n finished run folder(s) up to date on Windows, $skipped skipped. Any copied file is listed above."
}

# run and run-bg take the same arguments; only run-bg detaches.
check_run_args() {
    [ $# -gt 0 ] || fail "$MODE needs a script: $MODE <win> -- scripts/x.py ..."
    [ -x "$PY" ] || fail "no Linux virtual environment at $DEST/.venv; run: run_linux.ps1 -CheckOnly"
}

case "$MODE" in
    serve)     sync_in; check_env; serve ;;
    api)       sync_in; check_env; api_only ;;
    api-bg)    sync_in; check_env; api_background ;;
    check)     sync_in; check_env; say "Environment ready. (check mode: the app was not started)" ;;
    copy-back) copy_back ;;
    run)       shift 2
               [ "${1:-}" = "--" ] && shift
               check_run_args "$@"
               cd "$DEST"
               say "Running in $DEST: python $*"
               exec env PYTHONUNBUFFERED=1 PYTHONPATH="$DEST/src" "$PY" -u "$@" ;;
    run-bg)    shift 2
               [ "${1:-}" = "--" ] && shift
               check_run_args "$@"
               cd "$DEST"
               # One log and one pid file per script, named after it, so a job
               # started this way is findable afterwards without this shell.
               JOB=$(basename "${1%.py}")
               JOB_LOG="$DEST/outputs/logs/$JOB.log"
               JOB_PID=$(detach "$DEST/outputs/logs/$JOB.pid" "$JOB_LOG" \
                   env PYTHONUNBUFFERED=1 PYTHONPATH="$DEST/src" "$PY" -u "$@") \
                   || fail "$JOB did not start at all (see outputs/logs/$JOB.log)"
               say "Started in a session of its own: python $*"
               say "   pid $JOB_PID   log outputs/logs/$JOB.log"
               say "   It survives this shell and this window. Stop it: kill $JOB_PID" ;;
    *)         fail "unknown mode '$MODE' (serve, api, api-bg, check, copy-back, run or run-bg)" ;;
esac
