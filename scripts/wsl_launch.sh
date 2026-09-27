#!/usr/bin/env bash
# The Linux half of run_linux.ps1. Runs inside WSL; the PowerShell half only frees
# the Windows port and calls this. Usable on its own from a WSL shell too:
#
#   bash scripts/wsl_launch.sh <mode> <windows-project-dir-as-wsl-path> [port]
#
#   serve      sync code in, check imports, start Streamlit (the default)
#   check      sync code in, check imports, stop
#   copy-back  copy finished results from WSL back to the Windows folder
#   run        run one script in the WSL copy:  run <win> -- scripts/x.py --args
#              (no sync: the Model registry page runs "check" first, "copy-back"
#              after, each as its own stage so its log shows which step failed)
#
# The WSL copy lives in $CREDITSURV_WSL_DIR (default ~/creditsurv). The Windows
# folder is the source of truth for code; WSL is the source of truth for results
# produced there. Nothing here changes any Windows setting.
set -euo pipefail

MODE="${1:-serve}"
WIN="${2:?usage: wsl_launch.sh <serve|check|copy-back> <windows project dir> [port]}"
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

stop_old_linux_server() {
    # A Streamlit started by an earlier launch still holds the port inside WSL. Stop
    # it if (and only if) it is Streamlit; anything else is named and left running.
    local pids pid cmd
    pids=$(ss -Hltnp "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u || true)
    for pid in $pids; do
        cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
        if [[ "$cmd" == *streamlit* ]]; then
            say "Stopping the earlier Linux Streamlit on port $PORT (pid $pid)"
            kill "$pid" 2>/dev/null || true
            for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
        else
            fail "port $PORT in WSL is held by pid $pid ($cmd), which is not Streamlit. Stop it or pass -Port."
        fi
    done
}

serve() {
    stop_old_linux_server
    local ip
    ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    say ""
    say "=============================================================================="
    say " creditsurv is starting in Linux (WSL) from $DEST"
    say ""
    say "   Open:  http://localhost:$PORT"
    [ -n "$ip" ] && say "   (if localhost does not answer: http://$ip:$PORT)"
    say ""
    say " The header should read 'Running on Linux (WSL)'. Ctrl+C stops the server."
    say " Code edits on Windows are synced in and the app restarts on them by itself"
    say " (within ~20 s, whenever no job is running and no run is open on a page)."
    say "=============================================================================="
    say ""
    cd "$DEST"
    exec "$PY" -m streamlit run app/app.py --server.headless true \
        --server.address 0.0.0.0 --server.port "$PORT"
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

case "$MODE" in
    serve)     sync_in; check_env; serve ;;
    check)     sync_in; check_env; say "Environment ready. (check mode: the app was not started)" ;;
    copy-back) copy_back ;;
    run)       shift 2
               [ "${1:-}" = "--" ] && shift
               [ $# -gt 0 ] || fail "run needs a script: run <win> -- scripts/x.py ..."
               [ -x "$PY" ] || fail "no Linux virtual environment at $DEST/.venv; run: run_linux.ps1 -CheckOnly"
               cd "$DEST"
               say "Running in $DEST: python $*"
               exec env PYTHONUNBUFFERED=1 PYTHONPATH="$DEST/src" "$PY" -u "$@" ;;
    *)         fail "unknown mode '$MODE' (serve, check, copy-back or run)" ;;
esac
