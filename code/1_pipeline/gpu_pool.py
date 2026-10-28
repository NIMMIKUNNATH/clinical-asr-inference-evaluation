"""
gpu_pool.py - keep the GPU full: whenever fewer than MAX model instances are running, start the highest-priority
job whose prerequisites are met. Replaces the fixed-stage schedulers, where a stage waited for its slowest job.

GPU jobs, in priority order when a slot frees:
  1. AfriSpeech-Dialog matched large-v3 checkpoint pair (v3_oldref, v3_trial32)
  2. PriMock57 matched large-v3 checkpoint pair
  3. search workers: local 20-recording searches (AfriSpeech, Fareez) first, since the grid waits on them,
     then joint workers up to 4, then second local workers
  4. joint top-5 re-evaluation on full development sets, once the joint search has finished
  5. grid runs (joint / local20_fareez / local20_afrispeech on each held-out set) as soon as that configuration
     has been selected
CPU steps (scoring, selection, statistics) start in the background as soon as their inputs are complete.

State is read from disk (outputs, study journals, choice files), so the pool can be restarted at any time;
every GPU job is resumable. Log: logs/gpu_pool.log. Ends when everything, including the statistics, is done.

USAGE (WSL, osce venv):  python gpu_pool.py
"""
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
ROOT = Path("/mnt/d/fareez_optuna")
CODE = ROOT / "code" / "final"
LOG = ROOT / "logs"
HOME = Path.home() / "fareez_optuna"
PY = str(Path.home() / "osce" / "venv" / "bin" / "python")
EV = str(Path.home() / "eval_venv" / "bin" / "python")
MAX = 4
GPU_PATTERN = "search_joint.py|run_primock.py|final_run.py|joint_reevaluate.py"

STUDIES = {  # name -> (journal, target completed+pruned)
    "joint": ("joint_three_corpus_large_v3", "joint.journal", 25),
    "local_afrispeech": ("afrispeech_local20_large_v3", "afrispeech_local20.journal", 15),
    "local_fareez": ("fareez_local20_large_v3", "fareez_local20.journal", 15),
}
CHOICE = {"joint": ROOT / "results_joint" / "joint_choice.json",
          "local20_fareez": ROOT / "results_fareez_local20" / "choice.json",
          "local20_afrispeech": ROOT / "results_afrispeech_local20" / "choice.json"}
N_OUT = {"afrispeech": 49, "primock": 57, "fareez": 231}
OUTDIR = {"afrispeech": ROOT / "final_afrispeech", "primock": ROOT / "final_primock", "fareez": ROOT / "final"}


def log(msg):
    with open(LOG / "gpu_pool.log", "a") as f:
        f.write(f"{datetime.now():%m-%d %H:%M:%S} {msg}\n")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def gpu_procs():
    out = sh(f"pgrep -af '{GPU_PATTERN}' | grep -v pgrep")
    # the launcher shell that starts a job also matches the pattern; count only the python processes themselves
    return [l for l in out.splitlines() if "python" in l and "gpu_pool" not in l and "bash -c" not in l]


TRACKED = []      # processes this pool started, counted from the moment of launch


def busy():   # noqa: D401
    """Model instances occupying a slot: this pool's own live jobs, plus any job started by an earlier pool.
    A job is counted from the instant it is launched, because a job counted only once it appears in the process
    list let the pool start a fifth instance while the fourth was still loading, and five instances wedged the
    GPU once before."""
    TRACKED[:] = [p for p in TRACKED if p.poll() is None]
    mine = {p.pid for p in TRACKED}
    others = [l for l in gpu_procs() if l.split()[0].isdigit() and int(l.split()[0]) not in mine]
    return len(TRACKED) + len(others)


def running(fragment):
    return sum(fragment in p for p in gpu_procs())


def n_outputs(corpus, cfg):
    d = OUTDIR[corpus] / cfg
    return len(list(d.glob("*.json"))) if d.exists() else 0     # includes <rec>.failed.json


def run_done(corpus, cfg):
    return n_outputs(corpus, cfg) >= N_OUT[corpus]


_HELDOUT = []


def shard_done(cfg, shard, nshards=2):
    """True when every Fareez held-out recording of this shard already has an output or a failure record."""
    if not _HELDOUT:
        sys.path.insert(0, str(CODE))
        import final_run
        _HELDOUT.extend(final_run.recordings("heldout"))
    d = OUTDIR["fareez"] / cfg
    return all((d / f"{r}.json").exists() or (d / f"{r}.failed.json").exists()
               for r in _HELDOUT[shard::nshards])


def study_state(key):
    name, journal, target = STUDIES[key]
    if not (HOME / journal).exists():
        return 0, target
    try:
        s = optuna.load_study(study_name=name, storage=optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(str(HOME / journal))))
        n = sum(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED) for t in s.trials)
        return n, target
    except Exception:          # study not created yet
        return 0, target


def study_done(key):
    n, target = study_state(key)
    return n >= target


def launch(cmd, logname):
    """Start the job directly in its own session: a wrapper shell would linger and be counted as a second job."""
    out = open(LOG / logname, "a")
    TRACKED.append(subprocess.Popen(shlex.split(cmd), cwd=str(CODE), stdout=out, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True))
    log(f"START {cmd}  -> {logname}")


def cpu_step(tag, cmd):
    """Run a CPU step once, in the background; a marker file records that it has been started."""
    marker = LOG / f".pool_{tag}"
    if marker.exists():
        return False
    marker.write_text(cmd)
    subprocess.Popen(["bash", "-c", f"cd {CODE} && ({cmd}) > {LOG / ('pool_' + tag + '.log')} 2>&1; "
                                    f"echo done >> {marker}"])
    log(f"CPU {tag}: {cmd}")
    return True


def cpu_finished(tag):
    m = LOG / f".pool_{tag}"
    return m.exists() and "done" in m.read_text()


def gpu_candidates():
    """Ready GPU jobs in priority order: (identifier, command, log name)."""
    c = []
    for corpus, cfgs in (("afrispeech", ("oldref_v2", "trial32_v2", "v3_oldref", "v3_trial32")),
                         ("primock", ("v3_oldref", "v3_trial32"))):
        for cfg in cfgs:
            # AfriSpeech large-v2 pair must finish first only in the sense that it already occupies slots
            if not run_done(corpus, cfg) and not running(f"--corpus {corpus} --config {cfg}"):
                c.append((f"{corpus}:{cfg}", f"{PY} run_primock.py --corpus {corpus} --config {cfg}",
                          f"{'afri' if corpus == 'afrispeech' else 'extra_primock'}_{cfg}.log"))
    # grid on the small corpora, as soon as a configuration has been selected: about 16 h of audio for all three
    # configurations, against 44 h per configuration on Fareez, so these results arrive first
    grid_small = []
    for cfg in ("local20_fareez", "local20_afrispeech"):   # joint == primock_tuned, already scored everywhere
        if not CHOICE[cfg].exists():
            continue
        for corpus in ("afrispeech", "primock"):
            frag = f"run_primock.py --corpus {corpus} --config {cfg}"
            if not run_done(corpus, cfg) and not running(frag):
                grid_small.append((f"grid:{cfg}:{corpus}", f"{PY} {frag}", f"grid_{cfg}_{corpus}.log"))
    # searches
    for key, scope in (("local_afrispeech", "afrispeech"), ("local_fareez", "fareez")):
        if not study_done(key) and running(f"--scope {scope}") < 1:
            c.append((f"{key}:w0", f"{PY} search_joint.py --scope {scope} --trials 15 --worker 0",
                      f"local20_{scope}.log"))
    if not study_done("joint"):
        for w in range(4):
            if not any(f"search_joint.py --trials 25 --worker {w}" in p for p in gpu_procs()):
                c.append((f"joint:w{w}", f"{PY} search_joint.py --trials 25 --worker {w}", f"joint_w{w}.log"))
    for key, scope in (("local_afrispeech", "afrispeech"), ("local_fareez", "fareez")):
        if not study_done(key) and running(f"--scope {scope}") < 2:
            c.append((f"{key}:w1", f"{PY} search_joint.py --scope {scope} --trials 15 --worker 1",
                      f"local20_{scope}_w1.log"))
    # joint re-evaluation
    if study_done("joint") and running("search_joint.py --trials 25") == 0 and not CHOICE["joint"].exists():
        top5 = LOG / ".pool_top5"
        if top5.exists() and "done" in top5.read_text():
            for t in top5.read_text().split("\n")[0].split():
                if not (ROOT / "results_joint" / "reeval" / f"trial_{int(t):03d}.json").exists() \
                        and not running(f"joint_reevaluate.py --trial {t}"):
                    c.append((f"reeval:{t}", f"{PY} joint_reevaluate.py --trial {t}", f"reeval_{t}.log"))
    # the small-corpus grid runs after the re-evaluation above: selecting the joint configuration unblocks its
    # own grid runs, which are the largest remaining work
    c += grid_small

    # Fareez grid last: 44 h of audio per configuration
    grid_fareez = []
    for cfg in ("local20_fareez", "local20_afrispeech"):   # joint == primock_tuned, already scored everywhere
        if not CHOICE[cfg].exists():
            continue
        for s in (0, 1):
            frag = f"final_run.py --config {cfg} --set heldout --require-cache --shard {s} --nshards 2"
            # per shard, not per configuration: a finished shard exits at once, and counting only the
            # configuration's total made the pool relaunch it continuously while the other shard ran
            if not shard_done(cfg, s) and not running(frag):
                grid_fareez.append((f"grid:{cfg}:fareez{s}", f"{PY} {frag}", f"grid_{cfg}_fareez{s}.log"))
    return c + grid_fareez


def cpu_steps():
    """Start every CPU step whose inputs are complete. Returns True when the final statistics have finished."""
    afri_all = ["defaults", "tuned", "primock_tuned", "oldref_v2", "trial32_v2", "v3_oldref", "v3_trial32"]
    if all(run_done("afrispeech", c) for c in afri_all) and not running("--corpus afrispeech"):
        cpu_step("score_afrispeech", "; ".join(f"{EV} score_primock.py --corpus afrispeech --config {c}"
                                                for c in afri_all) + f"; {EV} afrispeech_summary.py")
    if all(run_done("primock", c) for c in ("v3_oldref", "v3_trial32")):
        cpu_step("score_primock_v3", f"{EV} score_primock.py --corpus primock --config v3_oldref; "
                                     f"{EV} score_primock.py --corpus primock --config v3_trial32")
    if study_done("local_afrispeech") and study_done("local_fareez") and running("--scope ") == 0:
        cpu_step("select_locals", f"{PY} select_configs.py --locals")
    if study_done("joint") and running("search_joint.py --trials 25") == 0:
        cpu_step("top5", f"{PY} select_configs.py --joint-top5 2>/dev/null | tail -1 > {LOG / '.pool_top5'}.tmp; "
                         f"mv {LOG / '.pool_top5'}.tmp {LOG / '.pool_top5'}")
        top5 = LOG / ".pool_top5"
        if cpu_finished("top5") and top5.read_text().split("\n")[0].strip():
            ts = top5.read_text().split("\n")[0].split()
            if all((ROOT / "results_joint" / "reeval" / f"trial_{int(t):03d}.json").exists() for t in ts):
                cpu_step("select_joint", f"{PY} select_configs.py --joint-select")
    grid_done = all(CHOICE[c].exists() and n_outputs("fareez", c) >= N_OUT["fareez"]
                    and run_done("primock", c) and run_done("afrispeech", c) for c in CHOICE)
    if grid_done and not gpu_procs():
        cpu_step("score_grid", "; ".join(
            f"{EV} metrics.py --config {c}; {EV} score_primock.py --corpus primock --config {c}; "
            f"{EV} score_primock.py --corpus afrispeech --config {c}" for c in CHOICE))
    if cpu_finished("score_grid") and cpu_finished("score_afrispeech") and cpu_finished("score_primock_v3"):
        cpu_step("stats", f"{EV} grid_analysis.py")
    return cpu_finished("stats")


def main():
    os.environ.setdefault("HF_TOKEN", sh("grep -m1 'HF_TOKEN=' ~/.bashrc | cut -d= -f2- | tr -d '\"\\''"))
    log(f"pool started, max {MAX} GPU jobs")
    last = None
    while True:
        if cpu_steps():
            log("ALL DONE - statistics written to results_grid/")
            break
        started = []
        if busy() < MAX:
            for ident, cmd, logname in gpu_candidates():
                # recount before every launch, including launcher shells whose python is still starting:
                # five model instances wedged this GPU once before, so the cap must never be exceeded
                if busy() >= MAX:
                    break
                launch(cmd, logname)
                started.append(ident)
                time.sleep(45)          # let the model load before counting again
        state = (busy(), tuple(started))
        if state != last:
            log(f"running {state[0]} GPU jobs" + (f"; started {', '.join(started)}" if started else ""))
            last = state
        time.sleep(30)


if __name__ == "__main__":
    main()
