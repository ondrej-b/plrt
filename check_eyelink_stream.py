#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_eyelink_stream.py -- is the EyeLink LSL stream carrying live data?
=======================================================================
taVNS project, October 2026

Run it right before plrt.py, after calibration and with the tracker
recording on the Host PC.  It subscribes to the stream that the Ubuntu
machine republishes from the EyeLink (name 'EyeLink', type 'Gaze') for a few
seconds and checks that what arrives is live eye data rather than a stream
that merely exists.  Session S002 of P01 (1 Oct 2026) is the case it is meant
to catch: LabRecorder recorded 8 minutes of the EyeLink stream, but every one
of its 456,934 samples was the same frozen sample, because the tracker was not
recording.

Checks
------
FAIL (do not start the session):
  * stream not found on the network;
  * no samples arrive;
  * eyelink_timestamp does not advance (frozen sample repeated, as in S002),
    or advances much slower than real time;
  * no required eye has a pupil (pupil area 0 in most samples), or the pupil
    value never changes.
WARN (session can run, but look at it):
  * a required eye loses the pupil in more than 20 % of samples;
  * the republisher delivers less than 80 % of the tracker samples
    (known: ~87 % at the moment, see project notes);
  * tracker clock steps backwards, channel labels unexpected, clock sync
    with the stream fails, or the data arrive more than 0.5 s late.

Usage
-----
    python check_eyelink_stream.py             one check (3 s of data)
    python check_eyelink_stream.py --watch     check every 2 s until Ctrl+C
    python check_eyelink_stream.py --eyes both require both eyes (default: any)

    --seconds S    data collected for the one-off check   (default 3)
    --interval S   window length in watch mode            (default 2)
    --eyes         any | left | right | both              (default any)
    --name/--type  LSL stream to check                    (default EyeLink / Gaze)
    --timeout S    how long to look for the stream        (default 5)
    --pause        wait for Enter at the end; offers watch mode (used by the .bat)

Exit code of the one-off check: 0 = OK, 1 = WARN, 2 = FAIL, so a batch file
or plrt.py can refuse to continue.

Watch mode prints one line per window and beeps when the data stop being
live.  An extra LSL consumer does not disturb LabRecorder.  If possible run
the watch on a machine other than the stimulus PC.

Needs pylsl and numpy only -- both ship with PsychoPy Standalone.
"""

import argparse
import sys
import time

import numpy as np

try:
    import pylsl
except ImportError:
    sys.exit("pylsl is not installed in this Python. Run the script with PsychoPy's "
             "python.exe (it bundles pylsl) or `pip install pylsl`.")

# Channel layout of the republished stream (verified on the P01 recordings).
DEFAULT_LABELS = ["leftEyeX", "leftEyeY", "rightEyeX", "rightEyeY",
                  "leftPupilArea", "rightPupilArea", "pixelsPerDegreeX",
                  "pixelsPerDegreeY", "eyelink_timestamp", "LSL_timestamp"]
EL_MISSING = -32768.0            # EyeLink MISSING_DATA code for gaze

# Thresholds
MIN_ADVANCE = 0.5                # tracker ms per ms of real time below this = stalled
MIN_PUPIL_VALID = 0.5            # an eye counts as tracked above this fraction
WARN_PUPIL_VALID = 0.8           # ... and gets a warning below this one
WARN_DELIVERED = 0.8             # fraction of tracker samples the republisher delivers
WARN_LAG_S = 0.5                 # data older than this when they arrive
STALL_WARN_S = 0.1               # no new tracker sample for this long -> warning
STALL_FAIL_S = 1.0               # ... for this long -> fail (frozen)

OK, INFO, WARN, FAIL = "OK", "info", "WARN", "FAIL"
SEVERITY = {OK: 0, INFO: 0, WARN: 1, FAIL: 2}


# --------------------------------------------------------------------------
# LSL plumbing
# --------------------------------------------------------------------------
def list_streams(wait=2.0):
    """One line per stream visible on the network (to help when ours is missing)."""
    infos = pylsl.resolve_streams(wait_time=wait)
    if not infos:
        return ["  (no LSL streams visible at all - network / multicast problem?)"]
    return ["  %-20s type %-10s host %-18s source_id %s"
            % (i.name(), i.type(), i.hostname(), i.source_id()) for i in infos]


def find_stream(name, stype, timeout):
    infos = pylsl.resolve_byprop("name", name, minimum=1, timeout=timeout)
    if stype:
        infos = [i for i in infos if i.type() == stype]
    return infos[0] if infos else None


def channel_labels(full_info):
    labels = []
    ch = full_info.desc().child("channels").child("channel")
    while not ch.empty():
        labels.append(ch.child_value("label"))
        ch = ch.next_sibling()
    return labels


def open_inlet(info, recover):
    inlet = pylsl.StreamInlet(info, max_buflen=30, recover=recover)
    inlet.open_stream(timeout=5.0)
    try:
        full = inlet.info(timeout=5.0)
    except Exception:
        full = info
    try:
        tc = inlet.time_correction(timeout=5.0)
    except Exception:
        tc = None
    # drop whatever was queued while connecting
    t_end = pylsl.local_clock() + 0.3
    while pylsl.local_clock() < t_end:
        inlet.pull_chunk(timeout=0.05, max_samples=8192)
    return inlet, full, tc


def collect(inlet, seconds, nch, tc):
    """Pull samples for `seconds`; also measure how old they are on arrival."""
    data, stamps, lags = [], [], []
    t_end = pylsl.local_clock() + seconds
    while pylsl.local_clock() < t_end:
        chunk, ts = inlet.pull_chunk(timeout=0.2, max_samples=8192)
        if ts:
            now = pylsl.local_clock()
            data.extend(chunk)
            stamps.extend(ts)
            if tc is not None:
                lags.append(now - (ts[-1] + tc))
    arr = np.asarray(data, dtype=float).reshape(-1, nch) if data else np.empty((0, nch))
    return arr, np.asarray(stamps, dtype=float), lags


# --------------------------------------------------------------------------
# The checks (pure function of the collected data)
# --------------------------------------------------------------------------
def analyse(data, stamps, cols, eyes="any", lags=None, watch_eyes=None):
    """Return (metrics dict, list of (level, message)).

    watch_eyes=None: the one-off pre-session check.  A list (possibly empty) switches
    to watch semantics: a window in which the watched eyes have little pupil is only a
    WARN (blinks, eyes closed); run_watch escalates when the pupil stays away.
    """
    res, m = [], {"n": len(stamps)}
    if len(stamps) < 10:
        res.append((FAIL, "only %d samples arrived - the stream exists but carries no data. "
                          "Is the republisher on the Ubuntu machine running?" % len(stamps)))
        return m, res

    span = stamps[-1] - stamps[0]
    m["push_hz"] = (len(stamps) - 1) / span if span > 0 else float("nan")

    el = data[:, cols["eyelink_timestamp"]]
    uniq, first = np.unique(el, return_index=True)
    m["distinct"] = uniq.size
    if uniq.size <= 1:
        res.append((FAIL, "FROZEN: eyelink_timestamp is constant (%d) in all %d samples - the "
                          "republisher repeats one old sample. The tracker is NOT recording: start "
                          "recording on the Host PC / check the Host-Ubuntu link."
                          % (el[0], len(el))))
        return m, res

    # tracker sampling interval: the smallest common step between distinct timestamps
    steps, counts = np.unique(np.diff(uniq), return_counts=True)
    common = steps[(steps > 0) & (counts >= 0.05 * counts.sum())]
    dt = float(common.min()) if common.size else float(np.median(np.diff(uniq)))
    m["tracker_hz"] = 1000.0 / dt

    # Forward steps between consecutive pushes. Jumps > 50 samples are discontinuities
    # (e.g. the switch between a frozen and a live sample), not delivered data.
    d = np.diff(el)
    fwd = d[(d > 0) & (d <= 50 * dt)]
    m["advance"] = fwd.sum() / (1000.0 * span) if span > 0 else 0.0
    m["delivered"] = fwd.size / (fwd.sum() / dt) if fwd.size else 0.0
    n_back = int(np.sum(d < 0))

    # longest time during which no new tracker sample arrived (pushes repeat the same one)
    change = np.r_[0, np.flatnonzero(d != 0) + 1, len(el)]
    m["stall_s"] = float(np.max(stamps[change[1:] - 1] - stamps[change[:-1]]))

    if m["advance"] < MIN_ADVANCE or m["stall_s"] >= STALL_FAIL_S:
        res.append((FAIL, "STALLED: no new tracker sample for %.1f s; the tracker clock advanced only "
                          "%.0f %% of real time - the republisher repeats old samples. Is the tracker "
                          "still recording?" % (m["stall_s"], 100 * m["advance"])))
        return m, res
    if m["stall_s"] >= STALL_WARN_S:
        res.append((WARN, "no new tracker sample for %.0f ms at one point (link hiccup?)"
                          % (1000 * m["stall_s"])))

    rows = data[first]                       # one row per tracker sample
    tracked = {}
    for eye, col in (("left", "leftPupilArea"), ("right", "rightPupilArea")):
        p = rows[:, cols[col]]
        valid = p > 0
        m["pupil_" + eye] = valid.mean()
        m["pupil_mean_" + eye] = float(np.sqrt(p[valid]).mean()) if valid.any() else float("nan")
        m["pupil_frozen_" + eye] = bool(valid.sum() > 10 and np.ptp(p[valid]) == 0)
        tracked[eye] = valid.mean() >= MIN_PUPIL_VALID and not m["pupil_frozen_" + eye]

    # --- tracker / republisher ---------------------------------------------
    res.append((OK, "live: tracker clock advances in real time (%.2f ms/ms), tracker rate %.0f Hz"
                    % (m["advance"], m["tracker_hz"])))
    lvl = OK if m["delivered"] >= 0.99 else (INFO if m["delivered"] >= WARN_DELIVERED else WARN)
    note = {OK: "", INFO: " (known republisher loss; the notebook fills the 1-2 ms holes)",
            WARN: " - more than the known loss of ~13 %: is the Ubuntu machine overloaded?"}[lvl]
    res.append((lvl, "republisher pushes %.0f Hz and delivers %.1f %% of tracker samples%s"
                     % (m["push_hz"], 100 * m["delivered"], note)))
    if n_back:
        res.append((WARN, "eyelink_timestamp stepped backwards %d x (tracker restarted?)" % n_back))
    if lags:
        lag = float(np.median(lags))
        m["lag_s"] = lag
        res.append((OK if lag < WARN_LAG_S else WARN,
                    "data arrive %.0f ms after they were stamped" % (1000 * lag)))

    # --- eyes ------------------------------------------------------------------
    m["eyes"] = [e for e in ("left", "right") if tracked[e]]
    for eye in ("left", "right"):
        if m["pupil_frozen_" + eye]:
            res.append((FAIL, "%s eye: pupil value never changes - frozen data" % eye))
    if watch_eyes is not None:
        for eye in watch_eyes:
            if m["pupil_" + eye] < MIN_PUPIL_VALID:
                res.append((WARN, "%s eye: pupil in only %.0f %% of this window (blinks, eyes "
                                  "closed, looking away?)" % (eye, 100 * m["pupil_" + eye])))
        return m, res

    for eye in ("left", "right"):
        v = m["pupil_" + eye]
        if m["pupil_frozen_" + eye]:
            continue
        elif v >= WARN_PUPIL_VALID:
            res.append((OK, "%s eye: pupil in %.0f %% of samples (mean sqrt(area) %.1f)"
                            % (eye, 100 * v, m["pupil_mean_" + eye])))
        elif v >= MIN_PUPIL_VALID:
            res.append((WARN, "%s eye: pupil only in %.0f %% of samples - blinking a lot, or check "
                              "the camera image / pupil threshold" % (eye, 100 * v)))
        else:
            res.append((INFO, "%s eye: no pupil (%.0f %% of samples)" % (eye, 100 * v)))
    if m["eyes"]:
        res.append((INFO, "recording is %s" % ("binocular" if len(m["eyes"]) == 2
                                               else "monocular (%s eye)" % m["eyes"][0])))

    required = {"any": [], "left": ["left"], "right": ["right"], "both": ["left", "right"]}[eyes]
    if not m["eyes"]:
        res.append((FAIL, "no eye has a usable pupil signal - is the participant in place and the "
                          "eye found on the Host PC?"))
    for eye in required:
        if not tracked[eye]:
            res.append((FAIL, "%s eye is required (--eyes %s) but has no usable pupil signal"
                              % (eye, eyes)))
    return m, res


def verdict(res):
    worst = max((SEVERITY[lvl] for lvl, _ in res), default=2)
    return {0: OK, 1: WARN, 2: FAIL}[worst]


def column_map(labels):
    """Map the channel names we need to indices; fall back to the known layout."""
    use = labels if all(k in labels for k in ("leftPupilArea", "rightPupilArea",
                                              "eyelink_timestamp")) else DEFAULT_LABELS
    return {lab: i for i, lab in enumerate(use)}, use is labels


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
TAGS = {OK: "[ OK ]", INFO: "[info]", WARN: "[WARN]", FAIL: "[FAIL]"}
PUPIL_LOST_FAIL_S = 10.0         # watch mode: no pupil this long in a row -> FAIL


def print_report(res):
    for lvl, msg in res:
        print("  %s %s" % (TAGS[lvl], msg))


def print_verdict(v):
    text = {OK: "OK - eye data are live, the session can start.",
            WARN: "WARNING - eye data are live, but look at the [WARN] lines above.",
            FAIL: "FAIL - do NOT start the session: see the [FAIL] lines above."}[v]
    bar = "=" * 78
    print("\n%s\n  %s\n%s" % (bar, text, bar))
    if v == FAIL:
        print("\a", end="", flush=True)


def watch_line(m, res):
    v = verdict(res)
    parts = [time.strftime("%H:%M:%S")]
    if "push_hz" in m:
        parts.append("push %4.0f Hz" % m["push_hz"])
    if "delivered" in m:
        parts.append("tracker %4.0f Hz, %3.0f %% delivered" % (m["tracker_hz"], 100 * m["delivered"]))
    if "pupil_left" in m:
        parts.append("pupil L %3.0f %%  R %3.0f %%" % (100 * m["pupil_left"], 100 * m["pupil_right"]))
    if "lag_s" in m:
        parts.append("lag %4.0f ms" % (1000 * m["lag_s"]))
    line = " | ".join(parts) + "  " + TAGS[v]
    if v != OK:
        line += "  " + "; ".join(msg for lvl, msg in res if lvl == v)
    return v, line


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run_once(args):
    print("Looking for LSL stream name=%r type=%r (up to %g s) ..." % (args.name, args.type, args.timeout))
    info = find_stream(args.name, args.type, args.timeout)
    if info is None:
        print("\n  %s stream %r not found. Visible streams:" % (TAGS[FAIL], args.name))
        print("\n".join(list_streams()))
        print("  -> Is the republisher running on the Ubuntu machine, and is this PC on the same subnet?")
        print_verdict(FAIL)
        return FAIL, None, None
    inlet, full, tc = open_inlet(info, recover=False)
    labels = channel_labels(full)
    cols, labels_ok = column_map(labels)
    nch = info.channel_count()
    print("Found %s (type %s) from host %s, %d channels, nominal %g Hz. Collecting %g s ..."
          % (info.name(), info.type(), info.hostname(), nch, info.nominal_srate(), args.seconds))
    data, stamps, lags = collect(inlet, args.seconds, nch, tc)
    m, res = analyse(data, stamps, cols, args.eyes, lags)
    if not labels_ok:
        res.append((WARN, "channel labels %s differ from the expected layout - assumed %s"
                          % (labels or "(none)", DEFAULT_LABELS)))
    if tc is None:
        res.append((WARN, "LSL clock synchronisation with the stream failed (time_correction)"))
    print()
    print_report(res)
    v = verdict(res)
    print_verdict(v)
    inlet.close_stream()
    return v, info, m.get("eyes")


def eye_words(eyes):
    return " + ".join(eyes) + (" eye" if len(eyes) == 1 else " eyes")


def run_watch(args, info=None, eyes_seen=None):
    if info is None:
        info = find_stream(args.name, args.type, args.timeout)
        if info is None:
            print("%s stream %r not found." % (TAGS[FAIL], args.name))
            print("\n".join(list_streams()))
            return
    inlet, full, tc = open_inlet(info, recover=True)
    cols, _ = column_map(channel_labels(full))
    nch = info.channel_count()
    # which eyes to watch: given by --eyes, else those the one-off check found,
    # else those seen in the first window that has a pupil
    watch_eyes = {"left": ["left"], "right": ["right"], "both": ["left", "right"]}.get(args.eyes, eyes_seen)
    print("Watching %s from %s every %g s - Ctrl+C to stop." % (info.name(), info.hostname(), args.interval))
    if watch_eyes:
        print("Pupil watched on: %s." % eye_words(watch_eyes))
    last, lost_s = None, 0.0
    try:
        while True:
            data, stamps, lags = collect(inlet, args.interval, nch, tc)
            m, res = analyse(data, stamps, cols, args.eyes, lags, watch_eyes=watch_eyes or [])
            if not watch_eyes and m.get("eyes"):
                watch_eyes = m["eyes"]
                print("Pupil watched on: %s." % eye_words(watch_eyes))
            if watch_eyes and "pupil_left" in m:
                lost = all(m["pupil_" + e] < 0.1 for e in watch_eyes)
                lost_s = lost_s + args.interval if lost else 0.0
                if lost_s >= PUPIL_LOST_FAIL_S:
                    res.append((FAIL, "no pupil for %.0f s - participant moved / tracker lost the eye?"
                                      % lost_s))
            v, line = watch_line(m, res)
            if v == FAIL:
                line = "!!! " + line + "\a"
            elif last == FAIL:
                line = "... recovered: " + line
            print(line, flush=True)
            last = v
    except KeyboardInterrupt:
        print("\nstopped.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check that the EyeLink LSL stream carries live eye data.")
    ap.add_argument("--name", default="EyeLink")
    ap.add_argument("--type", default="Gaze")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--eyes", choices=["any", "left", "right", "both"], default="any")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--pause", action="store_true")
    args = ap.parse_args(argv)

    if args.watch:
        run_watch(args)
        if args.pause:
            input("\nPress Enter to close.")
        return 0

    v, info, eyes_seen = run_once(args)
    if args.pause:
        ans = input("\nEnter = close   |   w + Enter = keep watching during the session: ")
        if ans.strip().lower().startswith("w"):
            run_watch(args, info, eyes_seen)
            input("\nPress Enter to close.")
    return SEVERITY[v]


if __name__ == "__main__":
    sys.exit(main())
