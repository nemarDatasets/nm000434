"""fix-20261030 (Oct 2026): walking / stop / freezing segments and gait events for the iEEG (Percept LFP) runs of nm000434.

Inputs (all regenerable from the dataset and the authors' public code):
  code/fix-20261030_authors_timepoints.tsv  authors' hard-coded values (extracted by fix-20261030_extract_authors_timepoints.py
      from github.com/PKlocke/SenseFOG-Documentation @ b87df3f, 'A) Subject Files/sub-XX-datafile.m'):
      signalpoint E (EEG artefact sample, EEG cut at the first 'M  1' marker, 1 kHz) and L (LFP artefact sample, LFP
      resampled to 1 kHz), and events_filt segments (task label, start, end in s)
  sourcedata/SenseFOG-main/sub-XX/ses-<cond>/motion/*_gaitfilt_cor.mat  GaitEvents.rf_events / lf_events
      (Heelstrike_Loc, Toe_Off_Loc, Midswing_Loc in s)
  sub-XX/motion/*_motion.tsv (used only for the check of step 4), code/conversion_report.json (LFP sample counts)

Time bases, from the authors' scripts (B) Preprocessing 1.3 / 1.4 / 1.5 / 1.6, C) Analysis 2.2.x):
  - LFP (= the BIDS ieeg data, Percept TimeDomainData unchanged, 250 Hz) is resampled to 1 kHz (1.3: resample), so
    1-kHz sample j lies at (j - 1) / 1000 s of the BIDS ieeg run.
  - IMU (= the BIDS motion data) is resampled to 1 kHz; IMU time t = i / 1000 for 1-kHz sample i (1.4: imutime).
    The IMU recording starts at the first 'M  1' marker of the EEG (1.2 / 1.5), EEG_pre = EEG from that marker.
  - 1.5 aligns on the artefact: delay = |L - E|; if L > E the LFP is cut to samples delay..end, else EEG_pre and IMU are
    cut to samples delay..end. The aligned 1-kHz axis: sample k <-> aligned time k / 1000 (analysis: index = 1000 * t).
  - events_filt times are on the IMU time axis before the cut: 1.6 subtracts delay / 1000 from them when E > L.
  - GaitEvents times are on the aligned axis (checked in step 4).
  So aligned time t_al -> BIDS ieeg onset: L > E: t_al + (delay - 2) / 1000 ; E > L: t_al - 1 / 1000.
  events_filt t -> t_al = t (L > E) or t - delay / 1000 (E > L).
Steps:
  1. rows for every events_filt segment of the run's task (label verbatim; duration = end - start, n/a if end < start);
  2. rows for every heel strike, toe off and midswing of each foot (rf = right, lf = left);
  3. rows whose onset falls outside the ieeg run are dropped (counted in the report);
  4. check per foot: each Midswing time, mapped to the BIDS motion axis (t_al - 1 ms; t_al + (delay - 2) ms when E > L), must sit within
     16 ms (2 IMU samples) of the local maximum of |gyroscope y| of that foot's sensor in the BIDS motion data (+/- 250 ms
     window). Gait rows of a foot are written only if at least 50 % of its midswings pass; segments are written regardless.
Usage: python fix-20261030_gait_events.py <bids_root> <dir with the content of the needed annexed files (same relpaths)>
"""
import csv, json, os, sys
import numpy as np
import scipy.io as sio

ROOT, DATA = sys.argv[1], sys.argv[2]
SES2KEY = {"walk": "Walk", "walkws": "WalkWS", "walkint": "WalkINT", "walkint2": "WalkINT_new"}
rep = json.load(open(os.path.join(ROOT, "code", "conversion_report.json")))
tp = list(csv.DictReader(open(os.path.join(ROOT, "code", "fix-20261030_authors_timepoints.tsv")), delimiter="\t"))


def ev_arr(x, f):
    return np.array([float(np.asarray(e[f]).ravel()[0]) if np.asarray(e[f]).size else np.nan for e in np.atleast_1d(x)])


def fnum(v, nd=4):
    return "n/a" if v is None or v != v else ("%.*f" % (nd, v)).rstrip("0").rstrip(".")


summary = []
for r in rep["ieeg"]:
    sub = r["stem"].split("_")[0]; ses = r["src"].split("/")[1][4:]; stem = r["stem"]
    if ses not in SES2KEY:
        continue
    key = SES2KEY[ses]
    sp = [t for t in tp if t["subject"] == sub and t["datafile_task"] == key and t["row_type"] == "signalpoint"]
    segs = [t for t in tp if t["subject"] == sub and t["datafile_task"] == key and t["row_type"] == "events_filt"]
    s = {"run": stem, "datafile_task": key}
    if len(sp) != 1:
        s["status"] = "no artefact samples in the authors' datafile"; summary.append(s); continue
    E, L = int(sp[0]["EEG_signal"]), int(sp[0]["LFP_signal"]); delay = abs(L - E); lfp_first = L > E
    al2lfp = (lambda t: t + (delay - 2) / 1000.0) if lfp_first else (lambda t: t - 0.001)
    filt2al = (lambda t: t) if lfp_first else (lambda t: t - delay / 1000.0)
    al2imu = (lambda t: t - 0.001) if lfp_first else (lambda t: t + (delay - 2) / 1000.0)
    fs = float(r["fs"]); dur = r["n"] / fs
    s.update(E=E, L=L, delay=delay, cut="LFP" if lfp_first else "EEG+IMU", ieeg_duration_s=dur)
    rows = []; dropped = 0
    for t in segs:
        st = float(t["start_s"]); en = float(t["end_s"]) if t["end_s"] != "n/a" else float("nan")
        on = al2lfp(filt2al(st)); d = en - st if en == en and en >= st else float("nan")
        if not (0 <= on <= dur):
            dropped += 1; continue
        rows.append([on, d, t["label"], "n/a", "datafile", int(t["index"])])
    s["segments"] = len(rows); s["segments_bad_end"] = sum(1 for x in rows if x[1] != x[1])
    gp = os.path.join(DATA, "sourcedata", "SenseFOG-main", sub, f"ses-{ses}", "motion", f"{sub}-ses-{ses}_gaitfilt_cor.mat")
    s["gait"] = {}
    if os.path.exists(gp):
        G = sio.loadmat(gp, squeeze_me=True)["GaitEvents"]
        tsk = stem.split("_task-")[1].split("_")[0]
        mrel = stem.replace(f"_task-{tsk}", f"_task-{tsk}_tracksys-APDMOpal")
        chans = [c["name"] for c in csv.DictReader(open(os.path.join(ROOT, sub, "motion", mrel + "_channels.tsv")), delimiter="\t")]
        M = np.loadtxt(os.path.join(DATA, sub, "motion", mrel + "_motion.tsv"), delimiter="\t", ndmin=2)
        for foot, fk in (("right", "rf_events"), ("left", "lf_events")):
            ev = G[fk].item() if (G[fk].dtype == object and G[fk].ndim == 0) else G[fk]
            ci = [i for i, c in enumerate(chans) if c.lower().startswith(foot) and c.endswith("GYRO_y")]
            ms = ev_arr(ev, "Midswing_Loc"); ms = ms[np.isfinite(ms)]
            x = np.abs(M[:, ci[0]]); offs = []
            for t in ms:
                tc = al2imu(t); i0 = int(round((tc - 0.25) * 128)); i1 = int(round((tc + 0.25) * 128))
                if 0 <= i0 and i1 < len(x):
                    offs.append((i0 + int(np.argmax(x[i0:i1 + 1]))) / 128.0 - tc)
            frac = float(np.mean(np.abs(np.array(offs)) <= 0.016)) if offs else 0.0
            g = {"sensor_channel": chans[ci[0]], "midswing_checked": len(offs), "midswing_within_16ms": frac,
                 "median_abs_offset_ms": float(np.median(np.abs(offs)) * 1e3) if offs else None}
            if frac >= 0.5:
                n0 = len(rows)
                for fld, lab in (("Heelstrike_Loc", "heel_strike"), ("Toe_Off_Loc", "toe_off"), ("Midswing_Loc", "midswing")):
                    for t in ev_arr(ev, fld):
                        if t != t:
                            continue
                        on = al2lfp(t)
                        if not (0 <= on <= dur):
                            dropped += 1; continue
                        rows.append([on, 0.0, lab, foot, "gaitfilt_cor", "n/a"])
                g["rows"] = len(rows) - n0
            else:
                g["rows"] = 0; g["note"] = "not written: midswings do not match this foot's gyroscope"
            s["gait"][foot] = g
    s["dropped_outside_run"] = dropped; s["rows"] = len(rows)
    rows.sort(key=lambda x: (x[0], x[2], x[3]))
    out = os.path.join(ROOT, sub, "ieeg", stem + "_events.tsv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["onset", "duration", "trial_type", "foot", "sample", "source", "datafile_index"])
        for on, d, lab, foot, src, idx in rows:
            w.writerow([fnum(on), fnum(d), lab, foot, int(round(on * fs)), src, idx])
    side = {
        "onset": {"Description": "Seconds from the first sample of this Percept LFP run, mapped from the authors' aligned time axis (see _note)", "Units": "s"},
        "duration": {"Description": "Segment rows: end - start from the authors' datafile (n/a if end < start); gait rows: 0", "Units": "s"},
        "trial_type": {"Description": "Segment label as written in the authors' datafile, or gait event",
                       "Levels": {"Walk": "authors' walking segment (events_filt task 'Walk')", "Selected_stop": "authors' self-selected stop segment",
                                  "Selected stop": "authors' label variant (not selected by the authors' scripts, which match 'Selected_stop')",
                                  "Selected_stop_start": "authors' label variant (not selected by the authors' scripts)",
                                  "Freezing_walk": "authors' freezing-of-gait segment while walking",
                                  "Freezing_walk_initiation": "authors' freezing segment at gait initiation",
                                  "heel_strike": "GaitEvents Heelstrike_Loc", "toe_off": "GaitEvents Toe_Off_Loc", "midswing": "GaitEvents Midswing_Loc"}},
        "foot": {"Description": "Gait rows: right (GaitEvents.rf_events) or left (lf_events); n/a for segments"},
        "sample": {"Description": "round(onset x 250), 0-based sample of this run"},
        "source": {"Description": "datafile: subjectdata.events_filt in sub-XX-datafile.m of github.com/PKlocke/SenseFOG-Documentation (b87df3f), copied to code/fix-20261030_authors_timepoints.tsv; gaitfilt_cor: sourcedata .../motion/*_gaitfilt_cor.mat"},
        "datafile_index": {"Description": "Segment rows: index i of events_filt.<task>(i) in the authors' datafile"},
        "_note": (f"fix-20261030 (code/fix-20261030_gait_events.py). Authors' artefact samples for this run (datafile task {key}): "
                  f"E = {E} (EEG, 1 kHz, counted from the first 'M  1' marker), L = {L} (LFP resampled to 1 kHz); "
                  + (f"L > E: the authors cut the LFP by {delay} samples, so onset = aligned time + {delay - 2} ms; segments are on the aligned axis."
                     if lfp_first else
                     f"E > L: the authors cut EEG and IMU by {delay} samples and subtract {delay} ms from the datafile segments, so onset = segment time - {delay + 1} ms and gait time - 1 ms."))}
    json.dump(side, open(os.path.join(ROOT, sub, "ieeg", stem + "_events.json"), "w"), indent=2, ensure_ascii=False)
    s["status"] = "written"; summary.append(s)
    print(stem, json.dumps({k: s[k] for k in ("cut", "delay", "segments", "rows", "dropped_outside_run")}), json.dumps({f: (round(g["midswing_within_16ms"], 2), g["rows"]) for f, g in s["gait"].items()}), flush=True)
json.dump(summary, open(os.path.join(ROOT, "code", "fix-20261030_gait_events_report.json"), "w"), indent=1)
print("DONE")
