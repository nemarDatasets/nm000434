"""d3-mendeley-sensefog (Mendeley Data 10.17632/c9ckcvjxc7.1, CC BY 4.0) -> raw BIDS (iEEG + EEG + motion) with scrubbed sourcedata.

Release "SenseFOG-main" (measured in Jobs ieeg-b3w11-sfprobe-1008004235 / sfprobe2-1008025000 / sfprobe3-1008025334):
  sub-XX/ses-<condition>/ieeg/*.json   Medtronic Percept PC session exports; BrainSenseTimeDomain = 2 bipolar STN streams at 250 Hz
  sub-XX/ses-<condition>/eeg/*.vhdr    BrainVision (58 ch: 48 EEG, 8 EMG TA/GA, 2 'IPG'; 1000 Hz INT_16, 0.1 uV); DataFile/MarkerFile still name
                                       the pre-rename files (e.g. PD01_gehen.eeg)
  sub-XX/ses-<condition>/motion/*.h5   APDM Opal HDF5 (3 IMUs, 128 Hz), absolute UTC time stamps; *_gaitfilt_cor.mat gait events
  *_lfpalg.mat / *_eegalg.mat / *_gaitalg.mat  the authors' already-aligned files for recordings without a usable raw file
                                       (sub-10 all: ECG-cleaned; sub-05 standing; sub-11 / sub-17 sitting LFP). They embed the Percept JSON.
Conversion (values never changed):
  - ieeg: each BrainSenseTimeDomain recording -> BrainVision float32, channel names = Percept stream labels, unit uV (Percept JSON convention).
  - eeg: .eeg bytes copied unchanged; vhdr/vmrk rewritten with the BIDS file names; vmrk 'New Segment' date -> day 01.
  - motion: APDM 'Calibrated' sensor arrays -> BIDS motion TSV (exact %.17g text), channel types ACCEL/GYRO/MAGN/ORNT/MISC(temp).
  - The release's ses-<condition> folders are conditions of one visit -> BIDS task-<condition> (walkint2 -> task-walkint run-2).
  - sourcedata/SenseFOG-main: the full release; Percept JSON, vmrk, h5 and .mat copies scrubbed (every date -> day 01 of its month;
    h5 time vectors shifted by the same whole days; Percept GroupName free text removed).
Usage: python b3w11_convert_sensefog.py <work_dir> <bids_root>
"""
import datetime
import glob
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import zlib
from collections import defaultdict

import h5py
import mne
import numpy as np
import scipy.io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b3w3_common import scrub_mat_header, sha256_file, wjson, write_vhdr, wtsv  # noqa: E402

mne.set_log_level("ERROR")
W, OUT = sys.argv[1], sys.argv[2]
S = glob.glob(os.path.join(W, "sourcedata_dl", "SenseFOG-main*"))[0]
SD = os.path.join(OUT, "sourcedata", "SenseFOG-main")
rep = {"ieeg": [], "eeg": [], "motion": [], "sourcedata": [], "privacy": defaultdict(int), "not_converted": []}

# ------------------------------------------------------------------ privacy helpers
ISO = re.compile(r"((?:18|19|20)\d\d-[01]\d-)([0-3]\d)")
ISO_BAD = re.compile(r"(?:18|19|20)\d\d-[01]\d-(?!01)[0-3]\d")
BV = re.compile(r"(New Segment,[^,\n]*,\d+,\d+,\d+,\d{6})(\d\d)(\d+)")
BV_BAD = re.compile(r"New Segment,[^,\n]*,\d+,\d+,\d+,\d{6}(?!01)\d\d")


def scrub_text_dates(t):
    t2, n = ISO.subn(lambda m: m.group(1) + "01", t)
    t3, n2 = BV.subn(lambda m: m.group(1) + "01" + m.group(3), t2)
    return t3, n + n2


def scrub_percept(txt):
    t, n = scrub_text_dates(txt)
    t, g = re.subn(r'("GroupName"\s*:\s*")([^"]+)(")', lambda m: m.group(1) + "[removed]" + m.group(3), t)
    json.loads(t)
    assert not ISO_BAD.search(t)
    return t, n, g


def u16(s):
    return b"".join(bytes([c]) + b"\x00" for c in s)


B_ISO_A = re.compile(rb"((?:18|19|20)\d\d-[01]\d-)[0-3]\d")
B_ISO_U = re.compile(rb"((?:1\x00[89]\x00|2\x000\x00)\d\x00\d\x00-\x00[01]\x00\d\x00-\x00)[0-3]\x00\d\x00")
B_BV_A = re.compile(rb"(?<!\d)((?:19|20)\d\d[01]\d)[0-3]\d(\d{12})(?!\d)")
B_BV_U = re.compile(rb"((?:1\x009\x00|2\x000\x00)\d\x00\d\x00[01]\x00\d\x00)[0-3]\x00\d\x00((?:\d\x00){12})")
B_BAD = [re.compile(rb"(?:18|19|20)\d\d-[01]\d-(?!01)[0-3]\d"),
         re.compile(rb"(?:1\x00[89]\x00|2\x000\x00)\d\x00\d\x00-\x00[01]\x00\d\x00-\x00(?!0\x001\x00)[0-3]\x00\d\x00")]


def scrub_bytes(b):
    n = 0
    b, k = B_ISO_A.subn(lambda m: m.group(1) + b"01", b); n += k
    b, k = B_ISO_U.subn(lambda m: m.group(1) + b"0\x001\x00", b); n += k
    b, k = B_BV_A.subn(lambda m: m.group(1) + b"01" + m.group(2), b); n += k
    b, k = B_BV_U.subn(lambda m: m.group(1) + b"0\x001\x00" + m.group(2), b); n += k
    return b, n


def mat_elements(b):
    """yield (type, payload bytes) of top-level MAT v5 elements (little-endian), decompressing miCOMPRESSED."""
    pos = 128
    while pos < len(b):
        typ, size = struct.unpack("<II", b[pos:pos + 8])
        if typ >> 16:
            yield typ & 0xFFFF, b[pos + 4:pos + 8]; pos += 8; continue
        data = b[pos + 8:pos + 8 + size]
        if typ == 15:
            yield 15, zlib.decompress(data); pos += 8 + size
        else:
            yield typ, data; pos += 8 + size + ((-size) % 8)


def scrub_mat(src, dst):
    b = open(src, "rb").read()
    assert b[126:128] == b"IM", "big-endian MAT not handled"
    b, hdr = scrub_mat_header(b)
    out = bytearray(b[:128]); pos = 128; n = 0
    while pos < len(b):
        typ, size = struct.unpack("<II", b[pos:pos + 8])
        if typ >> 16:
            out += b[pos:pos + 8]; pos += 8; continue
        data = b[pos + 8:pos + 8 + size]
        if typ == 15:
            d = zlib.decompressobj(); raw = d.decompress(data); assert not d.unused_data
            new, k = scrub_bytes(raw)
            if k:
                data = zlib.compress(new, 6); n += k
            out += struct.pack("<II", 15, len(data)) + data; pos += 8 + size
        else:
            new, k = scrub_bytes(data); n += k
            pad = (-size) % 8
            out += b[pos:pos + 8] + new + b[pos + 8 + size:pos + 8 + size + pad]; pos += 8 + size + pad
    open(dst, "wb").write(bytes(out))
    # verify: no date with day != 01 left; numeric content unchanged
    o = open(dst, "rb").read()
    for _, payload in mat_elements(o):
        for p in B_BAD:
            assert not p.search(payload), f"date left in {dst}"
    a = scipy.io.loadmat(src, squeeze_me=True, struct_as_record=False)
    c = scipy.io.loadmat(dst, squeeze_me=True, struct_as_record=False)
    nnum = [0]

    def cmp(x, y):
        if isinstance(x, scipy.io.matlab.mat_struct):
            for f in x._fieldnames:
                cmp(getattr(x, f), getattr(y, f))
        elif isinstance(x, np.ndarray) and x.dtype == object:
            for u, v in zip(x.flat, y.flat):
                cmp(u, v)
        elif isinstance(x, np.ndarray) and x.dtype.kind in "fiub":
            assert x.shape == y.shape and np.array_equal(x, y, equal_nan=True); nnum[0] += x.size
        elif isinstance(x, (int, float)):
            assert x == y or (x != x and y != y); nnum[0] += 1
    for k in a:
        if not k.startswith("__"):
            cmp(a[k], c[k])
    return hdr, n, nnum[0]


def scrub_h5(src, dst):
    shutil.copy2(src, dst)
    with h5py.File(dst, "r+") as h:
        sens = [k for k in h.keys() if k != "Annotations"]
        t0 = int(h[sens[0]]["Time"][0])
        day = datetime.datetime.fromtimestamp(t0 / 1e6, datetime.timezone.utc).day
        sh = (day - 1) * 86400 * 10 ** 6
        for k in sens:
            h[k]["Time"][...] = h[k]["Time"][()] - np.uint64(sh)
            h[k]["SyncValue"][...] = h[k]["SyncValue"][()] - np.uint64((day - 1) * 86400 * 2560)
        if "Annotations" in h:
            a = h["Annotations"][()]
            a["Time"] = a["Time"] - np.uint64(sh)
            h["Annotations"][...] = a
        d1 = datetime.datetime.fromtimestamp(int(h[sens[0]]["Time"][0]) / 1e6, datetime.timezone.utc)
        assert sh == 0 or d1.day == 1
    with h5py.File(src, "r") as a, h5py.File(dst, "r") as c:
        for k in sens:
            for q in ("Accelerometers", "Gyroscopes", "Magnetometers", "Orientation", "Temperature"):
                assert np.array_equal(a[k]["Calibrated"][q][()], c[k]["Calibrated"][q][()], equal_nan=True)
    return day - 1


# ------------------------------------------------------------------ release inventory
TASKDESC = {
    "sitting": "Sitting with eyes open (about 120 s), medication OFF, stimulation OFF (release documentation)",
    "standing": "Standing with eyes open (about 120 s), medication OFF, stimulation OFF (release documentation)",
    "walk": "Walking through a narrow corridor as a mono task (about 240 s), medication OFF, stimulation OFF (release documentation)",
    "walkws": "Walking interrupted by internally generated voluntary stops (about 240 s), medication OFF, stimulation OFF (release documentation)",
    "walkint": "Walking under dual-task conditions (serial subtraction), medication OFF, stimulation OFF (release documentation)",
}
sesdirs = sorted(glob.glob(os.path.join(S, "sub-*", "ses-*")))
by_sub = defaultdict(list)
for d in sesdirs:
    by_sub[d.split(os.sep)[-2]].append(d.split(os.sep)[-1][4:])


def task_run(sub, cond):
    m = re.match(r"([a-z]+?)(\d*)$", cond)
    base, k = m.group(1), m.group(2)
    if k:
        return base, int(k)
    if any(re.fullmatch(base + r"\d+", c) for c in by_sub[sub]):
        return base, 1
    return base, None


def stem_of(sub, cond):
    t, r = task_run(sub, cond)
    return f"{sub}_task-{t}" + (f"_run-{r}" if r else ""), t


sex = defaultdict(set)
leads = {}
# ------------------------------------------------------------------ iEEG (Percept)
for f in sorted(glob.glob(os.path.join(S, "sub-*", "ses-*", "ieeg", "*.json"))):
    sub, cond = f.split(os.sep)[-4], f.split(os.sep)[-3][4:]
    d = json.load(open(f))
    for k in ("Initial", "Final"):
        g = d.get("PatientInformation", {}).get(k, {}).get("PatientGender", "")
        if g.endswith("MALE"):
            sex[sub].add("F" if g.endswith("FEMALE") else "M")
    lc = d.get("LeadConfiguration", {}).get("Final") or d.get("LeadConfiguration", {}).get("Initial") or []
    leads.setdefault(sub, {l["Hemisphere"].split(".")[-1].upper(): (l["Model"].split(".")[-1], l["LeadLocation"].split(".")[-1]) for l in lc})
    td = d.get("BrainSenseTimeDomain") or []
    if not td:
        rep["not_converted"].append({"file": os.path.relpath(f, S), "reason": "Percept JSON without BrainSenseTimeDomain streams (sourcedata only)"})
        continue
    groups = defaultdict(list)
    for t in td:
        groups[t["FirstPacketDateTime"]].append(t)
    assert len(groups) == 1, f"{f}: {len(groups)} TD recordings"
    st = list(groups.values())[0]
    lens = {len(t["TimeDomainData"]) for t in st}
    assert len(lens) == 1, f"{f}: unequal stream lengths {lens}"
    fs = float(st[0]["SampleRateInHz"]); assert all(t["SampleRateInHz"] == st[0]["SampleRateInHz"] for t in st)
    X = np.array([t["TimeDomainData"] for t in st], dtype=np.float64)
    names = [t["Channel"] for t in st]
    stem, task = stem_of(sub, cond)
    D = os.path.join(OUT, sub, "ieeg"); os.makedirs(D, exist_ok=True)
    np.ascontiguousarray(X.T.astype("<f4")).tofile(os.path.join(D, stem + "_ieeg.eeg"))
    write_vhdr(os.path.join(D, stem + "_ieeg"), len(names), fs, names, ["µV"] * len(names),
               comment="b3w11_convert_sensefog.py: Medtronic Percept BrainSenseTimeDomain streams, values unchanged as float32 (uV)")
    back = np.fromfile(os.path.join(D, stem + "_ieeg.eeg"), dtype="<f4").reshape(-1, len(names)).T.astype(np.float64)
    err = float(np.max(np.abs(back - X))); scale = float(np.max(np.abs(X))) or 1.0
    rt = back.shape == X.shape and err <= 1e-6 * scale
    raw = mne.io.read_raw_brainvision(os.path.join(D, stem + "_ieeg.vhdr"), preload=True)
    rt = rt and np.allclose(raw.get_data() * 1e6, X, rtol=0, atol=1e-6 * scale) and raw.ch_names == names
    stim = d.get("Stimulation", {})
    si, sf = stim.get("InitialStimStatus", "n/a").split(".")[-1], stim.get("FinalStimStatus", "n/a").split(".")[-1]
    side = lambda c: "LEFT" if c.endswith("_LEFT") else ("RIGHT" if c.endswith("_RIGHT") else "n/a")
    wtsv(os.path.join(D, stem + "_channels.tsv"), ["name", "type", "units", "low_cutoff", "high_cutoff", "sampling_frequency", "group", "status", "status_description", "description"],
         [[c, "DBS", "µV", "n/a", "n/a", fs, side(c), "good", "n/a",
           "Percept BrainSense TimeDomain stream %s (bipolar pair of contact levels named in the label; %s STN lead %s); Percept Gain field %s"
           % (c, side(c).lower(), leads[sub].get(side(c), ("n/a",))[0], t.get("Gain", "n/a"))] for c, t in zip(names, st)])
    js = {
        "TaskName": task, "TaskDescription": TASKDESC.get(task, "n/a"),
        "SamplingFrequency": fs, "PowerLineFrequency": 50,
        "SoftwareFilters": "n/a", "HardwareFilters": "n/a",
        "iEEGReference": "bipolar: each channel is the Percept sensing pair named in its label (e.g. ONE_THREE_LEFT = contact levels 1 and 3 of the left lead)",
        "iEEGPlacementScheme": "bilateral subthalamic nucleus DBS leads (LeadLocation 'Stn' in the Percept export)",
        "Manufacturer": "Medtronic",
        "ManufacturersModelName": "%s (%s)" % (d["DeviceInformation"]["Final"].get("Neurostimulator", "n/a"), d["DeviceInformation"]["Final"].get("NeurostimulatorModel", "n/a")),
        "SoftwareVersions": "Percept device %s; programmer %s" % (d["DeviceInformation"]["Final"].get("DeviceVersion", "n/a"), d.get("ProgrammerVersion", "n/a")),
        "RecordingType": "continuous", "RecordingDuration": round(X.shape[1] / fs, 6),
        "DBSChannelCount": len(names), "ECOGChannelCount": 0, "SEEGChannelCount": 0,
        "ElectricalStimulationParameters": "Release documentation: all recordings in medication OFF and stimulation OFF. Percept session stimulation status in this export: initial %s, final %s." % (si, sf),
    }
    if si == "OFF" and sf == "OFF":
        js["ElectricalStimulation"] = False
    wjson(os.path.join(D, stem + "_ieeg.json"), js)
    rep["ieeg"].append(dict(src=os.path.relpath(f, S), stem=stem, ch=names, n=int(X.shape[1]), fs=fs, max_abs_err=err, rt=bool(rt)))
    print("IEEG", stem, names, X.shape, "rt", rt, flush=True)

# electrodes (no coordinates released; contact levels + stated target per lead)
for sub in sorted({r["stem"].split("_")[0] for r in rep["ieeg"]}):
    D = os.path.join(OUT, sub, "ieeg")
    rows = []
    for hemi in ("LEFT", "RIGHT"):
        model, loc = leads[sub].get(hemi, ("n/a", "n/a"))
        for k, lev in enumerate(("ZERO", "ONE", "TWO", "THREE")):
            rows.append([f"{hemi}_{lev}", "n/a", "n/a", "n/a", "n/a", hemi, model, "subthalamic nucleus (Percept LeadLocation '%s'; release: electrode contacts reside within the STN)" % loc])
    wtsv(os.path.join(D, f"{sub}_space-Other_electrodes.tsv"), ["name", "x", "y", "z", "size", "group", "lead_model", "region_stated"], rows)
    wjson(os.path.join(D, f"{sub}_space-Other_electrodes.json"), {
        "name": {"Description": "Lead (LEFT/RIGHT) and contact level ZERO..THREE as used in the Percept channel labels"},
        "lead_model": {"Description": "Medtronic lead model from the Percept LeadConfiguration (3387, 3389, B33005/B33015 SenSight)"},
        "region_stated": {"Description": "Target region stated by the device export and the release documentation; not a localisation"}})
    wjson(os.path.join(D, f"{sub}_space-Other_coordsystem.json"), {"iEEGCoordinateSystem": "Other", "iEEGCoordinateUnits": "n/a",
                                                                  "iEEGCoordinateSystemDescription": "No contact coordinates or imaging are released; contacts listed with x, y, z = n/a."})

# ------------------------------------------------------------------ EEG (BrainVision)
EMG = re.compile(r"^(TA|GA)_[LR]\d$")
for f in sorted(glob.glob(os.path.join(S, "sub-*", "ses-*", "eeg", "*.vhdr"))):
    sub, cond = f.split(os.sep)[-4], f.split(os.sep)[-3][4:]
    stem, task = stem_of(sub, cond)
    D = os.path.join(OUT, sub, "eeg"); os.makedirs(D, exist_ok=True)
    src_eeg, src_vmrk = f[:-5] + ".eeg", f[:-5] + ".vmrk"
    hdr = open(f, encoding="utf-8", errors="surrogateescape").read()
    dst = os.path.join(D, stem + "_eeg")
    hdr2 = re.sub(r"(?m)^DataFile=.*$", f"DataFile={stem}_eeg.eeg", hdr)
    hdr2 = re.sub(r"(?m)^MarkerFile=.*$", f"MarkerFile={stem}_eeg.vmrk", hdr2)
    hdr2, nd = scrub_text_dates(hdr2); rep["privacy"]["vhdr_dates"] += nd
    open(dst + ".vhdr", "w", encoding="utf-8", errors="surrogateescape").write(hdr2)
    mk = open(src_vmrk, encoding="utf-8", errors="surrogateescape").read()
    mk2 = re.sub(r"(?m)^DataFile=.*$", f"DataFile={stem}_eeg.eeg", mk)
    mk2, nd = scrub_text_dates(mk2); rep["privacy"]["vmrk_dates"] += nd
    assert not BV_BAD.search(mk2) and not ISO_BAD.search(mk2)
    open(dst + ".vmrk", "w", encoding="utf-8", errors="surrogateescape").write(mk2)
    shutil.copyfile(src_eeg, dst + ".eeg")
    same = sha256_file(src_eeg) == sha256_file(dst + ".eeg")
    nch = int(re.search(r"NumberOfChannels=(\d+)", hdr).group(1)); fs = 1e6 / float(re.search(r"SamplingInterval=(\S+)", hdr).group(1))
    chans = re.findall(r"(?m)^Ch\d+=([^,]*),([^,]*),([^,]*),(.*)$", hdr)
    names = [c[0] for c in chans]; res = [float(c[2]) for c in chans]
    assert re.search(r"BinaryFormat=INT_16", hdr) and len(names) == nch
    ints = np.fromfile(dst + ".eeg", dtype="<i2").reshape(-1, nch).T
    raw = mne.io.read_raw_brainvision(dst + ".vhdr", preload=True)
    exp = ints.astype(np.float64) * np.array(res)[:, None]
    rt = bool(same and raw.ch_names == names and raw.n_times == ints.shape[1] and np.allclose(raw.get_data() * 1e6, exp, rtol=0, atol=1e-6))
    typ = ["EMG" if EMG.match(n) else ("MISC" if n.startswith("IPG") else "EEG") for n in names]
    # amplifier settings from the [Comment] channel table, if present
    tab = re.findall(r"(?m)^\d+\s+(\S+)\s+\d+\s+[\d.]+ µV\s+(\S+)\s+(\S+)\s+(\S+)", hdr)
    hw = "n/a"
    if tab and len({t[1:] for t in tab}) == 1:
        lc, hc, notch = tab[0][1:]
        hw = {"Amplifier": {"LowCutoffTimeConstant_s": lc, "HighCutoff_Hz": hc, "Notch": notch,
                            "Description": "BrainVision Recorder amplifier setup listed in the vhdr [Comment] section"}}
    desc = {"EEG": "scalp EEG", "EMG": "surface EMG channel as labelled in the header (TA/GA, left/right; abbreviations not expanded in the release)",
            "MISC": "channel labelled IPG in the header (the release aligns EEG, LFP and IMU on the stimulation artefact at the start of each recording)"}
    wtsv(dst.replace("_eeg", "_channels") + ".tsv", ["name", "type", "units", "description", "status"],
         [[n, t, "µV", desc[t], "good"] for n, t in zip(names, typ)])
    marks = re.findall(r"(?m)^Mk\d+=([^,]*),([^,]*),(\d+),(\d+),(\d+)", mk2)
    ev = [[round((int(p) - 1) / fs, 6), 0 if int(sz) <= 1 else round(int(sz) / fs, 6), t, v.replace("\\1", ","), int(p) - 1] for t, v, p, sz, _ in marks if t != "New Segment"]
    wtsv(dst.replace("_eeg", "_events") + ".tsv", ["onset", "duration", "trial_type", "value", "sample"], ev)
    wjson(dst + ".json", {
        "TaskName": task, "TaskDescription": TASKDESC.get(task, "n/a"),
        "SamplingFrequency": fs, "PowerLineFrequency": 50, "SoftwareFilters": "n/a", "HardwareFilters": hw,
        "EEGReference": "n/a (not stated in the release)", "Manufacturer": "Brain Products",
        "SoftwareVersions": (re.search(r"BrainVision Recorder[^\n]*", hdr).group(0).strip() if "BrainVision Recorder" in hdr else "n/a"),
        "RecordingType": "continuous", "RecordingDuration": round(ints.shape[1] / fs, 6),
        "EEGChannelCount": typ.count("EEG"), "EMGChannelCount": typ.count("EMG"), "MiscChannelCount": typ.count("MISC"),
        "EOGChannelCount": 0, "ECGChannelCount": 0,
    })
    rep["eeg"].append(dict(src=os.path.relpath(f, S), stem=stem, old_datafile=re.search(r"DataFile=(.*)", hdr).group(1).strip(), nch=nch,
                           n=int(ints.shape[1]), events=len(ev), bytes_identical=same, rt=rt))
    print("EEG", stem, nch, ints.shape[1], "ev", len(ev), "rt", rt, flush=True)

# ------------------------------------------------------------------ motion (APDM Opal HDF5)
for f in sorted(glob.glob(os.path.join(S, "sub-*", "ses-*", "motion", "*.h5"))):
    sub, cond = f.split(os.sep)[-4], f.split(os.sep)[-3][4:]
    stem, task = stem_of(sub, cond)
    mstem = stem.replace(f"_task-{task}", f"_task-{task}_tracksys-APDMOpal")  # entity order: task, tracksys, run
    D = os.path.join(OUT, sub, "motion"); os.makedirs(D, exist_ok=True)
    with h5py.File(f, "r") as h:
        cases = [c.decode() for c in h.attrs["CaseIdList"]]; labels = [c.decode() for c in h.attrs["MonitorLabelList"]]
        lab = dict(zip(cases, labels))
        sens = [k for k in h.keys() if k != "Annotations"]
        srs = {int(h[k].attrs["SampleRate"]) for k in sens}; assert len(srs) == 1
        fs = float(srs.pop())
        n = {h[k]["Time"].shape[0] for k in sens}; assert len(n) == 1, f"{f}: unequal sensor lengths {n}"
        cols, rows = [], []
        for k in sorted(sens, key=lambda k: lab[k.lstrip("/")]):
            tp = lab[k].replace(" ", "")
            C = h[k]["Calibrated"]
            for q, typ, comps, unit in (("Accelerometers", "ACCEL", "xyz", "m/s^2"), ("Gyroscopes", "GYRO", "xyz", "rad/s"),
                                        ("Magnetometers", "MAGN", "xyz", "µT"), ("Orientation", "ORNT", "1234", "n/a")):
                A = C[q][()]
                assert A.shape[1] == len(comps)
                for j, c in enumerate(comps):
                    nm = f"{tp}_{typ}_{c}" if typ != "ORNT" else f"{tp}_ORNT_q{c}"
                    cols.append(A[:, j]); rows.append([nm, c if typ != "ORNT" else "n/a", typ, tp, unit, fs,
                                                       f"APDM Opal {k} ({lab[k]}) Calibrated/{q}" + (" column %s of the stored quaternion (order as stored)" % c if typ == "ORNT" else "")])
            cols.append(C["Temperature"][()]); rows.append([f"{tp}_TEMP", "n/a", "MISC", tp, "°C", fs, f"APDM Opal {k} ({lab[k]}) Calibrated/Temperature"])
        M = np.column_stack(cols)
    p = os.path.join(D, mstem + "_motion.tsv")
    with open(p, "w") as fh:
        for r in M:
            fh.write("\t".join("n/a" if v != v else repr(float(v)) for v in r) + "\n")
    back = np.array([[np.nan if v == "n/a" else float(v) for v in ln.rstrip("\n").split("\t")] for ln in open(p)])
    rt = bool(back.shape == M.shape and np.array_equal(back, M, equal_nan=True))
    wtsv(os.path.join(D, mstem + "_channels.tsv"), ["name", "component", "type", "tracked_point", "units", "sampling_frequency", "description"], rows)
    types = [r[2] for r in rows]
    wjson(os.path.join(D, mstem + "_motion.json"), {
        "TaskName": task, "TaskDescription": TASKDESC.get(task, "n/a"), "SamplingFrequency": fs,
        "TrackingSystemName": "APDM Opal", "Manufacturer": "APDM", "ManufacturersModelName": "Opal",
        "RecordingType": "continuous", "RecordingDuration": round(M.shape[0] / fs, 6),
        "MotionChannelCount": len(rows), "ACCELChannelCount": types.count("ACCEL"), "GYROChannelCount": types.count("GYRO"),
        "MAGNChannelCount": types.count("MAGN"), "ORNTChannelCount": types.count("ORNT"), "MISCChannelCount": types.count("MISC"),
        "TrackedPointsCount": len(sens),
        "SoftwareFilters": "n/a",
    })
    rep["motion"].append(dict(src=os.path.relpath(f, S), stem=stem, n=int(M.shape[0]), ncol=int(M.shape[1]), sensors=labels, rt=rt))
    print("MOTION", stem, M.shape, "rt", rt, flush=True)

# ------------------------------------------------------------------ sourcedata (scrubbed copy of the whole release)
for root, _, files in os.walk(S):
    for fn in files:
        src = os.path.join(root, fn); rel = os.path.relpath(src, S); dst = os.path.join(SD, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        ext = os.path.splitext(fn)[1].lower(); act = "copy"
        if ext == ".json":
            t, nd, ng = scrub_percept(open(src, encoding="utf-8", errors="surrogateescape").read())
            open(dst, "w", encoding="utf-8", errors="surrogateescape").write(t); act = f"percept_scrub dates={nd} groupnames={ng}"
            rep["privacy"]["percept_dates"] += nd; rep["privacy"]["percept_groupnames"] += ng
        elif ext in (".vmrk", ".vhdr"):
            t, nd = scrub_text_dates(open(src, encoding="utf-8", errors="surrogateescape").read())
            assert not BV_BAD.search(t) and not ISO_BAD.search(t)
            open(dst, "w", encoding="utf-8", errors="surrogateescape").write(t); act = f"text_scrub dates={nd}"
            rep["privacy"]["brainvision_text_dates"] += nd
        elif ext == ".h5":
            sh = scrub_h5(src, dst); act = f"h5_time_shift_days={sh}"
            rep["privacy"]["h5_shifted_files"] += int(sh > 0)
        elif ext == ".mat":
            hdr, nd, nnum = scrub_mat(src, dst); act = f"mat_scrub header={hdr} dates={nd} numeric_equal={nnum}"
            rep["privacy"]["mat_header_dates"] += int(hdr); rep["privacy"]["mat_embedded_dates"] += nd
        else:
            shutil.copyfile(src, dst)
        rep["sourcedata"].append({"file": rel, "action": act, "sha256_src": sha256_file(src)})
for fn in ("mendeley_dataset.json", "mendeley_files.json", "mendeley_folders.json"):
    p = os.path.join(W, "source_record", fn)
    if os.path.exists(p):
        os.makedirs(os.path.join(OUT, "sourcedata", "mendeley_record"), exist_ok=True)
        shutil.copyfile(p, os.path.join(OUT, "sourcedata", "mendeley_record", fn))
# final binary check over sourcedata text-like files
for root, _, files in os.walk(os.path.join(OUT, "sourcedata")):
    for fn in files:
        if fn.endswith((".json", ".vhdr", ".vmrk")) and "mendeley_record" not in root:
            t = open(os.path.join(root, fn), encoding="utf-8", errors="replace").read()
            assert not ISO_BAD.search(t) and not BV_BAD.search(t) and "Hormozi" not in t, fn

# ------------------------------------------------------------------ top level
subs = sorted({r["stem"].split("_")[0] for k in ("ieeg", "eeg", "motion") for r in rep[k]})
wtsv(os.path.join(OUT, "participants.tsv"), ["participant_id", "sex", "diagnosis"],
     [[s, (list(sex[s])[0] if len(sex[s]) == 1 else "n/a"), "Parkinson's disease"] for s in subs])
wjson(os.path.join(OUT, "participants.json"), {
    "sex": {"Description": "PatientGender field of the subject's Medtronic Percept exports (n/a where the device field is UNKNOWN or absent)", "Levels": {"M": "male", "F": "female"}},
    "diagnosis": {"Description": "Release documentation: idiopathic Parkinson's disease, bilateral STN-DBS with Medtronic Percept"}})
for task in sorted({r["stem"].split("task-")[1].split("_")[0] for k in ("eeg",) for r in rep[k]}):
    wjson(os.path.join(OUT, f"task-{task}_events.json"), {
        "trial_type": {"Description": "BrainVision marker type as recorded (e.g. Toggle, Moblab, Comment)"},
        "value": {"Description": "BrainVision marker description as recorded (e.g. 'T  1_on', 'M  1'); the release does not document their meaning"},
        "sample": {"Description": "0-based sample index at 1000 Hz"}})
os.makedirs(os.path.join(OUT, "code"), exist_ok=True)
for fn in (os.path.abspath(__file__), os.path.join(os.path.dirname(os.path.abspath(__file__)), "b3w3_common.py")):
    shutil.copy(fn, os.path.join(OUT, "code", os.path.basename(fn)))
rep["privacy"] = dict(rep["privacy"])
json.dump(rep, open(os.path.join(OUT, "code", "conversion_report.json"), "w"), indent=1)
rec = json.load(open(os.path.join(W, "source_record", "mendeley_dataset.json")))
AUTH = ["%s %s" % (c["first_name"], c["last_name"]) for c in rec["contributors"]]
h_ieeg = sum(r["n"] / r["fs"] for r in rep["ieeg"]) / 3600
h_eeg = sum(r["n"] / 1000.0 for r in rep["eeg"]) / 3600
wjson(os.path.join(OUT, "dataset_description.json"), {
    "Name": "SenseFOG: subthalamic LFP (Medtronic Percept), EEG/EMG and IMU kinematics during sitting, standing, walking, stopping and freezing of gait in Parkinson's disease",
    "BIDSVersion": "1.10.0", "DatasetType": "raw", "License": "CC-BY-4.0", "Authors": AUTH,
    "HowToAcknowledge": "Cite Klocke P, Loeffler M, Weiss D (2024), 'Supraspinal circuits contribute to the failure of antagonistic inhibition and freezing of gait in Parkinson's disease', Mendeley Data, V1, doi:10.17632/c9ckcvjxc7.1, and the associated article.",
    "ReferencesAndLinks": ["https://doi.org/10.17632/c9ckcvjxc7.1", "https://github.com/PKlocke/SenseFOG-Documentation"],
    "EthicsApprovals": ["Ethics Committee of Tübingen University (166/2020BO1), as stated in the release documentation"],
    "Keywords": ["Parkinson's disease", "freezing of gait", "subthalamic nucleus", "local field potential", "deep brain stimulation", "Medtronic Percept", "EEG", "EMG", "IMU", "gait"],
    "GeneratedBy": [{"Name": "b3w11_convert_sensefog.py", "Description": "Repackaging of the Mendeley release into BIDS with unchanged signal values; scrubbed originals in sourcedata/. Code in code/."}],
})
open(os.path.join(OUT, "README"), "w").write(f"""# SenseFOG: STN LFP, EEG/EMG and IMU during gait and freezing of gait in Parkinson's disease

Patients with idiopathic Parkinson's disease and bilateral subthalamic (STN) DBS with a Medtronic Percept PC
implantable pulse generator, recorded at the University Hospital Tübingen (Department of Neurology) while sitting,
standing, walking through a narrow corridor, walking with self-initiated stops and walking with a dual task
(serial subtraction), after overnight withdrawal of dopaminergic medication and with stimulation OFF (release
documentation). Signals: bipolar STN local field potentials streamed by the Percept (BrainSense, 250 Hz), 48-channel
scalp EEG with 8 surface EMG channels and 2 'IPG' channels (Brain Products, 1000 Hz), and three APDM Opal inertial
sensors (lumbar spine and both legs/feet, 128 Hz).

## Source
- {', '.join(AUTH)} (2024). "{rec['name']}". Mendeley Data, V1, doi:10.17632/c9ckcvjxc7.1. Licence CC BY 4.0.
  All 277 files were downloaded and matched the SHA-256 checksums listed by Mendeley.
- Documentation and analysis code: https://github.com/PKlocke/SenseFOG-Documentation (study design, inclusion
  criteria, task descriptions and the authors' hard-coded walking/stop/freezing time points live there).
- Ethics: Ethics Committee of Tübingen University (166/2020BO1).

## Contents
- {len(subs)} participants (`sub-XX` = release subject id).
- `ieeg/`: {len(rep['ieeg'])} Percept BrainSense TimeDomain recordings ({h_ieeg:.2f} h), 2 bipolar channels each, named
  with the Percept stream label (e.g. `ONE_THREE_LEFT` = contact levels 1 and 3 of the left lead). Values are the
  Percept `TimeDomainData` unchanged (float32 BrainVision); Percept JSON exports store these in µV. Missing packets
  are not filled in; the Percept packet sequence numbers are in the sourcedata JSON. No contact coordinates are
  released: `electrodes.tsv` lists the contact levels with x/y/z = n/a, the lead model and the stated target (STN).
- `eeg/`: {len(rep['eeg'])} BrainVision recordings ({h_eeg:.2f} h). The `.eeg` files are byte-identical to the release;
  the headers were rewritten only to point at the BIDS file names (the released headers named the pre-rename files,
  e.g. `PD01_gehen.eeg`). Channel types: EEG, EMG (TA_/GA_), MISC (IPG_1/IPG_2). `events.tsv` lists the recorded markers.
- `motion/`: {len(rep['motion'])} APDM Opal recordings: the 'Calibrated' accelerometer, gyroscope, magnetometer,
  orientation and temperature arrays of each sensor, written exactly (17 significant digits). Tracked points use the
  sensor labels stored in each file (Lumbar, Left/Right Leg or Foot).
- Task labels: the release's `ses-<condition>` folders are conditions of one visit and are BIDS tasks here:
  sitting, standing, walk, walkws (walking with stops), walkint (dual task; sub-14 has two: run-1, run-2).
- Not all modalities exist for every condition. Not converted (kept in sourcedata): the authors' pre-aligned files
  `*_lfpalg.mat`, `*_eegalg.mat`, `*_gaitalg.mat` (sub-10 all conditions, ECG-artefact-cleaned; sub-05 standing;
  sub-11 and sub-17 sitting LFP), the gait-event files `*_gaitfilt_cor.mat`, and the Percept export of sub-11
  sitting (no time-domain stream). EEG, LFP and IMU are not resampled or aligned to each other here; the release
  aligns them on the stimulation artefact at the start of each recording.

## Privacy
- Percept exports: every date (session, device clock, implant, date of birth, event logs, packet times) set to day
  01 of its month (year, month and time of day kept); free-text therapy group names removed. Patient name fields
  hold only placeholders in the release (e.g. 'NAME, FIRST', 'MUSTERMANN, MAX'); serial numbers are masked.
- BrainVision marker files: 'New Segment' recording date set to day 01. APDM HDF5 copies: the absolute time vectors
  (Time, SyncValue, Annotations) shifted back by whole days so they fall on day 01 of the month; sensor data unchanged.
- MATLAB files: dates embedded in the authors' structures and in the MAT header set to day 01; numeric contents verified equal.
- BIDS files carry no dates. No scans.tsv acquisition times.
""")
open(os.path.join(OUT, "CHANGES"), "w").write("1.0.0 2026-10-08\n  - BIDS conversion of Mendeley Data doi:10.17632/c9ckcvjxc7.1 for NEMAR.\n")
open(os.path.join(OUT, ".bidsignore"), "w").write("code/\n")
rt_all = all(r["rt"] for k in ("ieeg", "eeg", "motion") for r in rep[k])
print(json.dumps({"ieeg": len(rep["ieeg"]), "eeg": len(rep["eeg"]), "motion": len(rep["motion"]), "subjects": subs, "hours_ieeg": h_ieeg,
                  "not_converted": len(rep["not_converted"]), "privacy": rep["privacy"], "rt_all": rt_all}))
print("CONVERT_DONE")
