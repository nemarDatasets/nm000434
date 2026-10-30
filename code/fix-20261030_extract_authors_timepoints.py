"""fix-20261030 (Oct 2026): extract the authors' hard-coded time points from SenseFOG-Documentation into a TSV.

Source: https://github.com/PKlocke/SenseFOG-Documentation (commit b87df3f), files
'Matlab_scripts/A) Subject Files/sub-XX-datafile.m':
  subjectdata.signalpoint.<Task>.EEG_signal / .LFP_signal  sample of the stimulation-offset artefact (EEG after the
      first 'M  1' marker; LFP resampled to 1000 Hz), used by 'B) Preprocessing/1.5_Data_Alignment.m'
  subjectdata.events_filt.<Task>(i).task/.start/.end       activity segments in seconds, used by
      'B) Preprocessing/1.6_Sub_GrandActivity_Log.m'
MATLAB % comments are dropped (commented-out assignments are not used by the authors either); a later assignment to the
same field overrides an earlier one, as in MATLAB.
Usage: python fix-20261030_extract_authors_timepoints.py '<clone>/Matlab_scripts/A) Subject Files' <out.tsv>
"""
import csv, glob, os, re, sys

NUM = r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?"
SP = re.compile(r"^\s*subjectdata\.signalpoint\.(\w+)\.(EEG|LFP)_signal\s*=\s*(" + NUM + r")\s*;", re.M)
EV = re.compile(r"^\s*subjectdata\.events_filt\.(\w+)\((\d+)\)\.(task|start|end)\s*=\s*('([^']*)'|" + NUM + r")\s*;", re.M)


def strip_comments(txt):
    out = []
    for line in txt.splitlines():
        q = False; cut = len(line)
        for i, ch in enumerate(line):
            if ch == "'":
                q = not q
            elif ch == "%" and not q:
                cut = i; break
        out.append(line[:cut])
    return "\n".join(out)


def parse(path):
    txt = strip_comments(open(path, encoding="utf-8", errors="replace").read())
    sp = {}
    for task, kind, val in SP.findall(txt):
        sp.setdefault(task, {})[kind] = int(float(val))
    ev = {}
    for task, idx, field, val, sval in EV.findall(txt):
        d = ev.setdefault(task, {}).setdefault(int(idx), {})
        d[field] = sval if field == "task" else float(val)
    return sp, {t: [dict(index=i, **v) for i, v in sorted(d.items())] for t, d in ev.items()}


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    rows = []
    for p in sorted(glob.glob(os.path.join(src, "sub-*-datafile.m"))):
        sub = os.path.basename(p)[:6]
        sp, ev = parse(p)
        for task, d in sorted(sp.items()):
            rows.append([sub, task, "signalpoint", "n/a", "n/a", "n/a", "n/a", d.get("EEG", "n/a"), d.get("LFP", "n/a")])
        for task, l in sorted(ev.items()):
            for e in l:
                rows.append([sub, task, "events_filt", e["index"], e.get("task", "n/a"), repr(e["start"]) if "start" in e else "n/a",
                             repr(e["end"]) if "end" in e else "n/a", "n/a", "n/a"])
    with open(out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["subject", "datafile_task", "row_type", "index", "label", "start_s", "end_s", "EEG_signal", "LFP_signal"])
        w.writerows(rows)
    print(len(rows), "rows ->", out)
