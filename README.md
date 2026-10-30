[![DOI](https://img.shields.io/badge/DOI-10.82901%2Fnemar.nm000434-blue)](https://doi.org/10.82901/nemar.nm000434)

# SenseFOG: STN LFP, EEG/EMG and IMU during gait and freezing of gait in Parkinson's disease

Patients with idiopathic Parkinson's disease and bilateral subthalamic (STN) DBS with a Medtronic Percept PC
implantable pulse generator, recorded at the University Hospital Tübingen (Department of Neurology) while sitting,
standing, walking through a narrow corridor, walking with self-initiated stops and walking with a dual task
(serial subtraction), after overnight withdrawal of dopaminergic medication and with stimulation OFF (release
documentation). Signals: bipolar STN local field potentials streamed by the Percept (BrainSense, 250 Hz), 48-channel
scalp EEG with 8 surface EMG channels and 2 'IPG' channels (Brain Products, 1000 Hz), and three APDM Opal inertial
sensors (lumbar spine and both legs/feet, 128 Hz).

## Source
- Philipp Klocke, Moritz Loeffler, Daniel Weiss (2024). "Supraspinal circuits contribute to the failure of antagonistic inhibition and freezing of gait in Parkinson’s disease". Mendeley Data, V1, doi:10.17632/c9ckcvjxc7.1. Licence CC BY 4.0.
  All 277 files were downloaded and matched the SHA-256 checksums listed by Mendeley.
- Documentation and analysis code: https://github.com/PKlocke/SenseFOG-Documentation (study design, inclusion
  criteria, task descriptions and the authors' hard-coded walking/stop/freezing time points live there).
- Ethics: Ethics Committee of Tübingen University (166/2020BO1).

## Contents
- 12 participants (`sub-XX` = release subject id).
- `ieeg/`: 43 Percept BrainSense TimeDomain recordings (2.50 h), 2 bipolar channels each, named
  with the Percept stream label (e.g. `ONE_THREE_LEFT` = contact levels 1 and 3 of the left lead). Values are the
  Percept `TimeDomainData` unchanged (float32 BrainVision); Percept JSON exports store these in µV. Missing packets
  are not filled in; the Percept packet sequence numbers are in the sourcedata JSON. No contact coordinates are
  released: `electrodes.tsv` lists the contact levels with x/y/z = n/a, the lead model and the stated target (STN).
- `eeg/`: 45 BrainVision recordings (2.73 h). The `.eeg` files are byte-identical to the release;
  the headers were rewritten only to point at the BIDS file names (the released headers named the pre-rename files,
  e.g. `PD01_gehen.eeg`). Channel types: EEG, EMG (TA_/GA_), MISC (IPG_1/IPG_2). `events.tsv` lists the recorded markers.
- `motion/`: 43 APDM Opal recordings: the 'Calibrated' accelerometer, gyroscope, magnetometer,
  orientation and temperature arrays of each sensor, written exactly (17 significant digits). Tracked points use the
  sensor labels stored in each file (Lumbar, Left/Right Leg or Foot).
- Task labels: the release's `ses-<condition>` folders are conditions of one visit and are BIDS tasks here:
  sitting, standing, walk, walkws (walking with stops), walkint (dual task; sub-14 has two: run-1, run-2).
- Not all modalities exist for every condition. Not converted (kept in sourcedata): the authors' pre-aligned files
  `*_lfpalg.mat`, `*_eegalg.mat`, `*_gaitalg.mat` (sub-10 all conditions, ECG-artefact-cleaned; sub-05 standing;
  sub-11 and sub-17 sitting LFP), the gait-event files `*_gaitfilt_cor.mat`, and the Percept export of sub-11
  sitting (no time-domain stream). EEG, LFP and IMU are not resampled or aligned to each other here; the release
  aligns them on the stimulation artefact at the start of each recording.

## Gait events in ieeg/ (Oct 2026)
The walking-task iEEG runs (task walk, walkws, walkint; 24 runs) have an `events.tsv` with the authors' walking, stop and
freezing segments and the heel strike, toe off and midswing times of both feet (22,964 rows: 431 segments,
22,533 gait events; code/fix-20261030_gait_events.py, report code/fix-20261030_gait_events_report.json).
- Sources: segments = `subjectdata.events_filt` hard-coded in the authors' `sub-XX-datafile.m`
  (github.com/PKlocke/SenseFOG-Documentation, commit b87df3f; copied with the artefact samples to
  code/fix-20261030_authors_timepoints.tsv); gait events = sourcedata `*_gaitfilt_cor.mat` (GaitEvents).
- Time base: the authors align LFP, EEG and IMU on the stimulation-offset artefact with two hard-coded samples per run
  (E: EEG at 1 kHz counted from the first `M  1` marker, which marks the IMU start; L: LFP resampled to 1 kHz) and cut the
  earlier-starting stream by |L - E| samples (scripts 1.3-1.6). The onsets here undo that cut, so they are seconds from the
  first sample of the BIDS ieeg run; each run's `events.json` gives E, L and the shift. Checked on the BIDS data: the LFP
  offset artefact lies at (L - 1) / 1000 s and the IPG stimulation train in the EEG ends at the `M  1` sample + E (all 24
  runs, within about 0.2 s, the precision of the manual marks); the authors' midswing times fall on the gyroscope peaks
  of the same foot in the BIDS motion data (median 0.6-1.9 ms).
- Not written: gait events of the left foot of sub-15 task-walk (its midswing times do not match the left sensor on any
  of the tested time bases); 135 rows that fall outside the LFP run; sitting/standing runs (no events in the source);
  sub-10 (no BIDS ieeg run; the release holds only its pre-aligned files). Labels are kept as the authors wrote them:
  their scripts select 'Walk', 'Selected_stop' and 'Freezing_walk' ('Selected stop' and 'Selected_stop_start' occur
  once or twice and are not used by them); one segment (sub-13 walkws, index 34) has an end before its start and gets
  duration n/a.

## Privacy
- Percept exports: every date (session, device clock, implant, date of birth, event logs, packet times) set to day
  01 of its month (year, month and time of day kept); free-text therapy group names removed. Patient name fields
  hold only placeholders in the release (e.g. 'NAME, FIRST', 'MUSTERMANN, MAX'); serial numbers are masked.
- BrainVision marker files: 'New Segment' recording date set to day 01. APDM HDF5 copies: the absolute time vectors
  (Time, SyncValue, Annotations) shifted back by whole days so they fall on day 01 of the month; sensor data unchanged.
- MATLAB files: dates embedded in the authors' structures and in the MAT header set to day 01; numeric contents verified equal.
- BIDS files carry no dates. No scans.tsv acquisition times.
