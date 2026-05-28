import numpy as np
import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

manifest_path = "/media/maria/notsudata/AllenNeuropixels/manifest.json"
session_id = 739448407
choose_probe = "probeA"

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
session = cache.get_session_data(session_id)

probe_id = session.probes[
    session.probes.description == choose_probe
].index.values[0]

fs = float(session.probes.loc[probe_id, "lfp_sampling_rate"])
print("LFP sampling rate:", fs)

presentations = session.stimulus_presentations[
    session.stimulus_presentations.stimulus_name == "natural_scenes"
].copy()

presentations["frame"] = pd.to_numeric(presentations["frame"], errors="coerce")
presentations["start_time"] = pd.to_numeric(presentations["start_time"], errors="coerce")

if "stop_time" in presentations.columns:
    presentations["stop_time"] = pd.to_numeric(presentations["stop_time"], errors="coerce")
    presentations["_stim_end_time"] = presentations["stop_time"]
elif "duration" in presentations.columns:
    presentations["duration"] = pd.to_numeric(presentations["duration"], errors="coerce")
    presentations["_stim_end_time"] = presentations["start_time"] + presentations["duration"]
else:
    raise ValueError("No stop_time or duration column found")

presentations = presentations.dropna(
    subset=["frame", "start_time", "_stim_end_time"]
).copy()

presentations["frame"] = presentations["frame"].astype(int)

# Remove blank/empty stimulus.
presentations = presentations[presentations["frame"] >= 0].copy()

presentations["_stim_duration"] = (
    presentations["_stim_end_time"] - presentations["start_time"]
)

presentations["_lfp_sample_count_float"] = presentations["_stim_duration"] * fs
presentations["_lfp_sample_count_floor"] = np.floor(
    presentations["_lfp_sample_count_float"]
).astype(int)
presentations["_lfp_sample_count_round"] = np.round(
    presentations["_lfp_sample_count_float"]
).astype(int)

print("\nNumber of real natural-scenes presentations:", len(presentations))
print("Unique images:", presentations["frame"].nunique())

print("\nStimulus duration summary, seconds:")
print(presentations["_stim_duration"].describe())

print("\nLFP sample count summary, floor:")
print(presentations["_lfp_sample_count_floor"].describe())

print("\nValue counts for sample count floor:")
print(
    presentations["_lfp_sample_count_floor"]
    .value_counts()
    .sort_index()
)

print("\nValue counts for sample count round:")
print(
    presentations["_lfp_sample_count_round"]
    .value_counts()
    .sort_index()
)