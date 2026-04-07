from dotenv import load_dotenv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache


# --------------------------------------------------
# 1. Cache setup
# --------------------------------------------------
load_dotenv()

CACHE_PATH = os.getenv("LFP_DATA_PATH")
if CACHE_PATH is None:
    raise ValueError("LFP_DATA_PATH is not set in your .env file")

manifest_path = os.path.join(CACHE_PATH, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)


# --------------------------------------------------
# 2. Helper: find a session with natural_scenes
# --------------------------------------------------
def find_session_with_natural_scenes(cache, max_sessions=None, verbose=True):
    """
    Loop through sessions and return the first one that contains
    stimulus_name == 'natural_scenes' and has at least one probe.

    Returns
    -------
    dict with:
        session_id
        session
        natural_scenes_table
        durations
    """
    sessions = cache.get_session_table()
    session_ids = list(sessions.index.values)

    if max_sessions is not None:
        session_ids = session_ids[:max_sessions]

    for session_id in session_ids:
        try:
            if verbose:
                print(f"Checking session {session_id} ...")

            session = cache.get_session_data(session_id)
            stim = session.stimulus_presentations

            ns_table = stim[stim["stimulus_name"] == "natural_scenes"].copy()

            if ns_table.empty:
                continue

            # Real durations from the table
            if "duration" in ns_table.columns:
                durations = ns_table["duration"].dropna().values
            else:
                durations = (ns_table["stop_time"] - ns_table["start_time"]).dropna().values

            if len(durations) == 0:
                continue

            return {
                "session_id": session_id,
                "session": session,
                "natural_scenes_table": ns_table,
                "durations": durations,
            }

        except Exception as e:
            print(f"Skipping session {session_id} because of error: {e}")

    raise RuntimeError("No session with natural_scenes was found.")


# --------------------------------------------------
# 3. Helper: choose an image identity column
# --------------------------------------------------
def choose_image_id_column(ns_table: pd.DataFrame) -> str:
    """
    Try to find the column that identifies which natural scene image was shown.
    """
    candidates = [
        "frame",
        "image_id",
        "stimulus_condition_id",
        "image_name",
    ]

    for col in candidates:
        if col in ns_table.columns:
            return col

    raise ValueError(
        "Could not find an image identity column. "
        f"Available columns are: {list(ns_table.columns)}"
    )


# --------------------------------------------------
# 4. Helper: average LFP by image across repeated trials
# --------------------------------------------------
def average_lfp_by_image(
    session,
    probe_name="probeA",
    stimulus_name="natural_scenes",
    pre_time=0.05,
    post_time=None,
    fs=1250.0,
    drop_blank=True,
):
    """
    Align LFP to natural scene onset and average across repeats of the same image.

    Parameters
    ----------
    pre_time : float
        Seconds before stimulus onset.
    post_time : float or None
        Seconds after onset. If None, infer from median presentation duration.
    fs : float
        Sampling rate for alignment grid. Neuropixels LFP in NWB is decimated;
        cheat sheet notes NWB LFP includes every 2nd sample from ~2.5 kHz,
        so ~1250 Hz is a reasonable target grid. :contentReference[oaicite:1]{index=1}

    Returns
    -------
    result : dict
        Contains:
            session_id
            probe_id
            probe_name
            image_id_column
            duration_summary
            aligned_lfp_trials   (xarray.DataArray)
            mean_lfp_by_image    (xarray.DataArray)
            natural_scenes_table (DataFrame)
    """
    stim = session.stimulus_presentations
    ns_table = stim[stim["stimulus_name"] == stimulus_name].copy()

    if ns_table.empty:
        raise ValueError(f"No rows found for stimulus_name='{stimulus_name}'")

    # Compute duration from table
    if "duration" in ns_table.columns:
        ns_table["computed_duration"] = ns_table["duration"]
    else:
        ns_table["computed_duration"] = ns_table["stop_time"] - ns_table["start_time"]

    duration_summary = ns_table["computed_duration"].describe()
    median_duration = float(ns_table["computed_duration"].median())

    if post_time is None:
        post_time = median_duration

    image_id_col = choose_image_id_column(ns_table)

    # Optionally remove blank presentations
    if drop_blank:
        if image_id_col == "frame":
            ns_table = ns_table[ns_table["frame"] >= 0].copy()
        elif "is_blank" in ns_table.columns:
            ns_table = ns_table[~ns_table["is_blank"]].copy()

    if ns_table.empty:
        raise ValueError("After blank filtering, no natural_scenes rows remain.")

    # Probe lookup
    probe_matches = session.probes[session.probes["description"] == probe_name]
    if probe_matches.empty:
        raise ValueError(
            f"Probe '{probe_name}' not found. "
            f"Available: {list(session.probes['description'].values)}"
        )

    probe_id = probe_matches.index.values[0]
    lfp = session.get_lfp(probe_id)  # dims typically include time, channel

    # Alignment grid
    rel_time = np.arange(-pre_time, post_time, 1.0 / fs)

    trial_arrays = []
    trial_meta = []

    for presentation_id, row in ns_table.iterrows():
        onset = float(row["start_time"])
        image_id = row[image_id_col]

        # absolute times for this trial
        abs_times = onset + rel_time

        # select nearest LFP samples
        trial = lfp.sel(time=abs_times, method="nearest")

        # rename absolute time axis to relative time for stacking
        trial = trial.assign_coords(time=rel_time)

        trial_arrays.append(trial)
        trial_meta.append(
            {
                "presentation_id": presentation_id,
                "image_id": image_id,
                "start_time": onset,
                "duration": float(row["computed_duration"]),
            }
        )

    # stack into one xarray
    aligned = xr.concat(trial_arrays, dim="presentation_id")
    aligned = aligned.assign_coords(
        presentation_id=[m["presentation_id"] for m in trial_meta],
        image_id=("presentation_id", [m["image_id"] for m in trial_meta]),
        start_time=("presentation_id", [m["start_time"] for m in trial_meta]),
        stimulus_duration=("presentation_id", [m["duration"] for m in trial_meta]),
    )

    # Average across repeated presentations of the same image
    mean_by_image = aligned.groupby("image_id").mean(dim="presentation_id")

    return {
        "probe_id": probe_id,
        "probe_name": probe_name,
        "image_id_column": image_id_col,
        "duration_summary": duration_summary,
        "aligned_lfp_trials": aligned,
        "mean_lfp_by_image": mean_by_image,
        "natural_scenes_table": ns_table,
    }


# --------------------------------------------------
# 5. Run it
# --------------------------------------------------
found = find_session_with_natural_scenes(cache, verbose=True)

session_id = found["session_id"]
session = found["session"]
ns_table = found["natural_scenes_table"]

print("\nFound session with natural_scenes:")
print("session_id:", session_id)

print("\nNatural scenes duration summary:")
print(pd.Series(found["durations"]).describe())

print("\nAvailable probes:")
print(session.probes["description"].tolist())

result = average_lfp_by_image(
    session=session,
    probe_name="probeA",   # change if needed
    stimulus_name="natural_scenes",
    pre_time=0.05,
    post_time=None,        # inferred from median duration
    fs=1250.0,
    drop_blank=True,
)

print("\nImage identity column:", result["image_id_column"])
print("\nDuration summary from selected rows:")
print(result["duration_summary"])

print("\nAligned trial LFP:")
print(result["aligned_lfp_trials"])

print("\nMean LFP by image:")
print(result["mean_lfp_by_image"])

print(result["mean_lfp_by_image"].values.shape)
np.save("/home/maria/LFPExperiments/data/mean_lfp_by_image.npy", result["mean_lfp_by_image"].values)