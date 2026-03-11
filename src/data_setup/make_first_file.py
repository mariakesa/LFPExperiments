from dotenv import load_dotenv
import os

import numpy as np
import pandas as pd

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

load_dotenv()
CACHE_PATH=os.getenv("LFP_DATA_PATH")
print(CACHE_PATH)

output_dir=CACHE_PATH

manifest_path = os.path.join(output_dir, "manifest.json")

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

sessions = cache.get_session_table()
session=sessions.index.values[0]
session=831882777
session = cache.get_session_data(session)

print({session.probes.loc[probe_id].description : 
     list(session.channels[session.channels.probe_id == probe_id].ecephys_structure_acronym.unique())
     for probe_id in session.probes.index.values})

choose_probe= 'probeA'

probe_id = session.probes[session.probes.description == choose_probe].index.values[0]
print(probe_id)
lfp = session.get_lfp(probe_id)
print(lfp)