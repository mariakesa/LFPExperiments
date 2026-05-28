from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

def load_vit_animate_labels(vit_logits_path: str) -> np.ndarray:
    """
    Load ViT logits and derive animate/inanimate labels.

    Returns:
        image_labels: shape (n_images,)
            1 = animate
            0 = inanimate
    """
    vit_file = np.load(vit_logits_path, allow_pickle=True)

    # Your current object:
    # vit = np.load(..., allow_pickle=True)["natural_scenes"]
    vit = vit_file["natural_scenes"]

    if vit.ndim != 2:
        raise ValueError(f"Expected ViT logits with shape (n_images, 1000), got {vit.shape}")

    top1 = np.argmax(vit, axis=1)
    image_labels = (top1 <= 397).astype(np.int64)

    print(f"Loaded ViT logits: {vit.shape}")
    print(f"Image-level label counts: {np.bincount(image_labels)}")
    print("Convention: 0 = inanimate, 1 = animate")

    return image_labels


manifest_path = "/media/maria/notsudata/AllenNeuropixels/manifest.json"
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

sessions = cache.get_session_table()
probes = cache.get_probes()
channels = cache.get_channels()
units = cache.get_units()

session_id: int = 739448407

choose_probe: str = "probeA"

session = cache.get_session_data(session_id)

session.stimulus_presentations.head()

probe_id = session.probes[
        session.probes.description == choose_probe
    ].index.values[0]

lfp = session.get_lfp(probe_id)

session.get_stimulus_epochs()

session.get_stimulus_table(['natural_scenes']).head()

presentation_table = session.stimulus_presentations[session.stimulus_presentations.stimulus_name == 'natural_scenes']

presentation_times = presentation_table.start_time.values
presentation_ids = presentation_table.index.values

session.probes