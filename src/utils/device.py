import torch


def get_device(preferred: str | None = None) -> str:
    """Select and return an available compute device.

    This helper chooses a valid PyTorch device string among:
    `"cuda"`, `"mps"`, or `"cpu"`.

    Selection logic:
      - If `preferred` is provided and is one of {"cuda", "mps", "cpu"},
        that device is tried first and falls back to auto-detection if
        unavailable.
      - If `preferred` is `"auto"` or None, devices are auto-detected
        in priority order: CUDA > MPS > CPU.

    Args:
        preferred: Optional preferred device identifier. Supported values
            are `"cuda"`, `"mps"`, `"cpu"`, `"auto"`, or None.

    Returns:
        A string identifying the selected device: `"cuda"`, `"mps"`, or `"cpu"`.
    """
    if preferred is not None:
        preferred = preferred.lower()

    if preferred == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        else:
            print("[device] CUDA requested but not available. Falling back to auto.")
    elif preferred == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        else:
            print("[device] MPS requested but not available. Falling back to auto.")
    elif preferred == "cpu":
        return "cpu"

    if torch.cuda.is_available():
        print("[device] Using CUDA")
        return "cuda"
    if torch.backends.mps.is_available():
        print("[device] Using MPS")
        return "mps"

    print("[device] Using CPU")
    return "cpu"
