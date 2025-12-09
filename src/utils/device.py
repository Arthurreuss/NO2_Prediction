import torch


def get_device(preferred: str | None = None) -> str:
    """
    Returns a valid device string among: 'cuda', 'mps', 'cpu'.

    preferred:
      - 'cuda', 'mps', 'cpu' -> try that first, fall back if unavailable
      - 'auto' or None      -> auto-detect (cuda > mps > cpu)
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
