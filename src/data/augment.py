"""Training-time input masking: hide extra readings in each batch so the model learns to cope with gaps.

Only training inputs are touched; targets, validation and test data stay as they are.
"""
import torch


def sample_drop(shape: tuple[int, int, int], spec: dict, device) -> torch.Tensor:
    """Readings to hide, [batch, steps, sensors]. Each window draws its own amounts.

    spec["point_max"]:   share of single readings hidden, uniform in [0, point_max]
    spec["sensor_prob"]: chance a window also loses whole sensors
    spec["sensor_max"]:  share of sensors switched off in such a window, uniform in [0, sensor_max]
    """
    b, t, n = shape
    rate = torch.rand(b, 1, 1, device=device) * spec.get("point_max", 0.0)
    drop = torch.rand(b, t, n, device=device) < rate
    if spec.get("sensor_prob", 0.0) > 0:
        hit = torch.rand(b, 1, 1, device=device) < spec["sensor_prob"]
        share = torch.rand(b, 1, 1, device=device) * spec.get("sensor_max", 0.0)
        drop |= hit & (torch.rand(b, 1, n, device=device) < share)
    return drop


def hide_inputs(x: torch.Tensor, drop: torch.Tensor, imputation: str) -> torch.Tensor:
    """Re-fill normalized inputs x [batch, steps, sensors, C] after hiding `drop`, the same way as at test time.

    'mean' sets hidden readings to the train mean (0 after scaling). 'ffill' carries the last kept value
    in the window forward; with nothing earlier in the window it falls back to the mean.
    If x has a missing-indicator channel (C=2), hidden readings are marked 0 there.
    """
    v = x[..., 0]
    if imputation == "mean":
        v = v.masked_fill(drop, 0.0)
    elif imputation == "ffill":
        steps = torch.arange(v.shape[1], device=v.device).view(1, -1, 1).expand_as(v)
        last = torch.where(drop, -1, steps).cummax(dim=1).values
        v = torch.where(last >= 0, v.gather(1, last.clamp(min=0)), 0.0)
    else:
        raise ValueError(f"unknown input_imputation {imputation!r}")
    channels = [v]
    if x.shape[-1] == 2:
        channels.append(x[..., 1].masked_fill(drop, 0.0))
    return torch.stack(channels, dim=-1)
