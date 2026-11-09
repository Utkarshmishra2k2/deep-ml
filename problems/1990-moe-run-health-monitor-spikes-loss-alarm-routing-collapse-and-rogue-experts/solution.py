import numpy as np

def spike_steps(values, num_samples=100, factor=10.0):
    baseline = []
    reference = None
    flagged = []

    for step, value in enumerate(values):
        if not np.isfinite(value):
            continue

        if len(baseline) < num_samples:
            baseline.append(abs(value))
            if len(baseline) == num_samples:
                reference = max(baseline)
            continue

        if reference is not None and abs(value) >= factor * reference:
            flagged.append(step)

    return flagged

def loss_alarm_steps(losses, margin=0.1):
    best = None
    flagged = []

    for step, value in enumerate(losses):
        if not np.isfinite(value):
            continue

        if best is not None and value - best > margin:
            flagged.append(step)

        if best is None or value < best:
            best = value

    return flagged

def maxvio(counts):
    counts = np.asarray(counts)
    mean_count = counts.mean()

    if mean_count == 0:
        return None

    return counts.max() / mean_count - 1

def expert_norm_ratios(norm_sum, counts):
    norm_sum = np.asarray(norm_sum)
    counts = np.asarray(counts)

    active = np.flatnonzero(counts > 0)
    empty = np.flatnonzero(counts == 0).tolist()

    if len(active) == 0:
        return {
            "max_ratio": None, "min_ratio": None,
            "argmax": None, "argmin": None, "empty": empty
        }

    means = norm_sum[active] / counts[active]
    median = np.median(means)

    if median == 0:
        return {
            "max_ratio": None, "min_ratio": None,
            "argmax": None, "argmin": None, "empty": empty
        }

    max_mean = means.max()
    min_mean = means.min()
    argmax = int(active[np.flatnonzero(means == max_mean)[0]])
    argmin = int(active[np.flatnonzero(means == min_mean)[0]])

    return {
        "max_ratio": round(float(max_mean / median), 4),
        "min_ratio": round(float(min_mean / median), 4),
        "argmax": argmax,
        "argmin": argmin,
        "empty": empty,
    }

def health_report(
    loss, grad_norm, counts, norm_sum,
    maxvio_limit=1.0, rogue_ratio=10.0, dead_ratio=0.1,
    margin=0.1, num_samples=100, factor=10.0
):
    loss = np.asarray(loss)
    grad_norm = np.asarray(grad_norm)
    counts = np.asarray(counts)
    norm_sum = np.asarray(norm_sum)

    nonfinite = [
        step for step in range(len(loss))
        if not np.isfinite(loss[step]) or not np.isfinite(grad_norm[step])
    ]

    collapse = []
    rogue = []
    dead = []
    empty = []

    for step in range(counts.shape[0]):
        for layer in range(counts.shape[1]):
            layer_counts = counts[step, layer]
            vio = maxvio(layer_counts)
            ratios = expert_norm_ratios(norm_sum[step, layer], layer_counts)

            if vio is not None and vio > maxvio_limit:
                collapse.append([step, layer])

            if ratios["max_ratio"] is not None and ratios["max_ratio"] > rogue_ratio:
                rogue.append([step, layer, ratios["argmax"]])

            if ratios["min_ratio"] is not None and ratios["min_ratio"] < dead_ratio:
                dead.append([step, layer, ratios["argmin"]])

            for expert in ratios["empty"]:
                empty.append([step, layer, expert])

    return {
        "nonfinite": nonfinite,
        "loss_spikes": spike_steps(loss, num_samples, factor),
        "grad_spikes": spike_steps(grad_norm, num_samples, factor),
        "loss_alarm": loss_alarm_steps(loss, margin),
        "collapse": collapse,
        "rogue": rogue,
        "dead": dead,
        "empty": empty,
    }