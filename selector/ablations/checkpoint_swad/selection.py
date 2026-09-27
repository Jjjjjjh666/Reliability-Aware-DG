"""Replay the official LossValley queue rules on checkpoint validation losses.

Each checkpoint stands in for one SWAD segment. The official implementation
instead validates a dense average of all training steps in that segment.
"""

import math
from collections import deque


def loss_valley(losses, n_converge=3, n_tolerance=6, tolerance_ratio=0.3):
    if not losses or not all(math.isfinite(value) for value in losses):
        raise ValueError("LossValley needs a nonempty finite loss sequence")
    if n_converge < 1 or n_tolerance < 1:
        raise ValueError("LossValley window sizes must be positive")

    converge_q = deque(maxlen=n_converge)
    smooth_q = deque(maxlen=n_tolerance)
    converge_index = None
    threshold = None
    updates = []
    dead = False

    for index in range(len(losses)):
        if dead:
            break
        converge_q.append(index)
        smooth_q.append(index)

        if converge_index is None:
            if len(converge_q) < n_converge:
                continue
            minimum = min(range(len(converge_q)), key=lambda i: losses[converge_q[i]])
            if minimum != 0:
                continue
            converge_index = converge_q[0]
            threshold = sum(losses[i] for i in converge_q) / n_converge * (1 + tolerance_ratio)

            if n_tolerance < n_converge:
                updates.extend(list(converge_q)[1 : n_converge - n_tolerance + 1])
            elif n_tolerance > n_converge:
                earlier = list(smooth_q)[: n_tolerance - n_converge + 1]
                start_index = 0
                for i in reversed(range(len(earlier))):
                    if losses[earlier[i]] > threshold:
                        start_index = i + 1
                        break
                updates.extend(earlier[start_index + 1 :])
            continue

        if smooth_q[0] < converge_index:
            continue
        if min(losses[i] for i in smooth_q) > threshold:
            dead = True
            break
        updates.append(smooth_q[0])

    if converge_index is None:
        return {
            "indices": [len(losses) - 1], "converge_index": None,
            "threshold": None, "dead_valley": False, "fallback_last": True,
        }

    if not dead:
        smooth_q.popleft()
        while smooth_q:
            if min(losses[i] for i in smooth_q) > threshold:
                break
            updates.append(smooth_q.popleft())

    # AveragedModel is initialized from converge_index with n_averaged=0.
    # Its first update replaces that initialization, so preserve repeated
    # indices and use the initialization only if no update occurs.
    return {
        "indices": updates or [converge_index],
        "converge_index": converge_index,
        "threshold": threshold,
        "dead_valley": dead,
        "fallback_last": False,
    }
