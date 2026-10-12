"""
Batching of `GradientEngine` iterations: where the measurements live, how a
scan is split into batches, how each batch's frames reach the device, and the
largest batch that fits a CUDA memory budget.
"""

import copy

import torch


def data_device(device, requested, nbytes):
    """
    Return the device on which measurements of `nbytes` bytes are kept.

    Args:
        device (str or torch.device):
            Device of the model.

        requested (str or None):
            The engine's `dataDevice`. A device of the same type as `device`
            means `device` itself.

        nbytes (int):
            Size of the measurements in bytes.

    Returns:
        torch.device:
            `requested` if given, otherwise `device` unless the data would take
            more than half of its free CUDA memory, in which case the host.
    """
    device = torch.device(device)
    if requested is not None:
        requested = torch.device(requested)
        return device if requested.type == device.type else requested
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        if nbytes > 0.5 * free:
            return torch.device("cpu")
    return device


def split_indices(indices, size):
    """
    Split frame indices into consecutive batches of `size`.

    Args:
        indices (torch.Tensor):
            Frame indices of the scan.

        size (int or None):
            Frames per batch; None keeps the whole scan as one batch.

    Returns:
        list:
            Index tensors, one per batch; the last may be shorter.
    """
    return list(torch.split(indices, len(indices) if size is None else int(size)))


class MeasurementLoader:
    """
    Yields the measured frames of index batches on the model's device.

    Device-resident data are indexed directly. Host-resident data for a CUDA
    run are pinned once, gathered into one of two pinned buffers and copied on
    a side stream; the copy of the next batch overlaps the computation on the
    current one.

    Args:
        data (torch.Tensor):
            Measurements of shape `(frames, Np, Np)`, on `device` or the host.

        device (str or torch.device):
            Device of the model.

    Attributes:
        data (torch.Tensor):
            The measurements, pinned if they stay on the host of a CUDA run.
    """

    def __init__(self, data, device):
        self.device = torch.device(device)
        self.onDevice = data.device == torch.empty(0, device=self.device).device
        if not self.onDevice and self.device.type == "cuda":
            data = data.pin_memory()
        self.data = data
        self._buffers = None
        self._stream = None

    def __call__(self, batches):
        """
        Yield the measured frames of each index batch on `device`, in order.

        Args:
            batches (Sequence):
                Index tensors, one per batch.

        Yields:
            torch.Tensor:
                Measured frames of each batch.
        """
        data, device = self.data, self.device
        if self.onDevice:
            for batch in batches:
                yield data[batch]
            return
        host_batches = [batch.cpu() for batch in batches]
        if device.type != "cuda":
            for batch in host_batches:
                yield data[batch].to(device)
            return
        # Two buffers sized for the largest batch: one is filled on the host
        # while the other's copy is still in flight.
        largest = max(len(batch) for batch in host_batches)
        if self._buffers is None or self._buffers[0].shape[0] < largest:
            shape = (largest,) + tuple(data.shape[1:])
            self._buffers = [
                torch.empty(shape, dtype=data.dtype, pin_memory=True) for _ in range(2)
            ]
            self._stream = torch.cuda.Stream(device)
        buffers, stream = self._buffers, self._stream
        copies = [None, None]

        def start(position):
            slot = position % 2
            if copies[slot] is not None:
                copies[slot][1].synchronize()  # the buffer's last copy is done
            host = buffers[slot][: len(host_batches[position])]
            torch.index_select(data, 0, host_batches[position], out=host)
            with torch.cuda.stream(stream):
                copy = host.to(device, non_blocking=True)
                event = torch.cuda.Event()
                event.record(stream)
            copies[slot] = (copy, event)
            return copies[slot]

        pending = start(0)
        for position in range(len(host_batches)):
            copy, event = pending
            if position + 1 < len(host_batches):
                pending = start(position + 1)
            current = torch.cuda.current_stream(device)
            current.wait_event(event)
            # The copy was allocated on the side stream; keep its memory alive
            # until the compute stream is done with it.
            copy.record_stream(current)
            yield copy


def largest_batch_within(budget, frames, peak):
    """
    Return the largest batch size whose measured peak fits a memory budget.

    `peak(size)` is evaluated at batch sizes 1, 2, 4, ... until it exceeds
    `budget` (an out-of-memory error counts as exceeding it) or the batch holds
    all `frames`, then bisected between the last size that fits and the first
    that does not. The returned size was itself measured, because the peak is
    not exactly linear in the batch size (allocator rounding).

    Args:
        budget (float):
            Memory budget in bytes.

        frames (int):
            Frames in the scan, the largest possible batch.

        peak (Callable):
            Peak bytes of one iteration at a given batch size.

    Returns:
        dict:
            `"batch size"`, its `"peak bytes"`, and `"measurements"`, the peak
            of every size tried.

    Raises:
        MemoryError:
            If one frame per batch already exceeds the budget.
    """
    measured = {}

    def fits(size):
        if size not in measured:
            try:
                measured[size] = peak(size)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                measured[size] = float("inf")
        return measured[size] <= budget

    if not fits(1):
        raise MemoryError(
            f"memoryBudget {budget:.4g} B is below the {measured[1]:.4g} B "
            "one frame per batch needs."
        )
    low, high = 1, None
    while high is None and low < frames:
        size = min(2 * low, frames)
        if fits(size):
            low = size
        else:
            high = size
    while high is not None and high - low > 1:
        middle = (low + high) // 2
        if fits(middle):
            low = middle
        else:
            high = middle
    return {
        "batch size": low,
        "peak bytes": measured[low],
        "measurements": dict(sorted(measured.items())),
    }


def iteration_peak(engine, batch_size):
    """
    Return the peak allocated CUDA bytes of an engine iteration, restoring its state.

    Two iterations run and the second is measured: the optimizer state (Adam's
    moments) is first allocated in the first step, after that iteration's peak,
    and is resident in every later one. Parameters and optimizer state are
    restored afterwards; constraints and `afterStep` do run, and a random
    position order draws from NumPy's global generator.

    Args:
        engine (GradientEngine):
            A prepared engine on a CUDA device.

        batch_size (int):
            Frames per batch to measure.

    Returns:
        int:
            Peak bytes, without the copies made here to restore the state.
    """
    device = engine.device
    parameters = list(engine.model.parameters())
    saved = [parameter.detach().clone() for parameter in parameters]
    saved_bytes = sum(t.numel() * t.element_size() for t in saved)
    optimizer_state = copy.deepcopy(engine.optimizer.state_dict())
    previous = engine.effectiveBatchSize
    engine.effectiveBatchSize = batch_size
    try:
        engine.runIteration(0)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        engine.runIteration(1)
        torch.cuda.synchronize(device)
        # The copies made above are not part of a normal iteration.
        peak = torch.cuda.max_memory_allocated(device) - saved_bytes
    finally:
        engine.effectiveBatchSize = previous
        with torch.no_grad():
            for parameter, value in zip(parameters, saved):
                parameter.copy_(value)
        engine.optimizer.zero_grad(set_to_none=True)
        engine.optimizer.load_state_dict(optimizer_state)
    return peak
