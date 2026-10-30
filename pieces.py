"""Piece plan for chaining video model passes. No ComfyUI imports, so it is testable alone.

Wan-Animate-2 takes 4n + 1 frames per pass (step 4), LTX-2.5 takes 8n + 1 (step 8).
"""


def legal_length(frames: int, step: int = 4) -> int:
    """Smallest frame count of the form step * n + 1 (at least step + 1) that holds `frames`."""
    length = max(step + 1, frames)
    while (length - 1) % step != 0:
        length += 1
    return length


def plan_pieces(total_frames: int, piece_length: int, step: int = 4, overlap: int = 1) -> list[int]:
    """Frame length of each piece needed to cover `total_frames`.

    The first piece yields its full length. Every later piece repeats the last `overlap`
    frames of the piece before, so it yields `overlap` frames less than its length. The
    last piece is shortened to the smallest legal length that still covers the clip.
    """
    if total_frames < 1:
        raise ValueError("total_frames must be at least 1")
    if piece_length < step + 1 or (piece_length - 1) % step != 0:
        raise ValueError(f"piece_length must be {step}n + 1 and at least {step + 1}")
    if overlap < 1 or overlap >= piece_length - step:
        raise ValueError("overlap must be at least 1 and leave room for new frames in every piece")
    first = min(piece_length, legal_length(total_frames, step))
    lengths = [first]
    produced = first
    while produced < total_frames:
        remaining = total_frames - produced
        length = min(piece_length, legal_length(remaining + overlap, step))
        lengths.append(length)
        produced += length - overlap
    return lengths


def frames_produced(lengths: list[int], overlap: int = 1) -> int:
    """Frames a piece plan yields before trimming to the clip length."""
    return sum(lengths) - overlap * (len(lengths) - 1)


def piece_starts(lengths: list[int], overlap: int = 1) -> list[int]:
    """Clip frame where each piece starts: a later piece starts `overlap` frames before the end of the one before."""
    starts, start = [], 0
    for length in lengths:
        starts.append(start)
        start += length - overlap
    return starts


def plan_segments(total_frames: int, segment_frames: int, overlap: int) -> list[tuple[int, int]]:
    """(start, end) frame ranges that cover the clip, neighbours sharing `overlap` frames.

    Used by the streaming finish: each range is processed on its own and the shared
    frames are crossfaded, so memory depends on `segment_frames`, not on the clip length.
    """
    if total_frames < 1:
        raise ValueError("total_frames must be at least 1")
    if overlap < 0 or segment_frames <= overlap:
        raise ValueError("segment_frames must be larger than overlap, and overlap at least 0")
    if total_frames <= segment_frames:
        return [(0, total_frames)]
    step = segment_frames - overlap
    ranges, start = [], 0
    while True:
        end = min(start + segment_frames, total_frames)
        ranges.append((start, end))
        if end == total_frames:
            return ranges
        start += step
