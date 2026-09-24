from .router import RoutedBlock
from .schema import TextBlock


def merge_and_reorder(blocks: list[RoutedBlock], frame_timestamp: float | None = None) -> list[TextBlock]:
    # Buckets boxes into ~15px text-line bands before sorting left-to-right,
    # so minor y-jitter between boxes on the same line doesn't reorder them.
    ordered = sorted(blocks, key=lambda b: (round(b.box.y1 / 15), b.box.x1))
    return [
        TextBlock(
            text=b.text,
            box=b.box,
            confidence=b.confidence,
            source=b.source,
            frame_timestamp=frame_timestamp,
        )
        for b in ordered
    ]
