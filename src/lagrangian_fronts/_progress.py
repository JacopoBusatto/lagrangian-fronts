"""Small iterator wrapper for the CLI's active loop."""

import logging
import sys

from tqdm import tqdm


def track(iterable, *, desc, total, unit):
    """Count completed items; Python calls and redirected output stay quiet."""
    if (
        not total or not sys.stderr.isatty()
        or not logging.getLogger("lagrangian_fronts").isEnabledFor(logging.INFO)
    ):
        yield from iterable
        return
    with tqdm(
        total=total, desc=desc, unit=unit, file=sys.stderr,
        mininterval=0.2, miniters=1, dynamic_ncols=True, leave=False,
    ) as bar:
        for item in iterable:
            yield item
            bar.update(1)
