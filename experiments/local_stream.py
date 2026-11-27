import numpy as np

from relevance.streams import Stream


def local_drift_stream(
    n=20000,
    drift_at=10000,
    seed=0
):

    rng = np.random.default_rng(seed)

    X = rng.random((n, 10))

    regions = rng.integers(0, 2, size=n)

    # Separate the regions geometrically
    X[:, 0] = (
        regions * 0.8
        + rng.uniform(0, 0.2, size=n)
    )

    y = np.zeros(n, dtype=int)
    truth = np.zeros((n, 10), dtype=int)

    for t in range(n):

        region = regions[t]

        if t < drift_at:

            relevant = (
                [1, 2, 3]
                if region == 0
                else [4, 5, 6]
            )

        else:

            relevant = (
                [4, 5, 6]
                if region == 0
                else [1, 2, 3]
            )

        y[t] = int(
            X[t, relevant].sum() > 1.5
        )

        truth[t, relevant] = 1

    stream = Stream(
        X=X,
        y=y,
        truth=truth,
        drift_points=[drift_at],
        feature_names=[
            f"x{i}" for i in range(10)
        ]
    )

    return stream, regions