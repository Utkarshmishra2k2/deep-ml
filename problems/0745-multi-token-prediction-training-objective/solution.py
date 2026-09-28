import numpy as np

def mtp_loss(main_logits, main_targets, mtp_logits, mtp_targets, mtp_weight):
    """
    Compute the combined LM + depth-1 Multi-Token Prediction loss.

    Args:
        main_logits: array-like (N, V)
        main_targets: array-like (N,) integer token ids
        mtp_logits: array-like (N-1, V)
        mtp_targets: array-like (N-1,) integer token ids
        mtp_weight: float

    Returns:
        float: total loss rounded to 6 decimals
    """
    main_logits = np.asarray(main_logits, dtype=np.float64)
    main_targets = np.asarray(main_targets, dtype=np.int64)
    mtp_logits = np.asarray(mtp_logits, dtype=np.float64)
    mtp_targets = np.asarray(mtp_targets, dtype=np.int64)

    def mean_cross_entropy(logits, targets):
        # Shift each row for numerical stability.
        shifted_logits = logits - np.max(
            logits,
            axis=1,
            keepdims=True
        )

        # Negative log-probability of each target token.
        log_sum_exp = np.log(
            np.sum(np.exp(shifted_logits), axis=1)
        )
        target_logits = shifted_logits[
            np.arange(len(targets)),
            targets
        ]

        return np.mean(log_sum_exp - target_logits)

    main_loss = mean_cross_entropy(main_logits, main_targets)
    mtp_auxiliary_loss = mean_cross_entropy(mtp_logits, mtp_targets)

    total_loss = main_loss + mtp_weight * mtp_auxiliary_loss

    return round(float(total_loss), 6)
