def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    """
    Linear warmup (0→initial_lr) then linear decay (initial_lr→final_lr).
    Steps are 0-based; clamp at final_lr after total_steps.
    """
    # Warmup phase
    if warmup_steps > 0 and step < warmup_steps:
        return initial_lr * step / warmup_steps

    # After training is complete
    if step > total_steps:
        return float(final_lr)

    # Linear decay phase
    if total_steps == warmup_steps:
        return float(final_lr)

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = initial_lr + (final_lr - initial_lr) * progress

    return float(lr)