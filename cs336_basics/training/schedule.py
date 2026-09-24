import math

def cosine_learning_rate_schedule(
        it,
        max_lr,
        min_lr,
        warmup_iters,
        cosine_cycle_iters,
) -> float:

        if it < warmup_iters:
            return it * max_lr / warmup_iters

        elif warmup_iters <= it <= cosine_cycle_iters:
            return min_lr + 1/2 * (1 + math.cos((it - warmup_iters) * math.pi / (cosine_cycle_iters - warmup_iters))) * (max_lr - min_lr)

        else:
            return min_lr
