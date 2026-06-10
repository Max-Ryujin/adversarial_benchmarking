from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from adversarial_benchmarking.config import AttackConfig
from adversarial_benchmarking.logging_utils import (
    format_named_logits,
    get_logger,
    summarize_tensor,
)

logger = get_logger("attacks.apgd")


def dlr_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Difference of Logits Ratio loss from AutoAttack.

    More stable than CE for robustness evaluation on 3+ class tasks.
    """
    if logits.size(1) < 3:
        return F.cross_entropy(logits, labels)

    sorted_logits, sorted_indices = logits.sort(dim=1, descending=True)

    batch_indices = torch.arange(logits.size(0), device=logits.device)

    correct_logits = logits[batch_indices, labels]

    top1 = sorted_logits[:, 0]
    top2 = sorted_logits[:, 1]
    top3 = sorted_logits[:, 2]

    is_top1_correct = sorted_indices[:, 0] == labels

    other_logits = torch.where(is_top1_correct, top2, top1)

    loss = -(correct_logits - other_logits) / (top1 - top3 + 1e-12)

    return loss.mean()


def attack_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    targeted: bool,
    use_dlr: bool,
) -> torch.Tensor:
    if targeted:
        return -F.cross_entropy(logits, labels)
    if use_dlr:
        return dlr_loss(logits, labels)
    return F.cross_entropy(logits, labels)


def apgd_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    config: AttackConfig,
    *,
    use_dlr: bool = True,
    momentum: float = 0.75,
    eot_iter: int = 1,
) -> torch.Tensor:
    """
    Auto-PGD style attack.

    Improvements over vanilla PGD:
    - adaptive step size
    - momentum
    - oscillation detection
    - restart from best point
    - DLR loss
    - EOT support
    """

    epsilon = config.epsilon
    steps = config.steps

    logger.info(
        "Starting APGD attack: steps=%s epsilon=%.6f targeted=%s",
        steps,
        epsilon,
        config.targeted,
    )

    # ============================================================
    # Initialization
    # ============================================================

    if config.random_start:
        delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
        adv_images = torch.clamp(images + delta, 0.0, 1.0)
    else:
        adv_images = images.clone()

    x_best = adv_images.clone().detach()
    loss_best = torch.full(
        (images.size(0),),
        -float("inf"),
        device=images.device,
    )

    initial_step_size = config.step_size if config.step_size > 0 else min(epsilon, 2.0 * epsilon)
    step_size = torch.full(
        (images.size(0), 1, 1, 1),
        initial_step_size,
        device=images.device,
    )

    grad_prev = torch.zeros_like(images)
    x_prev = adv_images.clone()

    # ============================================================
    # APGD checkpoint schedule
    # ============================================================

    checkpoints = []
    p = [0.0, 0.22]

    while p[-1] < 1.0:
        next_p = p[-1] + max(p[-1] - p[-2] - 0.03, 0.06)
        p.append(min(next_p, 1.0))

    checkpoints = [int(steps * v) for v in p[1:]]

    loss_steps = []

    task = getattr(model, "task", None)
    debug_enabled = logger.isEnabledFor(logging.DEBUG)

    # ============================================================
    # Main loop
    # ============================================================

    for step in range(steps):

        adv_images.requires_grad_(True)

        grad = torch.zeros_like(adv_images)

        # --------------------------------------------------------
        # EOT gradients
        # --------------------------------------------------------

        for _ in range(eot_iter):

            logits = model(adv_images)

            loss = attack_loss(
                logits,
                labels,
                targeted=config.targeted,
                use_dlr=use_dlr,
            )

            grad += torch.autograd.grad(loss, adv_images)[0]

        grad /= float(eot_iter)

        with torch.no_grad():

            # ----------------------------------------------------
            # Track best point
            # ----------------------------------------------------

            current_loss = loss.detach()

            improved = current_loss > loss_best

            loss_best[improved] = current_loss
            x_best[improved] = adv_images.detach()[improved]

            loss_steps.append(current_loss.item())

            # ----------------------------------------------------
            # Momentum update
            # ----------------------------------------------------

            grad_momentum = (
                momentum * grad_prev
                + (1.0 - momentum) * grad
            )

            grad_prev = grad_momentum.clone()

            x_new = adv_images + step_size * grad_momentum.sign()

            # ----------------------------------------------------
            # Projection
            # ----------------------------------------------------

            delta = torch.clamp(
                x_new - images,
                min=-epsilon,
                max=epsilon,
            )

            x_new = torch.clamp(images + delta, 0.0, 1.0)

            # ----------------------------------------------------
            # Oscillation detection
            # ----------------------------------------------------

            if step in checkpoints and len(loss_steps) >= 10:

                recent = loss_steps[-10:]

                improvements = 0

                for i in range(len(recent) - 1):
                    if recent[i + 1] > recent[i]:
                        improvements += 1

                improvement_ratio = improvements / (len(recent) - 1)

                oscillating = improvement_ratio < 0.75

                if oscillating:

                    logger.debug(
                        "Reducing step size at step %s",
                        step,
                    )

                    step_size = step_size / 2.0

                    # restart from best point
                    x_new = x_best.clone()

            adv_images = x_new.detach()
            x_prev = adv_images.clone()

        # The per-step diagnostics below force host<->device syncs (`.item()`, `.cpu()`)
        # and extra reductions, so only compute them when DEBUG logging is on.
        if debug_enabled:
            predicted_indices = logits.argmax(dim=-1)

            logger.debug(
                "APGD step %s/%s loss=%.6f pred=%s step_size=%.6f grad=%s",
                step + 1,
                steps,
                loss.item(),
                predicted_indices.detach().cpu().tolist(),
                step_size.mean().item(),
                summarize_tensor(grad),
            )

            if task is not None:
                logger.debug(
                    "APGD step %s class logits: %s",
                    step + 1,
                    format_named_logits(
                        task.class_labels(),
                        logits[0],
                    ),
                )

    logger.info("Finished APGD attack")

    return x_best.detach()