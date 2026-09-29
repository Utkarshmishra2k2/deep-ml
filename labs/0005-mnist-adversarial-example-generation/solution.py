import torch
import torch.nn as nn


def generate_adversarial_example(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    criterion: nn.Module
) -> torch.Tensor:
    """
    Generate an untargeted adversarial example using
    Projected Gradient Descent (PGD).

    The returned tensor:
    1. Has the same shape as x
    2. Satisfies ||x_adv - x||_inf <= epsilon
    3. Contains pixels only in [0, 1]
    4. Attempts to make model(x_adv).argmax() != y
    """

    x_original = x.clone().detach()
    epsilon = float(epsilon)

    if epsilon <= 0.0:
        return x_original

    # Preserve the model's original training/evaluation state.
    was_training = model.training
    model.eval()

    # A good configuration for MNIST with epsilon around 0.15.
    num_steps = 30
    step_size = epsilon / 8.0

    try:
        # If the original input is already misclassified, no change is needed.
        with torch.no_grad():
            original_prediction = model(x_original).argmax(dim=1)

        if bool((original_prediction != y).all().item()):
            return x_original

        # Start from the original image.
        x_adv = x_original.clone()

        for _ in range(num_steps):
            # Create a fresh leaf tensor so gradients are computed only
            # with respect to the current adversarial image.
            x_adv = x_adv.detach()
            x_adv.requires_grad_(True)

            logits = model(x_adv)
            loss = criterion(logits, y)

            # Some criteria may return one loss per sample.
            if loss.ndim != 0:
                loss = loss.sum()

            # Exact gradient of the classification loss with respect to input.
            gradient = torch.autograd.grad(
                outputs=loss,
                inputs=x_adv,
                only_inputs=True
            )[0]

            # Untargeted attack: maximize the loss for the true class.
            x_adv = (
                x_adv.detach()
                + step_size * torch.sign(gradient.detach())
            )

            # Project into the L-infinity epsilon ball.
            perturbation = x_adv - x_original
            perturbation = torch.clamp(
                perturbation,
                min=-epsilon,
                max=epsilon
            )

            x_adv = x_original + perturbation

            # Enforce the valid image range.
            x_adv = torch.clamp(x_adv, min=0.0, max=1.0)

            # Stop as soon as the attack succeeds.
            with torch.no_grad():
                prediction = model(x_adv).argmax(dim=1)

            if bool((prediction != y).all().item()):
                break

        # Final safety projection.
        perturbation = torch.clamp(
            x_adv.detach() - x_original,
            min=-epsilon,
            max=epsilon
        )

        x_adv = torch.clamp(
            x_original + perturbation,
            min=0.0,
            max=1.0
        )

        return x_adv.detach()

    finally:
        # Restore the model's previous mode.
        model.train(was_training)