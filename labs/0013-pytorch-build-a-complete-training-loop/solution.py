import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def train_model(model, X_train, y_train, X_val, y_val,
                epochs, batch_size, lr):
    """
    Train a PyTorch model and return training history.
    """

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()

    history = []
    number_of_samples = X_train.shape[0]

    for epoch in range(epochs):
        # Training mode
        model.train()

        # Shuffle the training data
        shuffled_indices = torch.randperm(number_of_samples)

        total_train_loss = 0.0

        # Process mini-batches
        for start in range(0, number_of_samples, batch_size):
            end = start + batch_size
            batch_indices = shuffled_indices[start:end]

            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(X_batch)

            # Calculate loss
            loss = loss_function(logits, y_batch)

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Add batch loss, weighted by batch size
            total_train_loss += loss.item() * X_batch.shape[0]

        # Average loss across all training samples
        average_train_loss = total_train_loss / number_of_samples

        # Validation mode
        model.eval()

        with torch.no_grad():
            validation_logits = model(X_val)

            validation_loss = loss_function(
                validation_logits,
                y_val
            ).item()

            predictions = validation_logits.argmax(dim=1)

            validation_accuracy = (
                predictions == y_val
            ).float().mean().item()

        history.append({
            "epoch": epoch + 1,
            "train_loss": average_train_loss,
            "val_loss": validation_loss,
            "val_accuracy": validation_accuracy
        })

    return history