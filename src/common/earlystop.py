class EarlyStopper:
    def __init__(self, patience: float = 5, smooth_factor=0.9):
        assert patience == "inf" or patience > 0, "Patience must be a positive number."
        self.patience = float(patience)
        self.smooth_factor = smooth_factor
        self.counter = 0
        self.best_loss = None
        self.ema_loss = None

    def update(self, val_loss):
        if self.ema_loss is None:
            self.ema_loss = val_loss
        else:
            self.ema_loss = (
                self.smooth_factor * self.ema_loss + (1 - self.smooth_factor) * val_loss
            )

        if self.best_loss is None:
            self.best_loss = self.ema_loss

        if self.ema_loss < self.best_loss:
            self.best_loss = self.ema_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True  # Signal to stop
        return False  # Signal to continue


if __name__ == "__main__":
    # Usage Example:
    early_stopper = EarlyStopper(patience=5, smooth_factor=0.9)
    for epoch in range(100):  # Replace with your actual training loop
        val_loss = 0.1  # Replace with your actual validation loss
        if early_stopper.update(val_loss):
            print("Early stopping triggered.")
            break
