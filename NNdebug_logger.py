import numpy as np
import datetime

class DebugLogger:
    def __init__(self, log_file="training_debug.log"):
        self.log_file = log_file
        with open(self.log_file, 'w') as f:
            f.write(f"Training Debug Log - {datetime.datetime.now()}\n")
            f.write("=" * 50 + "\n\n")

    def log_epoch(self, epoch, loss, accuracy, learning_rate):
        with open(self.log_file, 'a') as f:
            f.write(
                f"EPOCH {epoch:5d} | "
                f"Loss={loss:.6f} | "
                f"Acc={accuracy:.6f} | "
                f"LR={learning_rate:.6e}\n"
            )

    def log_gradients(self, epoch, *layers):
        with open(self.log_file, 'a') as f:
            f.write(f"\nGRADIENT STATS - Epoch {epoch}\n")
            for i, layer in enumerate(layers, start=1):
                dw = layer.dweights
                db = layer.dbiases
                f.write(
                    f" Dense{i}: "
                    f"|dw| mean={np.mean(np.abs(dw)):.3e}, "
                    f"max={np.max(np.abs(dw)):.3e} | "
                    f"|db| mean={np.mean(np.abs(db)):.3e}\n"
                )

    def log_weights(self, epoch, *layers):
        with open(self.log_file, 'a') as f:
            f.write(f"\nWEIGHT STATS - Epoch {epoch}\n")
            for i, layer in enumerate(layers, start=1):
                w = layer.weights
                f.write(
                    f" Dense{i}: "
                    f"mean={np.mean(w):.4e}, "
                    f"std={np.std(w):.4e}, "
                    f"max={np.max(np.abs(w)):.4e}\n"
                )

    def log_predictions(self, epoch, predictions, targets, confidences):
        with open(self.log_file, 'a') as f:
            f.write(f"\nPREDICTION STATS - Epoch {epoch}\n")
            f.write(f" Pred dist : {np.bincount(predictions, minlength=3)}\n")
            f.write(f" Target dist: {np.bincount(targets, minlength=3)}\n")
            f.write(
                f" Confidence mean={np.mean(confidences):.4f}, "
                f"min={np.min(confidences):.4f}, "
                f"max={np.max(confidences):.4f}\n"
            )
            f.write("-" * 40 + "\n")
