import datetime

class TrainingLogger:
    def __init__(self, log_file):
        self.log_file = log_file

    def log(self, message):
        with open(self.log_file, "a") as file:
            file.write(message + "\n")

    def log_training_start(self, model_name, dataset, parameters, hardware):
        start_message = f"--- Deep Learning Training Log ---\n\n" \
                        f"Model Name: {model_name}\n" \
                        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" \
                        f"Dataset Used: {dataset}\n" \
                        f"\nTraining Parameters:\n{parameters}\n" \
                        f"\nHardware and Software:\n{hardware}\n" \
                        f"\n--- Epoch Log ---\n"
        self.log(start_message)

    def log_epoch(self, epoch, training_loss, validation_loss, learning_rate, other_metrics):
        epoch_message = f"Epoch {epoch} | Training Loss: {training_loss} | " \
                        f"Validation Loss: {validation_loss} | Learning Rate: {learning_rate} | " \
                        f"Other Metrics: {other_metrics}"
        self.log(epoch_message)

    def log_conclusion(self, conclusion):
        self.log("\n--- Conclusion and Next Steps ---\n" + conclusion)

# Example usage
logger = TrainingLogger("training_log.txt")
logger.log_training_start("MyModel", "MyDataset", "Parameters: Batch Size=32, ...", "GPU: NVIDIA RTX, ...")
# For each epoch in your training loop
logger.log_epoch(1, 0.45, 0.40, 0.001, "Accuracy: 85%")
# At the end of training
logger.log_conclusion("Final Thoughts: ..., Next Steps: ...")