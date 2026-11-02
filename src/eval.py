class Evaluator:
    def __init__(
        self,
        api,
        dataset,
        budget_epochs,
        total_epochs=200,
        training_schedule="200",
        is_random=False,
    ):
        self.api = api
        self.dataset = dataset
        self.budget = budget_epochs
        self.total_epochs = total_epochs
        self.training_schedule = training_schedule
        self.is_random = is_random

        self.used_epochs = 0
        self.training_progress = {}
        self.best_architecture = None
        self.best_validation_accuracy = -1.0
        self.trace = []

    def query_validation_accuracy(self, architecture_index, epochs):
        # leave out use eval=True
        info = self.api.evaluate_arch(
            architecture_index,
            self.dataset,
            iepoch=epochs - 1,
            hp=self.training_schedule,
            is_random=self.is_random,
        )
        return info["valid-accuracy"]

    def cost(self, architecture_index, target_epochs):
        trained_epochs = self.training_progress.get(architecture_index, 0)
        return max(0, target_epochs - trained_epochs)

    def remaining_budget(self):
        return self.budget - self.used_epochs

    def is_budget_exhausted(self):
        return self.used_epochs >= self.budget

    def can_afford(self, architecture_index, target_epochs):
        return self.cost(architecture_index, target_epochs) <= self.remaining_budget()

    def evaluate(self, architecture_index, epochs):
        if not 1 <= epochs <= self.total_epochs:
            raise ValueError(
                f"epochs must be in [1, {self.total_epochs}], got {epochs}"
            )

        additional_epochs = self.cost(architecture_index, epochs)

        if additional_epochs > self.remaining_budget():
            raise RuntimeError(
                "Budget exceeded: check can_afford() before evaluate()"
            )

        self.used_epochs += additional_epochs

        self.training_progress[architecture_index] = max(
            self.training_progress.get(architecture_index, 0),
            epochs,
        )

        validation_accuracy = self.query_validation_accuracy(
            architecture_index, epochs
        )

        if (
            epochs == self.total_epochs
            and validation_accuracy > self.best_validation_accuracy
        ):
            self.best_validation_accuracy = validation_accuracy
            self.best_architecture = architecture_index

        self.trace.append((self.used_epochs, self.best_architecture))

        return validation_accuracy