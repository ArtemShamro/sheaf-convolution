from comet_ml import start, CometExperiment


class CometLogger():
    def __init__(self):
        self._experiment = self.start_experiment()

    def start_experiment(self):
        _experiment = start(
            api_key="JJd7mzI6hcJQYysV3fxUNkRAQ",
            project_name="sheaf-diffusion",
            workspace="artem-d"
        )
        return _experiment

    def get_experiment(self) -> CometExperiment:
        return self._experiment

    def log_model_info(self, model):
        experiment = self.get_experiment()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
        experiment.log_other("model/total_params", total_params)
        experiment.log_other("model/trainable_params", trainable_params)
        print(
            f"Logged model parameters: total={total_params}, trainable={trainable_params}")

    def log_dataset_info(self, dataset):
        experiment = self.get_experiment()
        info = {
            "dataset/name": dataset.__class__.__name__,
            "dataset/num_nodes": getattr(dataset.data, "num_nodes", None),
            "dataset/num_edges": getattr(dataset.data, "num_edges", None),
            "dataset/num_features": getattr(dataset.data, "num_features", None),
        }
        for k, v in info.items():
            if v is not None:
                experiment.log_other(k, v)

    def end(self):
        self._experiment.end()
