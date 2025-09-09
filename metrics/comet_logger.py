from comet_ml import start, CometExperiment

_experiment = start(
    api_key="JJd7mzI6hcJQYysV3fxUNkRAQ",
    project_name="sheaf-diffusion",
    workspace="artem-d"
)

def get_experiment() -> CometExperiment:
    return _experiment

