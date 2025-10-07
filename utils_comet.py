import comet_ml
from comet_ml.api import Parameter
import pandas as pd


def export_synth_exp_group_table(workspace_name, project_name, exp_group: str) -> pd.DataFrame:
    api = comet_ml.API(cache=False)

    query_condition = (Parameter("exp_group") == exp_group)
    matching_api_experiments = api.query(
        workspace_name, project_name, query_condition)

    rows = []
    for exp in matching_api_experiments:
        row = {
            "model_name": str(exp.get_parameters_summary('model|name')['valueMax']),
            "model/total_params": int(exp.get_others_summary("model/total_params")[0]),

            "best_epoch": int(exp.get_others_summary("best_epoch")[0]),
            "best_val_ap": float(exp.get_others_summary("best_val_ap")[0]),
            "best_val_auc": float(exp.get_others_summary("best_val_auc")[0]),
            "test_ap_at_best_val": float(exp.get_others_summary("test_ap_at_best_val")[0]),
            "test_auc_at_best_val": float(exp.get_others_summary("test_auc_at_best_val")[0]),

            "dataset/name": str(exp.get_others_summary("dataset/name")[0]),
            "dataset/num_nodes": int(exp.get_others_summary("dataset/num_nodes")[0]),
            "dataset/num_edges": int(exp.get_others_summary("dataset/num_edges")[0]),
            "dataset/num_features": int(exp.get_others_summary("dataset/num_features")[0]),
            "dataset/nproj": int(exp.get_parameters_summary('dataset|nproj')['valueMax']),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df
