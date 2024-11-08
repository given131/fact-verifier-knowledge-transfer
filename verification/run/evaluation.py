import logging
import os

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from utils.log import json_log


def calc_metric(y_true, y_pred, config, epoch=None):
    # f1 score
    json_log("calc_metric", {"y_true": y_true, "y_pred": y_pred})
    f1 = f1_score(y_true, y_pred, average='micro')

    # classification report
    if config.class_num == 2:
        target_names = ['refutes', 'supports']
    elif config.class_num == 3:
        target_names = ['refutes', 'supports', 'nei']
    else:
        raise ValueError(f"class_num should be 2 or 3 but got {config.class_num}")
    try:
        c_report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    except ValueError as e:
        c_report = None
    log_data = {
        "f1": f1,
        "c_report": c_report
    }
    if epoch:
        log_data["epoch"] = epoch
    json_log("calc_metric", log_data)

    return f1, c_report


def save_prediction(claim_id_list, y_pred, config):
    prediction = {"claim_id": claim_id_list, "cleaned_truthfulness": y_pred}
    df = pd.DataFrame(prediction)

    run_dir = config.run_dir
    os.makedirs(run_dir / "verification", exist_ok=True)
    prediction_path = run_dir / "verification/verification_result.csv"

    logging.info(f"saving prediction result to {prediction_path}")
    df.to_csv(prediction_path, index=False)
