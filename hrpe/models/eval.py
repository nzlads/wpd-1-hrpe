"""
Scoring functions for evaluation model predictions
"""


def rmse(preds, truths):
    """
    Custom RMSE calc as per challenge specification
    :param preds: dataframe of predictions with time, value_max, value_min columns
    :param truths: dataframe of true values with time, value_max, value_min columns
    :return rmse value for the preds dataframe
    """

    sq_max_errors = ((preds["value_max"] - truths["value_max"]) ** 2).sum()
    sq_min_errors = ((preds["value_min"] - truths["value_min"]) ** 2).sum()

    return (sq_max_errors + sq_min_errors) ** 0.5


def score_model(preds, truths):
    """
    Skill score as per challenge specification
    :param preds: dataframe of predictions with time, value_max, value_min, value_mean columns
    :param truths: dataframe of true values with time, value_max, value_min, value_mean columns
    :return skill score for the dataframe
    """
    # see pdf
    EXPECTED_COLS = ["time", "value_max", "value_min", "value_mean"]
    for df in [preds, truths]:
        missing = [col for col in EXPECTED_COLS if col not in df.columns]
        assert (
            missing == []
        ), f"The following columns are missing from {df.name}: {missing}"

    pred = preds.copy().sort_values("time")
    truth = truths.copy().sort_values("time")

    assert all(
        pred["time"].values == truth["time"].values
    ), "The time col for preds and truths do not match"

    # Create benchmark table
    bench = pred[["time"]]
    bench.loc[:, "value_max"] = pred["value_mean"]
    bench.loc[:, "value_min"] = pred["value_mean"]

    pred_rmse = rmse(pred, truth)
    bench_rmse = rmse(bench, truth)

    return pred_rmse / bench_rmse
