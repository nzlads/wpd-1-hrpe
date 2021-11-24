

def rmse(preds, truths):
    """
    Root Mean Squared Error
    """
    return ((preds - truths) ** 2).mean() ** 0.5


def score_model(preds, truths):
    """
    preds : 4 cols date,max,min,mean
    truths : 4 cols date,max,min,mean
    """
    # see pdf
    # sqrt(rmse(max) + rmse(min))
