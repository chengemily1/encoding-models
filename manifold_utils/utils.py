import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

def get_stats(preds, actuals):
    preds = preds.T
    actuals = actuals.T
    r2 = r2_score(actuals, preds)
    print('R2: ', r2)
    correlations = [pearsonr(preds[i], actuals[i])[0] for i in range(len(preds))]
    print('Mean Correlation: ', np.mean(correlations))
    print('Std Correlation: ', np.std(correlations))
    print('Min Correlation: ', np.min(correlations))
    print('Max Correlation: ', np.max(correlations))

    return r2, correlations 