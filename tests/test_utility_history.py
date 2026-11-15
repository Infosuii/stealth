import numpy as np

from stealth.objective_models import fit_cpu
from stealth.models import lgbm


def test_core_history_retains_early_rows_without_inventing_teacher_labels():
    rng = np.random.default_rng(19)
    era = np.repeat(np.arange(3, 443), 5)
    X = rng.integers(0, 10, (len(era), 5)).astype(float)
    y = rng.normal(size=len(era))
    spec = dict(history=None, target='core')
    base = dict(cap=5, num_leaves=3, min_data_in_leaf=20, early_stopping_rounds=2)
    predictions = []
    for teacher in (np.full(len(era), np.nan), rng.normal(size=len(era))):
        model, info = fit_cpu(spec, base, X, y, era, teacher, width=40, gap=40,
                              seed=17, threads=1, bound=2/np.pi)
        assert info['train_min'] == 3 and info['train_max'] == 442
        assert info['inner_valid_min'] - info['inner_fit_max'] > 40
        predictions.append(lgbm.predict(model, X[:30]))
    np.testing.assert_array_equal(*predictions)
