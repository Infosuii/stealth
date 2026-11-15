import numpy as np
import pytest
from stealth.count_transfer import partitions


def test_equal_count_controls_preserve_validation_and_outer_rows():
    era=np.repeat(np.arange(2400,3960),3)
    results={p:partitions(era,history=1024,width=200,gap=40,floor=2800,policy=p)
             for p in ('ragged','spread','recent','full_span')}
    assert [len(np.unique(era[m])) for m,_ in results['ragged'][1]]==[384,584,784]
    for p,n in [('spread',384),('recent',384),('full_span',784)]:
        outer,folds=results[p]
        np.testing.assert_array_equal(outer,results['ragged'][0])
        for i,(train,valid) in enumerate(folds):
            assert len(np.unique(era[train]))==n
            assert era[valid].min()-era[train].max()>40
            np.testing.assert_array_equal(valid,results['ragged'][1][i][1])
    assert era[results['full_span'][1][0][0]].min()==2536
    assert era[results['full_span'][0]].min()==2936
    # The original latest source is unchanged when full-size histories are made available to older sources.
    np.testing.assert_array_equal(results['full_span'][1][-1][0],results['ragged'][1][-1][0])
    assert era[results['spread'][1][-1][0]].min()<era[results['recent'][1][-1][0]].min()


def test_missing_context_refuses_constant_full_history():
    era=np.repeat(np.arange(2800,3960),3)
    with pytest.raises(ValueError,match='earlier context'):
        partitions(era,history=1024,width=200,gap=40,floor=2800,policy='full_span')
