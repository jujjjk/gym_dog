"""Regression: removed compensation must not reappear during straight travel."""
import numpy as np
import pytest
from test_rs01_model18000 import contract_session
from mydog_policy.rs01_model18000_core import Rs01Model18000Core

@pytest.mark.parametrize('vy',[-.1,.1])
def test_no_lateral_trim_even_with_persistent_estimated_drift(contract_session,vy):
    c,s,_=contract_session
    core=Rs01Model18000Core(s,c)
    core.steps=1
    core.odometry.estimate=lambda *a,**k: dict(base_linear_velocity=np.array([.2,vy,0]),confidence=.95)
    for _ in range(150):
        out=core.tick(c.default,np.zeros(12),np.zeros(3),[0,0,-1],0,[.2,0,0])
        np.testing.assert_allclose(out['observation'][9:12],[.4,0,0],atol=1e-6)
        np.testing.assert_allclose(out['observation'][57:60],[.4,0,0],atol=1e-6)
