"""The documentation's phantom: MPAT's exact density agrees with Monte Carlo tracking sampled
from the same chain, and the docs' helper that splits a solve into its forward and backward
parts reproduces segment_bundles.  (The figures in docs/algorithm.md rely on both.)"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs" / "animations"))
P = pytest.importorskip("phantom")


@pytest.fixture(scope="module")
def chain():
    return P.build_chain()


@pytest.mark.parametrize("kw", [dict(include=False, exclude=False),        # endpoint ROIs
                                dict(exclude=False, ends=False)])          # two include ROIs
def test_monte_carlo_matches_exact(chain, kw):
    spec = P.arc_spec(**kw)
    exact = P.segment(chain, spec).density_t
    mc = P.monte_carlo_density(chain, spec, 20000, seed=3)
    assert abs(mc.sum() / exact.sum() - 1) < 0.05
    assert np.abs(mc - exact).sum() / exact.sum() < 0.1


def test_reactive_parts_match_segment(chain):
    spec = P.arc_spec()
    parts = P.reactive_parts(chain, spec)
    exact = P.segment(chain, spec).density_t
    np.testing.assert_allclose(parts["density"], exact, rtol=0, atol=1e-12 * exact.max())


def test_endpoint_mass(chain):
    res = P.segment(chain, P.arc_spec())
    assert res.info["endpoint_mass_check"] == pytest.approx(1.0, abs=1e-6)
