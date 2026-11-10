"""Sanity tests that confirm the repo skeleton is importable."""


def test_src_package_importable():
    import src
    assert src is not None


def test_dynamics_subpackage_importable():
    from src.dynamics import state
    assert state is not None


def test_aero_subpackage_importable():
    from src.aero import vtail
    assert vtail is not None             