"""Regrid module import smoke test."""

from utils.regrid import crop_global_latlon025, regrid_n320_to_latlon025, regrid_to_n320


def test_regrid_public_api_importable():
    assert callable(regrid_to_n320)
    assert callable(regrid_n320_to_latlon025)
    assert callable(crop_global_latlon025)
