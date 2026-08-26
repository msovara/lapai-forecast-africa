#!/usr/bin/env python3
"""Smoke-check africa_hw_mask broadcasts to (H,W)."""
from evaluation.masks import africa_hw_mask
from lapai_inference.cache_schema import lat_lon_mesh

lat, lon = lat_lon_mesh(181, 360)
m = africa_hw_mask(lat, lon)
assert m.shape == (181, 360), m.shape
print("shape", m.shape, "dtype", m.dtype, "mean", float(m.mean()), "sum", float(m.sum()))
