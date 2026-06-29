"""AIFS Single v2 field names (aligned with aifs-africa / C4E n320_gt6)."""

PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

PARAM_PL_CDS = ["z", "t", "u", "v", "w", "q"]

PARAM_SFC_DYNAMIC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw"]
PARAM_SFC_STATIC = ["lsm", "z", "slor", "sdor"]
PARAM_SFC = PARAM_SFC_DYNAMIC + PARAM_SFC_STATIC

STATIC_FORCING_VARS = ("lsm", "sdor", "slor", "z")
