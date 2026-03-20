# simulation/satellite_config.py
# Data only. No functions here.




generate_constellation = True         # True  -> Walker constellation mode, False -> Independent per-satellite mode

offnadir_limit_deg = 40.0

# Only used for constellation generation
time_delay_tip_cue = 1 * 60
nPlanes = 1
nSats = 1



SATELLITE_MODE = {
    "Tip": generate_constellation,
    "Cue": generate_constellation,
}


DEFAULT_CUE_ORBIT = {
    "hp": 615.7e3,
    "ha": 624.6e3,
    "i_deg": 97.8703,
    "RAAN_deg": 336.4191,
    "argp_deg": 110.0511,
    "M_deg": 250.1394,
}

DEFAULT_TIP_ORBIT = {
    "hp": 615.7e3,
    "ha": 624.6e3,
    "i_deg": 97.8703,
    "RAAN_deg": 336.4191,
    "argp_deg": 110.0511,
    "M_deg": 250.1394,
}

TIP_CUE_ORBIT_LINK = {
    "enabled": True,
    "reference": "Cue",
    "delay_seconds": time_delay_tip_cue,
}
# Used only for constellation mode.
# If enabled, the default Tip constellation orbit is derived from the Cue default
# constellation orbit by shifting mean anomaly based on this time delay.
# Independent satellites stay fully explicit and are never auto-shifted.

DEFAULT_TIP_SENSOR = {
    "resolution": 124,
    "sample_count": 512,
    "GSD": 10.0,
    "specular_weight": 0.2,
    "swath_m": 290e3,
    "offnadir_limit_deg": offnadir_limit_deg,
}

DEFAULT_CUE_SENSOR = {
    "resolution": 124,
    "sample_count": 512,
    "GSD": 0.31,
    "specular_weight": 0.2,
    "swath_m": 13.1e3,
    "offnadir_limit_deg": offnadir_limit_deg,
}



CONSTELLATION_CONFIG = {
    "Tip": {
        "nPlanes": nPlanes,
        "nSats": nSats,  # satellites per plane
        "orbit": {
            **DEFAULT_TIP_ORBIT,
            "M_deg": 407.0,
        },
        "sensor": {
            **DEFAULT_TIP_SENSOR,
        },
    },
    "Cue": {
        "nPlanes": nPlanes,
        "nSats": nSats,  # satellites per plane
        "orbit": {
            **DEFAULT_CUE_ORBIT,
            "M_deg": 407.0,
        },
        "sensor": {
            **DEFAULT_CUE_SENSOR,
        },
    },
}


INDEPENDENT_SATELLITES = {
    "Tip": [
        {
            "name": "Sentinel_2A",
            "orbit": {
                **DEFAULT_TIP_ORBIT,
                "hp": 783438.036,
                "ha": 804307.718,
                "i_deg": 98.56655207,
                "RAAN_deg": 336.82519372,
                "argp_deg": 99.38541835,
                "M_deg": 64.85941569,
            },
            "sensor": {
                **DEFAULT_TIP_SENSOR,
            },
        },
        {
            "name": "Sentinel_2B",
            "orbit": {
                **DEFAULT_TIP_ORBIT,
                "hp": 786879.705,
                "ha": 798660.456,
                "i_deg": 98.56731318,
                "RAAN_deg": 336.75430302,
                "argp_deg": 93.38537598,
                "M_deg": 107.93113283,
            },
            "sensor": {
                **DEFAULT_TIP_SENSOR,
            },
        },
        {
            "name": "Sentinel_2C",
            "orbit": {
                **DEFAULT_TIP_ORBIT,
                "hp": 782444.073,
                "ha": 803621.723,
                "i_deg": 98.56476774,
                "RAAN_deg": 336.82234106,
                "argp_deg": 86.08734118,
                "M_deg": 294.30991151,
            },
            "sensor": {
                **DEFAULT_TIP_SENSOR,
            },
        },
        {
            "name": "Landsat_7",
            "orbit": {
                **DEFAULT_TIP_ORBIT,
                "hp": 666076.545,
                "ha": 676172.368,
                "i_deg": 97.86772993,
                "RAAN_deg": 274.18031782,
                "argp_deg": 217.53372286,
                "M_deg": 220.69743630,
            },
            "sensor": {
                **DEFAULT_TIP_SENSOR,
            },
        },
        {
            "name": "Landsat_8",
            "orbit": {
                **DEFAULT_TIP_ORBIT,
                "hp": 687381.840,
                "ha": 698363.418,
                "i_deg": 98.22135057,
                "RAAN_deg": 332.37295064,
                "argp_deg": 356.58256641,
                "M_deg": 114.50104054,
            },
            "sensor": {
                **DEFAULT_TIP_SENSOR,
            },
        },
        {
            "name": "Landsat_9",
            "orbit": {
                **DEFAULT_TIP_ORBIT,
                "hp": 675069.402,
                "ha": 710688.388,
                "i_deg": 98.22286733,
                "RAAN_deg": 332.38836130,
                "argp_deg": 108.28651417,
                "M_deg": 182.76361215,
            },
            "sensor": {
                **DEFAULT_TIP_SENSOR,
            },
        },
        {
            "name": "Resourcesat_2",
            "orbit": {
                **DEFAULT_TIP_ORBIT,
                "hp": 814579.028,
                "ha": 826519.372,
                "i_deg": 98.79730113,
                "RAAN_deg": 326.97941826,
                "argp_deg": 113.40550442,
                "M_deg": 280.53202342,
            },
            "sensor": {
                **DEFAULT_TIP_SENSOR,
            },
        },
        {
            "name": "Resourcesat_2A",
            "orbit": {
                **DEFAULT_TIP_ORBIT,
                "hp": 806672.449,
                "ha": 819849.559,
                "i_deg": 98.78804805,
                "RAAN_deg": 333.41248847,
                "argp_deg": 39.34909999,
                "M_deg": 82.95023203,
            },
            "sensor": {
                **DEFAULT_TIP_SENSOR,
            },
        },
    ],

    "Cue": [
        {
            "name": "WorldView_2",
            "orbit": {
                **DEFAULT_CUE_ORBIT,
                "hp": 762348.426,
                "ha": 779492.244,
                "i_deg": 98.47560556,
                "RAAN_deg": 335.36747608,
                "argp_deg": 81.78668872,
                "M_deg": 77.66199509,
            },
            "sensor": {
                **DEFAULT_CUE_SENSOR,
            },
        },
        {
            "name": "WorldView_3",
            "orbit": {
                **DEFAULT_CUE_ORBIT,
                "hp": 607555.648,
                "ha": 631114.837,
                "i_deg": 97.86514327,
                "RAAN_deg": 336.41909538,
                "argp_deg": 81.38853681,
                "M_deg": 278.78299862,
            },
            "sensor": {
                **DEFAULT_CUE_SENSOR,
            },
        },
        {
            "name": "WorldView_Legion_1",
            "orbit": {
                **DEFAULT_CUE_ORBIT,
                "hp": 498512.806,
                "ha": 513950.817,
                "i_deg": 97.53374308,
                "RAAN_deg": 338.37424302,
                "argp_deg": 351.18214474,
                "M_deg": 120.55393871,
            },
            "sensor": {
                **DEFAULT_CUE_SENSOR,
            },
        },
        {
            "name": "WorldView_Legion_2",
            "orbit": {
                **DEFAULT_CUE_ORBIT,
                "hp": 488527.365,
                "ha": 522768.732,
                "i_deg": 97.53420476,
                "RAAN_deg": 338.45690536,
                "argp_deg": 108.45179562,
                "M_deg": 181.21846214,
            },
            "sensor": {
                **DEFAULT_CUE_SENSOR,
            },
        },
        {
            "name": "WorldView_Legion_3",
            "orbit": {
                **DEFAULT_CUE_ORBIT,
                "hp": 505008.161,
                "ha": 533153.991,
                "i_deg": 44.99852967,
                "RAAN_deg": 38.60086397,
                "argp_deg": 103.66206474,
                "M_deg": 127.04083064,
            },
            "sensor": {
                **DEFAULT_CUE_SENSOR,
            },
        },
        {
            "name": "WorldView_Legion_4",
            "orbit": {
                **DEFAULT_CUE_ORBIT,
                "hp": 510016.350,
                "ha": 528499.590,
                "i_deg": 45.03373835,
                "RAAN_deg": 130.06010940,
                "argp_deg": 147.33811694,
                "M_deg": 83.23906155,
            },
            "sensor": {
                **DEFAULT_CUE_SENSOR,
            },
        },
        {
            "name": "WorldView_Legion_5",
            "orbit": {
                **DEFAULT_CUE_ORBIT,
                "hp": 512642.657,
                "ha": 525654.733,
                "i_deg": 45.06681871,
                "RAAN_deg": 308.98569530,
                "argp_deg": 239.39320466,
                "M_deg": 300.57582957,
            },
            "sensor": {
                **DEFAULT_CUE_SENSOR,
            },
        },
        {
            "name": "WorldView_Legion_6",
            "orbit": {
                **DEFAULT_CUE_ORBIT,
                "hp": 505513.577,
                "ha": 534486.647,
                "i_deg": 44.97349837,
                "RAAN_deg": 218.16036407,
                "argp_deg": 47.92641334,
                "M_deg": 39.74172267,
            },
            "sensor": {
                **DEFAULT_CUE_SENSOR,
            },
        },
    ],

}