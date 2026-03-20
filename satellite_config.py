# simulation/satellite_config.py
# Data only. No functions here.


SATELLITE_MODE = {
    "Tip": False,
    "Cue": False,
}
# True  -> Walker constellation mode
# False -> Independent per-satellite mode

nPlanes = 4
nSats = 2

time_delay_tip_cue = 5 * 60
offnadir_limit_deg = 40.0

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
        },
        "sensor": {
            **DEFAULT_CUE_SENSOR,
        },
    },
}


INDEPENDENT_SATELLITES = {
    "Tip": [
        {
            "name": "Tip_1",
            "orbit": {
                **DEFAULT_TIP_ORBIT,
            },
            "sensor": {
                **DEFAULT_TIP_SENSOR,
            },
        },
        {
            "name": "Tip_2",
            "orbit": {
                **DEFAULT_TIP_ORBIT,
                "RAAN_deg": 20.0,
                "M_deg": 70.0,
            },
            "sensor": {
                **DEFAULT_TIP_SENSOR,
                "swath_m": 250e3,
                "GSD": 12.0,
            },
        },
    ],
    "Cue": [
        {
            "name": "Cue_1",
            "orbit": {
                **DEFAULT_CUE_ORBIT,
                "RAAN_deg": 45.0,
                "M_deg": 120.0,
            },
            "sensor": {
                **DEFAULT_CUE_SENSOR,
            },
        },
        {
            "name": "Cue_2",
            "orbit": {
                **DEFAULT_CUE_ORBIT,
                "RAAN_deg": 45.0,
                "M_deg": 140.0,
            },
            "sensor": {
                **DEFAULT_CUE_SENSOR,
                "resolution": 256,
                "sample_count": 1024,
                "GSD": 0.25,
                "swath_m": 10.0e3,
                "offnadir_limit_deg": offnadir_limit_deg,
            },
        },
    ],
}