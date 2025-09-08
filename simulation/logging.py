import os
from openpyxl import Workbook, load_workbook
import math
from datetime import timedelta
import pandas as pd

import os
import shutil
import atexit

def init_excel_log(path, header, sheet_name="Log"):
    if os.path.exists(path):
        os.remove(path)

    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name
    ws.append(header)
    wb.save(path)
    wb.close()

    # return writer metadata
    return {
        "path": path,
        "sheet": sheet_name,
        "header": header
    }


def append_excel_log(writer, row):
    path = writer["path"]
    sheet = writer["sheet"]
    header = writer["header"]

    wb = load_workbook(path)
    ws = wb[sheet] if sheet in wb.sheetnames else wb.active
    if ws.max_row == 0:
        ws.append(header)
    ws.append(row)
    wb.save(path)
    wb.close()




def log_tip_detection(writer, t_datetime, actor, whale_idx, target_coord, r, v, offnadir_tip_deg, gsd_m,
                      in_footprint):
    if should_log_event(writer, whale_idx, t_datetime, min_gap_sec=1000):
        row = [whale_idx,
            t_datetime.isoformat(), actor.name,
            target_coord[0], target_coord[1], target_coord[2],
            r[0], r[1], r[2],
            v[0], v[1], v[2],
            offnadir_tip_deg,
            gsd_m, int(in_footprint)
        ]
        append_excel_log(writer, row)


def log_cue_evaluation(writer, t_datetime, actor, whale_idx, target_coord, r, v,
                       offnadir_cue_deg, gsd_m, in_view, in_footprint, yaw, pitch, roll):

    if should_log_event(writer, whale_idx, t_datetime, min_gap_sec=1000):
        row = [
            whale_idx,
            t_datetime.isoformat(), actor.name,
            target_coord[0], target_coord[1], target_coord[2],
            r[0], r[1], r[2],
            v[0], v[1], v[2],
            offnadir_cue_deg, gsd_m,
            int(in_view), int(in_footprint),
            yaw, pitch, roll
        ]
        append_excel_log(writer, row)


def gsd_offnadir(Pn_m, H_m, offnadir_deg):
    """
    Compute exact off-nadir ground sampling distance (GSD).

    Parameters
    ----------
    Pn_m : float
        Nadir ground sampling distance [m]
    H_m : float
        Sensor altitude above ground [m]
    offnadir_deg : float
        Off-nadir angle [deg]

    Returns
    -------
    Ptheta_m : float
        Off-nadir ground sampling distance [m]
    """
    theta = math.radians(offnadir_deg)

    # Step 1: pixel angular width beta from nadir GSD
    beta = 2.0 * math.atan(Pn_m / (2.0 * H_m))

    # Step 2: exact off-nadir GSD
    Ptheta_m = H_m * (math.tan(theta + beta/2.0) - math.tan(theta - beta/2.0))

    return Ptheta_m


def should_log_event(writer, whale_idx, t_datetime, min_gap_sec=600):
    import pandas as pd
    from datetime import timedelta

    path = writer["path"]
    sheet = writer["sheet"]

    if not os.path.exists(path):
        return True

    df = pd.read_excel(path, sheet_name=sheet)
    if df.empty:
        return True

    df_target = df[df["target_id"] == whale_idx]
    if df_target.empty:
        return True

    last_time = pd.to_datetime(df_target["date"].max())
    if (t_datetime - last_time) > timedelta(seconds=min_gap_sec):
        return True

    return False

def at_exit(save_name):
    print("Runtime ended. ")
    results_dir = os.path.join("results", save_name)
    os.makedirs(results_dir, exist_ok=True)

    # files to rename + move
    files_map = {
        "sim_output_tip.xlsx": f"{save_name}_tip.xlsx",
        "sim_output_cue.xlsx": f"{save_name}_cue.xlsx",
    }

    for src, dst in files_map.items():
        print(src)
        if os.path.exists(src):
            shutil.move(src, os.path.join(results_dir, dst))
        else:
            print(f"Warning: {src} not found, skipping.")

    # also copy settings.py
    settings_file = "settings.py"
    if os.path.exists(settings_file):
        shutil.copy(settings_file, os.path.join(results_dir, settings_file))
    else:
        print("Warning: settings.py not found, skipping.")

    print(f"Saved results in results/{save_name}")