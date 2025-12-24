import os
import importlib.util

spec = importlib.util.find_spec("drjit")
if spec is None or not spec.submodule_search_locations:
    raise RuntimeError("drjit is not installed (or not importable) in this environment.")

drjit_pkg_dir = next(iter(spec.submodule_search_locations))  # ...\Lib\site-packages\drjit
os.add_dll_directory(drjit_pkg_dir)


import mitsuba as mi


mi.set_variant('scalar_rgb')

mi.set_log_level(mi.LogLevel.Trace)

print(" Start render")
img = mi.render(mi.load_dict(mi.cornell_box()))

_ = mi.util.convert_to_bitmap(img)
print("End render")


# mi.util.write_bitmap("my_first_render.png", img)