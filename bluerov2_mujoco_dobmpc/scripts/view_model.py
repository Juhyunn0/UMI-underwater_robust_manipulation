"""Open the MJCF model in the interactive MuJoCo viewer (needs a display).

Note: the world frame is NED, i.e. +z points DOWN - the scene will look
upside-down unless you orbit the camera below the horizon.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import mujoco.viewer

from bluerov2mj.mujoco_env import _XML

model = mujoco.MjModel.from_xml_path(_XML)
data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)
