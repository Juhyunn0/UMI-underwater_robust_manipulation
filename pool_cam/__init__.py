"""pool_cam — the pool-corner USB cameras: watching and recording, nothing more.

    pool_cam.py   the cameras themselves: a standalone preview/recorder window,
                  and --serve, the child process rov_gui runs for its POOL CAMS
                  tab (`./c3 gui --pool-cams`)
    protocol.py   the wire between the two (stdlib only)

Nothing here is an input to any controller, estimator or policy, and nothing in
rov_gui's control path imports it. Kept light on purpose: importing the
package must not import cv2 (see protocol.py).
"""
