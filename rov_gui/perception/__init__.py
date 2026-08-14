"""rov_gui.perception — object tracking and 6-DoF pose, as an optional add-on.

This package deliberately imports NOTHING at module scope. Importing it must
stay free on a laptop with no CUDA, no torch and no SAM2 checkpoints, because
``rov_gui`` itself has to keep starting there — the same rule
``sam2_live/__init__.py`` follows upstream, and the same rule that keeps
``rov_gui`` runnable without depthai (see ``rov_gui/imaging.py`` on why
``c3_camera`` is copied rather than imported).

The heavy imports live inside :meth:`PoseSession.start`, which runs on the
perception worker's thread, after the QApplication exists.
"""
