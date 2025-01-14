# performance tuning
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
cv2.setNumThreads(0)

import click

from mediapipe_apiserver.camera import ZED2Camera
from mediapipe_apiserver.detector import MediaPipeDetector
from mediapipe_apiserver.common.option import CameraOption
from mediapipe_apiserver.restful import controller_app

from loguru import logger

@click.group()
@click.pass_context
def cli(ctx):
    # root function, do nothing
    pass

@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option('--port', default=3000, help="listen port")
@click.option('--debug', default=False, help="toggle sanic debug")
@click.pass_context
def serve(ctx, port: int, debug: bool):
    # create instances
    opt = CameraOption(use_depth=False)
    cam = ZED2Camera("0", opt)
    cam.open()
    detector = MediaPipeDetector()

    # hook objects to controller for later access
    controller_app.ctx.camera = cam
    controller_app.ctx.detector = detector
    
    # controller_app.run never returns
    controller_app.run(host="0.0.0.0", port=port, single_process=True, auto_reload=False, debug=debug)


def entrypoint():
    # enter click logic
    cli()