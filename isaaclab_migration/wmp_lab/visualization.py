"""Capture actual Isaac Sim viewport frames through the installed official API."""
from pathlib import Path


def configure_viewport(resolution):
    """Set actual GUI render texture size through the current viewport API."""
    from omni.kit.viewport.utility import get_active_viewport
    viewport=get_active_viewport()
    if viewport is None:raise RuntimeError('GUI viewport unavailable')
    viewport.fill_frame=False
    viewport.resolution_scale=1.0
    viewport.resolution=tuple(resolution)
    print('VIEWPORT_RESOLUTION',viewport.resolution,flush=True)


def color_terrain(color):
    """Bind only a visual shader; collision geometry/material are unchanged."""
    import isaaclab.sim as sim_utils
    material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color),roughness=1.0,metallic=0.0)
    path='/World/ground/viewerMaterial'
    material.func(path,material)
    sim_utils.bind_visual_material('/World/ground',path)


class ViewportCapture:
    def __init__(self, directory, every_steps=100, follow_camera=False, fps=None):
        from omni.kit.viewport.utility import get_active_viewport
        self.viewport = get_active_viewport()
        if self.viewport is None:
            raise RuntimeError('Viewport unavailable: visualization requires a GUI launch')
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.every_steps = every_steps
        self.fps = fps
        self.follow_camera = follow_camera
        self.step_count = 0
        self.frame_count = 0

    def _wait_for_capture(self, future, sim, path):
        import asyncio
        import time
        task = asyncio.ensure_future(future.wait_for_result())
        deadline = time.monotonic() + 30
        while not task.done() and time.monotonic() < deadline:
            sim.render()
        if not task.done():
            raise RuntimeError(f'Viewport capture timed out: {path}')
        task.result()
        while time.monotonic() < deadline:
            try:
                with path.open('rb') as stream:
                    stream.seek(-12, 2)
                    if stream.read() == b'\x00\x00\x00\x00IEND\xaeB\x60\x82':
                        return path
            except (OSError, ValueError):
                pass
            sim.render()
            time.sleep(.001)
        raise RuntimeError(f'Viewport PNG write timed out: {path}')

    def on_step(self, env=None):
        from omni.kit.viewport.utility import capture_viewport_to_file
        self.step_count += 1
        due = (self.step_count * self.fps // 50 > self.frame_count) if self.fps else (self.step_count % self.every_steps == 0)
        if due:
            if self.follow_camera:
                if env is None or env.viewport_camera_controller is None:
                    raise RuntimeError('Following camera requires a GUI environment')
                env.viewport_camera_controller.update_view_to_asset_root('robot')
                env.sim.render()
            path = self.directory / f'frame_{self.step_count:06d}.png'
            self._wait_for_capture(capture_viewport_to_file(self.viewport, str(path)), env.sim, path)
            self.frame_count += 1

    def capture_final(self, sim, filename='training.png'):
        from omni.kit.viewport.utility import capture_viewport_to_file
        path = self.directory / filename
        path.unlink(missing_ok=True)
        return self._wait_for_capture(capture_viewport_to_file(self.viewport, str(path)), sim, path)
