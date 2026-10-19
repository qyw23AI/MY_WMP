#!/usr/bin/env python3
"""Inspect the exact training geometry. No robot, runner, checkpoint or optimizer."""
import argparse, sys, time, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'isaaclab_migration'))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--level', type=int, choices=range(10), default=9)
parser.add_argument('--seconds', type=float, default=0., help='0: stay open until manually closed')
parser.add_argument('--capture_dir', type=Path, default=ROOT/'isaaclab_migration/outputs/terrain_inspection')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args); app = launcher.app
try:
    import numpy as np, trimesh
    import isaaclab.sim as sim_utils
    from isaaclab.terrains.utils import create_prim_from_mesh
    from wmp_lab.competition import COMPETITION_TYPES, make_course, course_mesh, tile_center
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.01, device=args.device))
    floor = sim_utils.GroundPlaneCfg(color=(.20,.23,.27), size=(110.,50.))
    floor.func('/World/floor', floor, translation=(40.,12.,0.))
    light = sim_utils.DomeLightCfg(intensity=2200.); light.func('/World/light', light)
    material = sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.)
    metadata = []
    for kind_index, kind in enumerate(COMPETITION_TYPES):
        for level in range(10):
            course = make_course(kind, level)
            vertices, faces = course_mesh(course)
            center = tile_center(level, kind_index)
            vertices[:, :2] += center
            path = f'/World/courses/{kind}/level_{level}'
            create_prim_from_mesh(path, trimesh.Trimesh(vertices=vertices,faces=faces,process=False), physics_material=material)
            color = (.045,.05,.055) if kind=='tire_array' else (.16,.50,.25)
            shader = sim_utils.PreviewSurfaceCfg(diffuse_color=color,roughness=.85)
            shader.func(path+'/material',shader);sim_utils.bind_visual_material(path,path+'/material')
            # Ground-level blue disk marks the base's XY spawn location, visual only.
            marker = sim_utils.CylinderCfg(radius=.07, height=.006,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.05,.25,1.)))
            marker.func(f'/World/spawns/{kind}/level_{level}', marker,
                translation=(center[0]+course.spawn[0], center[1]+course.spawn[1], .006))
            metadata.append({**course.to_dict(), 'tile_center': center,
                             'world_spawn': [center[0]+course.spawn[0], center[1]+course.spawn[1], course.spawn[2]]})
    args.capture_dir.mkdir(parents=True,exist_ok=True)
    (args.capture_dir/'courses.json').write_text(json.dumps(metadata,indent=2))
    sim.reset()
    selected = {'kind':0, 'level':args.level}
    def show_courses(all_levels=False, all_types=False):
        from pxr import UsdGeom
        for kind_index, kind in enumerate(COMPETITION_TYPES):
            for level in range(10):
                show=all_levels or (level==selected['level'] and (all_types or kind_index==selected['kind']))
                for group in ('courses','spawns'):
                    image=UsdGeom.Imageable(sim.stage.GetPrimAtPath(f'/World/{group}/{kind}/level_{level}'))
                    image.MakeVisible() if show else image.MakeInvisible()
    def focus(all_types=False):
        show_courses(all_types=all_types)
        x,y=tile_center(selected['level'],selected['kind'])
        if all_types:sim.set_camera_view(eye=(x-17.,12.,16.),target=(x,12.,.1))
        else:sim.set_camera_view(eye=(x+2.5,y-4.,3.),target=(x-.4,y,.12))
    def overview():
        show_courses(all_levels=True)
        sim.set_camera_view(eye=(40.,-49.,55.),target=(40.,12.,0.))
    def lane():
        show_courses(all_levels=True)
        from pxr import UsdGeom
        for other_index, kind in enumerate(COMPETITION_TYPES):
            if other_index == selected['kind']:continue
            for level in range(10):
                for group in ('courses','spawns'):
                    UsdGeom.Imageable(sim.stage.GetPrimAtPath(f'/World/{group}/{kind}/level_{level}')).MakeInvisible()
        y=tile_center(0,selected['kind'])[1]
        sim.set_camera_view(eye=(40.,y-46.,43.),target=(40.,y,0.))
    overview()
    window = None
    if not args.headless:
        import omni.ui as ui
        window = ui.Window('ROBOCON terrain inspection - NO TRAINING',width=450,height=405)
        with window.frame:
            with ui.VStack(spacing=8):
                ui.Label('Exact training meshes | +X travel | 30 courses',height=22)
                ui.Label('Rows: stones / tires / wall. Left to right: level 0-9.',height=22)
                ui.Label('Level 9 = competition dimensions. No robot or policy.',height=22)
                ui.Label('Blue ground disk = spawn XY. Stones start below platform.',height=22)
                combo = ui.ComboBox(0,'Plum stones','Tire array','High wall',height=28)
                combo.model.add_item_changed_fn(lambda model,item:selected.update(kind=model.get_item_value_model().as_int))
                with ui.HStack(height=28):
                    ui.Label('Difficulty level',width=130)
                    slider=ui.IntSlider(min=0,max=9);slider.model.set_value(args.level)
                    slider.model.add_value_changed_fn(lambda model:selected.update(level=model.as_int))
                ui.Button('Focus selected course',clicked_fn=lambda:focus(False),height=30)
                ui.Button('Show three courses at selected level',clicked_fn=lambda:focus(True),height=30)
                ui.Button('Overview: all 10 levels',clicked_fn=overview,height=30)
                ui.Button('Show selected type: level 0 to 9',clicked_fn=lane,height=30)
                ui.Label('Mouse: orbit / pan / zoom. Level 9: 20cm stones,',height=20)
                ui.Label('55/33/16.5cm tires, 30cm wall (8cm thick).',height=20)
        for _ in range(20):sim.render()
        from wmp_lab.visualization import ViewportCapture
        capture=ViewportCapture(args.capture_dir)
        capture.capture_final(sim,'curriculum_layout.png')
        focus(True)
        for _ in range(5):sim.render()
        capture.capture_final(sim,'standard_courses.png')
        overview()
    print('TERRAIN_PREVIEW_READY '+json.dumps({'training':False,'robots':0,'courses':30,'initial_level':args.level,'output':str(args.capture_dir)}),flush=True)
    started=time.monotonic()
    while app.is_running() and (args.seconds==0 or time.monotonic()-started<args.seconds):
        sim.render();time.sleep(.02)
except BaseException:
    import traceback
    traceback.print_exc();sys.stdout.flush();sys.stderr.flush()
    raise
finally:
    app.close(wait_for_replicator=False)
