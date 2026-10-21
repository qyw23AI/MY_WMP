#!/usr/bin/env python3
"""Inspect the exact training geometry. No robot, runner, checkpoint or optimizer."""
import argparse, sys, time, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'isaaclab_migration'))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--level', type=int, choices=range(15), default=14)
parser.add_argument('--types', nargs='+', choices=['plum_stones','platform_plum_combo','plum_stones_flat', 'tire_array', 'high_wall', 'platform', 'ramp', 'platform_gap','stair_gap','stair_gap_ramp'],
                    help='Show only these terrain types, in the supplied order')
parser.add_argument('--seconds', type=float, default=0., help='0: stay open until manually closed')
parser.add_argument('--capture_dir', type=Path, default=ROOT/'isaaclab_migration/outputs/terrain_inspection')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args); app = launcher.app
try:
    import numpy as np, trimesh
    import isaaclab.sim as sim_utils
    from isaaclab.terrains.utils import create_prim_from_mesh
    from wmp_lab.competition import COMPETITION_TYPES, COMBO_TYPES, PLUM_TYPES, standard_level, maximum_level, level_count, make_course, course_mesh, tile_center
    kinds = ['plum_stones' if kind=='platform_plum_combo' else kind for kind in (args.types or list(COMPETITION_TYPES))]
    courses_count=sum(level_count(kind) for kind in kinds)
    max_level=max(maximum_level(kind) for kind in kinds)
    center_y = 4. * len(kinds)
    labels = {'plum_stones':'Platform + plum stones (combo)', 'plum_stones_flat':'Plum stones carved into level road',
              'tire_array':'Hollow tires with physical tread', 'high_wall':'High wall 5-36cm',
              'platform':'Platform 5-30cm', 'ramp':'Ramp 5-30deg', 'platform_gap':'Platform gaps 5-45cm',
              'stair_gap':'A: stairs + jump', 'stair_gap_ramp':'B: stairs + jump + downhill'}
    from wmp_lab.trimesh import box_trimesh
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.01, device=args.device))
    # Per-tile floors keep the jump lane's pit open; a global Z=0 plane would fill it.
    light = sim_utils.DomeLightCfg(intensity=650.); light.func('/World/light', light)
    material = sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.)
    metadata = []
    for kind_index, kind in enumerate(kinds):
        for level in range(level_count(kind)):
            course = make_course(kind, level)
            vertices, faces = course_mesh(course)
            center = tile_center(level, kind_index)
            vertices[:, :2] += center
            path = f'/World/courses/{kind}/level_{level}'
            create_prim_from_mesh(path, trimesh.Trimesh(vertices=vertices,faces=faces,process=False), physics_material=material)
            color = {'tire_array':(.045,.05,.055), 'platform':(.78,.48,.12),
                     'ramp':(.12,.55,.60), 'platform_gap':(.25,.45,.80),
                     'stair_gap':(.78,.48,.12), 'stair_gap_ramp':(.12,.55,.60),
                     'plum_stones_flat':(.15,.48,.72)}.get(kind, (.16,.50,.25))
            shader = sim_utils.PreviewSurfaceCfg(diffuse_color=color,roughness=.85)
            shader.func(path+'/material',shader)
            sim_utils.bind_visual_material(path,path+'/material',stronger_than_descendants=False)
            floor_v, floor_f = box_trimesh(np.array([8., 8., .02]),
                                          np.array([center[0], center[1], course.floor_height - .01]))
            floor_path = path+'/floor'
            create_prim_from_mesh(floor_path, trimesh.Trimesh(floor_v, floor_f, process=False), physics_material=material)
            floor_shader = sim_utils.PreviewSurfaceCfg(diffuse_color=(.24,.26,.29),roughness=1.)
            floor_shader.func(floor_path+'/material',floor_shader)
            sim_utils.bind_visual_material(floor_path,floor_path+'/material')
            # Ground-level blue disk marks the base's XY spawn location, visual only.
            marker = sim_utils.CylinderCfg(radius=.07, height=.006,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.05,.25,1.)))
            marker.func(f'/World/spawns/{kind}/level_{level}', marker,
                translation=(center[0]+course.spawn[0], center[1]+course.spawn[1], course.spawn[2]+.006))
            metadata.append({**course.to_dict(), 'tile_center': center,
                             'world_spawn': [center[0]+course.spawn[0], center[1]+course.spawn[1], course.spawn[2]]})
    args.capture_dir.mkdir(parents=True,exist_ok=True)
    (args.capture_dir/'courses.json').write_text(json.dumps(metadata,indent=2))
    sim.reset()
    selected = {'kind':0, 'level':min(args.level,maximum_level(kinds[0]))}
    def show_courses(all_levels=False, all_types=False):
        from pxr import UsdGeom
        for kind_index, kind in enumerate(kinds):
            for level in range(level_count(kind)):
                show=all_levels or (level==min(selected['level'],maximum_level(kind)) and (all_types or kind_index==selected['kind']))
                for group in ('courses','spawns'):
                    image=UsdGeom.Imageable(sim.stage.GetPrimAtPath(f'/World/{group}/{kind}/level_{level}'))
                    image.MakeVisible() if show else image.MakeInvisible()
    def focus(all_types=False):
        show_courses(all_types=all_types)
        selected['level']=min(selected['level'],maximum_level(kinds[selected['kind']]))
        x,y=tile_center(selected['level'],selected['kind'])
        if all_types and all(kind in COMBO_TYPES for kind in kinds):
            sim.set_camera_view(eye=(x+1.5,center_y-14.,12.),target=(x,center_y,.1))
        elif all_types and len({min(selected['level'],maximum_level(kind)) for kind in kinds})>1:
            overview()
        elif all_types:sim.set_camera_view(eye=(x-7.-3.*len(kinds),center_y-12.,8.+3.*len(kinds)),target=(x,center_y,-.5))
        elif kinds[selected['kind']]=='platform_gap':
            sim.set_camera_view(eye=(x+5.,y-7.,4.),target=(x,y,-1.))
        elif kinds[selected['kind']] in COMBO_TYPES:
            sim.set_camera_view(eye=(x+.5,y-6.,5.),target=(x,y,.15))
        elif kinds[selected['kind']] in PLUM_TYPES:
            sim.set_camera_view(eye=(x+.5,y-5.,4.),target=(x,y,.1))
        else:sim.set_camera_view(eye=(x+2.5,y-4.,3.),target=(x-.4,y,.12))
    def overview():
        show_courses(all_levels=True)
        mid=4.*(max_level+1)
        sim.set_camera_view(eye=(mid,center_y-(max_level+1)*8.4,(max_level+1)*7.),target=(mid,center_y,0.))
    def lane():
        show_courses(all_levels=True)
        from pxr import UsdGeom
        for other_index, kind in enumerate(kinds):
            if other_index == selected['kind']:continue
            for level in range(level_count(kind)):
                for group in ('courses','spawns'):
                    UsdGeom.Imageable(sim.stage.GetPrimAtPath(f'/World/{group}/{kind}/level_{level}')).MakeInvisible()
        y=tile_center(0,selected['kind'])[1]
        mid=4.*level_count(kinds[selected['kind']])
        sim.set_camera_view(eye=(mid,y-mid*1.15,mid*1.075),target=(mid,y,0.))
    def tread_detail():
        if 'tire_array' not in kinds:return
        selected['kind']=kinds.index('tire_array')
        selected['level']=min(selected['level'],maximum_level('tire_array'))
        show_courses()
        x,y=tile_center(selected['level'],selected['kind'])
        sim.set_camera_view(eye=(x-1.3,y-1.1,.8),target=(x-.55,y-.275,.10))
    overview()
    window = None
    if not args.headless:
        import omni.ui as ui
        window = ui.Window('ROBOCON terrain inspection - NO TRAINING',width=480,height=650)
        with window.frame:
            with ui.VStack(spacing=8):
                ui.Label(f'Exact training meshes | +X travel | {courses_count} courses',height=22)
                ui.Label('Plum: L0-14; other terrain types: L0-9.',height=22)
                ui.Label('Plum L10 = competition; L14 = target.',height=22)
                ui.Label('Blue disk = spawn XY. Gaps start on the first platform.',height=22)
                combo = ui.ComboBox(0,*(labels[kind] for kind in kinds),height=28)
                combo.model.add_item_changed_fn(lambda model,item:selected.update(kind=model.get_item_value_model().as_int))
                with ui.HStack(height=28):
                    ui.Label('Difficulty level',width=130)
                    slider=ui.IntSlider(min=0,max=max_level);slider.model.set_value(selected['level'])
                    slider.model.add_value_changed_fn(lambda model:selected.update(level=model.as_int))
                def choose_level(level):
                    selected['level']=level;slider.model.set_value(level);focus(False)
                with ui.HStack(height=30):
                    ui.Button('Competition level',clicked_fn=lambda:choose_level(standard_level(kinds[selected['kind']])))
                    ui.Button('Final target level',clicked_fn=lambda:choose_level(maximum_level(kinds[selected['kind']])))
                ui.Button('Focus selected course',clicked_fn=lambda:focus(False),height=30)
                ui.Button('Show all types at selected level',clicked_fn=lambda:focus(True),height=30)
                ui.Button('Overview: all levels',clicked_fn=overview,height=30)
                ui.Button('Show selected type: all levels',clicked_fn=lane,height=30)
                if 'tire_array' in kinds:
                    ui.Button('Inspect physical tire tread',clicked_fn=tread_detail,height=30)
                ui.Label('Adjacent staggered gaps: X -7.5 to 15cm, Y 2.5-25cm.',height=20)
                ui.Label('Plum tops fixed 20x20x20cm; entry gaps stay positive.',height=20)
                ui.Label('L9 wall: 36cm. L9 tire: 19.8cm, hole always open.',height=20)
                ui.Label('Tire dimples are collision geometry, fixed at all levels.',height=20)
                ui.Label('No ground contacts between plum stones are allowed.',height=20)
        for _ in range(20):sim.render()
        from wmp_lab.visualization import ViewportCapture
        capture=ViewportCapture(args.capture_dir)
        capture.capture_final(sim,'curriculum_layout.png')
        focus(True)
        for _ in range(5):sim.render()
        capture.capture_final(sim,'standard_courses.png')
        for kind_index, kind in enumerate(kinds):
            selected['kind'] = kind_index
            levels = sorted({0, standard_level(kind), min(args.level,maximum_level(kind))})
            for level in levels:
                selected['level']=level
                focus(False)
                for _ in range(8):sim.render()
                capture.capture_final(sim,f'{kind}_L{level}.png')
        selected['kind'] = 0
        selected['level'] = min(args.level,maximum_level(kinds[0]))
        if 'tire_array' in kinds:
            tread_detail()
            for _ in range(8):sim.render()
            capture.capture_final(sim,f'tire_tread_L{args.level}.png')
        selected['kind'] = 0
        focus(False)
    print('TERRAIN_PREVIEW_READY '+json.dumps({'training':False,'robots':0,'courses':courses_count,'types':kinds,'initial_level':args.level,'output':str(args.capture_dir)}),flush=True)
    started=time.monotonic()
    while app.is_running() and (args.seconds==0 or time.monotonic()-started<args.seconds):
        sim.render();time.sleep(.02)
except BaseException:
    import traceback
    traceback.print_exc();sys.stdout.flush();sys.stderr.flush()
    raise
finally:
    app.close(wait_for_replicator=False)
