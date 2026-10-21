#!/usr/bin/env python3
"""Environment integration check only: no policy, optimizer or training updates."""
import argparse,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'isaaclab_migration'))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT/'isaaclab_migration/outputs/terrain_inspection/environment_check.json')
AppLauncher.add_app_launcher_args(parser);args=parser.parse_args();args.enable_cameras=False
launcher=AppLauncher(args);app=launcher.app
try:
    import torch,numpy as np,random
    torch.manual_seed(1);np.random.seed(1);random.seed(1)
    from wmp_lab.a1_config import A1AMPCfg
    from wmp_lab.competition import configure_competition, COMPETITION_TYPES, PLUM_TYPES, TREAD_DEPTH_FRACTION, tile_center, maximum_level
    from wmp_lab.env_cfg import make_env_cfg
    from wmp_lab.env import WMPEnv,Adapter
    c=A1AMPCfg();configure_competition(c,'robocon','eval')
    # One column per kind exercises the same generator and all integration hooks.
    count = 2 * len(COMPETITION_TYPES)
    c.terrain.num_cols=len(COMPETITION_TYPES)
    c.terrain.terrain_proportions=[0.]*10+[1/len(COMPETITION_TYPES)]*len(COMPETITION_TYPES)
    c.terrain.border_size=1.;c.env.num_envs=count;c.depth.camera_num_envs=count
    cfg=make_env_cfg(c,device=args.device);cfg.camera_backend='raycast'
    cfg.sim.enable_scene_query_support=True
    env=WMPEnv(cfg);adapter=Adapter(env)
    env.reset()
    for _ in range(6):
        obs,reward,term,trunc,info=env.step(torch.zeros(count,12,device=env.device))
        assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
    assert obs['policy'].shape==(count,285)
    assert env.core.measured_heights.shape==(count,187)
    assert env.core.measured_forward_heights.shape==(count,525)
    assert len(adapter.depth_index_without_crawl_tilt)==count
    state=env.core.course
    assert set(state.kind.tolist())==set(range(len(COMPETITION_TYPES)))
    expected_levels=torch.tensor([maximum_level(COMPETITION_TYPES[k]) for k in state.kind.tolist()],device=env.device)
    torch.testing.assert_close(env.core.terrain_levels,expected_levels)
    stones=state.ids[state.kind[state.ids]==0]
    assert (env.core.env_origins[stones,2]==0).all()
    spawn_ground=state.heights(env.core.root_states[stones,None,:2],stones)
    assert (spawn_ground==0).all(),'Stone spawn must be on approach ground'
    before=env.robot.data.root_state_w.clone()
    env._reset_idx(torch.tensor([0],device=env.device));env.sync_state(env.core)
    torch.testing.assert_close(env.robot.data.root_state_w[1:],before[1:])
    torch.testing.assert_close(env.core.terrain_levels,expected_levels)
    raw=env.camera.data.output['distance_to_image_plane']
    # The training mesh is a static triangle mesh, not a convex-hull ring.
    from pxr import UsdPhysics,UsdGeom
    prim=env.sim.stage.GetPrimAtPath('/World/ground')
    meshes=[p for p in env.sim.stage.Traverse() if p.GetPath().HasPrefix(prim.GetPath()) and p.IsA(UsdGeom.Mesh)]
    collision=[p for p in meshes if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert collision,'Terrain collision mesh missing'
    # Query PhysX itself: holes and dimples must exist in collision, not just rendering.
    import omni.physx, math
    query=omni.physx.get_physx_scene_query_interface()
    col=env.terrain.column_types.index('tire_array');level=9
    spec=env.terrain.competition_cells[level,col];center=tile_center(level,col)
    tx,ty,height,_=spec.rings[0]
    samples=((tx,ty,0.), (tx+.22,ty,height*(1-TREAD_DEPTH_FRACTION)),
             (tx+.22/math.sqrt(2),ty+.22/math.sqrt(2),height))
    physical_heights=[]
    for x,y,expected in samples:
        hit=query.raycast_closest((center[0]+x,center[1]+y,1.),(0.,0.,-1.),2.)
        assert hit['hit'],'Missing tire/floor collision ray'
        z=1.-float(hit['distance']);physical_heights.append(z)
        assert abs(z-expected)<1e-4,f'Tire collision surface differs: {z} versus {expected}'
    col=env.terrain.column_types.index('plum_stones_flat')
    level=maximum_level('plum_stones_flat')
    spec=env.terrain.competition_cells[level,col];center=tile_center(level,col)
    a,b=spec.boxes[:2]
    last=spec.boxes[5]
    flat_samples=((spec.spawn[0],0.,.2), (a[0],a[1],.2),
                  ((a[0]+b[0])/2,a[1],0.), (spec.exit+.55,0.,.2), (0.,.6,-5.),
                  ((spec.entry+a[0]-.1)/2,a[1],0.),
                  ((last[0]+.1+spec.exit)/2,last[1],0.))
    flat_heights=[]
    for x,y,expected in flat_samples:
        hit=query.raycast_closest((center[0]+x,center[1]+y,1.),(0.,0.,-1.),7.)
        assert hit['hit'],'Missing carved plum road collision'
        z=1.-float(hit['distance']);flat_heights.append(z)
        assert abs(z-expected)<1e-4,f'Carved plum surface differs: {z} versus {expected}'
    flat_ids=state.ids[state.kind[state.ids]==COMPETITION_TYPES.index('plum_stones_flat')]
    assert torch.allclose(env.core.env_origins[flat_ids,2],torch.full_like(env.core.env_origins[flat_ids,2],.2))
    # All three milestone geometries must exist in actual training collision.
    plum_checks=[]
    for kind in PLUM_TYPES:
        col=env.terrain.column_types.index(kind)
        for level,gx,gy in ((0,-.075,.025),(10,.10,.20),(14,.15,.25)):
            spec=env.terrain.competition_cells[level,col];center=tile_center(level,col)
            boxes=sorted(spec.boxes[:6],key=lambda box:box[0])
            for a,b in zip(boxes,boxes[1:]):
                assert abs(b[0]-a[0]-.2-gx)<1e-7
                assert abs(abs(b[1]-a[1])-.2-gy)<1e-7
            samples=[(b[0],b[1],.2) for b in boxes]
            if level==0:
                # X projections overlap; tops remain separate across the open Y gap.
                x=(boxes[0][0]+boxes[1][0])/2
                samples += [(x,boxes[0][1],.2),(x,boxes[1][1],.2),(x,0.,0.)]
            samples += [((spec.entry+boxes[0][0]-.1)/2,boxes[0][1],0.),
                        ((boxes[-1][0]+.1+spec.exit)/2,boxes[-1][1],0.)] if kind=='plum_stones_flat' else []
            for x,y,expected in samples:
                hit=query.raycast_closest((center[0]+x,center[1]+y,1.),(0.,0.,-1.),2.)
                assert hit['hit'] and abs(1-float(hit['distance'])-expected)<1e-4
            plum_checks.append(dict(kind=kind,level=level,longitudinal_gap=gx,transverse_gap=gy))
    assert env.core.cfg.terrain.num_rows==15
    assert state.attempts.shape==(len(COMPETITION_TYPES),15)
    result={'status':'passed','training_updates':0,'optimizer_created':False,'num_envs':count,
            'curriculum_shape':list(state.attempts.shape),
            'kinds':state.kind.tolist(),'levels':env.core.terrain_levels.tolist(),'obs_shape':list(obs['policy'].shape),
            'height_shape':list(env.core.measured_heights.shape),'forward_height_shape':list(env.core.measured_forward_heights.shape),
            'depth_shape':list(raw.shape),'partial_reset':'passed','stone_spawn_ground':True,'collision_meshes':len(collision),
            'tire_physx_surface_heights':physical_heights,'tire_hole_and_tread_collision':'passed',
            'flat_plum_physx_surface_heights':flat_heights,'flat_plum_level_road':'passed',
            'flat_plum_entry_exit_gaps':'passed','adjacent_staggered_plum_geometry':plum_checks}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print('ENVIRONMENT_CHECK_PASSED '+json.dumps(result),flush=True)
    env.close()
except BaseException:
    import traceback
    traceback.print_exc();sys.stdout.flush();sys.stderr.flush();raise
finally:
    app.close(wait_for_replicator=False)
