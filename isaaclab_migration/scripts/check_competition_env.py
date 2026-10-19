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
    from wmp_lab.competition import configure_competition
    from wmp_lab.env_cfg import make_env_cfg
    from wmp_lab.env import WMPEnv,Adapter
    c=A1AMPCfg();configure_competition(c,'robocon','eval')
    # One column per kind exercises the same generator and all integration hooks.
    c.terrain.num_cols=3;c.terrain.terrain_proportions=[0.]*10+[1/3]*3
    c.terrain.border_size=1.;c.env.num_envs=6;c.depth.camera_num_envs=6
    cfg=make_env_cfg(c,device=args.device);cfg.camera_backend='raycast'
    env=WMPEnv(cfg);adapter=Adapter(env)
    env.reset()
    for _ in range(6):
        obs,reward,term,trunc,info=env.step(torch.zeros(6,12,device=env.device))
        assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
    assert obs['policy'].shape==(6,285)
    assert env.core.measured_heights.shape==(6,187)
    assert env.core.measured_forward_heights.shape==(6,525)
    assert len(adapter.depth_index_without_crawl_tilt)==6
    state=env.core.course
    assert set(state.kind.tolist())=={0,1,2}
    assert (env.core.terrain_levels==9).all()
    stones=state.ids[state.kind[state.ids]==0]
    assert (env.core.env_origins[stones,2]==0).all()
    spawn_ground=state.heights(env.core.root_states[stones,None,:2],stones)
    assert (spawn_ground==0).all(),'Stone spawn must be on approach ground'
    before=env.robot.data.root_state_w.clone()
    env._reset_idx(torch.tensor([0],device=env.device));env.sync_state(env.core)
    torch.testing.assert_close(env.robot.data.root_state_w[1:],before[1:])
    assert (env.core.terrain_levels==9).all()
    raw=env.camera.data.output['distance_to_image_plane']
    # The training mesh is a static triangle mesh, not a convex-hull ring.
    from pxr import UsdPhysics,UsdGeom
    prim=env.sim.stage.GetPrimAtPath('/World/ground')
    meshes=[p for p in env.sim.stage.Traverse() if p.GetPath().HasPrefix(prim.GetPath()) and p.IsA(UsdGeom.Mesh)]
    collision=[p for p in meshes if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert collision,'Terrain collision mesh missing'
    result={'status':'passed','training_updates':0,'optimizer_created':False,'num_envs':6,
            'kinds':state.kind.tolist(),'levels':env.core.terrain_levels.tolist(),'obs_shape':list(obs['policy'].shape),
            'height_shape':list(env.core.measured_heights.shape),'forward_height_shape':list(env.core.measured_forward_heights.shape),
            'depth_shape':list(raw.shape),'partial_reset':'passed','stone_spawn_ground':True,'collision_meshes':len(collision)}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print('ENVIRONMENT_CHECK_PASSED '+json.dumps(result),flush=True)
    env.close()
except BaseException:
    import traceback
    traceback.print_exc();sys.stdout.flush();sys.stderr.flush();raise
finally:
    app.close(wait_for_replicator=False)
