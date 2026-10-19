#!/usr/bin/env python3
"""Train or replay WMP/A1 with the existing local Isaac Lab installation."""
import argparse,sys,json,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'isaaclab_migration'))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
parser.add_argument('--camera_backend',choices=['raycast','rtx'],default='raycast')
parser.add_argument('--num_envs',type=int,default=128)
parser.add_argument('--camera_envs',type=int,default=None,help='Depth cameras: competition defaults to 25%% of environments, legacy profiles to all')
parser.add_argument('--iterations',type=int,default=1000)
parser.add_argument('--save_interval',type=int,default=100)
parser.add_argument('--rollout_steps',type=int,default=24)
parser.add_argument('--terrain',choices=['plane','mixed','robocon','robocon_mixed','plum_stones','tire_array','high_wall'],default='mixed')
parser.add_argument('--competition_stage',choices=['learn','standard','eval'],default='learn')
parser.add_argument('--episode_seconds',type=float,default=20.)
parser.add_argument('--wm_batch',type=int,default=2)
parser.add_argument('--wm_length',type=int,default=16)
parser.add_argument('--wm_start',type=int,default=256)
parser.add_argument('--wm_updates',type=int,default=1)
parser.add_argument('--amp_preload',type=int,default=100000)
parser.add_argument('--amp_replay_size',type=int,default=100000)
parser.add_argument('--depth_batch',type=int,default=32)
parser.add_argument('--depth_iters',type=int,default=1)
parser.add_argument('--checkpoint',type=Path)
parser.add_argument('--runtime_config',action='store_true',help='Override checkpoint batch/replay/update settings with explicit CLI resource settings; keep networks and optimizer states')
parser.add_argument('--reset_course_stats',action='store_true',help='Reset competition curriculum counters after loading a checkpoint under changed rules')
parser.add_argument('--compact_courses',action='store_true',help='One geometric lane per competition type; retain all 10 levels')
parser.add_argument('--video_interval',type=int,default=0)
parser.add_argument('--progress_interval',type=int,default=0,help='Write course progress and checkpoint at this interval without recording video')
parser.add_argument('--video_steps',type=int,default=1000)
parser.add_argument('--video_smoke',action='store_true')
parser.add_argument('--record_course_videos',action='store_true',help='Record six course clips from the loaded checkpoint, then exit without training')
parser.add_argument('--record_round',type=int,default=None,help='Cumulative training round label for --record_course_videos')
parser.add_argument('--log_dir',type=Path,default=ROOT/'isaaclab_migration/logs/a1_wmp')
parser.add_argument('--capture_dir',type=Path)
parser.add_argument('--capture_every_steps',type=int,default=20)
parser.add_argument('--capture_fps',type=int,default=0,help='Capture at real simulation-time FPS (50 Hz control loop); overrides capture interval')
parser.add_argument('--follow_camera',action='store_true')
parser.add_argument('--eval_terrain_col',type=int,default=None,help='Playback: put viewed robot on this terrain column')
parser.add_argument('--eval_terrain_level',type=int,default=6,help='Playback terrain difficulty row, 0-9')
parser.add_argument('--fixed_forward_speed',type=float,default=None,help='Playback: hold a forward command in m/s')
parser.add_argument('--play',action='store_true')
parser.add_argument('--probe',action='store_true')
parser.add_argument('--probe_full_wm_batch',action='store_true',help='Synthetic full-length WM memory probe after checkpoint saving')
parser.add_argument('--reward_schedule_scale',type=float,default=1.,help='Scale iteration-based reward curriculum milestones')
parser.add_argument('--require_wm_updates',action='store_true')
parser.add_argument('--seconds',type=float,default=30.,help='Playback wall seconds; zero runs until window closes')
parser.add_argument('--play_steps',type=int,default=0,help='Playback control steps; overrides wall seconds when positive')
AppLauncher.add_app_launcher_args(parser); args=parser.parse_args(); args.enable_cameras=args.camera_backend=='rtx'
if args.num_envs<4:parser.error('Use at least 4 environments for WMP terrain sampling')
if args.camera_envs is None:
    args.camera_envs=max(1,args.num_envs//4) if args.terrain not in ('plane','mixed') else args.num_envs
if not 1<=args.camera_envs<=args.num_envs:parser.error('--camera_envs must be between 1 and --num_envs')
if args.save_interval<1:parser.error('--save_interval must be positive')
if args.capture_every_steps<1 or args.play_steps<0 or not 0<=args.capture_fps<=50:parser.error('Capture interval/play steps or capture FPS is invalid')
if args.eval_terrain_col is not None and (not args.play or args.eval_terrain_col<0):parser.error('--eval_terrain_col requires --play and a nonnegative column (validated against the selected profile)')
if args.eval_terrain_level not in range(10):parser.error('--eval_terrain_level must be 0-9')
if args.fixed_forward_speed is not None and (not args.play or args.fixed_forward_speed<0):parser.error('--fixed_forward_speed requires --play and a nonnegative speed')
if args.record_course_videos and (not args.play or not args.checkpoint or args.headless or args.terrain!='robocon' or args.record_round is None):
    parser.error('--record_course_videos requires GUI --play, --checkpoint, --terrain robocon and --record_round')
if min(args.wm_batch,args.wm_length,args.wm_updates,args.amp_preload,args.amp_replay_size,args.depth_batch,args.depth_iters)<1:
    parser.error('Training batch, sequence, update and replay settings must be positive')
from wmp_lab.validation import MemoryMonitor
args.log_dir.mkdir(parents=True,exist_ok=True)
(args.log_dir/'pid').write_text(str(os.getpid())+'\n')
monitor=MemoryMonitor(0); launcher=AppLauncher(args); app=launcher.app
# Keep long GUI training runs active when the desktop is locked or the Isaac
# window is obscured.  Periodic course videos require a viewport, so this run
# cannot use AppLauncher headless mode.
import carb
_kit_settings=carb.settings.get_settings()
_kit_settings.set('/app/renderer/skipWhileMinimized',False)
_kit_settings.set('/rtx/ecoMode/enabled',False)
_kit_settings.set('/exts/isaacsim.core.throttling/enable_async',False)
_kit_settings.set('/exts/isaacsim.core.throttling/enable_manualmode',False)
print('KIT_BACKGROUND_SETTINGS',{
    'skipWhileMinimized':_kit_settings.get('/app/renderer/skipWhileMinimized'),
    'ecoMode':_kit_settings.get('/rtx/ecoMode/enabled'),
    'throttling_async':_kit_settings.get('/exts/isaacsim.core.throttling/enable_async'),
    'throttling_manual':_kit_settings.get('/exts/isaacsim.core.throttling/enable_manualmode'),
},flush=True)
try:
    import numpy as np,torch,random
    torch.set_num_threads(8); torch.manual_seed(1); np.random.seed(1); random.seed(1)
    from wmp_lab.a1_config import A1AMPCfg,A1AMPCfgPPO
    from wmp_lab.config import class_to_dict,scale_reward_schedule
    from wmp_lab.env_cfg import make_env_cfg
    from wmp_lab.env import WMPEnv,Adapter
    from wmp_lab.runner import Runner
    cfg=A1AMPCfg(); cfg.env.num_envs=args.num_envs; cfg.depth.camera_num_envs=args.camera_envs; cfg.env.episode_length_s=args.episode_seconds
    # Keep all paper terrain rows and columns; reduce only the outer border.
    cfg.terrain.num_rows=10; cfg.terrain.num_cols=20; cfg.terrain.border_size=5.
    cfg.terrain.mesh_type='plane' if args.terrain=='plane' else 'trimesh'
    if args.terrain=='plane':cfg.terrain.curriculum=False
    if args.terrain not in ('plane','mixed'):
        from wmp_lab.competition import configure_competition
        configure_competition(cfg,args.terrain,'eval' if args.play else args.competition_stage)
        if args.compact_courses:
            if args.terrain!='robocon':raise ValueError('Compact courses require robocon')
            cfg.terrain.num_cols=3;cfg.terrain.terrain_proportions=[0.]*10+[1/3]*3
    if args.eval_terrain_col is not None and args.eval_terrain_col>=cfg.terrain.num_cols:
        raise ValueError(f'Column {args.eval_terrain_col} outside selected terrain profile ({cfg.terrain.num_cols} columns)')
    if args.eval_terrain_col is not None:
        cfg.terrain.curriculum=False
        cfg.terrain.eval_terrain_col=args.eval_terrain_col
        cfg.terrain.eval_terrain_level=args.eval_terrain_level
    if args.fixed_forward_speed is not None:
        cfg.commands.ranges.lin_vel_x=[args.fixed_forward_speed]*2
        cfg.commands.ranges.flat_lin_vel_x=[args.fixed_forward_speed]*2
    if args.fixed_forward_speed is not None and getattr(cfg.terrain,'competition',False):
        cfg.terrain.competition_speed=(args.fixed_forward_speed,args.fixed_forward_speed)
    cfg.rewards.reward_curriculum_schedule=scale_reward_schedule(cfg.rewards.reward_curriculum_schedule,args.reward_schedule_scale)
    train=class_to_dict(A1AMPCfgPPO()); train['runner']['num_steps_per_env']=args.rollout_steps
    train['runner']['amp_num_preload_transitions']=args.amp_preload
    train['runner']['amp_motion_files']=[str(p) for p in (ROOT/'datasets/mocap_motions').glob('*.txt')]
    train['runner']['save_interval']=args.save_interval; train['algorithm']['amp_replay_buffer_size']=args.amp_replay_size
    train['depth_predictor'].update(training_iters=args.depth_iters,batch_size=args.depth_batch)
    if args.checkpoint:
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        train=saved['train_cfg']; w=saved['wm_config']
        if args.runtime_config:
            train['runner'].update(num_steps_per_env=args.rollout_steps,amp_num_preload_transitions=args.amp_preload)
            train['algorithm']['amp_replay_buffer_size']=args.amp_replay_size
            train['depth_predictor'].update(batch_size=args.depth_batch,training_iters=args.depth_iters)
        else:
            args.wm_batch=w['batch_size']; args.wm_length=w['batch_length']; args.wm_updates=w['train_steps_per_iter'];args.wm_start=w['train_start_steps']
        train['runner']['save_interval']=args.save_interval
        del saved
    lab_cfg=make_env_cfg(cfg,4,args.device); lab_cfg.sim.render_interval=20
    lab_cfg.camera_backend=args.camera_backend
    lab_cfg.viewer.origin_type='asset_root' if args.follow_camera else 'env';lab_cfg.viewer.asset_name='robot';lab_cfg.viewer.env_index=0;lab_cfg.viewer.eye=(-3.,0.,1.65);lab_cfg.viewer.lookat=(1.2,0.,.25)
    lab_cfg.sim.render.antialiasing_mode='FXAA'
    env=WMPEnv(lab_cfg)
    if args.video_interval:
        from pxr import UsdGeom
        for i in range(1,args.num_envs):
            UsdGeom.Imageable(env.sim.stage.GetPrimAtPath(f'/World/envs/env_{i}/Robot')).MakeInvisible()
    capture=None
    if args.capture_dir and not args.headless:
        from wmp_lab.visualization import ViewportCapture
        capture=ViewportCapture(args.capture_dir,args.capture_every_steps,args.follow_camera,args.capture_fps or None)
    probe_result=None
    if args.probe:
        from wmp_lab.probes import check_sim_contract
        probe_result=check_sim_contract(env)
        print('SIM_CONTRACT',probe_result,flush=True)
    import gc,ctypes
    gc.collect();ctypes.CDLL('libc.so.6').malloc_trim(0)
    adapter=Adapter(env,capture)
    args.log_dir.mkdir(parents=True,exist_ok=True)
    (args.log_dir/'config.json').write_text(json.dumps({'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'env':class_to_dict(cfg),'train':train},indent=2))
    runner=Runner(adapter,train,str(args.log_dir),args.device,wm_batch=args.wm_batch,wm_length=args.wm_length,wm_updates=args.wm_updates,wm_start=args.wm_start)
    if args.checkpoint:runner.load(str(args.checkpoint),load_optimizer=not args.play)
    if args.reset_course_stats:
        if not args.checkpoint or args.play or getattr(adapter,'course',None) is None:
            raise ValueError('--reset_course_stats requires competition training from a checkpoint')
        adapter.course.reset_statistics()
        print('COURSE_STATS_RESET',flush=True)
    if args.video_interval:
        if args.headless or args.terrain!='robocon':raise ValueError('Periodic course videos require GUI and --terrain robocon')
    if args.progress_interval and args.terrain!='robocon':raise ValueError('Course progress callback requires --terrain robocon')
    if args.video_interval or args.progress_interval or args.record_course_videos:
        from wmp_lab.training_video import CourseVideo
        if args.record_course_videos:runner.training_origin=10000
        else:runner.training_origin=getattr(runner,'training_origin',runner.current_learning_iteration)
        runner.iteration_callback=CourseVideo(runner,adapter,args.log_dir,args.video_interval or args.progress_interval or 300,args.video_steps,video_enabled=bool(args.video_interval or args.record_course_videos))
        if args.video_smoke:runner.iteration_callback.record(0,steps=50)
    if args.record_course_videos:
        runner.iteration_callback.record(args.record_round,steps=args.video_steps)
        print('COURSE_VIDEO_RECORD_ONLY_COMPLETE',args.record_round,flush=True)
        raise SystemExit(0)
    baseline_wm_updates=runner.wm_updates_done
    t=time.monotonic()
    if not args.play:
        before=next(runner._world_model.encoder.parameters()).detach().clone()
        runner.learn(args.iterations,init_at_random_ep_len=True)
        wm_delta=float((before-next(runner._world_model.encoder.parameters())).abs().max())
    else:
        if not args.checkpoint:raise ValueError('--play requires --checkpoint')
        obs=adapter.get_observations(); history=torch.zeros(args.num_envs,5,42,device=args.device)
        no_cmd=torch.cat((obs[:,53:59],obs[:,62:-187]),1)
        history[:,-1]=no_cmd
        latent=action=None; first=torch.ones(args.num_envs,device=args.device); actions_history=torch.zeros(args.num_envs,5,12,device=args.device)
        feature=torch.zeros(args.num_envs,512,device=args.device); steps=resets=0; state_rows=[]
        with torch.no_grad():
            while app.is_running() and ((steps<args.play_steps) if args.play_steps else (args.seconds==0 or time.monotonic()-t<args.seconds)):
                tick=time.monotonic()
                if adapter.global_counter%5==0:
                    prop=obs[:,53:86]
                    depth_image=runner.depth_predictor(env.core.get_forward_map(),prop)
                    depth_image[env.camera_env_ids]=env.core.depth_buffer[:,-2,...,None]
                    wmobs={'prop':prop,'image':depth_image,'is_first':first}
                    embed=runner._world_model.encoder(wmobs); latent,_=runner._world_model.dynamics.obs_step(latent,action,embed,first,sample=False)
                    feature=runner._world_model.dynamics.get_deter_feat(latent); first.zero_()
                actions=runner.alg.actor_critic.act_inference(obs,history.flatten(1),feature)
                if not torch.isfinite(actions).all():raise RuntimeError('Nonfinite playback actions')
                obs,_,_,dones,_,ids,_=adapter.step(actions)
                actions_history=torch.cat((actions_history[:,1:],actions[:,None]),1); actions_history[ids]=0; action=actions_history.flatten(1)
                first[ids]=1; history[ids]=0
                no_cmd=torch.cat((obs[:,53:59],obs[:,62:-187]),1)
                history=torch.cat((history[:,1:],no_cmd[:,None]),1)
                resets+=len(ids);steps+=1
                state_rows.append([steps,steps*.02,float(env.core.base_lin_vel[0,0]),float(env.core.commands[0,0]),float(env.core.root_states[0,2]),int(dones[0]),float(env.core.root_states[0,0]-env.core.env_origins[0,0]),int(env.core.terrain_levels[0]) if args.terrain!='plane' else -1,int(env.core.terrain_types[0]) if args.terrain!='plane' else -1])
                time.sleep(max(0.,.02-(time.monotonic()-tick)))
        wm_delta=None
    full_batch_probe=None
    if args.probe_full_wm_batch:
        if args.play:raise ValueError('--probe_full_wm_batch is for training diagnostics')
        b,l=args.wm_batch,args.wm_length
        data={'prop':torch.zeros(b,l,33),'image':torch.zeros(b,l,64,64,1),
              'action':torch.zeros(b,l,60),'reward':torch.zeros(b,l),'is_first':torch.zeros(b,l)}
        data['is_first'][:,0]=1
        _,_,metrics=runner._world_model._train(data)
        if not all(np.isfinite(np.asarray(v)).all() for v in metrics.values()):raise RuntimeError('Nonfinite WM full-batch probe')
        full_batch_probe={'status':'passed','batch':b,'sequence_length':l,'synthetic':True,'checkpoint_saved_before_probe':True}
        print('WM_FULL_BATCH_PROBE_PASSED',full_batch_probe,flush=True)
    if capture:capture.capture_final(env.sim,'playback.png' if args.play else 'training.png')
    raw=env.camera.data.output['distance_to_image_plane']
    # Save actual camera data independently of the desktop viewport.
    from PIL import Image
    depth=env.core.depth_buffer[0,-1].detach().cpu().numpy()
    Image.fromarray(((depth+.5)*255).clip(0,255).astype('uint8')).save(args.log_dir/'depth.png')
    np.save(args.log_dir/'depth_meters.npy',torch.nan_to_num(raw[0,...,0],posinf=2.).cpu().numpy())
    import resource
    report={'status':'passed','mode':'play' if args.play else 'train','num_envs':args.num_envs,'camera_envs':args.camera_envs,'terrain':args.terrain,'camera_backend':args.camera_backend,'iterations':runner.current_learning_iteration,'world_model_updates':runner.wm_updates_done,'new_world_model_updates':runner.wm_updates_done-baseline_wm_updates,'depth_predictor_updates':runner.depth_updates_done,'world_model_encoder_max_change':wm_delta,'elapsed_seconds':time.monotonic()-t,'peak_gpu_used_mib':monitor.stop(),'torch_peak_mib':torch.cuda.max_memory_allocated()/2**20,'peak_process_ram_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'obs_shape':list(env.core.obs_buf.shape),'depth_min':float(depth.min()),'depth_max':float(depth.max()),'joint_names':env.core.dof_names,'joint_ids':env.joint_ids}
    report['wm_full_batch_probe']=full_batch_probe
    if not args.play:
        report['world_model_effective_sequence_min']=min(runner.wm_effective_lengths) if runner.wm_effective_lengths else None
        report['world_model_effective_sequence_mean']=float(np.mean(runner.wm_effective_lengths)) if runner.wm_effective_lengths else None
        report['world_model_effective_sequence_max']=max(runner.wm_effective_lengths) if runner.wm_effective_lengths else None
        report['world_model_predicted_depth_samples']=runner.wm_predicted_depth_samples
    report['checkpoint_output_round_trip_verified']=getattr(runner,'round_trip_verified',False)
    report['checkpoint_output_max_errors']=getattr(runner,'round_trip_errors',None)
    if args.play:
        report.update(play_steps=steps,resets=resets,simulation_seconds=steps*.02,first_robot_mean_vx=float(np.mean([r[2] for r in state_rows])) if state_rows else None,first_robot_max_forward_offset_m=max((r[6] for r in state_rows),default=None))
        import csv
        with (args.log_dir/'state.csv').open('w') as stream:
            writer=csv.writer(stream);writer.writerow(['step','sim_seconds','vx','command_x','base_z','done','forward_offset_m','terrain_level','terrain_col']);writer.writerows(state_rows)
    report['sim_probes']=probe_result
    if args.require_wm_updates and (runner.wm_updates_done==0 or wm_delta is None or wm_delta<=0):raise RuntimeError('World model did not update')
    if not args.play:
        for module in [runner.alg.actor_critic,runner.alg.discriminator,runner._world_model,runner.depth_predictor]:
            if not all(torch.isfinite(p).all() for p in module.parameters()):raise RuntimeError('Nonfinite trained parameters')
    if not torch.isfinite(env.core.obs_buf).all():raise RuntimeError('Nonfinite observations')
    (args.log_dir/'validation.json').write_text(json.dumps(report,indent=2)); print('VALIDATION',json.dumps(report),flush=True)
    if runner.writer:runner.writer.close()
    env.close()
except SystemExit:
    raise
except BaseException:
    import traceback
    (args.log_dir/'failure.json').write_text(json.dumps({'traceback':traceback.format_exc(),'unix_time':time.time()},indent=2))
    traceback.print_exc();sys.stdout.flush();sys.stderr.flush()
    raise
finally:
    monitor.stop()
    if 'env' in globals():env.close()
    app.close(wait_for_replicator=False)
