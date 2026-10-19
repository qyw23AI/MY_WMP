"""Synchronous checkpoint videos using the training simulation, without optimizer updates.
Completed WM/AMP replay stays resident. Episodes and incomplete WM trajectories restart.
"""
import json,random,time
from pathlib import Path
import numpy as np
import torch
from .competition import COMPETITION_TYPES


def atomic_json(path,data):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)

class CourseVideo:
    def __init__(self,runner,adapter,directory,interval,steps,video_enabled=True):
        self.r=runner;self.a=adapter;self.root=Path(directory).resolve();self.interval=interval;self.steps=steps
        self.video_enabled=video_enabled
        self.origin=runner.training_origin
        self.initial_wm_updates=runner.wm_updates_done;self.initial_depth_updates=runner.depth_updates_done
        self.index=[]
        if video_enabled:
            videos=self.root/'videos';videos.mkdir(parents=True,exist_ok=True)
            self.index=json.loads((videos/'index.json').read_text()) if (videos/'index.json').exists() else []
            if not (videos/'index.md').exists():
                (videos/'index.md').write_text('首组正式录像将在本次训练完成300轮后自动生成。每300轮：三类地形各含课程前沿与9级，共六段20秒MP4。\n')
                atomic_json(videos/'index.json',self.index)
    def __call__(self,iteration):
        local=iteration-self.origin
        c=self.a.course
        state={k:v.tolist() if torch.is_tensor(v) else v for k,v in c.state_dict().items()}
        summaries={}
        for k,name in enumerate(COMPETITION_TYPES):
            n=c.attempts[k].sum().item();den=max(n,1);counts=c.totals[k].sum(0).tolist()
            top=int(c.attempts[k,9])
            summaries[name]=dict(attempts=n,strict_success_rate=float(c.strict_wins[k].sum())/den,
                failure_rate=counts[0]/den,violation_rate=counts[1]/den,timeout_rate=counts[2]/den,
                platform_contact_rates=[x/den for x in counts[3:5]],stone_contact_rates=[x/den for x in counts[5:11]],
                all_six_rate=counts[11]/den,frontier=int(c.frontier[k]),standard_attempts=top,
                standard_strict_success_rate=float(c.strict_wins[k,9])/top if top else None)
        atomic_json(self.root/'progress.json',dict(training_round=local,checkpoint_iteration=iteration,courses=summaries,
                    world_model_updates=self.r.wm_updates_done,depth_updates=self.r.depth_updates_done,
                    world_model_updates_since_start=self.r.wm_updates_done-self.initial_wm_updates,
                    depth_updates_since_start=self.r.depth_updates_done-self.initial_depth_updates,
                    torch_peak_mib=torch.cuda.max_memory_allocated()/2**20,course=state,updated_unix=time.time()))
        if local%self.interval:return False
        checkpoint=self.root/f'model_{iteration}.pt'
        if not checkpoint.exists():self.r.save(str(checkpoint))
        if self.video_enabled:self.record(local)
        return True
    @torch.inference_mode()
    def record(self,local,steps=None):
        from .visualization import ViewportCapture,configure_viewport
        import imageio.v2 as imageio
        from PIL import Image,ImageDraw
        r,a=self.r,self.a;env=a.env;c=a.course;core=env.core
        steps=self.steps if steps is None else steps
        rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all(),np.random.get_state(),random.getstate())
        replay_sizes=r.wm_dataset_size.copy() if hasattr(r,'wm_dataset_size') else None
        update_counts=(r.wm_updates_done,r.depth_updates_done)
        saved=c.state_dict();types=core.terrain_types.clone();levels=core.terrain_levels.clone();kind=c.kind.clone()
        stage=core.cfg.terrain.competition_stage;speed=core.cfg.terrain.competition_speed
        global_counter=core.global_counter
        counters=(core.total_env_steps_counter,core.common_step_counter,env.common_step_counter)
        modules=[r.alg.actor_critic,r.alg.discriminator,r._world_model,r.depth_predictor]
        modes=[m.training for m in modules]
        for m in modules:m.eval()
        configure_viewport((960,540))
        output=self.root/'videos'/f'round_{local:06d}';output.mkdir(parents=True,exist_ok=True)
        print('COURSE_VIDEO_START',local,flush=True)
        try:
            core.cfg.terrain.competition_stage='eval';core.cfg.terrain.competition_speed=(.2,.2)
            for k,name in enumerate(COMPETITION_TYPES):
                for label,level in [('frontier',int(saved['frontier'][k])),('standard',9)]:
                    torch.manual_seed(2027);np.random.seed(2027);random.seed(2027)
                    core.terrain_types.copy_(types);c.kind.copy_(kind)
                    core.terrain_types[0]=next(i for i,n in enumerate(core.terrain.column_types) if n==name);c.kind[0]=k
                    core.terrain_levels[:]=levels;core.terrain_levels[0]=level
                    core.env_origins[:]=core.terrain_origins[core.terrain_levels,core.terrain_types]
                    env.scene.env_origins.copy_(core.env_origins)
                    core.global_counter=0
                    c.evaluation_events=[]
                    core.episode_length_buf.zero_();a.reset()
                    capture=ViewportCapture(output/f'{name}_{label}_L{level}',5,True)
                    obs=a.get_observations();history=torch.zeros(a.num_envs,5,42,device=a.device)
                    actions_history=torch.zeros(a.num_envs,5,12,device=a.device)
                    first=torch.ones(a.num_envs,device=a.device);latent=action=None
                    feature=torch.zeros(a.num_envs,512,device=a.device)
                    rows=[]
                    with torch.no_grad():
                        for step in range(steps):
                            if step%5==0:
                                prop=obs[:,53:86]
                                depth=r.depth_predictor(core.get_forward_map(),prop)
                                depth[env.camera_env_ids]=core.depth_buffer[:,-2,...,None]
                                latent,_=r._world_model.dynamics.obs_step(latent,action,r._world_model.encoder({'prop':prop,'image':depth,'is_first':first}),first,sample=False)
                                feature=r._world_model.dynamics.get_deter_feat(latent);first.zero_()
                            actions=r.alg.actor_critic.act_inference(obs,history.flatten(1),feature)
                            if not torch.isfinite(actions).all():raise RuntimeError('Nonfinite evaluation action')
                            obs,_,_,dones,_,ids,_=a.step(actions)
                            capture.on_step(env)
                            actions_history=torch.cat((actions_history[:,1:],actions[:,None]),1);actions_history[ids]=0;action=actions_history.flatten(1)
                            first[ids]=1;history[ids]=0
                            history=torch.cat((history[:,1:],torch.cat((obs[:,53:59],obs[:,62:-187]),1)[:,None]),1)
                            rows.append([step+1,float(core.root_states[0,0]-c.centers[0,0]),float(core.base_lin_vel[0,0]),bool(dones[0])])
                    clip=capture.directory/'preview.mp4'
                    with imageio.get_writer(clip,fps=10,codec='libx264',quality=8,macro_block_size=2) as writer:
                        frames=sorted(capture.directory.glob('frame_*.png'))
                        for frame in frames:
                            im=Image.open(frame).convert('RGB');im.thumbnail((960,540))
                            ImageDraw.Draw(im).text((12,12),f'{name} | {label} L{level} | new round {local} | checkpoint {self.origin+local} | 0.20 m/s',fill='white',stroke_width=2,stroke_fill='black')
                            writer.append_data(np.asarray(im))
                    reader=imageio.get_reader(clip);reader.get_data(0);reader.close()
                    for frame in frames:frame.unlink()
                    atomic_json(capture.directory/'evaluation.json',dict(terrain=name,level=level,mode=label,round=local,checkpoint=self.origin+local,seed=2027,steps=steps,states=rows,episodes=c.evaluation_events))
                    self.index.append(dict(round=local,terrain=name,level=level,mode=label,video=str(clip.relative_to(self.root))))
                    atomic_json(self.root/'videos'/'index.json',self.index)
                    (self.root/'videos'/'index.md').write_text('\n'.join(f"- round {v['round']} · {v['terrain']} · {v['mode']} L{v['level']}: [MP4]({Path(v['video']).relative_to('videos')})" for v in self.index)+'\n')
                    print('COURSE_VIDEO_SAVED',clip,flush=True)
        finally:
            a.capture=None;core.cfg.terrain.competition_stage=stage;core.cfg.terrain.competition_speed=speed
            core.terrain_types.copy_(types);core.terrain_levels.copy_(levels);c.kind.copy_(kind);c.load_state_dict(saved)
            core.episode_length_buf.zero_();a.reset();c.load_state_dict(saved);core.global_counter=global_counter
            core.total_env_steps_counter,core.common_step_counter,env.common_step_counter=counters
            if hasattr(c,'evaluation_events'):del c.evaluation_events
            env.scene.env_origins.copy_(core.env_origins)
            torch.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1]);np.random.set_state(rng[2]);random.setstate(rng[3])
            for m,mode in zip(modules,modes):m.train(mode)
        if replay_sizes is not None and not np.array_equal(r.wm_dataset_size, replay_sizes):
            raise RuntimeError('Evaluation changed world-model replay sizes')
        if update_counts != (r.wm_updates_done,r.depth_updates_done):raise RuntimeError('Evaluation updated training models')
        for key,value in c.state_dict().items():
            if torch.is_tensor(value) and not torch.equal(value,saved[key]):raise RuntimeError('Evaluation changed course counters')
        print('COURSE_VIDEO_COMPLETE',local,flush=True)
