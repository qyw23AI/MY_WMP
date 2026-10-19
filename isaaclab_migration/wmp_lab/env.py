"""Isaac Lab simulation with real tiled depth cameras and WMP runner contract."""
import math, random, re
import numpy as np
import torch, trimesh
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, TiledCamera, TiledCameraCfg, RayCasterCamera, RayCasterCameraCfg, patterns
from isaaclab.terrains.utils import create_prim_from_mesh
from .config import asset_contract
from .core import Core
from .math import quat_rotate_inverse, torch_rand_float

class WMPEnv(DirectRLEnv):
    def __init__(self,cfg,**kwargs):
        self.core=None; super().__init__(cfg,**kwargs)
        if cfg.camera_backend=='rtx' and cfg.legacy.depth.camera_num_envs!=self.num_envs:
            paths=[str(p.GetPath()) for p in self.camera._sensor_prims]
            actual=np.array([int(re.search(r'/env_(\d+)/',p).group(1)) for p in paths],dtype=np.int64)
            if len(actual)!=cfg.legacy.depth.camera_num_envs or set(actual)!=set(self.camera_env_ids.cpu().tolist()):
                raise RuntimeError(f'Camera mapping mismatch: {len(actual)} actual versus {len(self.camera_env_ids)} selected')
            self.camera_env_ids=torch.as_tensor(actual,device=self.device,dtype=torch.long)
            self.camera_lookup.fill_(-1)
            self.camera_lookup[self.camera_env_ids]=torch.arange(len(actual),device=self.device)
            print('RTX_CAMERA_MAPPING',len(actual),int(actual.min()),int(actual.max()),flush=True)
        _,names,_,_=asset_contract(cfg.legacy)
        self.joint_ids,_=self.robot.find_joints(names,preserve_order=True)
        self.contact_ids=[self.contacts.body_names.index(n) for n in self.robot.body_names]
        from isaaclab.utils.math import quat_mul, quat_from_euler_xyz
        pose_ids=self.camera_env_ids if self.cfg.camera_backend=='rtx' else torch.arange(self.num_envs,device=self.device)
        self.camera_pitch=torch_rand_float(-5.,5.,(len(pose_ids),),self.device)*math.pi/180
        zero=torch.zeros_like(self.camera_pitch)
        orientation=quat_mul(self.robot.data.root_quat_w[pose_ids],quat_from_euler_xyz(zero,self.camera_pitch,zero))
        self.camera.set_world_poses(orientations=orientation,convention='world')
        self.core=Core(self); self.randomize_properties()
        initial=self.core.base_init_state.repeat(self.num_envs,1); initial[:,:3]+=self.core.env_origins
        self.write_root(initial,torch.arange(self.num_envs,device=self.device)); self.sync_state(self.core)
        self.scene.env_origins.copy_(self.core.env_origins)
        if self.viewport_camera_controller is not None:self.viewport_camera_controller.update_view_to_env()
        self.terminal_ids=torch.empty(0,device=self.device,dtype=torch.long)
        self.terminal_amp=torch.empty(0,30,device=self.device)
        self.depth_age=torch.zeros(len(self.camera_env_ids),device=self.device,dtype=torch.long)
        print('WMP policy joints',names,'Lab indices',self.joint_ids,flush=True)
    def _setup_scene(self):
        count=self.cfg.legacy.depth.camera_num_envs
        proportions=self.cfg.legacy.terrain.terrain_proportions
        from .competition import column_types, COMPETITION_TYPES
        kinds=column_types(self.cfg.legacy.terrain)
        cols=np.floor(np.arange(self.num_envs)/(self.num_envs/len(kinds))).astype(int)
        if hasattr(self.cfg.legacy.terrain,'eval_terrain_col'):
            if getattr(self.cfg.legacy.terrain,'competition',False):cols[:]=self.cfg.legacy.terrain.eval_terrain_col
            else:cols=(np.arange(self.num_envs)+self.cfg.legacy.terrain.eval_terrain_col)%len(kinds)
        # Preserve the original 25% camera coverage on competition courses.
        # Tilt and crawl still require direct depth; other robots use the
        # world-model depth predictor, including competition robots.
        required=np.array([i for i,col in enumerate(cols) if kinds[col] in ('tilt','crawl')],dtype=np.int64)
        if count<len(required):raise ValueError(f'{count} cameras cannot cover {len(required)} tilt/crawl environments; increase --camera_envs')
        required=np.unique(np.append(required,0))
        if count<len(required):raise ValueError('Camera subset must also include environment 0')
        other=np.setdiff1d(np.arange(self.num_envs),required)
        chosen=np.sort(np.concatenate((required,np.random.choice(other,count-len(required),replace=False))))
        self.camera_env_ids=torch.as_tensor(chosen,device=self.device,dtype=torch.long)
        self.camera_lookup=torch.full((self.num_envs,),-1,device=self.device,dtype=torch.long)
        self.camera_lookup[self.camera_env_ids]=torch.arange(count,device=self.device)
        self.robot=Articulation(self.cfg.robot); self.scene.articulations['robot']=self.robot
        self.contacts=ContactSensor(self.cfg.contact_sensor); self.scene.sensors['contacts']=self.contacts
        c=self.cfg.legacy; self.terrain=None
        if c.terrain.mesh_type=='plane':
            ground=sim_utils.GroundPlaneCfg(physics_material=self.cfg.sim.physics_material); ground.func('/World/ground',ground)
        else:
            from .terrain import Terrain
            self.terrain=Terrain(c.terrain,self.num_envs)
            v=self.terrain.vertices.copy(); v[:,:2]-=c.terrain.border_size
            create_prim_from_mesh('/World/ground',trimesh.Trimesh(vertices=v,faces=self.terrain.triangles,process=False),physics_material=self.cfg.sim.physics_material)
        # Original URDF contains an opaque visualization box centered at the camera.
        # RTX cameras see its inside; hide only that visual marker (no collision).
        from pxr import Usd, UsdGeom, Sdf
        root=self.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot')
        sim_utils.modify_collision_properties('/World/envs/env_0/Robot',sim_utils.CollisionPropertiesCfg(
            contact_offset=c.sim.physx.contact_offset,rest_offset=c.sim.physx.rest_offset))
        material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.18,.22,.25),roughness=1.)
        material.func('/World/ground/viewerMaterial',material)
        sim_utils.bind_visual_material('/World/ground','/World/ground/viewerMaterial')
        cache=UsdGeom.XformCache()
        base=self.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot/base')
        hidden=[]
        for prim in list(Usd.PrimRange(root,Usd.TraverseInstanceProxies())):
            if prim.GetTypeName()=='Cube' and '/base/visuals/' in str(prim.GetPath()):
                local=cache.ComputeRelativeTransform(prim,base)[0].ExtractTranslation()
                if np.linalg.norm(np.array(local)-np.array([.27,0.,.03]))<1e-5:
                    if prim.IsInstanceProxy():
                        # Edit the shared visual source once; keep base visuals/collisions instanced.
                        spec=prim.GetPrimStack()[0]
                        attr=spec.attributes.get('visibility') or Sdf.AttributeSpec(spec,'visibility',Sdf.ValueTypeNames.Token)
                        attr.default='invisible'
                    else:UsdGeom.Imageable(prim).MakeInvisible()
                    hidden.append(str(prim.GetPath()))
        if len(hidden)!=1:raise RuntimeError(f'Expected one camera marker, got {hidden}')
        print('HIDDEN_CAMERA_VISUAL',hidden,flush=True)
        self.scene.clone_environments(copy_from_source=False); self.scene.filter_collisions(global_prim_paths=['/World/ground'])
        # ROS optical axes: camera +Z points forward along robot +X, +Y down.
        if self.cfg.camera_backend=='rtx':
            from isaaclab.utils.math import convert_camera_frame_orientation_convention
            cam_asset=sim_utils.PinholeCameraCfg(focal_length=24.,horizontal_aperture=2*24*math.tan(math.radians(58)/2),clipping_range=(.01,2.))
            cam_asset.vertical_aperture=cam_asset.horizontal_aperture
            rot=convert_camera_frame_orientation_convention(torch.tensor([[.5,-.5,.5,-.5]]),origin='ros',target='opengl')[0].cpu().numpy()
            for env_id in chosen:
                cam_asset.func(f'/World/envs/env_{env_id}/Robot/base/depth_camera',cam_asset,translation=(.27,0.,.03),orientation=rot)
            camera_match='.*' if count==self.num_envs else '('+'|'.join(map(str,chosen.tolist()))+')'
            cam_cfg=TiledCameraCfg(prim_path=f'/World/envs/env_{camera_match}/Robot/base/depth_camera',update_period=.1,
                height=64,width=64,data_types=['distance_to_image_plane'],
                spawn=None,
                offset=TiledCameraCfg.OffsetCfg(pos=(.27,0.,.03),rot=(.5,-.5,.5,-.5),convention='ros'))
            self.camera=TiledCamera(cam_cfg)
        else:
            # Full 3-D triangle mesh rays include gaps, overhead crawl bars and tilt walls.
            # Restrict to terrain to prevent overlapping parallel robots contaminating images.
            cam_cfg=RayCasterCameraCfg(prim_path='/World/envs/env_.*/Robot/base',update_period=.1,
                mesh_prim_paths=['/World/ground'],max_distance=100.,data_types=['distance_to_image_plane'],
                offset=RayCasterCameraCfg.OffsetCfg(pos=(.27,0.,.03),rot=(.5,-.5,.5,-.5),convention='ros'),
                pattern_cfg=patterns.PinholeCameraPatternCfg(focal_length=24.,horizontal_aperture=2*24*math.tan(math.radians(58)/2),height=64,width=64))
            self.camera=RayCasterCamera(cam_cfg)
        self.scene.sensors['depth_camera']=self.camera
        if self.cfg.camera_backend=='rtx':
            camera_paths=sim_utils.find_matching_prim_paths('/World/envs/env_.*/Robot/base/depth_camera')
            print('RTX_CLONED_CAMERA_PRIMS',len(camera_paths),camera_paths[:3],flush=True)
        light=sim_utils.DomeLightCfg(intensity=1500.,color=(.75,.75,.75)); light.func('/World/Light',light)
    def sync_state(self,c,joints_only=False):
        d=self.robot.data; c.dof_pos=d.joint_pos[:,self.joint_ids].clone(); c.dof_vel=d.joint_vel[:,self.joint_ids].clone()
        if joints_only:return
        state=d.root_state_w.clone(); state[:,3:7]=state[:,[4,5,6,3]]; c.root_states=state; c.base_quat=state[:,3:7]
        bodies=d.body_state_w.clone(); bodies[:,:,3:7]=bodies[:,:,[4,5,6,3]]; c.rigid_body_states=bodies
        c.rigid_body_pos=bodies[:,:,:3]; c.rigid_body_lin_vel=bodies[:,:,7:10]
        c.contact_forces=self.contacts.data.net_forces_w[:,self.contact_ids].clone()
        if hasattr(c,'feet_indices'):c.sensor_forces=c.contact_forces[:,c.feet_indices]
        if hasattr(c,'gravity_vec'):
            c.base_lin_vel=quat_rotate_inverse(c.base_quat,state[:,7:10]); c.base_ang_vel=quat_rotate_inverse(c.base_quat,state[:,10:13]); c.projected_gravity=quat_rotate_inverse(c.base_quat,c.gravity_vec)
    def write_root(self,state,ids):
        pose=state[:,:7].clone(); pose[:,3:7]=state[:,[6,3,4,5]]
        self.robot.write_root_pose_to_sim(pose,env_ids=ids); self.robot.write_root_velocity_to_sim(state[:,7:13],env_ids=ids)
    def randomize_properties(self):
        c=self.core; view=self.robot.root_physx_view; ids=torch.arange(self.num_envs,dtype=torch.int32,device='cpu')
        masses=view.get_masses().clone(); original=masses.clone(); base=self.robot.body_names.index('base')
        masses[:,base]+=c.randomized_added_masses[:,0].cpu()
        other=[i for i in range(self.robot.num_bodies) if i!=base]
        masses[:,other]*=torch_rand_float(*c.cfg.domain_rand.link_mass_range,(self.num_envs,len(other)),'cpu')
        view.set_masses(masses,ids); inertia=view.get_inertias().clone(); inertia*=(masses/original).unsqueeze(-1); view.set_inertias(inertia,ids)
        com=view.get_coms().clone(); com[:,base,:3]+=c.randomized_com_pos.cpu(); view.set_coms(com,ids)
        mat=view.get_material_properties(); mat[:,:,0]=c.randomized_frictions.cpu(); mat[:,:,1]=c.randomized_frictions.cpu(); mat[:,:,2]=c.randomized_restitutions.cpu(); view.set_material_properties(mat,ids)
    def _pre_physics_step(self,actions):
        c=self.core; c.global_counter+=1; c.total_env_steps_counter+=1
        c.actions=actions.clamp(-6.,6.); self.substep=0; self.latency=random.randint(0,1)
    def _apply_action(self):
        c=self.core; self.sync_state(c,True)
        a=c.last_actions if c.cfg.domain_rand.randomize_action_latency and self.substep<self.latency else c.actions
        c.torques=c._compute_torques(a)
        if c.cfg.domain_rand.randomize_motor_strength:c.torques*=torch_rand_float(*c.cfg.domain_rand.motor_strength_range,c.torques.shape,self.device)
        self.robot.set_joint_effort_target(c.torques,joint_ids=self.joint_ids); self.substep+=1
    def _get_dones(self):
        c=self.core; self.sync_state(c); c.common_step_counter=self.common_step_counter
        c._post_physics_step_callback(); c.check_termination()
        return c.reset_buf & ~c.time_out_buf,c.time_out_buf
    def _get_rewards(self):
        c=self.core; c.compute_reward(); self.terminal_ids=c.reset_buf.nonzero(as_tuple=False).flatten(); self.terminal_amp=c.get_amp_observations()[self.terminal_ids].clone()
        return c.rew_buf
    def _reset_idx(self,ids):
        if self.core is None:return
        self.core.reset_idx(ids); self.robot.reset(ids); self.contacts.reset(ids)
        camera_ids=self.camera_lookup[ids]; camera_ids=camera_ids[camera_ids>=0]
        if self.cfg.camera_backend=='raycast':self.camera.reset(ids)
        elif len(camera_ids):self.camera.reset(camera_ids)
        self.episode_length_buf[ids]=0; self.scene.env_origins[ids]=self.core.env_origins[ids]
        if hasattr(self,'depth_age'):self.depth_age[camera_ids]=0
    def _get_observations(self):
        c=self.core; self.sync_state(c); c.measured_heights=c._get_heights(); c.measured_forward_heights=c._get_forward_heights()
        c.compute_observations(); c.obs_buf=c.obs_buf.clamp(-100,100); c.privileged_obs_buf=c.privileged_obs_buf.clamp(-100,100)
        if c.global_counter%5==0:
            raw=self.camera.data.output['distance_to_image_plane'][...,0]
            if self.cfg.camera_backend=='raycast':raw=raw[self.camera_env_ids]
            depth=torch.nan_to_num(raw,nan=2.,posinf=2.,neginf=0.).clamp(0,2)/2-.5
            fresh=self.depth_age<5
            c.depth_buffer[fresh]=depth[fresh,None].expand(-1,2,-1,-1)
            c.depth_buffer[~fresh]=torch.stack((c.depth_buffer[~fresh,-1],depth[~fresh]),1)
        self.depth_age+=1
        c.last_last_actions.copy_(c.last_actions); c.last_actions.copy_(c.actions); c.last_dof_pos.copy_(c.dof_pos); c.last_dof_vel.copy_(c.dof_vel); c.last_torques.copy_(c.torques); c.last_root_vel.copy_(c.root_states[:,7:13])
        self.extras=c.extras; self.extras['time_outs']=c.time_out_buf.clone()
        self.extras['depth']=c.depth_buffer[:,-2].clone() if c.global_counter%5==0 else None
        assert c.obs_buf.shape==(self.num_envs,285),c.obs_buf.shape
        return {'policy':c.obs_buf,'critic':c.privileged_obs_buf}

class Adapter:
    def __init__(self,env,capture=None):
        self.env=env; self.cfg=env.cfg.legacy; self.capture=capture
        self.device=env.device; self.num_envs=env.num_envs; self.depth_index=env.camera_env_ids.cpu().numpy()
        self.depth_index_inverse=env.camera_lookup.cpu().numpy()
        if getattr(env.cfg.legacy.terrain,'competition',False):
            kinds=env.terrain.column_types
            self.depth_index_without_crawl_tilt=np.array([i for i in self.depth_index
                if kinds[int(env.core.terrain_types[i])] not in ('tilt','crawl')],dtype=np.int64)
        else:
            self.depth_index_without_crawl_tilt=self.depth_index[self.depth_index<env.core.tilt_start_idx]
    @property
    def episode_length_buf(self):return self.env.episode_length_buf
    @episode_length_buf.setter
    def episode_length_buf(self,value):self.env.episode_length_buf.copy_(value)
    def __getattr__(self,name):return getattr(self.env.core,name)
    def reset(self):
        self.env.reset(); return self.get_observations(),self.get_privileged_observations()
    def get_observations(self):return self.env.core.obs_buf
    def get_privileged_observations(self):return self.env.core.privileged_obs_buf
    def step(self,actions):
        _,r,terminated,truncated,info=self.env.step(actions)
        if self.capture:
            self.capture.on_step(self.env)
            if self.capture.step_count%self.capture.every_steps==0:
                from PIL import Image
                depth=self.env.core.depth_buffer[0,-2].detach().cpu().numpy()
                Image.fromarray(((depth+.5)*255).clip(0,255).astype('uint8')).save(self.capture.directory/f'depth_{self.capture.step_count:06d}.png')
        return self.get_observations(),self.get_privileged_observations(),r,terminated|truncated,info,self.env.terminal_ids,self.env.terminal_amp
