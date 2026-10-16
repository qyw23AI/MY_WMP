"""Isaac Lab 2.3 configuration; import only after AppLauncher."""
import re
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from .config import ROOT, asset_contract

@configclass
class WMPEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 20.
    action_space = 12
    observation_space = 270
    state_space = 238
    sim = sim_utils.SimulationCfg(dt=0.005, render_interval=4)
    scene = InteractiveSceneCfg(num_envs=1024, env_spacing=3., replicate_physics=True)
    robot: ArticulationCfg = None
    contact_sensor = ContactSensorCfg(prim_path='/World/envs/env_.*/Robot/.*',update_period=0.,history_length=1)
    legacy: object = None
    stage: int = 3

def make_env_cfg(legacy, stage=3, device='cuda:0'):
    path,names,limits,root_link=asset_contract(legacy)
    c=WMPEnvCfg()
    c.legacy=legacy; c.stage=stage
    c.seed=1
    c.decimation=legacy.control.decimation
    c.episode_length_s=legacy.env.episode_length_s
    c.action_space=legacy.env.num_actions
    c.observation_space=legacy.env.num_observations
    c.state_space=legacy.env.num_privileged_obs
    c.scene.num_envs=legacy.env.num_envs; c.scene.env_spacing=legacy.env.env_spacing
    c.sim.dt=legacy.sim.dt; c.sim.device=device; c.sim.render_interval=c.decimation
    c.sim.gravity=tuple(legacy.sim.gravity)
    c.sim.physx.solver_type=legacy.sim.physx.solver_type
    c.sim.physx.bounce_threshold_velocity=legacy.sim.physx.bounce_threshold_velocity
    c.sim.physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=legacy.terrain.static_friction,
        dynamic_friction=legacy.terrain.dynamic_friction,restitution=legacy.terrain.restitution,
        friction_combine_mode='average',restitution_combine_mode='average')
    arm=legacy.asset.armature
    if isinstance(arm,dict):
        arm={re.escape(n):float(arm.get(n,arm.get(n.removesuffix('_joint'),0.))) for n in names}
    effort={re.escape(n):limits[n]['effort'] for n in names}
    velocity={re.escape(n):limits[n]['velocity'] for n in names} if path.suffix=='.urdf' else None
    q=legacy.init_state.rot
    c.robot=ArticulationCfg(
        prim_path='/World/envs/env_.*/Robot',
        spawn=sim_utils.UrdfFileCfg(asset_path=str(path),
            usd_dir=str(ROOT/'isaaclab_migration/.cache'/path.stem),
            make_instanceable=False,fix_base=legacy.asset.fix_base_link,merge_fixed_joints=legacy.asset.collapse_fixed_joints,
            root_link_name=root_link,activate_contact_sensors=True,
            self_collision=legacy.asset.self_collisions==0,
            replace_cylinders_with_capsules=legacy.asset.replace_cylinder_with_capsule,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(target_type='none',
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.,damping=0.)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=legacy.asset.disable_gravity,
                linear_damping=legacy.asset.linear_damping,angular_damping=legacy.asset.angular_damping,
                max_linear_velocity=legacy.asset.max_linear_velocity,max_angular_velocity=legacy.asset.max_angular_velocity,
                max_depenetration_velocity=legacy.sim.physx.max_depenetration_velocity),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=legacy.asset.self_collisions==0,
                solver_position_iteration_count=legacy.sim.physx.num_position_iterations,
                solver_velocity_iteration_count=legacy.sim.physx.num_velocity_iterations)),
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(legacy.init_state.pos),rot=(q[3],q[0],q[1],q[2]),
            joint_pos=legacy.init_state.default_joint_angles,joint_vel={'.*':0.}),
        soft_joint_pos_limit_factor=legacy.rewards.soft_dof_pos_limit,
        actuators={'effort':IdealPDActuatorCfg(joint_names_expr=['.*'],stiffness=0.,damping=0.,
            effort_limit=effort,effort_limit_sim=effort,velocity_limit_sim=velocity,armature=arm)})
    return c
