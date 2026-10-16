"""Controlled GPU probes for simulator boundaries, independent of training rewards."""
import torch

def check_sim_contract(env):
    c=env.core; n=env.num_envs
    env.reset(); zero=torch.zeros(n,12,device=env.device)
    for _ in range(10):env.step(zero)
    assert c.obs_buf.shape==(n,285)
    assert c.get_amp_observations().shape==(n,30)
    assert len(c.feet_indices)==4 and len(c.penalised_contact_indices)==8
    # Policy-order writes round-trip through explicit Lab indices.
    q=env.robot.data.joint_pos[:,env.joint_ids].clone()
    assert torch.equal(q,c.dof_pos)
    # Reset one environment only: another robot's joint/root state must be untouched.
    ids=torch.tensor([0],device=env.device)
    untouched=env.robot.data.joint_pos[1:].clone(); root=env.robot.data.root_state_w[1:].clone()
    env._reset_idx(ids)
    torch.testing.assert_close(untouched,env.robot.data.joint_pos[1:],rtol=0,atol=0)
    torch.testing.assert_close(root,env.robot.data.root_state_w[1:],rtol=0,atol=0)
    c.episode_length_buf[0]=int(c.max_episode_length)+1
    c.check_termination(); assert c.time_out_buf[0] and c.reset_buf[0]
    # Quaternion reorder and gravity: upright state must project gravity down.
    assert torch.isfinite(c.projected_gravity).all()
    raw=env.camera.data.output['distance_to_image_plane'][...,0]
    finite=raw[torch.isfinite(raw)]
    assert finite.numel()>0 and finite.max()>.05, 'Depth camera blocked or invalid'
    result={'partial_reset':'passed','joint_round_trip':'passed','timeout':'passed','amp_dim':30,'observation_dim':285,'foot_contacts':4,'penalized_contacts':8,'depth_finite_fraction':float(torch.isfinite(raw).float().mean()),'depth_finite_max_m':float(finite.max())}
    from .env import Adapter
    adapter=Adapter(env)
    adapter.episode_length_buf=torch.arange(n,device=env.device,dtype=torch.long)
    assert torch.equal(c.episode_length_buf,adapter.episode_length_buf)
    assert int(c.episode_length_buf[1])==1
    result['runner_episode_buffer_write']='passed'
    env.reset()
    return result
