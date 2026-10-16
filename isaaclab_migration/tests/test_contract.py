"""CPU checks comparing migrated tensor methods against original WMP sources."""
import ast,sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'isaaclab_migration'))
from wmp_lab.a1_config import A1AMPCfg
from wmp_lab.config import asset_contract
from wmp_lab.math import quat_apply,quat_rotate_inverse

def methods(path):
    tree=ast.parse(path.read_text()); cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    return {n.name:ast.dump(n,include_attributes=False) for n in cls.body if isinstance(n,ast.FunctionDef)}
class Contracts(unittest.TestCase):
    def test_scaled_reward_schedule(self):
        from wmp_lab.config import scale_reward_schedule
        self.assertEqual(scale_reward_schedule([[4000,10000,.1,1.]],8),[[32000,80000,.1,1.]])
        self.assertEqual(scale_reward_schedule([[4000,10000,.1,1.]],1),[[4000,10000,.1,1.]])
        for bad in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):scale_reward_schedule([[4000,10000,.1,1.]],bad)
    def test_original_tensor_methods(self):
        source=methods(ROOT/'legged_gym/envs/base/legged_robot.py')
        actual=methods(ROOT/'isaaclab_migration/wmp_lab/logic.py')
        checked=0
        for name,value in actual.items():
            if name=='init_tensors':continue
            self.assertEqual(value,source[name],name);checked+=1
        self.assertGreater(checked,35)
    def test_robot_joint_order(self):
        cfg=A1AMPCfg();path,names,limits,root=asset_contract(cfg)
        expected=[f'{leg}_{joint}_joint' for leg in ['FR','FL','RR','RL'] for joint in ['hip','thigh','calf']]
        self.assertEqual(names,expected);self.assertEqual(root,'base');self.assertTrue(path.exists())
        self.assertTrue(all(limits[n]['effort']>0 for n in names))
    def test_observation_contract(self):
        cfg=A1AMPCfg();self.assertEqual(cfg.env.num_observations,285)
        self.assertEqual(cfg.env.prop_dim+cfg.env.action_dim+cfg.env.privileged_dim+cfg.env.height_dim,285)
        self.assertEqual(cfg.env.forward_height_dim,525)
        self.assertEqual(5*(285-53-187-3),210)
    def test_optical_axes_and_depth_units(self):
        # Camera ROS +Z -> robot +X; ROS +Y -> robot -Z.
        q=torch.tensor([[.5,.5,-.5,.5]]) # xyzw corresponding to wxyz (.5,-.5,.5,-.5), same rotation up to sign
        # Use the exact xyzw boundary convention.
        q=torch.tensor([[-.5,.5,-.5,.5]])
        torch.testing.assert_close(quat_apply(q,torch.tensor([[0.,0.,1.]])),torch.tensor([[1.,0.,0.]]))
        torch.testing.assert_close(quat_apply(q,torch.tensor([[0.,1.,0.]])),torch.tensor([[0.,0.,-1.]]))
        depths=torch.tensor([0.,1.,2.]);torch.testing.assert_close(depths/2-.5,torch.tensor([-.5,0.,.5]))
        v=torch.randn(3,3);q=torch.tensor([[0.,0.,0.,1.]]).repeat(3,1)
        torch.testing.assert_close(quat_rotate_inverse(q,v),v)
if __name__=='__main__':unittest.main()
