"""CPU geometry, rule, height-query and curriculum checks (no optimizer/Isaac app)."""
import sys, unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'isaaclab_migration'))
from wmp_lab.competition import *
from wmp_lab.competition_tasks import CourseState
from wmp_lab.a1_config import A1AMPCfg


def fake_core(kind='plum_stones', level=9, count=1):
    cfg=A1AMPCfg();configure_competition(cfg,kind)
    courses={(r,c):make_course(kind,r) for r in range(10) for c in range(cfg.terrain.num_cols)}
    origins=torch.tensor([[[(r+.5)*8+courses[r,c].spawn[0],(c+.5)*8,courses[r,c].spawn[2]] for c in range(cfg.terrain.num_cols)] for r in range(10)])
    c=NS(cfg=cfg,num_envs=count,device='cpu',terrain_types=torch.zeros(count,dtype=torch.long),terrain_levels=torch.full((count,),level,dtype=torch.long),terrain_origins=origins,
         terrain=NS(column_types=[kind]*cfg.terrain.num_cols,competition_cells=courses,env_length=8.,env_width=8.),
         root_states=torch.zeros(count,13),rigid_body_pos=torch.zeros(count,4,3),feet_indices=torch.arange(4),
         base_lin_vel=torch.zeros(count,3),commands=torch.zeros(count,4),
         base_quat=torch.tensor([[0.,0.,0.,1.]]).repeat(count,1),
         forward_vec=torch.tensor([[1.,0.,0.]]).repeat(count,1),
         contact_forces=torch.zeros(count,4,3),projected_gravity=torch.tensor([[0.,0.,-1.]]).repeat(count,1),dt=.02,
         reset_buf=torch.zeros(count,dtype=torch.bool),time_out_buf=torch.zeros(count,dtype=torch.bool),episode_length_buf=torch.ones(count,dtype=torch.long))
    c.env_origins=origins[c.terrain_levels,c.terrain_types].clone()
    state=CourseState(c)
    return c,state


def place(c,state,root,feet,contacts=(1,1,1,1)):
    c.root_states[:,:3]=torch.tensor(root)+torch.cat((state.centers,torch.zeros(c.num_envs,1)),1)
    c.rigid_body_pos[:]=torch.tensor(feet)+torch.cat((state.centers,torch.zeros(c.num_envs,1)),1)[:,None]
    c.contact_forces[:,:,2]=torch.tensor(contacts)*20.
    c.reset_buf.zero_();c.time_out_buf.zero_()


class CompetitionTests(unittest.TestCase):
    def test_standard_dimensions_and_xml_alignment(self):
        stones=make_course('plum_stones',9);v,f=course_mesh(stones)
        self.assertAlmostEqual(float(np.ptp(v[:8*8,0])),3.5,places=5)
        np.testing.assert_allclose(np.array(stones.boxes)[:6,2:],[ [.2,.2,.2] ]*6)
        np.testing.assert_allclose(np.diff(np.array(stones.boxes[:3])[:,0])-.2,[.4,.4])
        np.testing.assert_allclose(np.array(stones.boxes[:3])[:,:2]+np.array(stones.boxes[3:6])[::-1,:2],[ [0,0] ]*3,atol=1e-8)
        tires=make_course('tire_array',9)
        np.testing.assert_allclose(np.ptp(ring_mesh(*tires.rings[0][:3])[0],axis=0),[.55,.55,.165],atol=1e-6)
        np.testing.assert_allclose(make_course('high_wall',9).boxes[0],[0,0,.08,.8,.30])
        # The 10300 course has no side walls; only the obstacle mesh is visible.
        self.assertAlmostEqual(float(np.abs(course_mesh(make_course('high_wall',9))[0][:,1]).max()),.4)
        self.assertAlmostEqual(float(np.abs(v[:,1]).max()),.4)
        # Validate against the user-approved corrected XML snapshot, independent of generators.
        import xml.etree.ElementTree as E
        xml=ROOT/'output/scene_audit/robocon2027.v2.xml'
        if xml.exists():
            r=E.parse(xml).getroot()
            for i,b in enumerate(stones.boxes[:6]):
                g=r.find(f".//geom[@name='stepping_stones_01_{i:02d}']")
                np.testing.assert_allclose(np.fromstring(g.get('pos'),sep=' ')[:2],[b[0]+4.575,b[1]+2],atol=1e-6)

    def test_all_levels_geometry(self):
        import trimesh
        for kind in COMPETITION_TYPES:
            previous=0
            for level in range(10):
                spec=make_course(kind,level);self.assertGreaterEqual(spec.height,previous);previous=spec.height
                v,f=course_mesh(spec);self.assertTrue(np.isfinite(v).all());self.assertLess(f.max(),len(v))
                # Every individual ring must be closed, outward and keep its central hole.
                if spec.rings:
                    rv,rf=ring_mesh(*spec.rings[0][:3]);mesh=trimesh.Trimesh(rv,rf,process=False)
                    self.assertTrue(mesh.is_watertight);self.assertGreater(mesh.volume,0)
        self.assertEqual(make_course('tire_array',9).rings[0][3],0)

    def test_query_matches_triangle_surface(self):
        # Independent vertical ray/triangle intersections via trimesh, not the height formula.
        import trimesh
        rng=np.random.default_rng(42)
        for kind in COMPETITION_TYPES:
            for level in (0,5,9):
                spec=make_course(kind,level);v,f=course_mesh(spec,ground=True)
                mesh=trimesh.Trimesh(v,f,process=False)
                xy=rng.uniform([-1.9,-.7],[1.9,.7],(180,2))
                origin=np.column_stack((xy,np.full(len(xy),1.)))
                hits,idx,_=mesh.ray.intersects_location(origin,np.tile([0.,0.,-1.],(len(xy),1)),multiple_hits=True)
                expected=np.zeros(len(xy));np.maximum.at(expected,idx,hits[:,2])
                q=HeightQuery({(0,0):spec},1,1,'cpu')
                actual=q(torch.tensor(xy,dtype=torch.float32)[None],torch.tensor([0]),torch.tensor([0]))[0].numpy()
                np.testing.assert_allclose(actual,expected,atol=2e-6)

    def test_tire_holes_are_legal_but_outside_ground_is_not(self):
        c,s=fake_core('tire_array');s.entered[:]=True
        feet=[[-.825,-.275,.02],[-.275,.275,.02],[.275,-.275,.02],[.825,.275,.02]]
        place(c,s,[0,0,.3],feet);s.update();self.assertFalse(s.failed.any())
        feet[0][1]=.7;feet[1][1]=.7
        place(c,s,[0,0,.3],feet);s.update();self.assertTrue(s.failed.all())

    def test_stone_top_coverage_no_repeat_reward(self):
        c,s=fake_core();spec=make_course('plum_stones',9)
        platform=spec.boxes[6]
        place(c,s,[platform[0],0,.5],[[platform[0],0,.22]]*4);s.update()
        for index,box in enumerate(spec.boxes[:6]):
            feet=[[box[0],box[1],.22]]*4
            place(c,s,[box[0],0,.5],feet);s.update()
            self.assertTrue(s.coverage[0,index]);before=s.reward.clone()
            place(c,s,[box[0],0,.5],feet);s.update()
            self.assertLessEqual(float(s.reward[0]),float(before[0]))
        self.assertTrue(s.coverage.all())
        s.coverage.zero_();place(c,s,[0,0,.4],[[0,0,.12]]*4);s.update();self.assertFalse(s.coverage.any())

    def test_strict_completion_and_no_side_bypass(self):
        c,s=fake_core();s.entered[:]=True
        feet=[[2.1,0,.02]]*4
        place(c,s,[2.3,0,.3],feet)
        for _ in range(16):s.update()
        self.assertTrue(s.failed.all());self.assertFalse(s.success.any())
        c,s=fake_core();s.entered[:]=True;s.coverage[:]=True
        for x in (-1.35,1.35):
            place(c,s,[x,0,.5],[[x,0,.22]]*4);s.update()
        place(c,s,[2.3,0,.3],feet)
        for _ in range(16):s.update()
        self.assertTrue(s.success.all());self.assertTrue(s.strict.all())
        c,s=fake_core('high_wall');place(c,s,[0,.6,.3],[[.2,.6,.02]]*4);s.update()
        self.assertTrue(s.failed.all())
        c,s=fake_core('high_wall');s.entered[:]=True
        place(c,s,[.6,0,.3],[[.2,0,.02]]*4)
        for _ in range(16):s.update()
        self.assertTrue(s.success.all())

    def test_wall_bypass_penalty_without_side_walls(self):
        c,s=fake_core('high_wall',level=0)
        place(c,s,[-1.,0.,.35],[[-1.,0.,.02]]*4)
        s.update();center_reward=float(s.reward[0])
        s.max_x[:] = -1.1
        place(c,s,[-1.,.32,.35],[[-1.,.32,.02]]*4)
        s.update();self.assertFalse(s.failed.any());self.assertEqual(float(s.reward[0]),center_reward)
        place(c,s,[0.,.60,.35],[[0.,.60,.02]]*4)
        s.update();self.assertTrue(s.violated.all());self.assertTrue(s.failed.all())
        self.assertLessEqual(float(s.reward[0]),-9.8)

    def test_original_clipped_tracking_and_progress_reward(self):
        from wmp_lab.logic import Logic
        from wmp_lab.competition_tasks import CompetitionMixin
        class RewardCore(CompetitionMixin,Logic):pass
        c=RewardCore();c.cfg=A1AMPCfg();c.commands=torch.tensor([[.2,0.,0.,0.]])
        c.base_lin_vel=torch.tensor([[.2,0.,0.]])
        c.course=NS(mask=torch.tensor([True]))
        self.assertGreater(float(c._reward_tracking_lin_vel()[0]),.99)
        c.base_lin_vel[0,0]=4.
        self.assertGreater(float(c._reward_tracking_lin_vel()[0]),.9)
        c,s=fake_core('high_wall',level=0)
        c.commands[0,0]=.2;c.base_lin_vel[0,0]=4.
        place(c,s,[-.8,0.,.35],[[-.8,0.,.02]]*4)
        s.update();self.assertAlmostEqual(float(s.reward[0]),.12,places=5)

    def test_ground_spawn_and_complete_platform_ascent(self):
        # Every level starts fully on approach ground, clear of the platform edge.
        for level in range(10):
            spec=make_course('plum_stones',level)
            self.assertEqual(spec.spawn[2],0.)
            self.assertAlmostEqual(spec.entry-spec.spawn[0],.7)
            q=HeightQuery({(0,0):spec},1,1,'cpu')
            sample=torch.tensor([[[spec.spawn[0]+dx,.13] for dx in (-.25,.25)]])
            self.assertTrue((q(sample,torch.tensor([0]),torch.tensor([0]))==0).all())
        c,s=fake_core();spec=make_course('plum_stones',9);edge=spec.entry
        place(c,s,[spec.spawn[0],0,.35],[[spec.spawn[0]+dx,y,.02] for dx in (-.2,.2) for y in (-.13,.13)])
        s.update();self.assertFalse(s.entered.any());self.assertFalse(s.failed.any())
        # Front feet reach the platform while both rear feet remain outside on ground.
        feet=[[edge+.20,y,.22] for y in (-.13,.13)]+[[edge-.10,y,.02] for y in (-.13,.13)]
        place(c,s,[edge+.05,0,.45],feet);s.update()
        self.assertTrue(s.platforms[0,0]);self.assertFalse(s.entered.any());self.assertFalse(s.failed.any())
        # Ascent is armed only after all four feet have cleared the approach ground.
        feet=[[-1.35+dx,y,.22] for dx in (-.2,.2) for y in (-.13,.13)]
        place(c,s,[-1.35,0,.5],feet);s.update();self.assertTrue(s.entered.all());self.assertFalse(s.failed.any())
        s.coverage[:]=True
        place(c,s,[1.35,0,.5],[[1.35,0,.22]]*4);s.update();self.assertTrue(s.platforms.all())
        place(c,s,[2.3,0,.3],[[2.1,0,.02]]*4)
        for _ in range(16):s.update()
        self.assertTrue(s.strict.all())
        # Landing on stones after skipping the entrance platform is rejected.
        c,s=fake_core();place(c,s,[-.75,0,.5],[[-.75,-.2,.22]]*4);s.update()
        self.assertTrue(s.failed.all());self.assertFalse(s.platforms[0,0])

    def test_type_lanes_and_difficulty_axis(self):
        from wmp_lab.terrain import Terrain
        c=A1AMPCfg();configure_competition(c,'robocon')
        c.terrain.num_cols=3;c.terrain.terrain_proportions=[0.]*10+[1/3]*3;c.terrain.border_size=1.
        terrain=Terrain(c.terrain,6)
        self.assertEqual(terrain.column_types,list(COMPETITION_TYPES))
        for col,kind in enumerate(COMPETITION_TYPES):
            centers=[]
            for level in range(10):
                spec=terrain.competition_cells[level,col]
                self.assertEqual(spec.kind,kind);self.assertEqual(spec.level,level)
                center=np.array(terrain.env_origins[level,col,:2])-np.array(spec.spawn[:2])
                np.testing.assert_allclose(center,tile_center(level,col),atol=1e-6)
                centers.append(center)
            np.testing.assert_allclose(np.diff(centers,axis=0),[[8.,0.]]*9)

    def test_curriculum_windows_and_standard_sampling(self):
        c,s=fake_core(level=0,count=50);s.success[:]=True;s.strict[:]=True
        s.finish(torch.arange(50));self.assertEqual(int(s.frontier[0]),1)
        s.select_levels(torch.arange(50));self.assertTrue((c.terrain_levels==1).all())
        s.success[:]=False;s.finish(torch.arange(50));self.assertEqual(int(s.frontier[0]),0)
        c,s=fake_core(level=9,count=50);s.frontier[0]=9;s.success[:]=True
        s.finish(torch.arange(50));s.select_levels(torch.arange(50));self.assertTrue((c.terrain_levels==9).all())
        c,s=fake_core(count=4000);c.cfg.terrain.competition_stage='standard'
        torch.manual_seed(1);s.select_levels(torch.arange(4000))
        self.assertTrue(.57<float((c.terrain_levels==9).float().mean())<.63)
        c.cfg.terrain.competition_stage='eval';before=c.terrain_levels.clone();s.select_levels(torch.arange(4000));torch.testing.assert_close(before,c.terrain_levels)

    def test_curriculum_checkpoint(self):
        c,s=fake_core();s.frontier[:]=torch.tensor([8,7,6]);s.attempts[0,9]=100
        saved=s.state_dict();s.frontier.zero_();s.attempts.zero_();s.load_state_dict(saved)
        self.assertEqual(s.frontier.tolist(),[8,7,6]);self.assertEqual(int(s.attempts[0,9]),100)
        s.reset_statistics()
        self.assertTrue(all(not getattr(s,name).any() for name in ('frontier','attempts','wins','strict_wins','window_n','window_wins','totals')))

    def test_profiles(self):
        for profile in ('robocon','robocon_mixed',*COMPETITION_TYPES):
            c=A1AMPCfg();configure_competition(c,profile)
            kinds=column_types(c.terrain)
            self.assertEqual(len(kinds),c.terrain.num_cols)
            self.assertTrue(set(kinds)&set(COMPETITION_TYPES))
        c=A1AMPCfg();configure_competition(c,'robocon_mixed')
        self.assertEqual(sum(k in COMPETITION_TYPES for k in column_types(c.terrain)),18)

if __name__=='__main__':unittest.main()
