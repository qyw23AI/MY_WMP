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


def fake_core(kind='plum_stones', level=None, count=1, device='cpu', foot_names=('FL_foot','FR_foot','RL_foot','RR_foot')):
    if level is None:level=maximum_level(kind)
    cfg=A1AMPCfg();configure_competition(cfg,kind)
    courses={(r,c):make_course(kind,r) for r in range(cfg.terrain.num_rows) for c in range(cfg.terrain.num_cols)}
    origins=torch.tensor([[[(r+.5)*8+courses[r,c].spawn[0],(c+.5)*8,courses[r,c].spawn[2]] for c in range(cfg.terrain.num_cols)] for r in range(cfg.terrain.num_rows)])
    c=NS(cfg=cfg,num_envs=count,device='cpu',terrain_types=torch.zeros(count,dtype=torch.long),terrain_levels=torch.full((count,),level,dtype=torch.long),terrain_origins=origins,
         terrain=NS(column_types=[kind]*cfg.terrain.num_cols,competition_cells=courses,env_length=8.,env_width=8.),
         root_states=torch.zeros(count,13),rigid_body_pos=torch.zeros(count,4,3),feet_indices=torch.arange(4),foot_names=foot_names,
         base_lin_vel=torch.zeros(count,3),commands=torch.zeros(count,4),
         base_quat=torch.tensor([[0.,0.,0.,1.]]).repeat(count,1),
         forward_vec=torch.tensor([[1.,0.,0.]]).repeat(count,1),
         contact_forces=torch.zeros(count,4,3),projected_gravity=torch.tensor([[0.,0.,-1.]]).repeat(count,1),dt=.02,
         reset_buf=torch.zeros(count,dtype=torch.bool),time_out_buf=torch.zeros(count,dtype=torch.bool),episode_length_buf=torch.ones(count,dtype=torch.long))
    c.env_origins=origins[c.terrain_levels,c.terrain_types].clone()
    c.device=device
    for name,value in vars(c).copy().items():
        if torch.is_tensor(value):setattr(c,name,value.to(device))
    state=CourseState(c)
    return c,state


def place(c,state,root,feet,contacts=(1,1,1,1)):
    c.root_states[:,:3]=torch.tensor(root,device=c.device)+torch.cat((state.centers,torch.zeros(c.num_envs,1,device=c.device)),1)
    c.rigid_body_pos[:]=torch.tensor(feet,device=c.device)+torch.cat((state.centers,torch.zeros(c.num_envs,1,device=c.device)),1)[:,None]
    c.contact_forces[:,:,2]=torch.tensor(contacts,device=c.device)*20.
    c.reset_buf.zero_();c.time_out_buf.zero_()


class CompetitionTests(unittest.TestCase):
    def test_standard_dimensions_and_xml_alignment(self):
        stones=make_course('plum_stones',10);v,f=course_mesh(stones)
        self.assertAlmostEqual(float(np.ptp(v[:8*8,0])),3.5,places=5)
        np.testing.assert_allclose(np.array(stones.boxes)[:6,2:],[ [.2,.2,.2] ]*6)
        np.testing.assert_allclose(np.diff(np.array(stones.boxes[:3])[:,0])-.2,[.4,.4])
        np.testing.assert_allclose(np.array(stones.boxes[:3])[:,:2]+np.array(stones.boxes[3:6])[::-1,:2],[ [0,0] ]*3,atol=1e-8)
        tires=make_course('tire_array',6)
        np.testing.assert_allclose(np.ptp(ring_mesh(*tires.rings[0][:3])[0],axis=0),[.55,.55,.165],atol=1e-6)
        np.testing.assert_allclose(make_course('high_wall',6).boxes[0],[0,0,.08,.8,.30])
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
            for level in range(level_count(kind)):
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
            for level in (0,standard_level(kind),maximum_level(kind)):
                spec=make_course(kind,level);v,f=course_mesh(spec,ground=True)
                mesh=trimesh.Trimesh(v,f,process=False)
                xy=rng.uniform([-1.9,-.7],[1.9,.7],(180,2))
                origin=np.column_stack((xy,np.full(len(xy),1.)))
                hits,idx,_=mesh.ray.intersects_location(origin,np.tile([0.,0.,-1.],(len(xy),1)),multiple_hits=True)
                expected=np.full(len(xy),spec.floor_height);np.maximum.at(expected,idx,hits[:,2])
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
        c,s=fake_core();spec=make_course('plum_stones',14)
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
        for level in range(level_count('plum_stones')):
            spec=make_course('plum_stones',level)
            self.assertEqual(spec.spawn[2],0.)
            self.assertAlmostEqual(spec.entry-spec.spawn[0],.7)
            q=HeightQuery({(0,0):spec},1,1,'cpu')
            sample=torch.tensor([[[spec.spawn[0]+dx,.13] for dx in (-.25,.25)]])
            self.assertTrue((q(sample,torch.tensor([0]),torch.tensor([0]))==0).all())
        c,s=fake_core();spec=make_course('plum_stones',14);edge=spec.entry
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
        c.terrain.num_cols=len(COMPETITION_TYPES);c.terrain.terrain_proportions=[0.]*10+[1/len(COMPETITION_TYPES)]*len(COMPETITION_TYPES);c.terrain.border_size=1.
        terrain=Terrain(c.terrain,6)
        self.assertEqual(terrain.column_types,list(COMPETITION_TYPES))
        for col,kind in enumerate(COMPETITION_TYPES):
            centers=[]
            for level in range(level_count(kind)):
                spec=terrain.competition_cells[level,col]
                self.assertEqual(spec.kind,kind);self.assertEqual(spec.level,level)
                center=np.array(terrain.env_origins[level,col,:2])-np.array(spec.spawn[:2])
                np.testing.assert_allclose(center,tile_center(level,col),atol=1e-6)
                centers.append(center)
            np.testing.assert_allclose(np.diff(centers,axis=0),[[8.,0.]]*(level_count(kind)-1))
            if kind not in PLUM_TYPES:
                self.assertTrue(all((level,col) not in terrain.competition_cells for level in range(10,15)))

    def test_mixed_type_origin_levels_and_explicit_eval_bounds(self):
        from wmp_lab.competition_tasks import CompetitionMixin
        class OriginBase:
            def _get_env_origins(self):
                self.terrain_levels=torch.tensor([14,14,14])
                self.terrain_types=torch.arange(3)
                self.terrain_origins=torch.zeros(15,3,3)
                self.terrain_origins[:,:,0]=torch.arange(15)[:,None]*8
                self.env_origins=torch.zeros(3,3)
                self.max_terrain_level=15
        class OriginProbe(CompetitionMixin,OriginBase):pass
        p=OriginProbe();p.device='cpu'
        p.terrain=NS(column_types=['plum_stones','high_wall','roughflat'])
        p.cfg=NS(terrain=NS(competition=True,competition_stage='eval'))
        p._get_env_origins()
        self.assertEqual(p.terrain_levels.tolist(),[14,9,9])
        self.assertEqual(p.max_terrain_level,10)
        p.cfg.terrain.eval_terrain_col=1;p.cfg.terrain.eval_terrain_level=14
        with self.assertRaises(ValueError):p._get_env_origins()
        p.cfg.terrain.eval_terrain_col=0;p._get_env_origins()
        self.assertEqual(p.terrain_levels.tolist(),[14,14,14])

    def test_curriculum_windows_and_standard_sampling(self):
        c,s=fake_core(level=0,count=50);s.success[:]=True;s.strict[:]=True
        s.finish(torch.arange(50));self.assertEqual(int(s.frontier[0]),1)
        s.select_levels(torch.arange(50));self.assertTrue((c.terrain_levels==1).all())
        s.success[:]=False;s.finish(torch.arange(50));self.assertEqual(int(s.frontier[0]),0)
        c,s=fake_core(level=14,count=50);s.frontier[0]=14;s.success[:]=True
        s.finish(torch.arange(50));s.select_levels(torch.arange(50));self.assertTrue((c.terrain_levels==14).all())
        c,s=fake_core(count=4000);c.cfg.terrain.competition_stage='standard'
        torch.manual_seed(1);s.select_levels(torch.arange(4000))
        self.assertTrue(.36<float((c.terrain_levels==14).float().mean())<.44)
        self.assertTrue(.36<float((c.terrain_levels==10).float().mean())<.44)
        self.assertTrue(set(c.terrain_levels.tolist())<={10,11,12,13,14})
        c.cfg.terrain.competition_stage='eval';before=c.terrain_levels.clone();s.select_levels(torch.arange(4000));torch.testing.assert_close(before,c.terrain_levels)

    def test_curriculum_checkpoint(self):
        c,s=fake_core();s.frontier[:3]=torch.tensor([8,7,6]);s.attempts[0,9]=100
        saved=s.state_dict();s.frontier.zero_();s.attempts.zero_();s.load_state_dict(saved)
        self.assertEqual(s.frontier.tolist(),[8,7,6]+[0]*(len(COMPETITION_TYPES)-3));self.assertEqual(int(s.attempts[0,9]),100)
        s.reset_statistics()
        self.assertTrue(all(not getattr(s,name).any() for name in ('frontier','attempts','wins','strict_wins','window_n','window_wins','totals')))

    def test_profiles(self):
        for profile in ('robocon','robocon_mixed','gap_combos','gap_training',*COMPETITION_TYPES):
            c=A1AMPCfg();configure_competition(c,profile)
            self.assertEqual(c.terrain.num_rows,max(level_count(k) for k in column_types(c.terrain)))
            kinds=column_types(c.terrain)
            self.assertEqual(len(kinds),c.terrain.num_cols)
            self.assertTrue(set(kinds)&set(COMPETITION_TYPES))
        c=A1AMPCfg();configure_competition(c,'robocon_mixed')
        self.assertEqual(sum(k in COMPETITION_TYPES for k in column_types(c.terrain)),42)

    def test_platform_and_ramp_dimensions_all_levels(self):
        import trimesh
        for level in range(10):
            platform = make_course('platform', level)
            np.testing.assert_allclose(platform.boxes[0][2:], [.8, .8, .05 + .25 * level / 9])
            ramp = make_course('ramp', level)
            self.assertEqual(ramp.boxes[0][2:4], [.8, .8])
            self.assertAlmostEqual(ramp.entry - ramp.spawn[0], .7)
            self.assertLess(ramp.exit + .6, 4.)
            for start, end, width, z0, z1 in ramp.ramps:
                angle = math.degrees(math.atan2(abs(z1 - z0), end - start))
                self.assertAlmostEqual(angle, 5 + 25 * level / 9)
                v, f = ramp_mesh(start, end, width, z0, z1)
                mesh = trimesh.Trimesh(v, f, process=False)
                self.assertTrue(mesh.is_watertight)
                self.assertGreater(mesh.volume, 0.)
            # Flat approach -> ascent -> flat crest -> descent -> flat exit.
            x = [-1.3, -1.2, -.8, -.4, 0., .4, .8, 1.2, 1.3]
            q = HeightQuery({(0, 0): ramp}, 1, 1, 'cpu')
            actual = q(torch.tensor([[[v, 0.] for v in x]]), torch.tensor([0]), torch.tensor([0]))
            np.testing.assert_allclose(actual[0], np.array([0, 0, .5, 1, 1, 1, .5, 0, 0]) * ramp.height, atol=1e-6)

    def test_paired_courses_require_ascent_and_ground_landing(self):
        for kind in ('platform', 'ramp'):
            for level in range(10):
                c, s = fake_core(kind, level)
                spec = make_course(kind, level)
                if kind == 'ramp':
                    place(c, s, [-.8, 0, spec.height / 2 + .3], [[-.8, 0, spec.height / 2 + .02]] * 4)
                    s.update()
                    self.assertTrue(s.coverage[0, 0])
                for x in (-.2, .2):
                    place(c, s, [x, 0, spec.height + .3], [[x, 0, spec.height + .02]] * 4)
                    s.update()
                    self.assertFalse(s.success.any())
                    self.assertFalse(s.failed.any())
                self.assertTrue(s.platforms.all())
                if kind == 'ramp':
                    place(c, s, [.8, 0, spec.height / 2 + .3], [[.8, 0, spec.height / 2 + .02]] * 4)
                    s.update()
                    self.assertTrue(s.coverage[0, :2].all())
                # Crossing the exit in flight cannot finish the task.
                place(c, s, [spec.exit + .5, 0, .5], [[spec.exit + .2, 0, .2]] * 4, contacts=(0, 0, 0, 0))
                for _ in range(16):s.update()
                self.assertFalse(s.success.any())
                feet = [[spec.exit + .2, 0, z] for z in (.02, .02, .08, .08)]
                place(c, s, [spec.exit + .5, 0, .3], feet, contacts=(1, 1, 0, 0))
                for _ in range(16):s.update()
                self.assertTrue(s.success.all())
                self.assertTrue(s.strict.all())

    def test_paired_bypass_and_skipped_slope_are_rejected(self):
        for kind in ('platform', 'ramp'):
            c, s = fake_core(kind)
            place(c, s, [0., .6, .3], [[0., .6, .02]] * 4)
            s.update();self.assertTrue(s.failed.all())
            c, s = fake_core(kind)
            spec = make_course(kind, 9)
            place(c, s, [spec.exit + .5, 0, .3], [[spec.exit + .2, 0, .02]] * 4)
            for _ in range(16):s.update()
            self.assertTrue(s.failed.all());self.assertFalse(s.success.any())
        # Crest contact without any uphill contact cannot arm a ramp task.
        c, s = fake_core('ramp');h = make_course('ramp', 9).height
        place(c, s, [0., 0., h + .3], [[0., 0., h + .02]] * 4)
        s.update();self.assertFalse(s.platforms.any())
        # Uphill and crest without downhill support cannot count as descent.
        place(c, s, [-.8, 0., h / 2 + .3], [[-.8, 0., h / 2 + .02]] * 4);s.update()
        for x in (-.2, .2):
            place(c, s, [x, 0., h + .3], [[x, 0., h + .02]] * 4);s.update()
        place(c, s, [1.7, 0., .3], [[1.4, 0., .02]] * 4)
        for _ in range(16):s.update()
        self.assertTrue(s.failed.all());self.assertFalse(s.success.any())

    def test_original_checkpoint_expands_new_course_counters(self):
        c, s = fake_core()
        s.frontier[:3] = torch.tensor([8, 7, 6])
        s.attempts[0, 9] = 100
        saved = s.state_dict()
        old = {'version':1, **{k:v[:3].clone() for k,v in saved.items() if torch.is_tensor(v)}}
        s.frontier.fill_(9);s.attempts.fill_(2)
        s.load_state_dict(old)
        self.assertEqual(s.frontier.tolist(), [0]*len(COMPETITION_TYPES))
        self.assertEqual(int(s.attempts[0, 9]), 0)
        self.assertFalse(s.attempts[3:].any())

    def test_new_courses_have_independent_learning_frontiers(self):
        for kind in ('platform', 'ramp', 'platform_gap', *COMBO_TYPES):
            c, s = fake_core(kind, level=0, count=50)
            k = COMPETITION_TYPES.index(kind)
            s.success[:] = True;s.strict[:] = True
            s.finish(torch.arange(50));s.select_levels(torch.arange(50))
            self.assertEqual(int(s.frontier[k]), 1)
            self.assertEqual(int(s.frontier.sum()), 1)
            self.assertTrue((c.terrain_levels == 1).all())

    def test_platform_gap_dimensions_spawn_and_deep_floor(self):
        for level in range(10):
            spec = make_course('platform_gap', level)
            boxes = np.asarray(spec.boxes)
            self.assertEqual(len(boxes), 6)
            np.testing.assert_allclose(boxes[:, 2:4], .8)
            np.testing.assert_allclose(boxes[:, 4], 0.)
            np.testing.assert_allclose(np.diff(boxes[:, 0]) - .8, .05 + .4 * level / 9)
            self.assertEqual(spec.floor_height, -5.)
            self.assertLess(abs(spec.entry), 4.);self.assertLess(spec.exit, 4.)
            self.assertEqual(spec.spawn, (boxes[0, 0], 0., 0.))
            q = HeightQuery({(0, 0): spec}, 1, 1, 'cpu')
            xy = np.concatenate((boxes[:, :2], (boxes[:-1, :2] + boxes[1:, :2]) / 2,
                                 [[0., .6], [spec.entry - .1, 0.], [spec.exit + .1, 0.]]))
            heights = q(torch.tensor(xy, dtype=torch.float32)[None], torch.tensor([0]), torch.tensor([0]))
            np.testing.assert_allclose(heights[0], [0.] * 6 + [-5.] * 8)
            # A1's standing footprint is fully on the starting 80cm top.
            feet = [[[spec.spawn[0] + dx, y] for dx in (-.25, .25) for y in (-.13, .13)]]
            self.assertTrue((q(torch.tensor(feet), torch.tensor([0]), torch.tensor([0])) == 0).all())

    def test_platform_gap_complete_ordered_jumps_all_levels(self):
        for level in range(10):
            c, s = fake_core('platform_gap', level)
            spec = make_course('platform_gap', level)
            x0 = spec.boxes[0][0]
            place(c, s, [x0, 0., .35], [[x0, 0., .02]] * 4);s.update()
            self.assertTrue(s.entered.all());self.assertTrue(s.coverage[0, 0])
            for index in range(1, 6):
                x1 = spec.boxes[index][0]
                x_air = (x0 + x1) / 2
                place(c, s, [x_air, 0., .50], [[x_air, 0., .20]] * 4, contacts=(0, 0, 0, 0));s.update()
                self.assertTrue(s.jump_airborne.all());self.assertFalse(s.success.any())
                place(c, s, [x1, 0., .35], [[x1, 0., .02]] * 4);s.update()
                self.assertEqual(int(s.jump_stage[0]), index)
                self.assertTrue(s.coverage[0, index]);self.assertFalse(s.failed.any())
                before = s.reward.clone();s.update();self.assertLess(float(s.reward[0]), float(before[0]))
                x0 = x1
            for _ in range(16):s.update()
            self.assertTrue(s.success.all());self.assertTrue(s.strict.all())
            self.assertTrue(s.coverage.all());self.assertTrue(s.platforms.all())
            s.reset(torch.tensor([0]))
            self.assertFalse(s.coverage.any());self.assertFalse(s.jump_ready.any())
            self.assertFalse(s.jump_airborne.any());self.assertEqual(int(s.jump_stage[0]), 0)

    def test_platform_gap_rejects_walking_skipping_side_and_falling(self):
        def start():
            c, s = fake_core('platform_gap')
            spec = make_course('platform_gap', 9);x = spec.boxes[0][0]
            place(c, s, [x, 0., .35], [[x, 0., .02]] * 4);s.update()
            return c, s, spec
        c, s, spec = start();x = spec.boxes[1][0]
        place(c, s, [x, 0., .35], [[x, 0., .02]] * 4);s.update()
        self.assertTrue(s.failed.all());self.assertFalse(s.success.any())
        c, s, spec = start();x = spec.boxes[2][0]
        place(c, s, [x, 0., .5], [[x, 0., .2]] * 4, contacts=(0, 0, 0, 0));s.update()
        place(c, s, [x, 0., .35], [[x, 0., .02]] * 4);s.update()
        self.assertTrue(s.failed.all());self.assertFalse(s.success.any())
        for y, z in ((.6, .35), (0., -.2)):
            c, s, spec = start();x = spec.boxes[0][0]
            place(c, s, [x, y, z], [[x, y, z - .3]] * 4, contacts=(0, 0, 0, 0));s.update()
            self.assertTrue(s.failed.all());self.assertFalse(s.success.any())
        # Contacts below the top or against the platform side cannot register a landing.
        c, s, spec = start();x = spec.boxes[1][0]
        place(c, s, [x, 0., .5], [[x, 0., .2]] * 4, contacts=(0, 0, 0, 0));s.update()
        place(c, s, [x, 0., .25], [[x, 0., -.05]] * 4);s.update()
        self.assertTrue(s.failed.all());self.assertFalse(s.coverage[0, 1])

    def test_platform_gap_training_mesh_has_no_flat_bridge(self):
        from wmp_lab.terrain import Terrain
        import trimesh
        c = A1AMPCfg();configure_competition(c, 'platform_gap')
        c.terrain.num_cols = 1;c.terrain.border_size = 1.
        terrain = Terrain(c.terrain, 4)
        level = 9;spec = terrain.competition_cells[level, 0]
        spec_center = np.asarray(tile_center(level, 0)) + c.terrain.border_size
        mesh = trimesh.Trimesh(terrain.vertices, terrain.triangles, process=False)
        x = [b[0] for b in spec.boxes] + [(a[0] + b[0]) / 2 for a, b in zip(spec.boxes, spec.boxes[1:])]
        origins = np.column_stack((np.array(x) + spec_center[0], np.full(len(x), spec_center[1]), np.ones(len(x))))
        hits, ids, _ = mesh.ray.intersects_location(origins, np.tile([0., 0., -1.], (len(x), 1)), multiple_hits=True)
        expected = np.full(len(x), -np.inf);np.maximum.at(expected, ids, hits[:, 2])
        np.testing.assert_allclose(expected, [0.] * 6 + [-5.] * 5, atol=1e-6)

    def test_five_course_checkpoint_retains_stats_when_gap_is_added(self):
        c, s = fake_core('platform_gap')
        s.frontier[:5] = torch.tensor([1, 2, 3, 4, 5]);s.attempts[4, 5] = 22
        old = {'version':1, **{k:v[:5].clone() for k,v in s.state_dict().items() if torch.is_tensor(v)}}
        s.frontier.fill_(9);s.attempts.fill_(2);s.load_state_dict(old)
        self.assertEqual(s.frontier.tolist(), [0, 0, 0, 4, 5]+[0]*(len(COMPETITION_TYPES)-5))
        self.assertEqual(int(s.attempts[4, 5]), 22);self.assertFalse(s.attempts[5].any())

    def test_combo_geometry_and_pits_all_levels(self):
        for kind in COMBO_TYPES:
            for level in range(level_count(kind)):
                spec = make_course(kind, level);b = np.asarray(spec.boxes)
                np.testing.assert_allclose(b[:, 3], .8)
                np.testing.assert_allclose(b[1:3, 2], .32)
                np.testing.assert_allclose(b[1:5, 4], np.array([1, 2, 3, 3]) * COMBO_RISES[level])
                np.testing.assert_allclose(b[3:5, 2:4], .8)
                self.assertAlmostEqual(b[4, 0] - b[3, 0] - .8, COMBO_GAPS[level])
                self.assertEqual(spec.parameters['turn_degrees'], 0)
                self.assertGreater(spec.parameters['route_start'], -4.)
                self.assertLess(spec.parameters['route_end'], 4.)
                q = HeightQuery({(0, 0): spec}, 1, 1, 'cpu')
                xy = b[:, :2].tolist() + [[(b[3, 0]+b[4, 0])/2, 0.]]
                expected = b[:, 4].tolist() + [-5.]
                if spec.ramps:
                    start, end, width, z0, z1 = spec.ramps[0]
                    self.assertAlmostEqual(start, b[4, 0]+.4)
                    self.assertAlmostEqual(end, b[5, 0]-.6)
                    self.assertAlmostEqual(math.degrees(math.atan2(z0-z1, end-start)), COMBO_ANGLES[level])
                    xy.append([(start+end)/2, 0.]);expected.append(spec.height/2)
                else:self.assertEqual(len(b), 5)
                sides = [[x, y] for x, _ in xy for y in (-.5, .5)]
                all_xy = torch.tensor([xy+sides], dtype=torch.float32)
                np.testing.assert_allclose(q(all_xy, torch.tensor([0]), torch.tensor([0]))[0],
                                           expected+[-5.]*len(sides), atol=1e-6)

    def combo_before_landing(self, kind, level=6, device='cpu'):
        c, s = fake_core(kind, level, device=device);spec = make_course(kind, level)
        for index in (0, 1, 2, 3):
            x, _, _, _, h = spec.boxes[index]
            place(c, s, [x, 0., h+.35], [[x, 0., h+.02]]*4);s.update()
            self.assertFalse(s.failed.any())
        x = (spec.boxes[3][0]+spec.boxes[4][0])/2;h = spec.height
        place(c, s, [x, 0., h+.5], [[x, 0., h+.2]]*4, contacts=(0,0,0,0));s.update()
        self.assertTrue(s.combo_crossing.all());self.assertFalse(s.failed.any())
        return c, s, spec

    def test_gap_training_weights_and_combo_reward(self):
        cfg=A1AMPCfg();configure_competition(cfg,'gap_training')
        from collections import Counter
        self.assertEqual(Counter(column_types(cfg.terrain)),
                         dict(stair_gap=6,stair_gap_ramp=6,slope=3,stairup=3,gap=2))
        for kind in COMBO_TYPES:
            c,s=fake_core(kind,0);spec=make_course(kind,0)
            x=spec.boxes[0][0]
            for dx in (0.,.05,.10):
                place(c,s,[x+dx,0.,.35],[[x+dx,0.,.02]]*4);s.update()
                self.assertFalse(s.failed.any())
                self.assertEqual(float(s.reward[0]),0.)
            c,s,spec=self.combo_before_landing(kind,0)
            x=spec.boxes[4][0];h=spec.height
            c.commands[:,0]=.25
            place(c,s,[x,0.,h+.35],[[x,0.,h+.02]]*4);s.update()
            self.assertEqual(float(s.reward[0]),.5)
            if kind=='stair_gap':
                self.assertEqual(float(c.commands[0,0]),0.)
                for _ in range(15):s.update()
                self.assertTrue(s.success.all());self.assertTrue(s.strict.all())
                self.assertEqual(float(s.reward[0]),2.)

    def test_combo_asynchronous_landing_and_separate_finishes(self):
        for kind in COMBO_TYPES:
            for level in range(10):
                c, s, spec = self.combo_before_landing(kind, level)
                x = spec.boxes[4][0];h = spec.height
                feet = [[x,0.,z] for z in (h+.02,h+.02,h+.2,h+.2)]
                place(c,s,[x,0.,h+.35],feet,contacts=(1,1,0,0));s.update()
                self.assertFalse(s.failed.any());self.assertTrue(s.coverage[0,3])
                self.assertFalse(s.strict_landed[0])
                feet = [[x,0.,z] for z in (h+.2,h+.2,h+.02,h+.02)]
                place(c,s,[x,0.,h+.35],feet,contacts=(0,0,1,1));s.update()
                self.assertTrue(s.coverage[0,3]);self.assertFalse(s.failed.any())
                place(c,s,[x,0.,h+.35],[[x,0.,h+.02]]*4)
                for _ in range(16):s.update()
                if kind=='stair_gap':self.assertTrue(s.success.all())
                else:
                    self.assertFalse(s.success.any())
                    start,end,_,_,_=spec.ramps[0];x=(start+end)/2
                    place(c,s,[x,0.,h/2+.35],[[x,0.,h/2+.02]]*4);s.update()
                    self.assertTrue(s.coverage[0,4]);self.assertFalse(s.failed.any())
                    self.assertEqual(float(s.reward[0]),.5)
                    x=spec.boxes[5][0]
                    place(c,s,[x,0.,.35],[[x,0.,.02]]*4)
                    for _ in range(16):s.update()
                    self.assertTrue(s.success.all());self.assertTrue(s.strict.all())

    def test_combo_hazards_remain_fatal_but_off_target_contact_is_not(self):
        for kind in COMBO_TYPES:
            for mode in ('front_gap_drop','rear_deep_gap_drop','side','rear_outside'):
                c,s,spec=self.combo_before_landing(kind)
                x=spec.boxes[4][0];h=spec.height
                feet=[[x,0.,h+.02]]*4
                contacts=(1,1,1,1)
                if mode=='front_gap_drop':
                    feet[0]=[(spec.boxes[3][0]+x)/2,0.,h-.03]
                elif mode=='rear_deep_gap_drop':
                    feet[3]=[(spec.boxes[3][0]+x)/2,0.,h-.25]
                elif mode=='side':feet[3]=[x,.5,h-.03]
                else:feet[3]=[x,.5,h+.02]
                place(c,s,[x,0.,h+.35],feet,contacts);s.update()
                self.assertEqual(bool(s.failed[0]),mode!='rear_outside')
                if mode=='rear_outside':
                    self.assertTrue(s.strict_invalid[0])
                    self.assertFalse(c.reset_buf[0])
                else:
                    self.assertTrue(c.reset_buf[0])
                    self.assertEqual(float(s.reward[0]),0.)  # no task-specific failure penalty
    def test_combo_stage_rewards_without_launch_bonus(self):
        for kind in COMBO_TYPES:
            c,s=fake_core(kind,0);spec=make_course(kind,0)
            for index in (1,2,3):
                x,_,_,_,h=spec.boxes[index]
                place(c,s,[x,0.,h+.35],[[x,0.,h+.02]]*4);s.update()
                self.assertAlmostEqual(float(s.reward[0]),.5 if index<3 else 0.)
            self.assertTrue(s.jump_ready[0])
            self.assertTrue(s.coverage[0,2])

    def test_all_terrain_timeout_is_20_seconds(self):
        # Exercise the real check_termination override without launching Isaac Lab.
        from wmp_lab.competition_tasks import CompetitionMixin
        from wmp_lab.logic import Logic
        class TerminationProbe(CompetitionMixin, Logic):
            pass
        def make_probe(kind):
            core,course=fake_core(kind,0)
            probe=TerminationProbe();probe.__dict__.update(vars(core))
            probe.course=course;course.c=probe
            probe.max_episode_length=1000  # 20s at 50 Hz
            probe.termination_contact_indices=torch.empty(0,dtype=torch.long)
            approach=make_course(kind,0).boxes[0]
            place(probe,course,[approach[0],0.,.35],[[approach[0],0.,.02]]*4,contacts=(0,0,0,0))
            return probe,course
        for kind in ('stair_gap','stair_gap_ramp','platform'):
            core,course=make_probe(kind)
            core.episode_length_buf.fill_(1000);core.check_termination()
            self.assertFalse(core.time_out_buf.any(), kind)
            self.assertFalse(core.reset_buf.any(), kind)
            core.episode_length_buf.fill_(1001);core.check_termination()
            self.assertTrue(core.time_out_buf.all(), kind)
            self.assertTrue(core.reset_buf.all(), kind)
            self.assertFalse(course.failed.any(), kind)  # timeout is not a task failure

    def test_combo_skipped_stairs_normal_completion_only(self):
        c,s=fake_core('stair_gap',9);spec=make_course('stair_gap',9)
        x=spec.boxes[3][0];h=spec.height
        place(c,s,[x,0.,h+.35],[[x,0.,h+.02]]*4);s.update()
        self.assertFalse(s.failed.any());self.assertFalse(s.coverage[0,:2].any())
        x=spec.boxes[4][0]
        place(c,s,[x,0.,h+.35],[[x,0.,h+.02]]*4)
        for _ in range(16):s.update()
        self.assertTrue(s.success.all());self.assertFalse(s.strict.any())
        self.assertEqual(float(s.reward[0]),0.)
        c,s,spec=self.combo_before_landing('stair_gap_ramp')
        x=spec.boxes[4][0];h=spec.height
        place(c,s,[x,0.,h+.35],[[x,0.,h+.02]]*4);s.update()
        x=spec.boxes[5][0];place(c,s,[x,0.,.35],[[x,0.,.02]]*4)
        for _ in range(16):s.update()
        self.assertFalse(s.success.any());self.assertFalse(s.coverage[0,4])

    def test_combo_ordinary_completion_after_off_target_touch_has_no_strict_reward(self):
        c,s,spec=self.combo_before_landing('stair_gap',0)
        launch=spec.boxes[3][0];target=spec.boxes[4][0];h=spec.height
        # Launch platform is a legal route support, even after crossing starts.
        place(c,s,[launch,0.,h+.35],[[launch,0.,h+.02]]*4);s.update()
        self.assertFalse(s.strict_invalid[0]);self.assertFalse(s.failed[0])
        feet=[[target,0.,h+.02]]*4
        feet[0]=[target,.5,h+.02]  # front non-target contact, not a physical fall
        place(c,s,[target,0.,h+.35],feet);s.update()
        self.assertTrue(s.strict_invalid[0]);self.assertFalse(s.failed[0])
        place(c,s,[target,0.,h+.35],[[target,0.,h+.02]]*4)
        for _ in range(16):s.update()
        self.assertTrue(s.success[0]);self.assertFalse(s.strict[0])
        self.assertEqual(float(s.reward[0]),0.)

    def test_platform_gap_allows_sequential_foot_contacts(self):
        c,s=fake_core('platform_gap');spec=make_course('platform_gap',9)
        x0,x1=spec.boxes[0][0],spec.boxes[1][0]
        place(c,s,[x0,0.,.35],[[x0,0.,.02]]*4);s.update()
        x=(x0+x1)/2
        place(c,s,[x,0.,.5],[[x,0.,.2]]*4,contacts=(0,0,0,0));s.update()
        for z,contacts in (((.02,.02,.2,.2),(1,1,0,0)),((.2,.2,.02,.02),(0,0,1,1))):
            place(c,s,[x1,0.,.35],[[x1,0.,h] for h in z],contacts);s.update()
            self.assertFalse(s.failed.any())
        self.assertEqual(int(s.jump_stage[0]),1)

    def test_combo_standard_sampling_and_six_course_checkpoint(self):
        c,s=fake_core('stair_gap',count=4000);c.cfg.terrain.competition_stage='standard'
        s.select_levels(torch.arange(4000))
        self.assertTrue(.36<float((c.terrain_levels==6).float().mean())<.44)
        self.assertTrue(.36<float((c.terrain_levels==9).float().mean())<.44)
        self.assertTrue(set(c.terrain_levels.tolist())<={6,7,8,9})
        s.frontier[:6]=torch.tensor([1,2,3,4,5,6])
        old={'version':1, **{k:v[:6].clone() for k,v in s.state_dict().items() if torch.is_tensor(v)}}
        s.frontier.fill_(9);s.load_state_dict(old)
        self.assertEqual(s.frontier.tolist(),[0,0,0,4,5,6]+[0]*(len(COMPETITION_TYPES)-6))

    def test_combo_training_mesh_pits_and_road(self):
        from wmp_lab.terrain import Terrain
        import trimesh
        c=A1AMPCfg();configure_competition(c,'gap_combos')
        c.terrain.num_cols=2;c.terrain.terrain_proportions=[.5 if name in COMBO_TYPES else 0. for name in ALL_TYPES];c.terrain.border_size=1.
        terrain=Terrain(c.terrain,4)
        mesh=trimesh.Trimesh(terrain.vertices,terrain.triangles,process=False)
        for col in range(2):
            spec=terrain.competition_cells[9,col];b=spec.boxes
            xy=[[box[0],0.] for box in b]+[[(b[3][0]+b[4][0])/2,0.]]
            expected=[box[4] for box in b]+[-5.]
            if spec.ramps:
                r=spec.ramps[0];xy.append([(r[0]+r[1])/2,0.]);expected.append(spec.height/2)
            sides=[[x,y] for x,_ in xy for y in (-.6,.6)]
            xy=np.asarray(xy+sides)+np.asarray(tile_center(9,col))+c.terrain.border_size
            rays=np.column_stack((xy,np.ones(len(xy))))
            hits,ids,_=mesh.ray.intersects_location(rays,np.tile([0.,0.,-1.],(len(xy),1)),multiple_hits=True)
            actual=np.full(len(xy),-np.inf);np.maximum.at(actual,ids,hits[:,2])
            np.testing.assert_allclose(actual,expected+[-5.]*len(sides),atol=2e-6)

    def test_plum_fixed_dimensions_spacing_only_and_flat_connections(self):
        for kind in PLUM_TYPES:
            self.assertEqual(level_count(kind),15)
            with self.assertRaises(ValueError):make_course(kind,15)
            with self.assertRaises(ValueError):make_course(kind,-1)
            previous = -float("inf")
            for level in range(level_count(kind)):
                spec=make_course(kind,level);b=np.asarray(spec.boxes)
                np.testing.assert_allclose(b[:6,2:],[ [.2,.2,.2] ]*6)
                ordered=b[:6][np.argsort(b[:6,0])]
                long_gap=np.diff(ordered[:,0])-.2
                np.testing.assert_allclose(long_gap,STONE_LONG_GAPS[level])
                np.testing.assert_allclose(np.abs(np.diff(ordered[:,1]))-.2,STONE_ROW_GAPS[level])
                np.testing.assert_allclose(np.diff(b[:3,0])-.2,.2+2*STONE_LONG_GAPS[level])
                self.assertGreaterEqual(float(long_gap[0]),previous-1e-7);previous=float(long_gap[0])
                self.assertAlmostEqual(b[3,1]-b[0,1]-.2,STONE_ROW_GAPS[level])
                if kind=='plum_stones':
                    np.testing.assert_allclose(b[6:8,2:],[ [.8,.8,.2] ]*2)
                    self.assertAlmostEqual(b[0,0]-.1-(b[6,0]+.4),STONE_PLATFORM_GAPS[level])
                else:
                    np.testing.assert_allclose(b[6:8,2:],[ [1.1,.8,.2] ]*2)
                    self.assertAlmostEqual(b[6,0]+b[6,2]/2,spec.entry)
                    self.assertAlmostEqual(b[7,0]-b[7,2]/2,spec.exit)
                    self.assertAlmostEqual(b[0,0]-.1-spec.entry,STONE_PLATFORM_GAPS[level])
                    self.assertAlmostEqual(spec.exit-(b[5,0]+.1),STONE_PLATFORM_GAPS[level])
                    self.assertEqual(spec.spawn[2],.2)
                q=HeightQuery({(0,0):spec},1,1,'cpu')
                midpoint=(b[0,0]+b[1,0])/2
                xy=torch.tensor([[[midpoint,b[0,1]],[midpoint,.6],[spec.spawn[0],0.]]],dtype=torch.float32)
                np.testing.assert_allclose(q(xy,torch.tensor([0]),torch.tensor([0]))[0],[0.,-5.,spec.spawn[2]])
                if kind=='plum_stones_flat':
                    finish=torch.tensor([[[spec.exit+.55,0.],[b[0,0],b[0,1]]]],dtype=torch.float32)
                    np.testing.assert_allclose(q(finish,torch.tensor([0]),torch.tensor([0]))[0],[.2,.2])
        self.assertEqual(STONE_LONG_GAPS[0],-.075)
        self.assertEqual(STONE_ROW_GAPS[0],.025)
        self.assertEqual(STONE_LONG_GAPS[10],.10)
        self.assertEqual(STONE_ROW_GAPS[10],.20)
        self.assertEqual(STONE_LONG_GAPS[14],.15)
        self.assertEqual(STONE_ROW_GAPS[14],.25)
        self.assertEqual(make_course('platform_plum_combo',6).kind,'plum_stones')

    def test_negative_long_gap_overlaps_projection_only(self):
        for kind in PLUM_TYPES:
            spec=make_course(kind,0);boxes=np.asarray(spec.boxes)
            stones=boxes[:6][np.argsort(boxes[:6,0])]
            a,b=stones[:2]
            self.assertAlmostEqual(.2-(b[0]-a[0]),.075)
            self.assertAlmostEqual(abs(a[1]-b[1])-.2,.025)
            for i in range(6):
                for j in range(i):
                    # Disjoint XY footprints imply disjoint collision solids.
                    self.assertTrue((np.abs(stones[i,:2]-stones[j,:2])>=.2-1e-7).any())
            self.assertTrue(np.all(np.asarray(STONE_PLATFORM_GAPS)>0))
            self.assertAlmostEqual(np.diff(boxes[:3,0])[0]-.2,.05)
            x=(a[0]+b[0])/2
            q=HeightQuery({(0,0):spec},1,1,'cpu')
            xy=torch.tensor([[[x,a[1]],[x,b[1]],[x,0.]]],dtype=torch.float32)
            np.testing.assert_allclose(q(xy,torch.tensor([0]),torch.tensor([0]))[0],[.2,.2,0.])
            c,s=fake_core(kind,0)
            if kind=='plum_stones':
                px=boxes[6,0];place(c,s,[px,0.,.55],[[px,0.,.22]]*4);s.update()
            feet=[[a[0],a[1],.22]]*2+[[b[0],b[1],.22]]*2
            place(c,s,[x,0.,.55],feet);s.update();self.assertFalse(s.failed.any())
            feet[3]=[x,0.,.17]
            place(c,s,[x,0.,.55],feet,contacts=(1,1,1,0));s.update()
            self.assertTrue(s.failed.all());self.assertTrue(c.reset_buf.all())

    def test_flat_plum_road_gaps_are_open_and_any_dropped_foot_resets(self):
        import trimesh
        for level in range(level_count('plum_stones_flat')):
            spec=make_course('plum_stones_flat',level)
            first,last=spec.boxes[0],spec.boxes[5]
            gaps=[((spec.entry+first[0]-.1)/2,first),
                  ((last[0]+.1+spec.exit)/2,last)]
            vertices,faces=course_mesh(spec,ground=True)
            mesh=trimesh.Trimesh(vertices,faces,process=False)
            origins=np.array([[x,box[1],1.] for x,box in gaps])
            hits,indices,_=mesh.ray.intersects_location(origins,np.array([[0.,0.,-1.]]*2),multiple_hits=True)
            heights=np.full(2,-np.inf);np.maximum.at(heights,indices,hits[:,2])
            np.testing.assert_allclose(heights,0.,atol=1e-6)
            for x,box in gaps:
                for dangling in (False,True):
                    c,s=fake_core('plum_stones_flat',level)
                    # Exit gap still fails after all six tops were covered.
                    if box is last:s.coverage[:]=True;s.entered[:]=True
                    feet=[[box[0],box[1],.22]]*3+[[x,box[1],.17 if dangling else .02]]
                    place(c,s,[box[0],0.,.55],feet,contacts=(1,1,1,0 if dangling else 1));s.update()
                    self.assertTrue(s.failed.all());self.assertTrue(c.reset_buf.all())
                    self.assertFalse(s.success.any())

    def test_plum_single_and_combo_complete_all_levels(self):
        for kind in PLUM_TYPES:
            for level in range(level_count(kind)):
                c,s=fake_core(kind,level);spec=make_course(kind,level)
                x=spec.spawn[0]
                ground_h=spec.spawn[2]
                place(c,s,[x,0.,ground_h+.35],[[x,0.,ground_h+.02]]*4);s.update()
                self.assertFalse(s.failed.any())
                if kind=='plum_stones':
                    x=spec.boxes[6][0]
                    place(c,s,[x,0.,.55],[[x,0.,.22]]*4);s.update()
                for box in sorted(spec.boxes[:6],key=lambda b:b[0]):
                    x,y=box[:2]
                    place(c,s,[x,y,.55],[[x,y,.22]]*4);s.update()
                    self.assertFalse(s.failed.any())
                self.assertTrue(s.coverage.all())
                if kind=='plum_stones':
                    x=spec.boxes[7][0]
                    place(c,s,[x,0.,.55],[[x,0.,.22]]*4);s.update()
                x=spec.exit+.55
                place(c,s,[x,0.,ground_h+.35],[[x,0.,ground_h+.02]]*4)
                for _ in range(16):
                    s.update()
                    if c.reset_buf.any():break
                self.assertTrue(s.success.all());self.assertFalse(s.failed.any())

    def test_plum_one_grounded_or_dangling_foot_resets(self):
        for kind in PLUM_TYPES:
            for level in (0,10,14):
                for dangling in (False,True):
                    c,s=fake_core(kind,level);spec=make_course(kind,level)
                    if kind=='plum_stones':
                        x=spec.boxes[6][0]
                        place(c,s,[x,0.,.55],[[x,0.,.22]]*4);s.update()
                    a,b=spec.boxes[0],spec.boxes[1]
                    gap_x=(a[0]+b[0])/2
                    feet=[[a[0],a[1],.22]]*3+[[gap_x,a[1],.02 if not dangling else .17]]
                    place(c,s,[a[0],a[1],.55],feet,contacts=(1,1,1,0 if dangling else 1));s.update()
                    self.assertTrue(s.failed.all());self.assertTrue(c.reset_buf.all());self.assertFalse(s.success.any())
                    self.assertTrue((s.reward<0).all())

    def test_tire_height_only_hollow_and_physical_tread(self):
        import trimesh
        reference=None
        for level in range(10):
            spec=make_course('tire_array',level)
            self.assertTrue(all(ring[3]==0 for ring in spec.rings))
            v,f=ring_mesh(*spec.rings[0][:3])
            local=v.copy();local[:,:2]-=np.asarray(spec.rings[0][:2]);local[:,2]/=spec.height
            if reference is None:reference=local
            else:np.testing.assert_allclose(local,reference,atol=1e-6)
            self.assertTrue(trimesh.Trimesh(v,f,process=False).is_watertight)
            q=HeightQuery({(0,0):spec},1,1,'cpu')
            xy=torch.tensor([[ring[:2] for ring in spec.rings]])
            self.assertTrue((q(xy,torch.tensor([0]),torch.tensor([0]))==0).all())
        self.assertAlmostEqual(TIRE_HEIGHTS[6],.165);self.assertAlmostEqual(TIRE_HEIGHTS[9],.198)
        spec=make_course('tire_array',9);x,y,h,_=spec.rings[0]
        q=HeightQuery({(0,0):spec},1,1,'cpu')
        # At angle zero the crest has a real dimple, at pi/4 it retains full height.
        xy=torch.tensor([[[x+.22,y],[x+.22/math.sqrt(2),y+.22/math.sqrt(2)]]])
        z=q(xy,torch.tensor([0]),torch.tensor([0]))[0]
        self.assertAlmostEqual(float(z[0]),h*(1-TREAD_DEPTH_FRACTION),places=6)
        self.assertAlmostEqual(float(z[1]),h,places=6)
        # Independent mesh collision surface agrees at the carved point.
        v,f=course_mesh(spec,ground=True);mesh=trimesh.Trimesh(v,f,process=False)
        origin=np.array([[x+.22,y,1.]])
        hits,_,_=mesh.ray.intersects_location(origin,np.array([[0.,0.,-1.]]))
        self.assertAlmostEqual(float(hits[:,2].max()),float(z[0]),places=6)

    def test_wall_height_target_and_named_checkpoint_round_trip(self):
        self.assertEqual(make_course('high_wall',6).height,.30)
        self.assertEqual(make_course('high_wall',9).height,.36)
        c,s=fake_core('plum_stones_flat');s.frontier[:]=torch.arange(len(COMPETITION_TYPES))
        s.attempts[:,6]=torch.arange(len(COMPETITION_TYPES))+10
        saved=s.state_dict();s.frontier.zero_();s.attempts.zero_();s.load_state_dict(saved)
        self.assertEqual(s.frontier.tolist(),list(range(len(COMPETITION_TYPES))))
        self.assertEqual(s.attempts[:,6].tolist(),list(range(10,10+len(COMPETITION_TYPES))))
        # Metadata, rather than row order, identifies a saved course.
        order=list(reversed(range(len(COMPETITION_TYPES))))
        shuffled={k:v[order].clone() if torch.is_tensor(v) else v for k,v in saved.items()}
        shuffled['course_types']=[saved['course_types'][i] for i in order]
        shuffled['course_revisions']=[saved['course_revisions'][i] for i in order]
        s.frontier.zero_();s.load_state_dict(shuffled)
        self.assertEqual(s.frontier.tolist(),list(range(len(COMPETITION_TYPES))))

    def test_old_plum_geometry_stats_reset_without_resetting_other_courses(self):
        c,s=fake_core();s.frontier.fill_(5);s.attempts.fill_(7)
        old=s.state_dict()
        old['course_revisions'][0]=2
        old['course_revisions'][COMPETITION_TYPES.index('plum_stones_flat')]=3
        s.load_state_dict(old)
        for k,name in enumerate(COMPETITION_TYPES):
            self.assertEqual(int(s.frontier[k]),0 if name in PLUM_TYPES else 5)
            self.assertTrue((s.attempts[k,:level_count(name)]==(0 if name in PLUM_TYPES else 7)).all())
            self.assertFalse(s.attempts[k,level_count(name):].any())

    def test_ten_level_checkpoint_expands_and_resets_changed_plum_only(self):
        c,s=fake_core();s.frontier.fill_(4);s.attempts[:,:10]=3
        s.attempts[:,10:]=0
        old=s.state_dict();old['version']=2
        for name,value in list(old.items()):
            if torch.is_tensor(value) and value.ndim>1:old[name]=value[:,:10].clone()
        for name in PLUM_TYPES:old['course_revisions'][COMPETITION_TYPES.index(name)]=4
        s.load_state_dict(old)
        for k,name in enumerate(COMPETITION_TYPES):
            self.assertEqual(int(s.frontier[k]),0 if name in PLUM_TYPES else 4)
            self.assertTrue((s.attempts[k,:10]==(0 if name in PLUM_TYPES else 3)).all())
            self.assertFalse(s.attempts[k,10:].any())
        s.frontier[0]=14;s.attempts[0,14]=9
        saved=s.state_dict();s.reset_statistics();s.load_state_dict(saved)
        self.assertEqual(int(s.frontier[0]),14);self.assertEqual(int(s.attempts[0,14]),9)

if __name__=='__main__':unittest.main()
