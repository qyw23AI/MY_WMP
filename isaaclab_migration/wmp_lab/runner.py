"""Original WMP algorithm with explicit config, CPU replay and complete checkpoints."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
from ruamel.yaml import YAML
from dreamer.models import WorldModel
from .runner_original import WMPRunner
from .config import ROOT

class Runner(WMPRunner):
    def __init__(self,*args,wm_batch=2,wm_length=16,wm_updates=1,wm_start=256,**kwargs):
        self.wm_overrides=dict(batch_size=wm_batch,batch_length=wm_length,train_steps_per_iter=wm_updates,train_start_steps=wm_start,compile=False)
        self.wm_updates_done=0; self.depth_updates_done=0
        self.wm_effective_lengths=[]; self.wm_predicted_depth_samples=0
        super().__init__(*args,**kwargs)
    def _build_world_model(self):
        cfg=YAML(typ='safe').load((ROOT/'dreamer/configs.yaml').read_text())['defaults']; cfg.update(self.wm_overrides)
        cfg['device']=self.device; cfg['num_actions']=12*self.env.cfg.depth.update_interval
        self.wm_config=SimpleNamespace(**cfg)
        self._world_model=WorldModel(self.wm_config,{'prop':(33,),'image':(64,64,1)},True).to(self.device)
        self.wm_feature_dim=cfg['dyn_deter']
    def train_world_model(self):
        # Original sampler reads CPU replay and transfers only sampled trajectories.
        result=super().train_world_model()
        if result:self.wm_updates_done+=self.wm_config.train_steps_per_iter
        return result
    def train_depth_predictor(self):
        eligible=self.env.depth_index_without_crawl_tilt
        eligible=eligible[self.wm_dataset_size[eligible]>1]
        if len(eligible)==0:return 0.
        loss_total=0.
        for _ in range(self.depth_predictor_cfg['training_iters']):
            ids=np.random.choice(eligible,self.depth_predictor_cfg['batch_size'])
            ts=np.array([np.random.randint(int(self.wm_dataset_size[i])) for i in ids])
            f=self.wm_dataset['forward_height_map'][ids,ts].to(self.device)
            p=self.wm_dataset['prop'][ids,ts].to(self.device)
            target=self.wm_dataset['image'][self.env.depth_index_inverse[ids],ts].to(self.device)
            loss=(self.depth_predictor(f,p)-target).square().mean()
            self.depth_predictor_opt.zero_grad(); (loss*self.depth_predictor_cfg['loss_scale']).backward()
            torch.nn.utils.clip_grad_norm_(self.depth_predictor.parameters(),1.); self.depth_predictor_opt.step()
            loss_total+=loss.item(); self.depth_updates_done+=1
        return loss_total/self.depth_predictor_cfg['training_iters']
    def signatures(self):
        with torch.no_grad():
            policy=self.alg.actor_critic.act_inference(torch.zeros(1,285,device=self.device),torch.zeros(1,210,device=self.device),torch.zeros(1,512,device=self.device)).cpu()
            encoder=self._world_model.encoder({'prop':torch.zeros(1,33,device=self.device),'image':torch.zeros(1,64,64,1,device=self.device)}).cpu()
        return {'policy':policy,'encoder':encoder}
    def save(self,path,infos=None):
        torch.save(dict(model_state_dict=self.alg.actor_critic.state_dict(),optimizer_state_dict=self.alg.optimizer.state_dict(),
            world_model_dict=self._world_model.state_dict(),wm_optimizer_state_dict=self._world_model._model_opt._opt.state_dict(),
            depth_predictor=self.depth_predictor.state_dict(),depth_optimizer=self.depth_predictor_opt.state_dict(),
            discriminator=self.alg.discriminator.state_dict(),amp_optimizer=self.alg.optimizer.state_dict(),
            signatures=self.signatures(),amp_normalizer=self.alg.amp_normalizer,iter=self.current_learning_iteration,infos=infos,
            train_cfg={'runner':self.cfg,'algorithm':self.alg_cfg,'policy':self.policy_cfg,'depth_predictor':self.depth_predictor_cfg},
            wm_config=vars(self.wm_config),wm_updates_done=self.wm_updates_done,depth_updates_done=self.depth_updates_done),path)
    def load(self,path,load_optimizer=True,**kwargs):
        d=torch.load(path,map_location=self.device,weights_only=False)
        self.alg.actor_critic.load_state_dict(d['model_state_dict']); self._world_model.load_state_dict(d['world_model_dict'])
        for module,key in [(self.alg.actor_critic,'model_state_dict'),(self._world_model,'world_model_dict')]:
            for name,value in module.state_dict().items():
                if not torch.equal(value,d[key][name]):raise RuntimeError(f'Checkpoint tensor mismatch: {key}/{name}')
        self.depth_predictor.load_state_dict(d['depth_predictor']); self.alg.discriminator.load_state_dict(d['discriminator'])
        self.alg.amp_normalizer=d['amp_normalizer']
        if load_optimizer:
            self.alg.optimizer.load_state_dict(d['optimizer_state_dict']); self._world_model._model_opt._opt.load_state_dict(d['wm_optimizer_state_dict']); self.depth_predictor_opt.load_state_dict(d['depth_optimizer'])
        self.round_trip_verified=False
        if 'signatures' in d:
            self.round_trip_errors={}
            for key,value in self.signatures().items():
                self.round_trip_errors[key]=float((value-d['signatures'][key].cpu()).abs().max())
                torch.testing.assert_close(value,d['signatures'][key].cpu(),rtol=1e-5,atol=1e-5)
            self.round_trip_verified=True
            print('CHECKPOINT_OUTPUT_ROUND_TRIP_PASSED',flush=True)
        self.current_learning_iteration=d['iter']; self.wm_updates_done=d['wm_updates_done']; self.depth_updates_done=d['depth_updates_done']
