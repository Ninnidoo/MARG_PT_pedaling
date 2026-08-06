from __future__ import annotations
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from src.stage2_encoder_only.coarse_to_fine import (Stage2CoarseToFineModel, coarse_to_fine_loss, decode_pedals, normalized_depth_targets, pedal_regions)
from src.stage2_encoder_only.evaluate_coarse_to_fine import _average
from src.stage2_encoder_only.train_coarse_to_fine import build_optimizer, train_step
from src.stage2_encoder_only.training import create_grad_scaler, set_deterministic_seed
from src.stage2_encoder_only.train import EarlyStopping, build_last_checkpoint, restore_training_state

class E(nn.Module):
 def __init__(self): super().__init__();self.config=SimpleNamespace(hidden_size=6,dropout_rate=0.);self.emb=nn.Embedding(256,6);self.p=nn.Linear(6,6)
 def forward(self,input_ids,attention_mask):
  b,l=input_ids.shape;return SimpleNamespace(last_hidden_state=self.p(self.emb(input_ids).view(b,l//8,8,6).mean(2)))
class CTFTests(unittest.TestCase):
 def batch(self):
  x=torch.randint(0,256,(2,16));return {"input_ids":x,"token_attention_mask":torch.ones_like(x),"pedal_targets":torch.tensor([[[0,1,63,127],[126,0,127,1]],[[1,126,0,127],[63,127,0,1]]]),"note_mask":torch.ones(2,2,dtype=torch.bool)}
 def model(self): return Stage2CoarseToFineModel(E())
 def test_regions(self): self.assertTrue(torch.equal(pedal_regions(torch.tensor([0,1,63,126,127,-100])),torch.tensor([0,1,1,1,2,-100])))
 def test_depth_normalization_and_decode(self):
  self.assertTrue(torch.allclose(normalized_depth_targets(torch.tensor([1,126])),torch.tensor([0.,1.])))
  r=torch.tensor([[[[9.,0.,0.],[0.,9.,0.],[0.,0.,9.],[0.,9.,0.]]]]);d=torch.tensor([[[0.,-20.,20.,0.]]]);self.assertTrue(torch.equal(decode_pedals(r,d),torch.tensor([[[0,1,127,63]]])))
 def test_zero_intermediate_loss(self):
  t=torch.tensor([[[0,127,0,127]]]);l=coarse_to_fine_loss(torch.randn(1,1,4,3,requires_grad=True),torch.randn(1,1,4,requires_grad=True),t);self.assertEqual(l.intermediate_count,0);self.assertEqual(float(l.depth_smooth_l1),0.)
 def test_ignored_positions(self):
  t=torch.tensor([[[0,-100,127,-100]]]);l=coarse_to_fine_loss(torch.randn(1,1,4,3),torch.randn(1,1,4),t);self.assertTrue(torch.isfinite(l.total))
 def test_overlap_raw_outputs(self):
  a=_average(3,[(0,torch.ones(2,4,3)),(1,torch.full((2,4,3),3.))]);self.assertEqual(float(a[1,0,0]),2.)
 def test_deterministic_initialization(self):
  set_deterministic_seed(7);a=self.model();set_deterministic_seed(7);b=self.model();self.assertTrue(all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters())))
 def test_heads_and_encoder_gradients_update(self):
  m=self.model();o=build_optimizer(m,.01,.01,0.);before=[p.detach().clone() for p in m.parameters()];metrics=train_step(m,self.batch(),o,create_grad_scaler(False,"cpu"),False,1.);self.assertGreater(metrics["encoder_gradient_norm"],0);self.assertGreater(metrics["head_gradient_norm"],0);self.assertTrue(all(any(p.grad is not None and p.grad.abs().sum()>0 for p in h.parameters()) for h in list(m.region_heads)+list(m.depth_heads)));self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,m.parameters())))
 def test_nonpedal_fields_are_unchanged_by_masking(self):
  original=torch.tensor([[5,6,7,8,9,10,11,12]]);masked=original.clone();masked[:,4:]=1;self.assertTrue(torch.equal(original[:,:4],masked[:,:4]))
 def test_checkpoint_resume_epoch_boundary(self):
  m=self.model();o=build_optimizer(m,.01,.01,0.);s=create_grad_scaler(False,"cpu");e=EarlyStopping(4,.0001);payload=build_last_checkpoint(m,o,s,3,17,e,{"loss_configuration":"ctf"});target=self.model();target_opt=build_optimizer(target,.01,.01,0.);target_scale=create_grad_scaler(False,"cpu");start,step=restore_training_state(payload,target,target_opt,target_scale,EarlyStopping(4,.0001));self.assertEqual((start,step),(4,17))
 def test_slots_decode_independently(self):
  r=torch.zeros(1,1,4,3);r[0,0,0,0]=2;r[0,0,1,1]=2;r[0,0,2,2]=2;r[0,0,3,1]=2;d=torch.tensor([[[0.,0.,0.,2.]]]);self.assertTrue(torch.equal(decode_pedals(r,d),torch.tensor([[[0,63,127,111]]])))
if __name__=="__main__":unittest.main()
