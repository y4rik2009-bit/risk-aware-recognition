import unittest
import numpy as np
import torch
from risk import crc_threshold, differentiable_threshold, corc_threshold, losses, empirical_cvar
from data import source_pools, boards


class CalibrationTests(unittest.TestCase):
    def test_bruteforce_and_ties(self):
        rng = np.random.default_rng(4)
        for _ in range(20):
            p = rng.choice([0.1, 0.2, 0.6, 0.9], (30, 4))
            y = (rng.random((30, 4)) > 0.6).astype(float); y[:, 0] = 1
            for alpha in [0.05, 0.1, 0.3]:
                q = crc_threshold(p, y, alpha)
                candidates = [0, *np.unique(p[y.astype(bool)]), 1]
                feasible = [c for c in candidates if (losses(p,y,c).sum()+1)/31 <= alpha+1e-12]
                self.assertAlmostEqual(q, max(feasible))
                tq = differentiable_threshold(torch.tensor(p),torch.tensor(y),alpha)
                self.assertAlmostEqual(q,float(tq))

    def test_active_score_gradient(self):
        p = torch.linspace(0.1,0.9,100,dtype=torch.float64).reshape(25,4).requires_grad_()
        y = torch.ones_like(p)
        q = differentiable_threshold(p,y,0.2); q.backward()
        direction = torch.sin(torch.arange(100,dtype=torch.float64)).reshape(25,4)
        eps = 1e-6
        fd = (differentiable_threshold(p.detach()+eps*direction,y,0.2)-
              differentiable_threshold(p.detach()-eps*direction,y,0.2))/(2*eps)
        self.assertAlmostEqual(float(fd),float((p.grad*direction).sum()),places=7)
        self.assertEqual(int((p.grad!=0).sum()),1)

    def test_cvar_correction(self):
        rng=np.random.default_rng(3); p=rng.random((200,4)); y=np.ones_like(p)
        q=corc_threshold(p,y,0.2,0.8,0.05)
        transformed=0.05+np.maximum(losses(p,y,q)-0.05,0)/0.2
        bound=(transformed.sum()+0.05+0.95/0.2)/201
        self.assertLessEqual(bound,0.2+1e-10)
        self.assertAlmostEqual(empirical_cvar([0,0,0.5,1],0.5),0.75)

    def test_infeasible_and_invalid(self):
        self.assertEqual(crc_threshold([[0.5]],[[1]],0.1),0)
        with self.assertRaises(ValueError): crc_threshold([[np.nan]],[[1]],0.1)
        with self.assertRaises(ValueError): crc_threshold([[0.1]],[[0]],0.1)

    def test_source_separation_and_reproducibility(self):
        x,y,a,b=source_pools(42); self.assertFalse(set(a)&set(b))
        p,q=boards(x,y,a,20,42); r,s=boards(x,y,a,20,42)
        np.testing.assert_array_equal(p,r); np.testing.assert_array_equal(q,s)
        self.assertTrue(np.isin(q.sum(1),[1,2,3]).all())


if __name__=='__main__': unittest.main()
