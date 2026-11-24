"""CPU-only gates for the interface-bundle plan; no Protenix/GPU needed."""
import unittest

from extra_training.joint_reranker.group_schedule import PoseRef, plan_epoch, scaled_group_loss


def poses(pdb, iface, n, crop, seed_base=0):
    return [PoseRef(pdb, iface, str(seed_base + i // 5), i % 5, i % 5, crop,
                    f"{pdb}/free_{seed_base + i // 5}") for i in range(n)]


class GroupScheduleTest(unittest.TestCase):
    def setUp(self):
        self.cap = {256: 20, 384: 8, 512: 5}

    def test_75_split_20_20_20_15(self):
        steps = plan_epoch(poses('abc', 'abc_HA', 75, 256), self.cap,
                           world=3, seed=2, epoch=0)
        batches = [b for s in steps for b in s.per_rank if b]
        self.assertEqual(len(batches), 1)
        self.assertEqual(sorted(len(g) for g in batches[0].groups), [15, 20, 20, 20])
        self.assertEqual(sum(len(g) for g in batches[0].groups), 75)

    def test_multiple_interfaces_same_pdb_not_mixed(self):
        ps = poses('abc', 'abc_HA', 75, 256) + poses('abc', 'abc_LA', 75, 256)
        s = plan_epoch(ps, self.cap, world=3, seed=42, epoch=0)
        self.assertEqual(s[0].n_interfaces, 2)
        for batch in s[0].per_rank:
            if batch:
                for group in batch.groups:
                    self.assertEqual({x.interface for x in group}, {batch.interface})

    def test_global_gradient_normalization_with_variable_k(self):
        ps = (poses('deep', 'deep_HA', 75, 256) + poses('short', 'short_HA', 5, 512)
              + poses('medium', 'medium_HA', 18, 384))
        st = plan_epoch(ps, self.cap, world=3, seed=11, epoch=0)[0]
        pae_sum = rank_sum = 0.0
        for rank, batch in enumerate(st.per_rank):
            if batch is None:
                continue
            for gi, group in enumerate(batch.groups):
                a, b = st.loss_coefficients(rank, gi)
                pae_sum += a
                rank_sum += b
        self.assertAlmostEqual(pae_sum, 1.0)
        self.assertAlmostEqual(rank_sum, 1.0)
        self.assertEqual(st.n_poses, 98)
        self.assertEqual(st.n_interfaces, 3)

    def test_per_pose_ce_and_per_interface_rank_not_per_group(self):
        ps = poses('deep', 'deep_HA', 75, 256) + poses('short', 'short_HA', 5, 256)
        step = plan_epoch(ps, self.cap, world=3, seed=1, epoch=0)[0]
        weights = {}
        for rank, batch in enumerate(step.per_rank):
            if not batch:
                continue
            sw = 0.0
            for gi, group in enumerate(batch.groups):
                pw, rw = step.loss_coefficients(rank, gi)
                self.assertAlmostEqual(pw / len(group), 1 / step.n_poses)
                sw += rw
            weights[batch.interface] = sw
        self.assertEqual(len(weights), 2)
        self.assertAlmostEqual(list(weights.values())[0], list(weights.values())[1])

    def test_all_candidates_once_uneven_final_step(self):
        ps = [p for i in range(5) for p in poses(str(i), str(i)+'_HA', 5, 256)]
        steps = plan_epoch(ps, self.cap, world=3, seed=42, epoch=0)
        self.assertEqual(len(steps), 2)
        self.assertEqual(len(steps[-1].per_rank), 3)
        self.assertEqual(sum(x is None for x in steps[-1].per_rank), 1)
        items = [p.identity for s in steps for b in s.per_rank if b for g in b.groups for p in g]
        self.assertEqual(set(items), {p.identity for p in ps})
        self.assertEqual(len(items), 25)
        self.assertEqual(sum(s.n_interfaces for s in steps), 5)

    def test_plan_stable_under_manifest_reordering(self):
        ps = poses('a','a_HA',75,256) + poses('b','b_HA',15,384) + poses('c','c_HA',5,512)
        x = plan_epoch(ps, self.cap, world=3, seed=3, epoch=1)
        y = plan_epoch(list(reversed(ps)), self.cap, world=3, seed=3, epoch=1)
        self.assertEqual(x, y)
        z = plan_epoch(ps, self.cap, world=3, seed=3, epoch=2)
        self.assertNotEqual(x, z)

    def test_dynamic_divisors_across_steps(self):
        ps = []
        for i, n in enumerate([75, 5, 75, 5]):
            ps += poses(str(i), f'{i}_HA', n, 256)
        steps = plan_epoch(ps, self.cap, world=3, seed=3, epoch=1)
        self.assertEqual(len(steps), 2)
        self.assertNotEqual(steps[0].n_poses, steps[1].n_poses)
        for step in steps:
            self.assertAlmostEqual(sum(step.loss_coefficients(r, gi)[0]
                for r,b in enumerate(step.per_rank) if b for gi in range(len(b.groups))), 1.0)

    def test_no_pose_drop_for_odd_k_and_singleton(self):
        p = poses('p','p_HA',5,256)
        st = plan_epoch(p, {256:2}, world=1, seed=1, epoch=0)[0]
        self.assertEqual(sorted(len(g) for g in st.per_rank[0].groups), [1,2,2])
        self.assertEqual(st.n_poses, 5)

    def test_duplicate_and_unknown_crop_refused(self):
        p = poses('a', 'a_HA', 5, 256)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            plan_epoch(p+[p[0]], self.cap, world=3, seed=1, epoch=0)
        with self.assertRaisesRegex(ValueError,'verified'):
            plan_epoch(poses('a','a_HA',5,768),self.cap,world=3,seed=1,epoch=0)

    def test_manual_rank_sum_matches_two_global_mean_gradients(self):
        import torch
        ps = (poses('deep', 'deep_HA', 75, 256)
              + poses('shallow', 'shallow_HA', 5, 512)
              + poses('medium', 'medium_HA', 18, 384))
        step = plan_epoch(ps, self.cap, world=3, seed=7, epoch=0)[0]
        alpha = 0.4

        def pae(w, p):
            return (w * (p.pose_index + 1) / 5.0 - 0.7).square()

        def ranking(w, group):
            return (w * len(group) / 20.0 - 0.3).square()

        whole = torch.tensor(1.3, requires_grad=True)
        all_poses = [p for b in step.per_rank if b for g in b.groups for p in g]
        global_pae = sum(pae(whole, p) for p in all_poses) / step.n_poses
        global_rank = (sum(sum(ranking(whole, g) for g in b.groups) / len(b.groups)
                           for b in step.per_rank if b) / step.n_interfaces)
        expected_grad = torch.autograd.grad(global_pae + alpha * global_rank, whole)[0]

        summed_manual_grad = 0.0
        for rank, batch in enumerate(step.per_rank):
            if batch is None:
                continue
            local = torch.tensor(1.3, requires_grad=True)
            for gi, group in enumerate(batch.groups):
                p_scale, r_scale = step.loss_coefficients(rank, gi)
                (p_scale * sum(pae(local, p) for p in group) / len(group)
                 + alpha * r_scale * ranking(local, group)).backward()
            summed_manual_grad += local.grad
        torch.testing.assert_close(summed_manual_grad, expected_grad)

    def test_scaled_helper_does_not_double_alpha_and_retains_autograd(self):
        from types import SimpleNamespace
        import torch
        step = plan_epoch(poses('x', 'x_HA', 5, 256), self.cap,
                          world=3, seed=1, epoch=0)[0]
        rank = next(r for r, b in enumerate(step.per_rank) if b)
        w = torch.tensor(0.5, requires_grad=True)
        result = SimpleNamespace(pae=w.square(), ranking=(w - 2).square(),
                                 total=w.square() + 99 * (w - 2).square())
        loss = scaled_group_loss(step, rank, 0, result, alpha=0.25)
        loss.backward()
        self.assertAlmostEqual(float(w.grad), 2 * 0.5 + 0.25 * (2 * (0.5 - 2)))

    def test_same_seed_different_pose_indices_wont_be_confused(self):
        p = poses('a','a_HA',5,256)
        self.assertEqual(len({(x.seed,x.sample_index) for x in p}),5)
        self.assertEqual(len({(x.seed,x.pose_index) for x in p}),5)


if __name__ == '__main__':
    unittest.main()
