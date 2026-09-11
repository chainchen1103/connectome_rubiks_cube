import unittest
from types import SimpleNamespace
import numpy as np
from connectome_lab.dynamics import LIFNetwork, LIFConfig, transmitter_signs
from connectome_lab.plasticity import LocalPlasticity, PlasticityConfig


def graph(nt="acetylcholine"):
    return SimpleNamespace(n_neurons=2, n_edges=1, neurons=[
        SimpleNamespace(id="a", neurotransmitter=nt),
        SimpleNamespace(id="b", neurotransmitter="unknown")],
        source=np.array([0]), target=np.array([1]), synapse_count=np.array([1.0]))


class DynamicsTests(unittest.TestCase):
    def test_exact_leak(self):
        n = LIFNetwork(graph())
        n.voltage[:] = .5
        n.step(np.zeros(2))
        np.testing.assert_allclose(n.voltage, .5 * np.exp(-.1))

    def test_threshold_reset_and_full_refractory(self):
        n = LIFNetwork(graph(), LIFConfig(refractory_ms=2))
        self.assertTrue(n.step([20, 0])[0])
        self.assertEqual(n.voltage[0], 0)
        self.assertFalse(n.step([20, 0])[0])
        self.assertFalse(n.step([20, 0])[0])
        self.assertTrue(n.step([20, 0])[0])

    def test_direction_delay_and_inhibition(self):
        for nt, sign in [("acetylcholine", 1), ("GABA", -1)]:
            n = LIFNetwork(graph(nt))
            n.step([20, 0])
            self.assertEqual(n.synaptic_current[1], 0)
            n.step([0, 0])
            self.assertEqual(np.sign(n.synaptic_current[1]), sign)
            self.assertEqual(n.synaptic_current[0], 0)

    def test_validation(self):
        with self.assertRaises(ValueError):
            LIFConfig(dt_ms=0)
        with self.assertRaises(ValueError):
            LIFNetwork(graph()).step([np.nan, 0])
        with self.assertRaises(ValueError):
            transmitter_signs(graph(), overrides={"missing": -1})

    def test_stdp_causality(self):
        for first, second, expected in [([1, 0], [0, 1], 1), ([0, 1], [1, 0], -1)]:
            n = LIFNetwork(graph())
            p = LocalPlasticity(n, PlasticityConfig(rule="stdp"))
            before = n.weights.copy()
            p.observe(first)
            p.observe(second)
            self.assertEqual(np.sign(n.weights[0] - before[0]), expected)

    def test_reward_delay_freeze_bounds(self):
        n = LIFNetwork(graph())
        p = LocalPlasticity(n, PlasticityConfig(learning_rate=100))
        before = n.weights.copy()
        p.observe([1, 0]); p.observe([0, 1])
        np.testing.assert_array_equal(before, n.weights)
        e = p.eligibility.copy()
        p.observe([0, 0])
        self.assertLess(p.eligibility[0], e[0])
        p.reward(1)
        self.assertEqual(n.weights[0], 8)
        p.reward(-1)
        self.assertEqual(n.weights[0], 0)
        self.assertEqual(n.signs[0], 1)

    def test_mask_and_none(self):
        for config, mask in [(PlasticityConfig(rule="none"), None),
                             (PlasticityConfig(rule="stdp"), [False])]:
            n = LIFNetwork(graph())
            before = n.weights.copy()
            p = LocalPlasticity(n, config, mask)
            p.observe([1, 0]); p.observe([0, 1]); p.reward(1)
            np.testing.assert_array_equal(before, n.weights)


if __name__ == "__main__":
    unittest.main()
