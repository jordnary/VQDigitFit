"""Check quantization semantics and the three independent gradient paths."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vqdigitfit import VQVAE, code_stats, image_metrics, seed_all, torch, F


class ModelTests(unittest.TestCase):
    def setUp(self):
        seed_all(42)
        self.model = VQVAE()
        self.x = torch.rand(2, 1, 28, 28)

    def test_shapes_and_finite_output(self):
        self.assertEqual(tuple(self.model.encoder(self.x).shape), (2, 32, 7, 7))
        self.assertEqual(tuple(self.model.codebook.weight.shape), (64, 32))
        recon, code, commit, indices = self.model(self.x)
        self.assertEqual(recon.shape, self.x.shape)
        self.assertEqual(tuple(indices.shape), (2, 7, 7))
        self.assertTrue(torch.isfinite(recon).all())
        self.assertTrue(torch.all((indices >= 0) & (indices < 64)))
        self.assertTrue(torch.isfinite(code + commit))

    def test_nearest_second_and_tie_order(self):
        with torch.no_grad():
            self.model.codebook.weight.zero_()
            self.model.codebook.weight[:, 0] = torch.arange(64)
        z = torch.zeros(1, 32, 7, 7)
        z[:, 0] = 5.2
        q1, i1 = self.model.quantize(z)
        q2, i2 = self.model.quantize(z, "second")
        self.assertTrue(torch.all(i1 == 5))
        self.assertTrue(torch.all(i2 == 6))
        self.assertTrue(torch.all(q1[:, 0] == 5))
        self.assertTrue(torch.all(q2[:, 0] == 6))
        z[:, 0] = 5.5
        self.assertTrue(torch.all(self.model.quantize(z)[1] == 5))
        self.assertTrue(torch.all(self.model.quantize(z, "second")[1] == 6))

    def test_straight_through_and_loss_gradient_isolation(self):
        pred, code, commit, _ = self.model(self.x)
        F.l1_loss(pred, self.x).backward(retain_graph=True)
        self.assertGreater(float(self.model.encoder[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.model.decoder[0].weight.grad.abs().sum()), 0)
        self.assertIsNone(self.model.codebook.weight.grad)
        self.model.zero_grad(set_to_none=True)
        code.backward(retain_graph=True)
        self.assertGreater(float(self.model.codebook.weight.grad.abs().sum()), 0)
        self.assertIsNone(self.model.encoder[0].weight.grad)
        self.model.zero_grad(set_to_none=True)
        commit.backward()
        self.assertGreater(float(self.model.encoder[0].weight.grad.abs().sum()), 0)
        self.assertIsNone(self.model.codebook.weight.grad)

    def test_random_replay_and_inference_keeps_weights(self):
        self.model.eval()
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        with torch.inference_mode():
            a = self.model(self.x, "random", torch.Generator().manual_seed(44))
            b = self.model(self.x, "random", torch.Generator().manual_seed(44))
            c = self.model(self.x, "random", torch.Generator().manual_seed(45))
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertTrue(torch.equal(a[3], b[3]))
        self.assertFalse(torch.equal(a[3], c[3]))
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))

    def test_metric_definitions(self):
        values = image_metrics(torch.full_like(self.x, 0.5), torch.zeros_like(self.x))
        self.assertEqual(values["mae"], [0.5, 0.5])
        self.assertEqual(values["mse"], [0.25, 0.25])
        self.assertAlmostEqual(values["psnr_db"][0], 6.0206, places=4)
        self.assertAlmostEqual(code_stats(torch.ones(64))["perplexity"], 64.0)
        self.assertEqual(code_stats(torch.tensor([49, 0]))["active_codes"], 1)


if __name__ == "__main__":
    unittest.main()
