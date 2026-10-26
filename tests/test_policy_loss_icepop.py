"""CPU regression coverage for ICEPOP importance-sampling masking."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


def _load_policy_loss():
    # Import the local loss module without importing distributed-training dependencies.
    models_dir = Path(__file__).resolve().parents[1] / "openrlhf-kd" / "openrlhf" / "models"
    package_name = "_llm_dojo_isolated_models"
    package = types.ModuleType(package_name)
    package.__path__ = [str(models_dir)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(package_name + ".loss", models_dir / "loss.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.PolicyLoss


PolicyLoss = _load_policy_loss()


class TestICEPOPMasking(unittest.TestCase):
    def test_rejected_infinite_ratio_does_not_poison_loss_or_gradients(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=str(dtype)):
                old = torch.full((2, 2), -1, dtype=dtype)
                current = old.clone().requires_grad_(True)
                rollout = torch.tensor([[-1001, -1], [-1, -1]], dtype=dtype)
                advantages = torch.ones_like(old)
                action_mask = torch.ones_like(old)
                loss_fn = PolicyLoss(
                    enable_vllm_is_correction=True,
                    vllm_is_truncated_threshold=[0.5, 5.0],
                    use_icepop=True,
                )
                # exp(old[0, 0] - rollout[0, 0]) = exp(1000) = inf.
                loss, _, _, _ = loss_fn(current, old, advantages, action_mask, rollout)
                loss.backward()

                torch.testing.assert_close(loss, torch.tensor(-0.75, dtype=dtype))
                self.assertTrue(torch.isfinite(current.grad).all())
                self.assertEqual(current.grad[0, 0].item(), 0.0)
                torch.testing.assert_close(
                    current.grad.reshape(-1)[1:],
                    torch.full((3,), -0.25, dtype=dtype),
                )

    def test_finite_rejected_ratio_and_accepted_weights_unchanged(self):
        old = torch.full((1, 3), -1.0)
        current = old.clone().requires_grad_(True)
        rollout = torch.tensor([[-11.0, -1.0, -1.0]])
        loss_fn = PolicyLoss(
            enable_vllm_is_correction=True,
            vllm_is_truncated_threshold=[0.5, 5.0],
            use_icepop=True,
        )
        loss, _, _, _ = loss_fn(
            current, old, torch.ones_like(old), torch.ones_like(old), rollout
        )
        loss.backward()

        torch.testing.assert_close(loss, torch.tensor(-2.0 / 3.0))
        torch.testing.assert_close(
            current.grad, torch.tensor([[0.0, -1.0 / 3.0, -1.0 / 3.0]])
        )


if __name__ == "__main__":
    unittest.main()
