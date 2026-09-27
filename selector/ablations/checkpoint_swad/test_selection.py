import math
import tempfile
import unittest
from pathlib import Path

from selector.ablations.checkpoint_swad.selection import loss_valley


class LossValleyTests(unittest.TestCase):
    def test_official_queue_order_and_repeated_segment_updates(self):
        chosen = loss_valley([3, 2, 1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(chosen["converge_index"], 2)
        self.assertAlmostEqual(chosen["threshold"], 2.6)
        self.assertEqual(chosen["indices"], [2, 3, 2, 3])
        self.assertTrue(chosen["dead_valley"])

    def test_never_converged_uses_last_snapshot(self):
        chosen = loss_valley([5, 4, 3, 2, 1])
        self.assertEqual(chosen["indices"], [4])
        self.assertTrue(chosen["fallback_last"])

    def test_nonfinite_loss_fails(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            loss_valley([1.0, math.nan, 2.0])


class AveragingTests(unittest.TestCase):
    def test_lazy_registered_buffer_is_resized_before_loading(self):
        import torch
        from selector.ablations.checkpoint_swad.run_one import load_model_state

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.register_buffer("q", torch.Tensor())

        model = Model()
        load_model_state(model, {"weight": torch.tensor([2.0]),
                                 "q": torch.tensor([0.2, 0.3, 0.5])})
        torch.testing.assert_close(model.q, torch.tensor([0.2, 0.3, 0.5]))

    def test_repeated_checkpoint_weights_and_first_snapshot_buffers(self):
        import torch
        from selector.ablations.checkpoint_swad.run_one import average_checkpoints

        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.BatchNorm1d(1)
            for step, weight, running_mean in ((0, 1.0, 7.0), (100, 3.0, 9.0)):
                model.weight.data.fill_(weight)
                model.running_mean.fill_(running_mean)
                torch.save({"model_hparams": {"batch_size": 32},
                            "model_dict": model.state_dict()},
                           Path(directory) / f"model_step{step}.pkl")
            average_checkpoints(model, directory, [0, 100, 0], {"batch_size": 32})
            self.assertAlmostEqual(model.weight.item(), 5 / 3, places=6)
            self.assertEqual(model.running_mean.item(), 7.0)

    def test_average_selected_buffers_uses_multiplicity_without_changing_parameters(self):
        import torch
        from selector.ablations.checkpoint_swad.run_average_buffers import average_selected_buffers

        with tempfile.TemporaryDirectory() as directory:
            source = torch.nn.BatchNorm1d(1)
            for step, running_mean, batches in ((0, 7.0, 4), (100, 9.0, 12)):
                source.running_mean.fill_(running_mean)
                source.num_batches_tracked.fill_(batches)
                torch.save({"model_hparams": {"batch_size": 32},
                            "model_dict": source.state_dict()},
                           Path(directory) / f"model_step{step}.pkl")
            model = torch.nn.BatchNorm1d(1)
            model.weight.data.fill_(42.0)
            average_selected_buffers(
                model, directory, [0, 100, 0], {"batch_size": 32}
            )
            self.assertAlmostEqual(model.running_mean.item(), 23 / 3, places=6)
            self.assertEqual(model.num_batches_tracked.item(), 4)
            self.assertEqual(model.weight.item(), 42.0)


if __name__ == "__main__":
    unittest.main()
