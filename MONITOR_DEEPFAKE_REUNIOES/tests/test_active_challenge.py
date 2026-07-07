import unittest
from types import SimpleNamespace

import numpy as np

from src.liveness.active import ActiveChallenge


class ActiveChallengeTests(unittest.TestCase):
    def test_classifies_center_orientation(self):
        face = SimpleNamespace(
            kps=np.array([[20, 20], [40, 20], [30, 30], [23, 40], [37, 40]], dtype=np.float32)
        )
        self.assertEqual(ActiveChallenge.orientation(face), "CENTRO")

    def test_start_creates_four_action_sequence(self):
        challenge = ActiveChallenge()
        challenge.start()
        self.assertTrue(challenge.active)
        self.assertEqual(len(challenge.sequence), 4)
        self.assertEqual(challenge.sequence[1:3], ["CENTRO", "PISQUE"])


if __name__ == "__main__":
    unittest.main()
