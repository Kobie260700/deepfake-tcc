import unittest

from src.capture.screen import CaptureRegion
from src.ui.status_overlay import overlay_position


class OverlayPositionTests(unittest.TestCase):
    def test_places_overlay_left_of_capture_when_space_exists(self):
        region = CaptureRegion(2200, 100, 1000, 700)
        x, y = overlay_position(region, 500, 250, (0, 0, 3840, 1080))
        self.assertLessEqual(x + 500, region.left)
        self.assertEqual(y, region.top)

    def test_fallback_stays_inside_virtual_screen(self):
        region = CaptureRegion(0, 0, 1920, 1080)
        x, y = overlay_position(region, 500, 250, (0, 0, 1920, 1080))
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + 500, 1920)
        self.assertLessEqual(y + 250, 1080)


if __name__ == "__main__":
    unittest.main()
