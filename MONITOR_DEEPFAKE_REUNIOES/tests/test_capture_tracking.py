import unittest
from unittest.mock import patch

from src.capture.screen import CaptureRegion
from src.monitor import tracked_region_provider


class CaptureTrackingTests(unittest.TestCase):
    def test_tracks_selected_subregion_when_window_moves(self):
        selected = CaptureRegion(300, 240, 640, 480)
        original_window = CaptureRegion(100, 100, 1200, 800)
        moved_window = CaptureRegion(180, 150, 1200, 800)

        with patch("src.monitor.window_region", return_value=moved_window):
            provider = tracked_region_provider(123, selected, original_window)
            tracked = provider()

        self.assertEqual(tracked, CaptureRegion(380, 290, 640, 480))

    def test_tracks_whole_window_when_no_subregion_was_selected(self):
        moved_window = CaptureRegion(180, 150, 1200, 800)

        with patch("src.monitor.window_region", return_value=moved_window):
            provider = tracked_region_provider(
                123,
                CaptureRegion(100, 100, 1200, 800),
                None,
            )
            tracked = provider()

        self.assertEqual(tracked, moved_window)

    def test_scales_subregion_when_window_is_resized(self):
        selected = CaptureRegion(300, 300, 600, 400)
        original_window = CaptureRegion(100, 100, 1200, 800)
        resized_window = CaptureRegion(50, 50, 600, 400)

        with patch("src.monitor.window_region", return_value=resized_window):
            provider = tracked_region_provider(123, selected, original_window)
            tracked = provider()

        self.assertEqual(tracked, CaptureRegion(150, 150, 300, 200))


if __name__ == "__main__":
    unittest.main()
