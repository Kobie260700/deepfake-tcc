import unittest

from src.capture.windows import DesktopWindow, list_visible_windows


class WindowSelectionTests(unittest.TestCase):
    def test_enumerates_visible_windows(self):
        windows = list_visible_windows()

        self.assertIsInstance(windows, list)
        for window in windows:
            self.assertIsInstance(window, DesktopWindow)
            self.assertTrue(window.title)
            self.assertGreater(window.region.width, 100)
            self.assertGreater(window.region.height, 100)


if __name__ == "__main__":
    unittest.main()
