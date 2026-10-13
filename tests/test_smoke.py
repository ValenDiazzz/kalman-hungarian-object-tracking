"""Check that the initial package can be imported without scientific dependencies."""

import importlib
import unittest


class SmokeTests(unittest.TestCase):
    def test_package_imports(self):
        package = importlib.import_module("kalman_hungarian_object_tracking")
        self.assertEqual(package.__name__, "kalman_hungarian_object_tracking")


if __name__ == "__main__":
    unittest.main()
