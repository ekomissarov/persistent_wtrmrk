import unittest
import cv2
import numpy as np
import watermark as wm


class WatermarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(13)
        cls.source = np.full((800, 800, 3), 250, np.uint8)
        cls.source[160:] = cv2.GaussianBlur(
            rng.integers(0, 256, (640, 800, 3), dtype=np.uint8), (5, 5), 0)
        cv2.putText(cls.source, 'SAMPLE DOCUMENT', (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cls.marked = wm.embed(cls.source, 'TEST-2026')

    def test_exact(self):
        self.assertEqual(wm.extract(self.marked), 'TEST-2026')

    def test_jpeg(self):
        bgr = cv2.cvtColor(self.marked, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        self.assertTrue(ok)
        jpeg = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        self.assertEqual(wm.extract(jpeg), 'TEST-2026')

    def test_resize(self):
        self.assertEqual(wm.extract(cv2.resize(self.marked, (600, 600))), 'TEST-2026')

    def test_flat_paper_is_unchanged(self):
        self.assertTrue(np.array_equal(self.source[:20], self.marked[:20]))

    def test_white_page_does_not_support_watermark(self):
        white = np.full((800, 800, 3), 255, np.uint8)
        self.assertTrue(np.array_equal(wm.embed(white, 'TEST-2026'), white))


if __name__ == '__main__':
    unittest.main()
