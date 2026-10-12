"""Run: python -m unittest -v test_watermark.py"""
import unittest
import io

import cv2
import numpy as np
from PIL import Image

import watermark as wm


class WatermarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(51)
        h, w = 800, 1000
        yy, xx = np.mgrid[:h, :w]
        luma = 90 + 50*np.sin(xx/37) + 45*np.cos(yy/67) + rng.normal(0, 15, (h,w))
        cls.original = np.empty((h,w,3), dtype=np.uint8)
        for c in range(3):
            cls.original[:, :, c] = np.clip(luma + 10*c + np.sin((xx+yy)/19)*10, 0, 255)
        cls.text = 'Photo archive'
        cls.marked = wm.embed(cls.original, cls.text)

    def test_hamming_recovers_one_bad_bit_per_word(self):
        data = wm._frame('1234567890123456')
        coded = wm._encode_hamming(data)
        coded[0::15] ^= 1
        self.assertEqual(wm._unframe(wm._decode_hamming(coded)), '1234567890123456')

    def test_utf8_and_length(self):
        self.assertEqual(wm._unframe(wm._decode_hamming(wm._encode_hamming(wm._frame('Привет')))), 'Привет')
        with self.assertRaises(ValueError):
            wm._frame('X'*17)

    def test_unmarked_has_no_watermark(self):
        self.assertIsNone(wm.extract(self.original))

    def test_original_and_jpeg(self):
        self.assertEqual(wm.extract(self.marked), self.text)
        ok, buf = cv2.imencode('.jpg', cv2.cvtColor(self.marked, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 55])
        self.assertTrue(ok)
        attacked = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        self.assertEqual(wm.extract(attacked), self.text)

    def test_resize_and_crop(self):
        resized = cv2.resize(self.marked, None, fx=.75, fy=.75, interpolation=cv2.INTER_AREA)
        self.assertEqual(wm.extract(resized), self.text)
        cropped = self.marked[155:-155, 120:-120]
        self.assertEqual(wm.extract(cropped), self.text)

    def test_simulated_print_scan(self):
        rng = np.random.default_rng(2026)
        scanned = cv2.resize(self.marked, None, fx=.8, fy=.8, interpolation=cv2.INTER_AREA)
        scanned = cv2.GaussianBlur(scanned, (5, 5), .75).astype(np.float32)
        scanned = np.clip(255*(scanned/255)**1.06*.97 + 4 + rng.normal(0, 1.5, scanned.shape), 0, 255).astype(np.uint8)
        self.assertEqual(wm.extract(scanned), self.text)

    def test_rotated(self):
        h, w = self.marked.shape[:2]
        m = cv2.getRotationMatrix2D((w/2, h/2), 1., 1.)
        rotated = cv2.warpAffine(self.marked, m, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
        self.assertEqual(wm.extract(rotated), self.text)

    def test_requires_sufficient_area(self):
        with self.assertRaises(ValueError):
            wm.embed(np.zeros((250, 800, 3), dtype=np.uint8), 'test')


if __name__ == '__main__':
    unittest.main()
