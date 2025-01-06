import os
import unittest
import shutil
import numpy as np
import cv2
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data_loader import data_generator, load_images
import signal

class TimeoutException(Exception):
    pass

def timeout(seconds=10):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutException(f"Test timed out after {seconds} seconds")
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
        return wrapper
    return decorator

class TestDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "tests/mock_data"
        cls.color_path = os.path.join(cls.test_dir, "color")
        cls.gray_path = os.path.join(cls.test_dir, "gray")
        os.makedirs(cls.color_path, exist_ok=True)
        os.makedirs(cls.gray_path, exist_ok=True)

        # Create the mock images
        for i in range(5):
            color_img = np.ones((100, 100, 3), dtype=np.uint8) * (i + 1) * 50
            gray_img = np.ones((100, 100), dtype=np.uint8) * (i + 1) * 50

            cv2.imwrite(os.path.join(cls.color_path, f"color_{i}.png"), color_img)
            cv2.imwrite(os.path.join(cls.gray_path, f"gray_{i}.png"), gray_img)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def test_load_images(self):
        print("Running test_load_images...")
        target_size = (256, 256)
        gray_images, color_images = load_images(
            self.color_path, self.gray_path, target_size=target_size
        )

        self.assertEqual(gray_images.shape, (5, 256, 256, 1))
        self.assertEqual(color_images.shape, (5, 256, 256, 3))
        self.assertTrue(np.all(gray_images[0] >= 0) and np.all(gray_images[0] <= 1))
        self.assertTrue(np.all(color_images[0] >= 0) and np.all(color_images[0] <= 1))
        print("test_load_images passed!")

    @timeout(5)
    def test_data_generator(self):
        print("Starting test_data_generator...")
        target_size = (256, 256)
        batch_size = 2
        generator = data_generator(
            self.color_path, self.gray_path, target_size=target_size, batch_size=batch_size, max_iterations=1
        )
        print("Created generator. Fetching batch...")  # Debugging line
        gray_batch, color_batch = next(generator)
        print("Batch fetched. Validating shapes...")  # Debugging line
        self.assertEqual(gray_batch.shape, (batch_size, 256, 256, 1))
        self.assertEqual(color_batch.shape, (batch_size, 256, 256, 3))
        self.assertTrue(np.all(gray_batch[0] >= 0) and np.all(gray_batch[0] <= 1))
        self.assertTrue(np.all(color_batch[0] >= 0) and np.all(color_batch[0] <= 1))
        print("test_data_generator passed!")  # Debugging line


    def test_empty_directory(self):
        print("Running test_empty_directory...")
        empty_color_path = os.path.join(self.test_dir, "empty_color")
        empty_gray_path = os.path.join(self.test_dir, "empty_gray")
        os.makedirs(empty_color_path, exist_ok=True)
        os.makedirs(empty_gray_path, exist_ok=True)

        with self.assertRaises(IndexError):  # Fail on empty data generator
            next(data_generator(empty_color_path, empty_gray_path))

        with self.assertRaises(ValueError): # Fail on empty load_images
            load_images(empty_color_path, empty_gray_path)

        shutil.rmtree(empty_color_path)
        shutil.rmtree(empty_gray_path)
        print("test_empty_directory passed!")


if __name__ == "__main__":
    unittest.main()
