import numpy as np
from syris import config as cfg
from syris.imageprocessing import Tiler
from unittest import TestCase


def _get_image(y_range, x_range, ar_type=cfg.NP_FLOAT):
    y_points, x_points = np.mgrid[y_range[0]:y_range[1],
                                  x_range[0]:x_range[1]]
    if ar_type == cfg.NP_CPLX:
        result = np.cast[ar_type](
            x_points * y_points + x_points * y_points * 1j)
    else:
        result = np.cast[ar_type](x_points * y_points)

    return result


class TestImageTiling(TestCase):

    def setUp(self):
        self.data = []

        sizes = [8, 32]
        tile_counts = [2, 4]
        for j in range(len(sizes)):
            for i in range(len(sizes)):
                size = sizes[j], sizes[i]
                for t_j in range(len(tile_counts)):
                    for t_i in range(len(tile_counts)):
                        tiles_count = tile_counts[t_j], tile_counts[t_i]
                        self.data.append((size, tiles_count))

    def _check_tiles(self, tiles, size, tiles_count, outlier):
        for j in range(tiles_count[0]):
            for i in range(tiles_count[1]):
                if outlier:
                    self.assertEqual(tiles[j, i][0],
                                     size[0] / tiles_count[0] * (j - 0.5))
                    self.assertEqual(tiles[j, i][1],
                                     size[1] / tiles_count[1] * (i - 0.5))
                else:
                    self.assertEqual(
                        tiles[j, i][0], j * size[0] / tiles_count[0])
                    self.assertEqual(
                        tiles[j, i][1], i * size[1] / tiles_count[1])

    def test_invalid_tiles(self):
        size = 16, 16
        tiles_count = 4, 3
        self.assertRaises(ValueError, Tiler, size, tiles_count, True)

        self.assertRaises(ValueError, Tiler, size, tiles_count, False)

        tiles_count = 3, 4
        self.assertRaises(ValueError, Tiler, size, tiles_count, True)

        self.assertRaises(ValueError, Tiler, size, tiles_count, False)

    def test_tile_shape(self):
        for size, tiles_count in self.data:
            tiler = Tiler(size, tiles_count, True)
            self.assertEqual(2 * size[0] / tiles_count[0],
                             tiler.tile_shape[0])
            self.assertEqual(2 * size[1] / tiles_count[1],
                             tiler.tile_shape[1])

            tiler = Tiler(size, tiles_count, False)
            self.assertEqual(size[0] / tiles_count[0],
                             tiler.tile_shape[0])
            self.assertEqual(size[1] / tiles_count[1],
                             tiler.tile_shape[1])

    def test_tile_indices(self):
        sizes = [256, 1024]
        tile_counts = [2, 4]
        for j in range(len(sizes)):
            for i in range(len(sizes)):
                size = sizes[j], sizes[i]
                for t_j in range(len(tile_counts)):
                    for t_i in range(len(tile_counts)):
                        tiles_count = tile_counts[t_j], tile_counts[t_i]
                        outlier = True
                        tiler = Tiler(size, tiles_count, outlier)
                        self._check_tiles(tiler.tile_indices, size,
                                          tiles_count, outlier)

                        outlier = False
                        tiler = Tiler(size, tiles_count, outlier)
                        self._check_tiles(tiler.tile_indices, size,
                                          tiles_count, outlier)
            
    def test_insert(self):
        shape = 16, 32
        tiles_count = 4, 2
        tile_shape = [shape[i] / tiles_count[i] for i in range(len(shape))]
        
        tilers = [Tiler(shape, tiles_count, outlier=True, supersampling=1),
                  Tiler(shape, tiles_count, outlier=False, supersampling=1),
                  Tiler(shape, tiles_count, outlier=True, supersampling=2),
                  Tiler(shape, tiles_count, outlier=False, supersampling=2),
                  Tiler(shape, tiles_count, outlier=True, supersampling=4),
                  Tiler(shape, tiles_count, outlier=False, supersampling=4)]
        for tiler in tilers:
            self.assertEqual(tiler.overall_image.shape, shape)
            for j in range(tiles_count[0]):
                for i in range(tiles_count[1]):
                    tile = np.random.random(tile_shape).\
                        astype(tiler.overall_image.dtype)
                    tiler.insert(tile, (j, i))
                    np.testing.assert_equal(
                            tiler.overall_image[j * tile_shape[0]:
                                                tile_shape[0] * (j + 1),
                                                i * tile_shape[1]:
                                                tile_shape[1] * (i + 1)],
                                                tile)
