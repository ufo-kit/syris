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

        sizes = [256, 1024]
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

    def test_tile_size(self):
        for size, tiles_count in self.data:
            tiler = Tiler(size, tiles_count, True)
            self.assertEqual(2 * size[0] / tiles_count[0],
                             tiler.tile_size[0])
            self.assertEqual(2 * size[1] / tiles_count[1],
                             tiler.tile_size[1])

            tiler = Tiler(size, tiles_count, False)
            self.assertEqual(size[0] / tiles_count[0],
                             tiler.tile_size[0])
            self.assertEqual(size[1] / tiles_count[1],
                             tiler.tile_size[1])

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

    def test_create_tiles(self):
        for size, tiles_count in self.data:
            tiler = Tiler(size, tiles_count, True)
            shape = np.array(tiles_count + tiler.tile_size)
            ar = tiler.create_tiles(cplx=False)
            diff = np.sum(shape - np.array(ar.shape))
            self.assertEqual(diff, 0)
            self.assertEqual(ar.dtype, cfg.NP_FLOAT)
            ar = tiler.create_tiles(cplx=True)
            self.assertEqual(ar.dtype, cfg.NP_CPLX)

            tiler = Tiler(size, tiles_count, True)
            shape = np.array(tiles_count + tiler.tile_size)
            ar = tiler.create_tiles(cplx=False)
            diff = np.sum(shape - np.array(ar.shape))
            self.assertEqual(diff, 0)
            self.assertEqual(ar.dtype, cfg.NP_FLOAT)
            ar = tiler.create_tiles(cplx=True)
            self.assertEqual(ar.dtype, cfg.NP_CPLX)

    def _compare_reconstruction(self, tiler, cplx):
        ar_type = cfg.NP_CPLX if cplx else cfg.NP_FLOAT
        tiles = tiler.create_tiles(cplx=cplx)
        for j in (range(tiler.tiles_count[0])):
            for i in (range(tiler.tiles_count[1])):
                y_start, x_start = tiler.tile_indices[j, i]
                tiles[j, i] = _get_image((y_start, y_start +
                                          tiler.tile_size[0]),
                                         (x_start, x_start + tiler.tile_size[
                                          1]),
                                         ar_type=ar_type)

        orig = _get_image((0, tiler.size[0]), (0, tiler.size[1]),
                          ar_type=ar_type)
        reco = tiler.reconstruct(tiles)
        self.assertAlmostEqual(np.sum(reco - orig), 0)

    def test_reconstruct(self):
        for size, tiles_count in self.data:
            tiler = Tiler(size, tiles_count, False)
            self._compare_reconstruction(tiler, False)
            self._compare_reconstruction(tiler, True)

            tiler = Tiler(size, tiles_count, True)
            self._compare_reconstruction(tiler, False)
            self._compare_reconstruction(tiler, True)
