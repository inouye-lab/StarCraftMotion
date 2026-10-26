"""Small failure-path tests; no reference data or filesystem output required."""
import unittest
import numpy as np
import pyarrow as pa

from convert_to_parquet import _build_list_array
from schema import equal_values, rebuild_attributes


class ReleaseGuardsTest(unittest.TestCase):
    def test_offset_overflow_is_rejected_before_allocation(self):
        class Oversized:
            size = 2 ** 31
        with self.assertRaises(OverflowError):
            _build_list_array([Oversized()], np.uint8, pa.uint8())

    def test_float_comparison_preserves_bits_and_accepts_nan_payloads(self):
        self.assertFalse(equal_values(np.array([-0.0], np.float16),
                                      np.array([0.0], np.float16)))
        nan1 = np.array([0x7E01], np.uint16).view(np.float16)
        nan2 = np.array([0x7FFF], np.uint16).view(np.float16)
        self.assertTrue(equal_values(nan1, nan2))
        self.assertFalse(equal_values(np.array([1.0], np.float16),
                                      np.array([1.0], np.float32)))

    def test_hdf5_attributes_restore_int64_and_full_xyz_float32(self):
        # Values that would change under a float16 downcast.
        xyz = np.array([10.123456, 20.234567, 11.984375], np.float32)
        row = dict(chunk_start_frame=720, chunk_end_frame=865, map_frame_start=45,
                   map_frame_skip=16, current_frame_idx=16,
                   player_1_start_location=xyz, player_2_start_location=xyz,
                   playable_area=[24, 4, 176, 140], player_info='[]', replay_info='{}',
                   map_name='Example', replay_hash='abc', segment_idx=5,
                   map_H=176, map_W=200, n_units=217, n_timesteps=145)
        attrs = rebuild_attributes(row)
        self.assertEqual(np.asarray(attrs['chunk_start_frame']).dtype, np.dtype('int64'))
        self.assertEqual(attrs['map_size'].dtype, np.dtype('int64'))
        self.assertEqual(attrs['map_size'].tolist(), [176, 200])
        self.assertEqual(attrs['playable_area'].shape, (4,))
        self.assertEqual(attrs['player_1_start_location'].dtype, np.dtype('float32'))
        np.testing.assert_array_equal(attrs['player_1_start_location'], xyz)
        self.assertEqual(attrs['player_1_start_location'].shape, (3,))


class RoundTripComparisonTest(unittest.TestCase):
    def make_row(self):
        from schema import DATASETS, LIST_DTYPES, array_shape
        row = dict(split='train', map_name='Example', replay_hash='abc', segment_idx=5,
                   n_timesteps=145, n_units=2, map_T=2, map_H=2, map_W=3,
                   chunk_start_frame=720, chunk_end_frame=865, map_frame_start=45,
                   map_frame_skip=16, current_frame_idx=16,
                   player_1_start_location=np.array([10.123456, 20.234567, 11.984375], np.float32),
                   player_2_start_location=np.array([10.123456, 20.234567, 11.984375], np.float32),
                   playable_area=np.array([0, 0, 2, 3], np.int64),
                   player_info='[]', replay_info='{}')
        for column in DATASETS:
            n = int(np.prod(array_shape(column, row)))
            row[column] = (np.arange(n) % 7).astype(LIST_DTYPES[column])
        row['unit_tag'] = np.array([2**63 + 3, 2**64 - 2], np.uint64)
        row['coordinate'][0] = np.nan
        return row

    def test_roundtrip_metadata_and_values_in_memory(self):
        import h5py
        from parquet_to_hdf5 import populate
        from compare_hdf5 import compare_open_files
        row = self.make_row()
        with h5py.File('memory-a.h5', 'w', driver='core', backing_store=False) as a, \
             h5py.File('memory-b.h5', 'w', driver='core', backing_store=False) as b:
            populate(a, row)
            populate(b, row)
            self.assertIsNone(compare_open_files(a, b))
            np.testing.assert_array_equal(a['unit_data/global/unit_tag'][:], row['unit_tag'])
            self.assertEqual(a.attrs['player_1_start_location'].dtype, np.dtype('float32'))
            self.assertEqual(a.attrs['player_1_start_location'].shape, (3,))
            self.assertEqual(len(a.attrs), 17)
            del b.attrs['current_frame_idx']
            b.attrs['current_frame_idx'] = np.int32(16)
            self.assertIn('attribute dtype', compare_open_files(a, b))

    def test_comparator_detects_one_changed_dataset_value(self):
        import h5py
        from parquet_to_hdf5 import populate
        from compare_hdf5 import compare_open_files
        row = self.make_row()
        with h5py.File('memory-c.h5', 'w', driver='core', backing_store=False) as a, \
             h5py.File('memory-d.h5', 'w', driver='core', backing_store=False) as b:
            populate(a, row)
            populate(b, row)
            b['unit_data/repeated/health'][0, 0] = np.float16(99)
            self.assertIn('dataset values: unit_data/repeated/health', compare_open_files(a, b))


if __name__ == '__main__':
    unittest.main()
