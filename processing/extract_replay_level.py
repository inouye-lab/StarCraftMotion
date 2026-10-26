#!/usr/bin/python
"""Extract raw, high-fidelity replay data into HDF5 format.

Replaces the quantized uint8 NPZ pipeline with full-precision floats,
raw SC2 integers, and original map dimensions. Produces one HDF5 file
per replay so downstream scenario-level clipping never needs the SC2
engine again.
"""

from __future__ import absolute_import
from __future__ import division

import sys
import os
import time
import math
import enum
import json
import tempfile
import logging
from pathlib import Path

import numpy as np
import h5py
import joblib

from absl import app
from absl import flags

from pysc2 import run_configs
from pysc2.lib import features
from pysc2.lib import point_flag
from pysc2.lib import replay

from s2clientprotocol import sc2api_pb2 as sc_pb

FLAGS = flags.FLAGS

# Extractor-specific flags
flags.DEFINE_string("replay_dir", None, "Directory of .SC2Replay files.")
flags.DEFINE_string("replay", None, "Single replay file.")
flags.DEFINE_integer("max_replays", -1, "Maximum number of replays to process (-1 = all).")
flags.DEFINE_integer("num_parallel_jobs", 1, "Number of parallel joblib workers.")
flags.DEFINE_string("data_save_dir",
                    f"data/replay-level-{time.strftime('%Y_%m_%d_%H%M%S')}",
                    "Output directory for per-replay HDF5 files.")
flags.DEFINE_string("map_data_dir", "map_data/static",
                    "Directory for static map HDF5 files.")
flags.DEFINE_integer("step_mul", 1, "Game steps per observation.")
flags.DEFINE_integer("max_steps", -1, "Maximum steps per replay (-1 = unlimited).")
flags.DEFINE_integer("map_frame_skip", 16,
                     "Frames between dynamic map snapshots (creep, FOW).")
flags.DEFINE_string("map_path", None, "Override map path for replay.")
flags.DEFINE_integer("start_idx", 0,
                     "Start index into sorted replay list (inclusive).")
flags.DEFINE_integer("end_idx", -1,
                     "End index into sorted replay list (exclusive, -1 = end).")

# pysc2-required dummy flags
point_flag.DEFINE_point("feature_screen_size", "0",
                        "Resolution for screen feature layers.")
point_flag.DEFINE_point("feature_minimap_size", "0",
                        "Resolution for minimap feature layers.")
flags.DEFINE_integer("feature_camera_width", 0,
                     "Width of the feature layer camera.")
point_flag.DEFINE_point("rgb_screen_size", "0",
                        "Resolution for RGB screen layers.")
point_flag.DEFINE_point("rgb_minimap_size", "0",
                        "Resolution for RGB minimap layers.")
flags.DEFINE_bool("full_screen", False, "Whether to run full screen.")
point_flag.DEFINE_point("window_size", "640,480",
                        "Screen size if not full screen.")

logger = logging.getLogger('replay_level_extraction')
logger.setLevel(logging.DEBUG)
logger.propagate = False


# Raw SC2 unit type IDs for each race's starting town hall
_TOWN_HALL_IDS = np.array([18, 59, 86], dtype=np.uint32)  # CommandCenter, Nexus, Hatchery


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def extract_raw_map_data_from_buffer(image_data_object):
    """Unpack a protobuf ImageData into a 2-D numpy array of shape (H, W).

    The raw buffer is in Fortran order with shape (size.x, size.y),
    which we transpose to (H, W) = (size.y, size.x) for storage.
    """
    extracted_data = np.frombuffer(
        image_data_object.data,
        dtype=features.Feature.dtypes[image_data_object.bits_per_pixel],
    )
    expected = image_data_object.size.x * image_data_object.size.y
    if extracted_data.shape[0] != expected:
        extracted_data = extracted_data[:expected]
    raw = extracted_data.reshape(
        image_data_object.size.x, image_data_object.size.y, order='F')
    return raw.T  # (H, W)


def save_static_map_if_needed(game_info, map_data_dir):
    """Write pathing_grid, placement_grid, and height_map for a map.

    Uses atomic write (temp file + os.rename) so parallel workers are safe.
    """
    map_name = game_info.map_name.replace(' ', '_')
    map_dir = Path(map_data_dir)
    map_dir.mkdir(parents=True, exist_ok=True)
    dest = map_dir / f"{map_name}.h5"
    if dest.exists():
        return str(dest)

    start_raw = game_info.start_raw
    pathing = extract_raw_map_data_from_buffer(start_raw.pathing_grid)
    placement = extract_raw_map_data_from_buffer(start_raw.placement_grid)
    height = extract_raw_map_data_from_buffer(start_raw.terrain_height)

    fd, tmp_path = tempfile.mkstemp(dir=str(map_dir), suffix='.h5.tmp')
    os.close(fd)
    try:
        with h5py.File(tmp_path, 'w') as f:
            f.create_dataset('pathing_grid', data=pathing.astype(bool),
                             compression='gzip')
            f.create_dataset('placement_grid', data=placement.astype(bool),
                             compression='gzip')
            f.create_dataset('height_map', data=height.astype(np.uint8),
                             compression='gzip')
        os.rename(tmp_path, str(dest))
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return str(dest)


def initiate_start_replay(replay_data, observed_player_id=1, disable_fog=False):
    """Create a StartReplayRequest with raw interface options."""
    interface = sc_pb.InterfaceOptions()
    interface.raw = True
    interface.raw_affects_selection = False
    interface.raw_crop_to_playable_area = False
    interface.score = True
    interface.show_cloaked = True
    interface.show_burrowed_shadows = True
    interface.show_placeholders = True

    start_replay = sc_pb.RequestStartReplay(
        replay_data=replay_data,
        options=interface,
        disable_fog=disable_fog,
        observed_player_id=observed_player_id,
    )
    return start_replay


def kill_controller(controller, timeout=5):
    """Force-close a pysc2 controller connection."""
    client = controller._client
    if client._sock:
        client._sock.close()
        time.sleep(1)
    Status = enum.Enum("Status", sc_pb.Status.items())
    client._status = Status.quit


# ---------------------------------------------------------------------------
# ReplayLevelExtractor
# ---------------------------------------------------------------------------

class ReplayLevelExtractor:
    """Manages per-replay state across three SC2 engine passes.

    Uses pre-allocated numpy arrays to avoid the ~30 GB Python dict overhead
    that accumulating per-frame dicts would cause on long replays. Arrays are
    allocated at construction with estimated T and N, grown if needed, and
    trimmed to exact size after Pass 1 completes.
    """

    # Names and dtypes/shapes of all (T, N, ...) unit arrays.
    # (attr_name, dtype, extra_dims)  — extra_dims=() means shape (T, N).
    _UNIT_ARRAYS = [
        ('is_alive',          bool,        ()),
        ('is_flying',         bool,        ()),
        ('is_burrowed',       bool,        ()),
        ('is_carried',        bool,        ()),
        ('unit_type',         np.uint32,   ()),
        ('ability_id',        np.uint32,   ()),
        ('coordinate',        np.float16,  (3,)),
        ('target_pos',        np.float16,  (2,)),
        ('radius',            np.float16,  ()),
        ('heading',           np.float16,  ()),
        ('health',            np.float16,  ()),
        ('health_max',        np.float16,  ()),
        ('shield',            np.float16,  ()),
        ('energy',            np.float16,  ()),
        ('build_progress',    np.float16,  ()),
        ('mineral_contents',  np.uint16,   ()),
        ('vespene_contents',  np.uint16,   ()),
        ('visible_status',    np.uint8,    ()),
    ]

    _T_BUFFER = 100   # safety margin added to T estimate
    _N_INIT = 512     # initial unit-slot capacity (doubled when exceeded)
    _TARGET_NONE = np.uint32(0xFFFFFFFF)  # sentinel for "no target"

    def __init__(self, game_info, replay_info, replay_name,
                 map_frame_skip=16, game_duration_loops=0):
        self.game_info = game_info
        self.replay_info = replay_info
        self.replay_name = replay_name
        self.map_name = game_info.map_name.replace(' ', '_')
        self.map_frame_skip = map_frame_skip

        # Full map dimensions
        self.H = game_info.start_raw.map_size.y
        self.W = game_info.start_raw.map_size.x

        # Playable area bounds
        pa = game_info.start_raw.playable_area
        self.playable_area = [pa.p0.x, pa.p0.y, pa.p1.x, pa.p1.y]

        # Persistent tag → row mapping
        self._tag_to_row = {}
        self._next_row = 0

        # Frame counter (incremented each process_omniscient_frame call)
        self._t = 0

        # --- Pre-allocate arrays with estimated dimensions ---
        self._T_alloc = game_duration_loops + self._T_BUFFER
        self._N_alloc = self._N_INIT
        self._allocate_unit_arrays(self._T_alloc, self._N_alloc)

        # target_id uses sentinel fill instead of zeros
        self.target_id = np.full(
            (self._T_alloc, self._N_alloc), self._TARGET_NONE, dtype=np.uint32)

        # Global per-unit arrays (N,) — grown alongside unit arrays
        self.unit_tag = np.zeros(self._N_alloc, dtype=np.uint64)
        self.unit_owner = np.zeros(self._N_alloc, dtype=np.uint8)
        self._owner_set = np.zeros(self._N_alloc, dtype=bool)

        # Deferred target_tag resolution: list of (t, row, target_tag)
        self._pending_targets = []

        # Player data (T, 2) — column 0 = P1, column 1 = P2 (filled later)
        self.player_minerals = np.zeros((self._T_alloc, 2), dtype=np.uint16)
        self.player_vespene = np.zeros((self._T_alloc, 2), dtype=np.uint16)
        self.player_food_used = np.zeros((self._T_alloc, 2), dtype=np.uint8)
        self.player_food_cap = np.zeros((self._T_alloc, 2), dtype=np.uint8)

        # Dynamic map arrays — use ceil(T_est / map_frame_skip) for T_map
        self._T_map_alloc = int(math.ceil(self._T_alloc / self.map_frame_skip))
        self.creep = np.zeros((self._T_map_alloc, self.H, self.W), dtype=bool)
        self.fow_player_1 = np.zeros((self._T_map_alloc, self.H, self.W), dtype=bool)
        self.fow_player_2 = np.zeros((self._T_map_alloc, self.H, self.W), dtype=bool)

        # FOW pass data (still accumulated as lists — lightweight)
        self._fow_states = {1: [], 2: []}
        self._fow_maps = {1: [], 2: []}
        self._player2_data = []

        # Set after finalize
        self.T = 0
        self.N = 0

    # ------------------------------------------------------------------
    # Array management helpers
    # ------------------------------------------------------------------

    def _allocate_unit_arrays(self, T, N):
        """Allocate all (T, N, ...) unit arrays."""
        for name, dtype, extra in self._UNIT_ARRAYS:
            shape = (T, N) + extra
            setattr(self, name, np.zeros(shape, dtype=dtype))

    def _grow_T(self):
        """Double the T dimension of all time-indexed arrays."""
        new_T = self._T_alloc * 2
        logger.debug(f'Growing T: {self._T_alloc} -> {new_T}')

        for name, dtype, extra in self._UNIT_ARRAYS:
            old = getattr(self, name)
            new_shape = (new_T, self._N_alloc) + extra
            new_arr = np.zeros(new_shape, dtype=dtype)
            new_arr[:self._T_alloc] = old
            setattr(self, name, new_arr)

        # target_id: fill new slots with sentinel
        old = self.target_id
        new_arr = np.full((new_T, self._N_alloc), self._TARGET_NONE, dtype=np.uint32)
        new_arr[:self._T_alloc] = old
        self.target_id = new_arr

        # Player arrays
        for attr in ('player_minerals', 'player_vespene',
                     'player_food_used', 'player_food_cap'):
            old = getattr(self, attr)
            new_arr = np.zeros((new_T, 2), dtype=old.dtype)
            new_arr[:self._T_alloc] = old
            setattr(self, attr, new_arr)

        # Dynamic map arrays
        new_T_map = int(math.ceil(new_T / self.map_frame_skip))
        for attr in ('creep', 'fow_player_1', 'fow_player_2'):
            old = getattr(self, attr)
            new_arr = np.zeros((new_T_map, self.H, self.W), dtype=old.dtype)
            new_arr[:self._T_map_alloc] = old
            setattr(self, attr, new_arr)
        self._T_map_alloc = new_T_map

        self._T_alloc = new_T

    def _grow_N(self):
        """Double the N dimension of all unit-indexed arrays."""
        new_N = self._N_alloc * 2
        logger.debug(f'Growing N: {self._N_alloc} -> {new_N}')

        for name, dtype, extra in self._UNIT_ARRAYS:
            old = getattr(self, name)
            new_shape = (self._T_alloc, new_N) + extra
            new_arr = np.zeros(new_shape, dtype=dtype)
            new_arr[:, :self._N_alloc] = old
            setattr(self, name, new_arr)

        # target_id: fill new slots with sentinel
        old = self.target_id
        new_arr = np.full((self._T_alloc, new_N), self._TARGET_NONE, dtype=np.uint32)
        new_arr[:, :self._N_alloc] = old
        self.target_id = new_arr

        # Global per-unit arrays (N,)
        for attr in ('unit_tag', 'unit_owner', '_owner_set'):
            old = getattr(self, attr)
            new_arr = np.zeros(new_N, dtype=old.dtype)
            new_arr[:self._N_alloc] = old
            setattr(self, attr, new_arr)

        self._N_alloc = new_N

    def _ensure_capacity(self, row):
        """Grow arrays if the row index would exceed current N capacity."""
        while row >= self._N_alloc:
            self._grow_N()

    def _ensure_frame_capacity(self):
        """Grow arrays if the current frame index would exceed T capacity."""
        while self._t >= self._T_alloc:
            self._grow_T()

    def get_or_assign_row(self, tag):
        """Assign a persistent row index per unique tag."""
        row = self._tag_to_row.get(tag)
        if row is None:
            row = self._next_row
            self._tag_to_row[tag] = row
            self._next_row += 1
            self._ensure_capacity(row)
            self.unit_tag[row] = tag
        return row

    # ------------------------------------------------------------------
    # Pass 1: Omniscient — write directly into numpy arrays
    # ------------------------------------------------------------------

    def process_omniscient_frame(self, obs, step):
        """Extract all unit data from an omniscient (fog-disabled) frame."""
        t = self._t
        self._ensure_frame_capacity()

        raw = obs.observation.raw_data

        for unit in raw.units:
            row = self.get_or_assign_row(unit.tag)

            self.is_alive[t, row] = True
            self.is_flying[t, row] = unit.is_flying
            self.is_burrowed[t, row] = unit.is_burrowed
            # is_carried stays False (default)
            self.unit_type[t, row] = unit.unit_type
            self.coordinate[t, row] = [unit.pos.x, unit.pos.y, unit.pos.z]
            self.radius[t, row] = unit.radius
            self.heading[t, row] = unit.facing
            self.health[t, row] = unit.health
            self.health_max[t, row] = unit.health_max
            self.shield[t, row] = unit.shield
            self.energy[t, row] = unit.energy
            self.build_progress[t, row] = unit.build_progress
            self.mineral_contents[t, row] = unit.mineral_contents
            self.vespene_contents[t, row] = unit.vespene_contents

            # Global per-unit arrays (first appearance wins)
            if not self._owner_set[row]:
                self.unit_owner[row] = unit.owner
                self._owner_set[row] = True

        # Player 1 common data (omniscient pass uses observed_player_id=1)
        pc = obs.observation.player_common
        self.player_minerals[t, 0] = pc.minerals
        self.player_vespene[t, 0] = pc.vespene
        self.player_food_used[t, 0] = pc.food_used
        self.player_food_cap[t, 0] = pc.food_cap

        # Creep map every map_frame_skip frames
        if t % self.map_frame_skip == 0:
            map_t = t // self.map_frame_skip
            if map_t < self._T_map_alloc:
                creep = extract_raw_map_data_from_buffer(raw.map_state.creep)
                self.creep[map_t] = creep.astype(bool)

        self._t += 1

    # ------------------------------------------------------------------
    # Finalize omniscient pass — trim arrays and resolve deferred targets
    # ------------------------------------------------------------------

    def finalize_omniscient_pass(self):
        """Trim pre-allocated arrays to actual (T, N) and resolve targets."""
        self.T = self._t
        self.N = self._next_row

        if self.T == 0 or self.N == 0:
            return

        # Resolve deferred target_tag -> target row
        for t, row, target_tag in self._pending_targets:
            target_row = self._tag_to_row.get(target_tag, self._TARGET_NONE)
            self.target_id[t, row] = target_row
        del self._pending_targets

        # Re-initialize for FOW passes (orders are extracted there)
        self._pending_targets = []

        # Trim unit arrays to exact (T, N)
        for name, dtype, extra in self._UNIT_ARRAYS:
            old = getattr(self, name)
            trimmed = old[:self.T, :self.N].copy()
            setattr(self, name, trimmed)

        # Trim target_id separately
        self.target_id = self.target_id[:self.T, :self.N].copy()

        # Trim global arrays to exact N
        self.unit_tag = self.unit_tag[:self.N].copy()
        self.unit_owner = self.unit_owner[:self.N].copy()
        del self._owner_set

        # Trim player data to exact T
        self.player_minerals = self.player_minerals[:self.T].copy()
        self.player_vespene = self.player_vespene[:self.T].copy()
        self.player_food_used = self.player_food_used[:self.T].copy()
        self.player_food_cap = self.player_food_cap[:self.T].copy()

        # Trim dynamic map arrays
        self.T_map = int(math.ceil(self.T / self.map_frame_skip))
        self.creep = self.creep[:self.T_map].copy()
        self.fow_player_1 = self.fow_player_1[:self.T_map].copy()
        self.fow_player_2 = self.fow_player_2[:self.T_map].copy()

    # ------------------------------------------------------------------
    # Pass 2/3: FOW
    # ------------------------------------------------------------------

    def process_fow_frame(self, obs, step, player_id):
        """Extract per-unit visibility and FOW map from a fog-of-war frame."""
        raw = obs.observation.raw_data
        tag_states = {}
        for unit in raw.units:
            # display_type: 1=Visible, 2=Snapshot, 3=Hidden, 4=Placeholder
            dt = unit.display_type
            if dt == 1:
                tag_states[unit.tag] = 2  # Visible
            elif dt == 2:
                tag_states[unit.tag] = 1  # Snapshot
            else:
                tag_states[unit.tag] = 0  # treat as unseen

            # Extract orders (only available in FOW passes)
            if unit.tag in self._tag_to_row:
                row = self._tag_to_row[unit.tag]
                t = step - 1  # frame index matching omniscient pass convention
                if len(unit.orders) > 0:
                    order = unit.orders[0]
                    self.ability_id[t, row] = order.ability_id
                    if order.HasField('target_unit_tag'):
                        self._pending_targets.append((t, row, order.target_unit_tag))
                    if order.HasField('target_world_space_pos'):
                        self.target_pos[t, row] = [
                            order.target_world_space_pos.x,
                            order.target_world_space_pos.y]

                # Extract passengers (only available in FOW passes)
                for passenger in unit.passengers:
                    if passenger.tag in self._tag_to_row:
                        p_row = self._tag_to_row[passenger.tag]
                        self.is_alive[t, p_row] = True
                        self.is_carried[t, p_row] = True
                        self.unit_type[t, p_row] = passenger.unit_type
                        self.coordinate[t, p_row] = [unit.pos.x, unit.pos.y, unit.pos.z]
                        self.heading[t, p_row] = unit.facing
                        self.health[t, p_row] = passenger.health
                        self.health_max[t, p_row] = passenger.health_max
                        self.shield[t, p_row] = passenger.shield
                        self.energy[t, p_row] = passenger.energy
                        self.build_progress[t, p_row] = 1.0

        self._fow_states[player_id].append(tag_states)

        # FOW map every map_frame_skip frames
        frame_idx = len(self._fow_states[player_id]) - 1
        if frame_idx % self.map_frame_skip == 0:
            vis_map = extract_raw_map_data_from_buffer(raw.map_state.visibility)
            fow_bool = (vis_map == 2)  # 2 = currently visible
            self._fow_maps[player_id].append((frame_idx, fow_bool))

        # Player 2 common data from Pass 3
        if player_id == 2:
            pc = obs.observation.player_common
            self._player2_data.append(
                (pc.minerals, pc.vespene, pc.food_used, pc.food_cap))

    # ------------------------------------------------------------------
    # Post-FOW processing
    # ------------------------------------------------------------------

    def finalize_fow_passes(self):
        """Fill FOW map arrays and player 2 data after both FOW passes."""
        # Fill FOW maps
        for fi, (frame_idx, arr) in enumerate(self._fow_maps[1]):
            map_t = frame_idx // self.map_frame_skip
            if map_t < self.T_map:
                self.fow_player_1[map_t] = arr
        del self._fow_maps[1]

        for fi, (frame_idx, arr) in enumerate(self._fow_maps[2]):
            map_t = frame_idx // self.map_frame_skip
            if map_t < self.T_map:
                self.fow_player_2[map_t] = arr
        del self._fow_maps[2]

        # Fill player 2 data
        for t, (minerals, vespene, food_used, food_cap) in enumerate(self._player2_data):
            if t < self.T:
                self.player_minerals[t, 1] = minerals
                self.player_vespene[t, 1] = vespene
                self.player_food_used[t, 1] = food_used
                self.player_food_cap[t, 1] = food_cap
        del self._player2_data

        # Resolve deferred target_tag -> target row from FOW order extraction
        for t, row, target_tag in self._pending_targets:
            target_row = self._tag_to_row.get(target_tag, self._TARGET_NONE)
            self.target_id[t, row] = target_row
        del self._pending_targets

    def compute_visible_status(self):
        """Compute visible_status = p1_state * 3 + p2_state for each unit/frame."""
        for t in range(self.T):
            p1_states = self._fow_states[1][t] if t < len(self._fow_states[1]) else {}
            p2_states = self._fow_states[2][t] if t < len(self._fow_states[2]) else {}
            for tag, row in self._tag_to_row.items():
                p1 = p1_states.get(tag, 0)
                p2 = p2_states.get(tag, 0)
                self.visible_status[t, row] = p1 * 3 + p2
        del self._fow_states

    # ------------------------------------------------------------------
    # HDF5 output
    # ------------------------------------------------------------------

    def save_to_hdf5(self, filepath):
        """Write all extracted data to an HDF5 file (atomic write)."""
        out_dir = os.path.dirname(filepath)
        fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix='.h5.tmp')
        os.close(fd)
        try:
            self._write_hdf5(tmp_path)
            os.rename(tmp_path, filepath)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def _write_hdf5(self, filepath):
        T, N = self.T, self.N
        H, W = self.H, self.W

        with h5py.File(filepath, 'w') as f:
            # -- Metadata as attributes --
            f.attrs['replay_name'] = self.replay_name
            f.attrs['map_name'] = self.map_name
            f.attrs['max_unique_units'] = N
            f.attrs['map_size'] = [H, W]
            f.attrs['total_frames'] = T
            f.attrs['playable_area'] = self.playable_area  # [x0, y0, x1, y1]

            # Serialize player_info and replay_info
            try:
                pi = []
                for p in self.replay_info.player_info:
                    pi.append({
                        'player_id': p.player_info.player_id,
                        'race_requested': p.player_info.race_requested,
                        'race_actual': p.player_info.race_actual,
                        'player_result': p.player_result.result,
                        'player_mmr': p.player_mmr,
                        'player_apm': p.player_apm,
                    })
                f.attrs['player_info'] = json.dumps(pi)
            except Exception:
                f.attrs['player_info'] = '[]'

            try:
                ri = {
                    'game_duration_loops': self.replay_info.game_duration_loops,
                    'game_duration_seconds': self.replay_info.game_duration_seconds,
                    'game_version': self.replay_info.game_version,
                    'data_build': self.replay_info.data_build,
                    'base_build': self.replay_info.base_build,
                }
                f.attrs['replay_info'] = json.dumps(ri)
            except Exception:
                f.attrs['replay_info'] = '{}'

            # -- Player start locations (town hall at frame 0) --
            is_town_hall = np.isin(self.unit_type[0, :N], _TOWN_HALL_IDS)
            for player_id in (1, 2):
                mask = (
                    (self.unit_owner[:N] == player_id)
                    & is_town_hall
                    & self.is_alive[0, :N]
                )
                indices = np.where(mask)[0]
                if len(indices) > 0:
                    f.attrs[f'player_{player_id}_start_location'] = (
                        self.coordinate[0, indices[0]].astype(np.float32))
                else:
                    logger.warning(
                        f'{self.replay_name}: no town hall for player {player_id}')
                    f.attrs[f'player_{player_id}_start_location'] = (
                        np.full(3, np.nan, dtype=np.float32))

            # -- Player data (T, 2) --
            pg = f.create_group('player_data')
            p_chunks = (min(T, 256), 2)
            pg.create_dataset('minerals', data=self.player_minerals,
                              chunks=p_chunks, compression='gzip')
            pg.create_dataset('vespene', data=self.player_vespene,
                              chunks=p_chunks, compression='gzip')
            pg.create_dataset('food_used', data=self.player_food_used,
                              chunks=p_chunks, compression='gzip')
            pg.create_dataset('food_cap', data=self.player_food_cap,
                              chunks=p_chunks, compression='gzip')

            # -- Dynamic map data (T_map, H, W) --
            mg = f.create_group('map_data/dynamic')
            m_chunks = (1, H, W)
            mg.create_dataset('creep', data=self.creep,
                              chunks=m_chunks, compression='gzip')
            mg.create_dataset('fow_player_1', data=self.fow_player_1,
                              chunks=m_chunks, compression='gzip')
            mg.create_dataset('fow_player_2', data=self.fow_player_2,
                              chunks=m_chunks, compression='gzip')

            # -- Unit data: global (N,) --
            ug = f.create_group('unit_data/global')
            ug.create_dataset('unit_tag', data=self.unit_tag)
            ug.create_dataset('unit_owner', data=self.unit_owner)

            # -- Unit data: repeated (T, N, ...) --
            ur = f.create_group('unit_data/repeated')
            comp = 4

            def chunks_2d():
                return (min(T, 64), min(N, 128))

            def chunks_3d(d):
                return (min(T, 64), min(N, 128), d)

            ur.create_dataset('is_alive', data=self.is_alive,
                              chunks=chunks_2d(), compression='gzip',
                              compression_opts=comp)
            ur.create_dataset('is_flying', data=self.is_flying,
                              chunks=chunks_2d(), compression='gzip',
                              compression_opts=comp)
            ur.create_dataset('is_burrowed', data=self.is_burrowed,
                              chunks=chunks_2d(), compression='gzip',
                              compression_opts=comp)
            ur.create_dataset('is_carried', data=self.is_carried,
                              chunks=chunks_2d(), compression='gzip',
                              compression_opts=comp)
            ur.create_dataset('visible_status', data=self.visible_status,
                              chunks=chunks_2d(), compression='gzip',
                              compression_opts=comp)
            ur.create_dataset('unit_type', data=self.unit_type,
                              chunks=chunks_2d(), compression='gzip',
                              compression_opts=comp)
            ur.create_dataset('ability_id', data=self.ability_id,
                              chunks=chunks_2d(), compression='gzip',
                              compression_opts=comp)
            ur.create_dataset('target_id', data=self.target_id,
                              chunks=chunks_2d(), compression='gzip',
                              compression_opts=comp)
            ur.create_dataset('coordinate', data=self.coordinate,
                              chunks=chunks_3d(3), compression='gzip',
                              compression_opts=comp)
            ur.create_dataset('target_pos', data=self.target_pos,
                              chunks=chunks_3d(2), compression='gzip',
                              compression_opts=comp)
            ur.create_dataset('radius', data=self.radius,
                              chunks=chunks_2d(), compression='gzip',
                              compression_opts=comp)
            ur.create_dataset('heading', data=self.heading,
                              chunks=chunks_2d(), compression='gzip',
                              compression_opts=comp)
            ur.create_dataset('health', data=self.health,
                              chunks=chunks_2d(), compression='gzip',
                              compression_opts=comp)
            ur.create_dataset('health_max', data=self.health_max,
                              chunks=chunks_2d(), compression='gzip',
                              compression_opts=comp)
            ur.create_dataset('shield', data=self.shield,
                              chunks=chunks_2d(), compression='gzip',
                              compression_opts=comp)
            ur.create_dataset('energy', data=self.energy,
                              chunks=chunks_2d(), compression='gzip',
                              compression_opts=comp)
            ur.create_dataset('build_progress', data=self.build_progress,
                              chunks=chunks_2d(), compression='gzip',
                              compression_opts=comp)
            ur.create_dataset('mineral_contents', data=self.mineral_contents,
                              chunks=chunks_2d(), compression='gzip',
                              compression_opts=comp)
            ur.create_dataset('vespene_contents', data=self.vespene_contents,
                              chunks=chunks_2d(), compression='gzip',
                              compression_opts=comp)


# ---------------------------------------------------------------------------
# Replay extraction orchestrator
# ---------------------------------------------------------------------------

def extract_replay(replay_file):
    """Run all three passes on a single replay and save to HDF5."""
    replay_name = Path(replay_file).stem

    # Skip if output already exists
    out_path = Path(FLAGS.data_save_dir) / f'{replay_name}.h5'
    if out_path.exists():
        logger.info(f'Skipping {replay_name} (output already exists)')
        return str(out_path)

    logger.info(f'Starting {replay_file}')

    if not replay_file or not replay_file.lower().endswith("sc2replay"):
        logger.error(f'Invalid replay file: {replay_file}')
        return None

    # Load replay
    run_config = run_configs.get()
    replay_data = run_config.replay_data(replay_file)
    version = replay.get_replay_version(replay_data)
    run_config = run_configs.get(version=version)

    # ---- Pass 1: Omniscient ----
    start_replay_req = initiate_start_replay(
        replay_data, observed_player_id=1, disable_fog=True)

    with run_config.start(want_rgb=False) as controller:
        info = controller.replay_info(replay_data)

        if info.game_duration_loops < 2:
            logger.warning(
                f'Skipping {replay_file}: too short ({info.game_duration_loops} frames)')
            kill_controller(controller)
            return None

        map_path = FLAGS.map_path or info.local_map_path
        if map_path:
            start_replay_req.map_data = run_config.map_data(
                map_path, len(info.player_info))
        controller.start_replay(start_replay_req)

        game_info = controller.game_info()

        # Save static map data
        try:
            save_static_map_if_needed(game_info, FLAGS.map_data_dir)
        except Exception as e:
            logger.warning(f'Could not save static map: {e}')

        extractor = ReplayLevelExtractor(
            game_info=game_info,
            replay_info=info,
            replay_name=replay_name,
            map_frame_skip=FLAGS.map_frame_skip,
            game_duration_loops=info.game_duration_loops,
        )

        try:
            step = 0
            start_time = time.time()
            while True:
                controller.step(FLAGS.step_mul)
                step += FLAGS.step_mul
                obs = controller.observe()
                extractor.process_omniscient_frame(obs, step)

                if step % 1000 == 0:
                    elapsed = time.time() - start_time
                    start_time = time.time()
                    logger.debug(
                        f'Pass 1 (omniscient): step={step}, '
                        f'elapsed={elapsed:.1f}s, '
                        f'units={len(obs.observation.raw_data.units)}')

                if obs.player_result or (FLAGS.max_steps != -1 and step >= FLAGS.max_steps):
                    break
        finally:
            kill_controller(controller)

    extractor.finalize_omniscient_pass()
    logger.info(f'Pass 1 done: T={extractor.T}, N={extractor.N}')

    if extractor.T == 0 or extractor.N == 0:
        logger.warning(f'Skipping {replay_file}: no data extracted')
        return None

    # ---- Pass 2: Player 1 FOW ----
    _run_fow_pass(extractor, run_config, replay_data, replay_file,
                  player_id=1)
    logger.info(f'Pass 2 (P1 FOW) done: {len(extractor._fow_states[1])} frames')

    # ---- Pass 3: Player 2 FOW ----
    _run_fow_pass(extractor, run_config, replay_data, replay_file,
                  player_id=2)
    logger.info(f'Pass 3 (P2 FOW) done: {len(extractor._fow_states[2])} frames')

    # Frame count assertion
    for pid in [1, 2]:
        fow_frames = len(extractor._fow_states[pid])
        if fow_frames != extractor.T:
            logger.warning(
                f'Frame mismatch: omniscient={extractor.T}, '
                f'P{pid} FOW={fow_frames}')

    # Finalize and save
    extractor.finalize_fow_passes()
    extractor.compute_visible_status()

    save_dir = Path(FLAGS.data_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(save_dir / f'{replay_name}.h5')
    extractor.save_to_hdf5(out_path)
    logger.info(f'Saved {out_path}')
    return out_path


def _run_fow_pass(extractor, run_config, replay_data, replay_file, player_id):
    """Run a single fog-of-war pass for a given player."""
    start_replay_req = initiate_start_replay(
        replay_data, observed_player_id=player_id, disable_fog=False)

    with run_config.start(want_rgb=False) as controller:
        info = controller.replay_info(replay_data)
        map_path = FLAGS.map_path or info.local_map_path
        if map_path:
            start_replay_req.map_data = run_config.map_data(
                map_path, len(info.player_info))
        controller.start_replay(start_replay_req)

        try:
            step = 0
            start_time = time.time()
            while True:
                controller.step(FLAGS.step_mul)
                step += FLAGS.step_mul
                obs = controller.observe()
                extractor.process_fow_frame(obs, step, player_id)

                if step % 1000 == 0:
                    elapsed = time.time() - start_time
                    start_time = time.time()
                    logger.debug(
                        f'Pass {"2" if player_id == 1 else "3"} '
                        f'(P{player_id} FOW): step={step}, '
                        f'elapsed={elapsed:.1f}s')

                if obs.player_result or (FLAGS.max_steps != -1 and step >= FLAGS.max_steps):
                    break
        finally:
            kill_controller(controller)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _safe_extract(replay_file):
    """Wrapper for joblib — must be top-level to be picklable."""
    try:
        return extract_replay(replay_file)
    except Exception as e:
        logger.error(f'Failed on {replay_file}: {e}', exc_info=True)
        return None


def main(unused_argv):
    # Logger setup
    log_dir = Path(FLAGS.data_save_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime('%Y_%m_%d_%H%M%S')
    verbose_fh = logging.FileHandler(log_dir / f'verbose-{ts}.log', mode='w+')
    verbose_fh.setLevel(logging.DEBUG)
    file_fh = logging.FileHandler(log_dir / f'extraction-{ts}.log', mode='w+')
    file_fh.setLevel(logging.INFO)
    console_fh = logging.StreamHandler()
    console_fh.setLevel(logging.DEBUG)

    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            datefmt='%m-%d %H:%M')
    verbose_fh.setFormatter(fmt)
    file_fh.setFormatter(fmt)
    console_fh.setFormatter(
        logging.Formatter('%(levelname)s - %(asctime)s - %(message)s',
                          datefmt='%m-%d %H:%M'))
    logger.addHandler(verbose_fh)
    logger.addHandler(file_fh)
    logger.addHandler(console_fh)

    # Validate flags
    if FLAGS.replay_dir and FLAGS.replay:
        raise RuntimeError('Specify --replay or --replay_dir, not both.')

    if FLAGS.replay_dir:
        replay_dir = os.path.abspath(FLAGS.replay_dir)
        files = [os.path.join(replay_dir, f)
                 for f in os.listdir(replay_dir)
                 if f.lower().endswith('sc2replay')]
        replay_file_list = sorted(files)
    elif FLAGS.replay:
        replay_file_list = [os.path.abspath(FLAGS.replay)]
    else:
        raise RuntimeError('Must specify --replay or --replay_dir.')

    if FLAGS.max_replays >= 0:
        replay_file_list = replay_file_list[:FLAGS.max_replays]

    # Slice to [start_idx, end_idx) for multi-device splitting
    start_idx = FLAGS.start_idx
    end_idx = FLAGS.end_idx if FLAGS.end_idx >= 0 else len(replay_file_list)
    replay_file_list = replay_file_list[start_idx:end_idx]

    logger.info(f'Processing {len(replay_file_list)} replays '
                f'(indices {start_idx}:{end_idx})')

    root_save_dir = Path(FLAGS.data_save_dir)
    root_save_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    run_config_dict = {
        'replay_files': replay_file_list,
        'flags': {
            'data_save_dir': FLAGS.data_save_dir,
            'map_data_dir': FLAGS.map_data_dir,
            'step_mul': FLAGS.step_mul,
            'max_steps': FLAGS.max_steps,
            'map_frame_skip': FLAGS.map_frame_skip,
            'num_parallel_jobs': FLAGS.num_parallel_jobs,
        },
    }
    with open(root_save_dir / 'run_config.json', 'w') as fh:
        json.dump(run_config_dict, fh, indent=2, sort_keys=True)

    if FLAGS.num_parallel_jobs == 1:
        for rf in replay_file_list:
            _safe_extract(rf)
    else:
        joblib.Parallel(n_jobs=FLAGS.num_parallel_jobs, backend='multiprocessing')(
            joblib.delayed(_safe_extract)(rf) for rf in replay_file_list
        )

    logger.info('Extraction complete.')


if __name__ == '__main__':
    app.run(main)
