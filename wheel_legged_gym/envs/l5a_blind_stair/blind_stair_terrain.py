import numpy as np
from isaacgym import terrain_utils


class BlindStairTerrain:
    """Square-pit stair terrain for blind stair-climbing.

    Each sub-terrain is an 8m x 8m square. The robot starts near the center
    low platform and can meet stairs in any horizontal direction after yaw
    randomization. Heights are generated as concentric square stair rings:
    0, 1*h, 2*h, ... num_steps*h.
    """

    def __init__(self, cfg, num_robots) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size / cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        for k in range(self.cfg.num_sub_terrains):
            row, col = np.unravel_index(k, (cfg.num_rows, cfg.num_cols))
            terrain = self._make_pit_stairs_subterrain(row, col)
            self._add_terrain_to_map(terrain, row, col)

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw,
                cfg.horizontal_scale,
                cfg.vertical_scale,
                cfg.slope_treshold,
            )

    def _make_pit_stairs_subterrain(self, row: int, col: int):
        terrain = terrain_utils.SubTerrain(
            "blind_pit_stairs",
            width=self.width_per_env_pixels,
            length=self.length_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )

        h_scale = self.cfg.horizontal_scale
        v_scale = self.cfg.vertical_scale
        step_height = getattr(self.cfg, "blind_stair_step_height", 0.08)
        step_width = getattr(self.cfg, "blind_stair_step_width", 0.25)
        num_steps = getattr(self.cfg, "blind_stair_num_steps", 10)
        pit_half_size = getattr(self.cfg, "blind_stair_pit_half_size", 0.65)

        length = terrain.length
        width = terrain.width
        cx = length // 2
        cy = width // 2
        xs = (np.arange(length) - cx) * h_scale
        ys = (np.arange(width) - cy) * h_scale
        xx, yy = np.meshgrid(xs, ys, indexing="ij")

        # Chebyshev distance forms square rings, so all four directions contain stairs.
        ring_distance = np.maximum(np.abs(xx), np.abs(yy)) - pit_half_size
        ring_idx = np.floor(np.maximum(ring_distance, 0.0) / step_width).astype(np.int32)
        ring_idx = np.clip(ring_idx, 0, num_steps)

        # All sub-terrains use the same 8cm x 25cm x 10-step geometry.
        height_m = ring_idx.astype(np.float32) * step_height
        terrain.height_field_raw[:, :] = np.round(height_m / v_scale).astype(np.int16)

        return terrain

    def _add_terrain_to_map(self, terrain, row: int, col: int) -> None:
        start_x = self.border + row * self.length_per_env_pixels
        end_x = self.border + (row + 1) * self.length_per_env_pixels
        start_y = self.border + col * self.width_per_env_pixels
        end_y = self.border + (col + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (row + 0.5) * self.env_length
        env_origin_y = (col + 0.5) * self.env_width
        env_origin_z = 0.0
        self.env_origins[row, col] = [env_origin_x, env_origin_y, env_origin_z]
