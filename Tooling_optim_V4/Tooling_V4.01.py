import pandas as pd
import numpy as np
import struct

class Workpiece:
    def __init__(self, zs: list[float], xs: list[float]):
        if zs[0] < zs[-1]:
            zs = zs[::-1]
            xs = xs[::-1]
        self.zs_1d = zs
        self.xs_1d = xs
        self.Ps_2d = np.vstack((zs, xs)).T
        self.x_from_z = lambda z: np.interp(z, self.zs_1d, self.xs_1d)

class Toolpath:
    def __init__(self, zs: list[float], xs: list[float], z_min = None, z_max = None):
        if z_min:
            zs = [z for z in zs if z >= z_min]
            xs = xs[-len(zs):]
        if z_max:
            zs = [z for z in zs if z <= z_max]
            xs = xs[:len(zs)]
        self.zs_1d = zs
        self.xs_1d = xs
        self.Ps_2d = np.vstack((zs, xs)).T

class Tool:
    def __init__(self,
                 z_min: float,
                 z_max: float,
                 dz_tool: float,
                 tool_tip_x0: float,
                 tool_tip_z0: float,
                 tool_radius_init: float,
                 n_phi: int):
        self.z_min = z_min
        self.z_max = z_max
        self.dz_tool = dz_tool
        self.tool_tip_x0 = tool_tip_x0
        self.tool_tip_z0 = tool_tip_z0
        self.tool_radius_init = tool_radius_init
        self.n_phi = n_phi

        self.initialize_geometry()

    def initialize_geometry(self):
        self.phis_1d = np.linspace(0, 2 * np.pi, self.n_phi, endpoint=True)
        self.zs_1d = np.linspace(self.z_min, self.z_max, int((self.z_max - self.z_min) / self.dz_tool) + 1)
        self.phis_2d, self.zs_2d = np.meshgrid(self.phis_1d, self.zs_1d)
        self.rs_2d = np.ones_like(self.zs_2d) * self.tool_radius_init

    def z_at_position(self, tip_pos_z: float, type):
        dz = tip_pos_z - self.tool_tip_z0
        if type == '1d':
            return self.zs_1d + dz
        elif type == '2d':
            return self.zs_2d + dz

def compute_single_distance(z: float,
                            tool_axis_x: float,
                            phi: float,
                            wp: Workpiece) -> float:


def compute_distances(tool: Tool, tp: Toolpath, wp: Workpiece):
    for tip_pos in tp.Ps_2d:   # rows are (z, x)
        tip_z, tip_x = tip_pos

        # tool axis X in workpiece frame
        axis_x = tip_x - tool.tool_tip_x0

        # Z of each tool station in workpiece frame
        tool_zs_1d = tool.z_at_position(tip_z, type='1d')

        for tool_z_idx, tool_zi in enumerate(tool_zs_1d):
            # optional: skip if outside wp range
            if tool_zi < wp.zs_1d[-1] or tool_zi > wp.zs_1d[0]:
                continue

            for phi_idx, tool_phii in enumerate(tool.phis_1d):
                r_allowed = compute_single_distance(
                    z=tool_zi,
                    tool_axis_x=axis_x,
                    phi=tool_phii,
                    wp=wp
                )
                if r_allowed < tool.rs_2d[tool_z_idx, phi_idx]:
                    tool.rs_2d[tool_z_idx, phi_idx] = r_allowed


if __name__ == "__main__":

    wp_df = pd.read_csv(r".\data\AD_HORN\Profile__ST0667014_01_AA.02_converted_20221112061505.csv")
    wp_z = wp_df["Z"].to_numpy()
    wp_x = wp_df["X"].to_numpy()
    wp = Workpiece(wp_z, wp_x)

    tool = Tool(z_min=0,z_max=100,dz_tool=0.5,
                tool_tip_x0= 3,tool_tip_z0 = 0,
                tool_radius_init = 100,
                n_phi = 120)

    tp = Toolpath(zs=wp_z,xs=wp_x,z_min=-189)
    compute_distances(tool, tp, wp)

    compute_distances(tool, tp, wp)

    export_tool_stl(tool, filename="tool_envelope.stl", origin=(0.0, 0.0, 0.0))
