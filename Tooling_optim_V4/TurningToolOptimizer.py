##########################
### IMPORTS ###
##########################

from typing import Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
from Functions.Plot import plot_wireframe, plot_surface

##########################
### CLASSES ###
##########################

class Workpiece:
    def __init__(self, zs: list[float], xs: list[float]):
        if zs[0] < zs[-1]:
            zs = zs[::-1]
            xs = xs[::-1]
        self.zs_1d = zs
        self.xs_1d = xs
        self.Ps_1d = np.vstack((zs, xs)).T

        if self.zs_1d[0] > self.zs_1d[-1]:
            self.x_from_z = lambda z: np.interp(z, self.zs_1d[::-1], self.xs_1d[::-1])
        else:
            self.x_from_z = lambda z: np.interp(z, self.zs_1d, self.xs_1d)

    def plot_3d(self,fig=None,n_thetas=60):
        if fig is None:
            fig = go.Figure()

        thetas_1d = np.linspace(0, 2 * np.pi, n_thetas)
        zs_2d, thetas_2d = np.meshgrid(self.zs_1d, thetas_1d)
        rs_2d = np.tile(self.xs_1d, (n_thetas, 1))
        xs_2d = rs_2d * np.cos(thetas_2d)
        ys_2d = rs_2d * np.sin(thetas_2d)

        plot_wireframe(xs_2d, ys_2d, zs_2d, fig=fig, linecolor="rgba(255,255,255,0.2)")

        return fig

class Toolpath:
    def __init__(self, zs: list[float], xs: list[float], z_min = None, z_max = None, dz_forced = None):
        if z_min:
            xs = xs[zs>=z_min]
            zs = zs[zs>=z_min]
        if z_max:
            xs = xs[zs <= z_max]
            zs = zs[zs <= z_max]
        if dz_forced: # resample to uniform dz
            z_new = np.linspace(np.max(zs), np.min(zs), np.abs(int((np.max(zs)-np.min(zs))/dz_forced))+1)
            if zs[0] > zs[-1]:
                x_new = np.interp(z_new, zs[::-1], xs[::-1])
            else:
                x_new = np.interp(z_new, zs, xs)
            zs = z_new
            xs = x_new
        self.zs_1d = zs
        self.xs_1d = xs
        self.Ps_1d = np.vstack((zs, xs)).T

    def plot_3d(self, fig=None):
        if fig is None:
            fig = go.Figure()

        fig.add_trace(go.Scatter3d(x=self.xs_1d, y=np.zeros_like(self.xs_1d), z=self.zs_1d,
                                   mode="lines+markers",
                                   line=dict(color="rgba(0,120,0,0.5)", width=2),
                                   marker=dict(size=2, color="rgba(0,120,0,1)"),
                                   name="Toolpath"))

class Tool:
    def __init__(self,
                 z_min: float,
                 z_max: float,
                 dz_tool: float,
                 tool_CL_x0: float,
                 tool_CL_z0: float,
                 tool_radius_init: float,
                 n_phi: int):

        if z_max >= z_min:
            self.z_min, self.z_max = z_max, z_min
        else:
            self.z_min, self.z_max = z_min, z_max
        self.dz_tool = dz_tool
        self.tool_CL_x0 = tool_CL_x0
        self.tool_CL_z0 = tool_CL_z0
        self.tool_radius_init = tool_radius_init
        self.n_phi = n_phi

        self.initialize_geometry()

    def initialize_geometry(self):
        self.phis_1d = np.linspace(0, 2 * np.pi, self.n_phi, endpoint=True)
        self.zs_1d = np.arange(self.z_min, self.z_max + self.dz_tool, -self.dz_tool)
        self.phis_2d, self.zs_2d = np.meshgrid(self.phis_1d, self.zs_1d)
        self.rs_2d = np.ones_like(self.zs_2d) * self.tool_radius_init

    def export_as_stl(self, path=None, solid_name="tool"):
        """
        Export the tool surface as a very simple ASCII STL.
        Only the lateral surface is exported (no end caps).
        """
        if path is None:
            path = "tool.stl"

        # Coordinates in tool frame (no CL offset here, keep it simple)
        xs = self.rs_2d * np.cos(self.phis_2d)
        ys = self.rs_2d * np.sin(self.phis_2d)
        zs = self.zs_2d

        # Remove last phi column (2π) to avoid duplicate seam
        xs = xs[:, :-1]
        ys = ys[:, :-1]
        zs = zs[:, :-1]

        n_z, n_phi = xs.shape

        def write_facet(f, v0, v1, v2):
            # Compute normal from triangle vertices
            v0 = np.array(v0)
            v1 = np.array(v1)
            v2 = np.array(v2)
            n = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(n)
            if norm > 0:
                n = n / norm
            else:
                n = np.array([0.0, 0.0, 0.0])

            f.write(f"  facet normal {n[0]} {n[1]} {n[2]}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]} {v0[1]} {v0[2]}\n")
            f.write(f"      vertex {v1[0]} {v1[1]} {v1[2]}\n")
            f.write(f"      vertex {v2[0]} {v2[1]} {v2[2]}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")

        with open(path, "w") as f:
            f.write(f"solid {solid_name}\n")

            # Build quads between z_i and z_{i+1}, then split into 2 triangles
            for i in range(n_z - 1):
                for j in range(n_phi):
                    j_next = (j + 1) % n_phi

                    v00 = (xs[i, j],     ys[i, j],     zs[i, j])
                    v01 = (xs[i, j_next], ys[i, j_next], zs[i, j_next])
                    v10 = (xs[i+1, j],   ys[i+1, j],   zs[i+1, j])
                    v11 = (xs[i+1, j_next], ys[i+1, j_next], zs[i+1, j_next])

                    # Triangle 1
                    write_facet(f, v00, v10, v11)
                    # Triangle 2
                    write_facet(f, v00, v11, v01)

            f.write(f"endsolid {solid_name}\n")

    def plot_3d(self,fig=None,n_thetas=120,CL_xz = (0,0)):
        if fig is None:
            fig = go.Figure()

        xs_2d = self.rs_2d * np.cos(self.phis_2d) + (CL_xz[0]-self.tool_CL_x0)
        ys_2d = self.rs_2d * np.sin(self.phis_2d)
        zs_2d = self.zs_2d + (CL_xz[1]-self.tool_CL_z0)

        plot_surface(xs_2d, ys_2d, zs_2d, fig=fig)
        plot_wireframe(xs_2d, ys_2d, zs_2d, fig=fig, linecolor="rgba(0,0,0,1)")

        # plot tool tip
        fig.add_trace(go.Scatter3d(x=[CL_xz[0]], y=[0], z=[CL_xz[1]],mode="markers",marker=dict(color="red",size=3),name="Tool Tip"))

        # plot tool axis
        fig.add_trace(go.Scatter3d(x=[CL_xz[0]-self.tool_CL_x0, CL_xz[0]-self.tool_CL_x0],
                                   y=[0,0],
                                   z=[self.z_min + (CL_xz[1]-self.tool_CL_z0), self.z_max + (CL_xz[1]-self.tool_CL_z0)],
                                   mode="lines", line=dict(color="yellow", width=3), name="Tool Axis"))


        return fig

##########################
### MATHS FUNCTIONS ###
##########################

def compute_single_distance(tool_axis_x: float,
                                phi: float,
                                R: float) -> Optional[float]:

    method = 1

    if method == 1:
        if R > abs(tool_axis_x * np.cos(phi)):
            d = -tool_axis_x * np.cos(phi) + (R ** 2 - tool_axis_x ** 2 * np.sin(phi) ** 2) ** 0.5
            if d < 0:
                print("Warning: negative distance computed!")
                return None
            return d
        else:
            return None
    elif method==2:

        x0 = tool_axis_x
        y0 = 0.0

        ux = np.cos(phi)
        uy = np.sin(phi)

        # 1) Distance from origin to infinite line through (x0, 0) along (ux, uy)
        #    d_min = |P0 x u| in 2D
        d_min = abs(x0 * uy)  # since y0 = 0

        # If the line misses the circle, the ray also misses.
        if d_min > R:
            return None

        # 2) Parameter of closest approach along the ray
        p0_dot_u = x0 * ux + y0 * uy  # = x0 * cos(phi)
        t_star = -p0_dot_u

        # Squared distance at closest approach
        d_min_sq = x0 * x0 - p0_dot_u * p0_dot_u

        # Safety check (should be consistent with d_min > R check)
        if d_min_sq > R * R:
            return None

        # 3) Offset from closest approach to intersection(s)
        offset = np.sqrt(R * R - d_min_sq)

        # Two intersection parameters along the ray
        t1 = t_star - offset
        t2 = t_star + offset

        # We only care about intersections in front of the starting point (t >= 0)
        candidates = [t for t in (t1, t2) if t >= 0]

        if not candidates:
            return None

        d = min(candidates)

        # Additional sanity check
        if d < 0:
            print("Warning: negative distance computed!")
            return None

        return d



def compute_distances_and_update_tool(tool: Tool, CL_pos, wp: Workpiece):
    CL_z, CL_x = CL_pos

    tool_zs_1d = tool.zs_1d + (CL_z - tool.tool_CL_z0)
    for tool_z_idx, tool_zi in enumerate(tool_zs_1d):
        # optional: skip if outside wp range
        if tool_zi < np.amin(wp.zs_1d) or tool_zi > np.amax(wp.zs_1d):
            continue

        for phi_idx, tool_phii in enumerate(tool.phis_1d):

            r_allowed = compute_single_distance(tool_axis_x=CL_x-tool.tool_CL_x0,phi=tool_phii,R = wp.x_from_z(tool_zi))
            if r_allowed:
                if r_allowed < tool.rs_2d[tool_z_idx, phi_idx]:
                    tool.rs_2d[tool_z_idx, phi_idx] = r_allowed
