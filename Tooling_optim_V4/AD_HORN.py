from TurningToolOptimizer import Tool, Workpiece, Toolpath, compute_distances_and_update_tool, plot_wireframe
import numpy as np
from tqdm import tqdm
import pandas as pd
import plotly.graph_objects as go


def plot_tool_and_workpiece(tp: Toolpath, tool: Tool, wp: Workpiece, CL_pos,filename, PLOT=False, PLOT_SHOW=False, PLOT_EXPORT=False, PLOT_PNG=False):
    if PLOT:
        fig = go.Figure()
        # fig.add_trace(go.Scatter3d(x=[wp.x_from_z(z_end_simulation)], y=[0], z=[z_end_simulation], mode="markers", marker=dict(color="red", size=5), name="Tool Tip"))
        # fig = tool.plot_3d(fig=fig, CL_xz = (wp.x_from_z(z_end_simulation) , z_end_simulation))
        wp.plot_3d(fig=fig)
        tp.plot_3d(fig=fig)
        tool.plot_3d(fig=fig, CL_xz=(CL_pos[1], CL_pos[0]))

        if "Format the Plot":
            k = 30

            # compute axis ranges
            xr_max = 50
            zr_min = -800
            zr_max = 500

            zr = zr_max - zr_min

            fig.update_scenes(
                xaxis=dict(range=[-xr_max, xr_max]),
                yaxis=dict(range=[-xr_max, xr_max]),
                zaxis=dict(range=[zr_min, zr_max])
            )

            layout = go.Layout(
                template="plotly_dark",
                scene=dict(
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=zr / (2 * xr_max))
                ),
                scene_camera=dict(
                    eye=dict(x=0, y=3, z=-3),
                    up=dict(x=1, y=0, z=0)
                ),
            )
            fig.update_layout(layout)
            axis_clean = dict(
                showgrid=False,  # no grid lines
                zeroline=False,  # no axis zero line
                showbackground=False,  # no axis background planes
                showticklabels=False,  # no tick numbers
                ticks="",  # no tick marks
                title=""  # remove axis title
            )

            fig.update_scenes(
                xaxis=axis_clean,
                yaxis=axis_clean,
                zaxis=axis_clean
            )

        fig.layout.scene.camera.projection.type = "orthographic"

        if PLOT_SHOW:
            fig.show(renderer="browser")
        if PLOT_EXPORT:
            fig.write_html(file=filename+".html", include_plotlyjs="cdn")
        if PLOT_PNG:
            fig.write_image(file=filename+".png",
                            width=1200, height=800, scale=2)

        return fig

def main ():
    ##########################
    ### CONFIG ###
    ##########################

    PLOT_SHOW = False
    PLOT_EXPORT = False
    PLOT_PNG = True
    PLOT_NB_INTERMEDIATE = 0
    EXPORT_STL = False
    EXPORT_plot = False
    PLOT = PLOT_SHOW or PLOT_EXPORT or PLOT_PNG

    dz_simulation_toolpath = 2

    ##########################
    ### OBJECTS ###
    ##########################
    #--> TOOL
    tool = Tool(z_min=0, z_max=200, dz_tool=0.5,tool_CL_x0=3, tool_CL_z0=0,tool_radius_init=40,n_phi=120)

    #--> WORKPIECE (side 1)
    def load_workpiece_AD_HORN_side_1():
        wp_df = pd.read_csv("./data/AD_HORN/Profile__ST0667014_01_AA.02_converted_20221112061505_0.01.csv")
        wp_z = wp_df["Z"].to_numpy()
        wp_x = -wp_df["X"].to_numpy() # minus to have positive values
        # remove points with same Z:
        _, unique_indices = np.unique(wp_z, return_index=True)
        wp_z = wp_z[unique_indices]
        wp_x = wp_x[unique_indices]
        if wp_z[-1] > wp_z[0]:
            wp_z = wp_z[::-1]
            wp_x = wp_x[::-1]
        return Workpiece(wp_z, wp_x)
    wp_side1 = load_workpiece_AD_HORN_side_1()

    #--> TOOLPATH (side 1)
    tp_side1 = Toolpath(zs=wp_side1.zs_1d, xs=wp_side1.xs_1d, z_min=-189, dz_forced=dz_simulation_toolpath)

    #--> WORKPIECE (side 2) will be created by mirroring side 1
    def load_workpiece_AD_HORN_side_2(wp_side1: Workpiece):
        wp_z = - wp_side1.zs_1d # mirror in YZ plane
        wp_x = wp_side1.xs_1d.copy()
        wp_z = wp_z - np.max(wp_z) # shift to zero point
        if wp_z[-1] > wp_z[0]:
            wp_z = wp_z[::-1]
            wp_x = wp_x[::-1]
        return Workpiece(wp_z, wp_x)
    wp_side2 = load_workpiece_AD_HORN_side_2(wp_side1)

    #--> TOOLPATH (side 2)
    tp_side2 = Toolpath(zs=wp_side2.zs_1d, xs=wp_side2.xs_1d, z_min=-161, dz_forced=dz_simulation_toolpath)


    for side_idx, (wp, tp) in enumerate( [(wp_side1, tp_side1), (wp_side2, tp_side2)], start=1 ):
        print(f"--- Processing side {side_idx} ---")

        plot_cur_intermediate = 0
        for z_idx,CL_pos in enumerate(tqdm(tp.Ps_1d)): # iterate through toolpath positions
            compute_distances_and_update_tool(tool, CL_pos, wp) # compute distances and update tool shape
            do_plot = (z_idx/len(tp.Ps_1d) >= plot_cur_intermediate/PLOT_NB_INTERMEDIATE if PLOT_NB_INTERMEDIATE else False) or z_idx == len(tp.Ps_1d)-1
            if do_plot:
                plot_cur_intermediate += 1
                plot_tool_and_workpiece(tp, tool, wp, CL_pos,filename = f"./data/AD_HORN/Toolpath_and_Workpiece_side{side_idx}_frame{plot_cur_intermediate:04d}" , PLOT=PLOT, PLOT_SHOW=PLOT_SHOW, PLOT_EXPORT=PLOT_EXPORT, PLOT_PNG=PLOT_PNG)
        if EXPORT_STL:
            tool.export_as_stl(path=f"./data/AD_HORN/Tool_after_side{side_idx}", solid_name="AD_HORN_tool")

if __name__ == "__main__":
    main()