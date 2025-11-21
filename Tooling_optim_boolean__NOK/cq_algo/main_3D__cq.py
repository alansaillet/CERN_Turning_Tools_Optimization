import numpy as np
import plotly.graph_objects as go
import cadquery as cq
from tqdm import tqdm

def arange2(start,end,pitch):
    return np.arange(start,end,np.abs(pitch)*np.sign(end-start))
class ToolEnvelopeCalculation:
    def __init__(self,partprofile__Rmach, partprofile_xmin, partprofile_xmax):
        self.part__Rmachprofile__Rmach = partprofile__Rmach
        self.part__Rmachprofile_xmin = partprofile_xmin
        self.part__Rmachprofile_xmax = partprofile_xmax

        self.deltaD = 2

        self.tootip_radius = 50

        self.build_part_cad__Rmach()
        self.build_tool_cad__Rtool()

    def build_part_cad__Rmach(self):
        # Generate profile points
        outer_r = 110

        x_values = arange2(0, -144, self.deltaD)
        profile_points = [(x, self.part__Rmachprofile__Rmach(x)) for x in x_values if not np.isnan(profile_cavity_1_3GHz__Rmach(x))]

        profile_points.append((x_values[-1], outer_r))
        profile_points.append((x_values[0], outer_r))
        profile_points.append((x_values[0], profile_points[-3][-1]))

        wire = cq.Workplane("XY").polyline(profile_points).close()
        self.part__Rmach = wire.revolve(360, (0, 0, 0), (1, 0, 0))

    def build_tool_cad__Rtool(self):
        x_start = -20
        x_end = 120

        # Calculate the extrusion length
        extrusion_length = x_end - x_start

        # Create a workplane at Z = x_end
        self.tool = (
            cq.Workplane("YZ")
            .workplane(offset=x_start)
            .circle(self.tootip_radius*4)
            .extrude(extrusion_length)
        )

    def get_part_cad__Rtool(self, d):
        # d: tool position
        # x__Rtool: geometric axis (in the configuration at tool position d)
        translation_vector = (-d, 0, -self.part__Rmachprofile__Rmach(d + 0) + self.tootip_radius)
        part__Rtool = self.part__Rmach.translate(translation_vector)
        return part__Rtool

    def update_tool_at_d(self, d):
        #self.tool = self.get_part_cad__Rtool(d).cut(self.tool)
        self.tool = self.tool.cut(self.get_part_cad__Rtool(d))

    def update_tool_on_drange(self, d_start,d_end):
        for d in tqdm(list(arange2(d_start, d_end, self.deltaD))+ [d_end]) :
            self.update_tool_at_d(d)

    def export_tool(self):
        cq.exporters.export(self.tool, "tool_result.step")

if __name__ == "__main__":
    def profile_cavity_1_3GHz__Rmach(x):
        cavity_length = 144
        cavity_halflength = cavity_length/2
        def profile_cavity_1_3GHz__centered(x):
            if x < -cavity_halflength:
                return np.nan
            elif -cavity_halflength <= x <= -57:
                return 39.05
            elif -57 < x <= -48.807:
                return 51.85 - np.sqrt((1 - (x - (-57))**2/9**2)*(12.8)**2)
            elif -48.807 < x <= -40.006:
                return  46.553 + (x - (-48.807))*np.tan(72.275 * np.pi / 180)
            elif -40.006 < x <= 0:
                return 61.3 + np.sqrt(42**2 - (x - 0)**2)
            else:
                return profile_cavity_1_3GHz__centered(-x)

        return profile_cavity_1_3GHz__centered(x+cavity_halflength)

    tool_envelope_calculation = ToolEnvelopeCalculation(profile_cavity_1_3GHz__Rmach,-144/2,0)
    tool_envelope_calculation.update_tool_on_drange(0,-50)
    tool_envelope_calculation.export_tool()