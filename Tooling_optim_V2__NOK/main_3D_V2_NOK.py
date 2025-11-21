import numpy as np
import plotly.graph_objects as go

class ToolEnvelopeCalculation:
    def __init__(self,partprofile__Rmach, partprofile_xmin, partprofile_xmax):
        self.partprofile__Rmach = partprofile__Rmach
        self.partprofile_xmin = partprofile_xmin
        self.partprofile_xmax = partprofile_xmax

        self.deltaD = 4
        self.deltaθ = np.pi/21

        self.tootip_radius = 50

    def partprofile_Rtool(self,x__Rtool,d):
        # d: tool position
        # x__Rtool: geometric axis (in the configuration at tool position d)
        x__Rmach = d + x__Rtool
        y__Rmach = self.partprofile__Rmach(d + x__Rtool)
        y__Rtool = y__Rmach - self.partprofile__Rmach(d + 0) + self.tootip_radius
        # y__Rmach - offset to put it at y0 at x=0 in Rtool + y__tool at x=0 in Rtool
        return y__Rtool

    def maxtoolradius_at_d_at_xRtool(self,x__Rtool, θ, d):
        # 2D
        Cy = self.partprofile_Rtool(x__Rtool, d) - self.partprofile__Rmach(x__Rtool + d)
        R = self.partprofile__Rmach(x__Rtool + d)

        # 3D
        tan_theta = np.tan(np.pi / 2 - θ)
        a = 1 + tan_theta ** 2
        b = 2 * Cy * tan_theta
        c = Cy ** 2 - R ** 2
        delta = b ** 2 - 4 * a * c

        z_solution_1 = (-b + np.sqrt(delta)) / (2 * a)
        z_solution_2 = (-b - np.sqrt(delta)) / (2 * a)

        y_solution_1 = tan_theta * z_solution_1
        y_solution_2 = tan_theta * z_solution_2

        R_solution_1 = np.sqrt(y_solution_1 ** 2 + z_solution_1 ** 2)
        R_solution_2 = np.sqrt(y_solution_2 ** 2 + z_solution_2 ** 2)

        if np.sign(z_solution_1) == np.sign(θ):
            Rsol = R_solution_1
        else:
            Rsol = R_solution_2

        if np.isnan(Rsol):
            #TODO: fix it
            Rsol = self.partprofile__Rmach(-1)

        return Rsol

    def maxtoolradius_on_dmindmaxrange_at_xRtool_atθ(self, x__Rtool, θ, dmin,dmax):
        Rsols = []

        for d in list(np.arange(dmin, dmax, self.deltaD)) + [dmax]:
            Rsols.append(self.maxtoolradius_at_d_at_xRtool(x__Rtool, θ, d))

        return min(Rsols)

    def maxtoolenvelope_on_dmindmaxrange(self,dmin,dmax):

        x_values = np.array(list(np.arange(-20, -self.partprofile_xmin + 20, self.deltaD)))
        thetas = np.arange(-np.pi, np.pi, self.deltaθ)
        thetas = [*thetas, thetas[0]]

        X, THETA = np.meshgrid(x_values, thetas)
        R = THETA * 0

        for x_idx, x in enumerate(x_values):
            for θ_idx, θ in enumerate(thetas):
                R[θ_idx, x_idx] = self.maxtoolradius_on_dmindmaxrange_at_xRtool_atθ(x, θ, dmin,dmax)
        X = X
        Y = R * np.cos(THETA)
        Z = R * np.sin(THETA)

        return X,Y,Z,THETA, R

    def plot_maxtoolenvelope_on_dmindmaxrange(self,dmin,dmax):
        X, Y, Z, THETA, R = self.maxtoolenvelope_on_dmindmaxrange(dmin,dmax)

        fig = go.Figure()

        # Définition des couleurs
        thetas = THETA[:, 0]
        S = X * 0
        mid_idx = np.floor(len(thetas) / 2)
        for θ_idx in range(0, len(thetas)):
            θ_idxopp = int((θ_idx + mid_idx) % len(thetas))

            P1 = np.array([X[θ_idx, :], Y[θ_idx, :], Z[θ_idx, :]])
            P2 = np.array([X[θ_idxopp, :], Y[θ_idxopp, :], Z[θ_idxopp, :]])
            dist = np.sum((P2 - P1) ** 2, axis=0) ** 0.5
            S[θ_idx, :] = dist

        fig.add_trace(go.Surface(x=X, y=Y, z=Z,surfacecolor=S, opacity=0.6, colorscale="RdBu"))


        x_values = X[0, :]
        for x_idx, x in enumerate(x_values):
            xis = np.zeros_like(thetas) + x
            yis = R[:, x_idx] * np.cos(thetas)
            zis = R[:, x_idx] * np.sin(thetas)
            fig.add_trace(
                go.Scatter3d(x=xis, y=yis, z=zis, opacity=0.5, mode="lines",
                             line=dict(color="rgba(0,0,0,0.5)", width=1), name=""))
        for θ_idx, θ in enumerate(thetas):
            xis = x_values
            yis = R[θ_idx, :] * np.cos(θ)
            zis = R[θ_idx, :] * np.sin(θ)

        # Plot of the tool
        linescolor = "rgba(0,0,0,1)"
        fig.add_trace(go.Scatter3d(x=xis, y=yis, z=zis, opacity=0.5, mode="lines",
                                   line=dict(color=linescolor, width=2), name=""))

        fig.update_scenes(aspectmode="data")
        fig.update_layout(template="plotly_dark")
        fig.show()

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

    tool_envelope_calculation = ToolEnvelopeCalculation(profile_cavity_1_3GHz__Rmach,-144,0)
    tool_envelope_calculation.plot_maxtoolenvelope_on_dmindmaxrange(0,-40)