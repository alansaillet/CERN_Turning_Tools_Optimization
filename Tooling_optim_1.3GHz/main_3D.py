import numpy as np
import plotly.graph_objs as go
import trimesh as trimesh

if 1: #1.3GHz
    LongueurDemiCavité = 150/2
    #LongueurDemiCavité = 413.8/2 #1.3GHz complète
    LongueurOutil = LongueurDemiCavité *2
    def P_centré(x):
        if x < -LongueurDemiCavité:
            return [x, np.inf]
        elif -LongueurDemiCavité <= x <= -57:
            return [x, 39.05]
        elif -57 < x <= -48.807:
            return [x, 51.85 - np.sqrt((1 - (x - (-57))**2/9**2)*(12.8)**2)]
        elif -48.807 < x <= -40.006:
            return [x, 46.553 + (x - (-48.807))*np.tan(72.275 * np.pi / 180)]
        elif -40.006 < x <= 0:
            return [x, 61.3 + np.sqrt(42**2 - (x - 0)**2)]
        else:
            return [x, P_centré(-x)[1]]
    deltaX_end = -LongueurDemiCavité

else: #400MHz
    LongueurDemiCavité = 512.5 #400MHz complète
    LongueurOutil = LongueurDemiCavité
    def P_centré(x):
        if x < -LongueurDemiCavité:
            return [x, np.inf]
        elif -LongueurDemiCavité <= x <= -162.75:
            return [x, 150]
        elif -162.75 < x <= -139.179:
            return [x, 175 - np.sqrt(25 ** 2 - (x - (-162.75)) ** 2)]
        elif -139.179 < x <= -101.09:
            return [x, 166.67 + (x - (-139.179)) * np.tan(70.537 * np.pi / 180)]
        elif -101.09 < x <= 0:
            return [x, 239.7 + np.sqrt(104.3 ** 2 - (x - (-2.75)) ** 2)]
        else:
            return [x, P_centré(-x)[1]]
    deltaX_end = -(LongueurDemiCavité + 162.75)

deltaD = 2
deltaθ = np.pi/41

deltaD = 4
deltaθ = np.pi/21

deltaD = 1
deltaθ = np.pi/71

EXPORT = True
PLOT = True
SAVEPLOT = False
GENERATE_ANIMATION = False
SAVE_PNG_nobg = True

DepthFactor = 0.5 #length of the tool in the z direction
output_name = "output_trimesh_144_halfway"

def P(x):
    return P_centré(x - LongueurDemiCavité)

def Pm(x, d):
    return [x, P(x + d)[1] + P(0 + d)[1] - P(0 + 0)[1]*1.8] #P(0 + 0)[1]

def Outil_shape_at_d(x,θ,d):
    # 2D
    Cy = Pm(x, d)[1] - P(x+d)[1]
    R = P(x + d)[1]

    # 3D
    tan_theta = np.tan(np.pi / 2 - θ)
    a = 1 + tan_theta ** 2
    b = 2*Cy * tan_theta
    c = Cy**2 - R**2
    delta = b**2-4*a*c

    z_solution_1 = (-b + np.sqrt(delta)) / (2*a)
    z_solution_2 = (-b - np.sqrt(delta)) / (2*a)

    y_solution_1 = tan_theta * z_solution_1
    y_solution_2 = tan_theta * z_solution_2

    R_solution_1 = np.sqrt(y_solution_1 ** 2 + z_solution_1 ** 2)
    R_solution_2 = np.sqrt(y_solution_2 ** 2 + z_solution_2 ** 2)
    # Check: sign of theta should be the same as the sign of z!
    # mask_solutions = np.sign(z_solution_1) == np.sign(θ)
    # print(np.sign(z_solution_1) , np.sign(θ),np.sign(z_solution_1) == np.sign(θ))
    Rsol = R_solution_1

    """from scipy.optimize import minimize
    def y_ligne(z):
        y = np.tan(np.pi / 2 - θ) * z
        return y
    def y_cercle(z):
        if 0>0:
            y = Cy + np.sqrt(R ** 2 - z ** 2)
        else:
            y = Cy - np.sqrt(R ** 2 - z ** 2)
        return y
    def costf(z):
        return (y_ligne(z)-y_cercle(z))**2
    z = minimize(costf,0).x[0]
    y = y_cercle(z)
    Rsol = np.sqrt(y ** 2 + z ** 2)
    print(Rsol)"""

    if np.sign(z_solution_1) == np.sign(θ):
        Rsol = R_solution_1
    else:
        Rsol = R_solution_2

        # R_solution = (mask_solutions * R_solution_1) + (~mask_solutions * R_solution_2)
    if np.isnan(Rsol):
        Rsol=P_centré(-1)[1]
    return [x, Rsol]
def Outil_shape_to_d(x,θ,dmax):
    Rsols = []

    for d in list(np.arange(0, dmax, deltaD))+[dmax]:
        Rsols.append(Outil_shape_at_d(x, θ, d)[1])

    return [x,min(Rsols)]

'''def Outil(x,θ):
    return Outil_shape_to_d(x,θ,LongueurDemiCavité + 2)'''

#x_values = np.array(list(np.arange(-LongueurOutil, 0, deltaD))+[0])
x_values = np.array(list(np.arange(-LongueurOutil*DepthFactor, 20, deltaD)))
thetas = np.arange(-np.pi, np.pi, deltaθ)
thetas = [*thetas, thetas[0]]

X,THETA = np.meshgrid(x_values, thetas)
R=THETA*0

###EXPORT###
if EXPORT:

    X, THETA = np.meshgrid(x_values, thetas)
    R = THETA * 0

    for x_idx, x in enumerate(x_values):
        for θ_idx, θ in enumerate(thetas):
            R[θ_idx, x_idx] = Outil_shape_to_d(x, θ, LongueurDemiCavité*2)[1]

    R[-1,:] = R[0,:] #making sure the surface is completely closed

    X = X
    Y = R * np.cos(THETA)
    Z = R * np.sin(THETA)

    # Create a numpy array containing the vertices and faces
    vertices = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

    # Calculate the indices for the faces
    faces = []
    num_points_0 = np.shape(X)[0]
    num_points_1 = np.shape(X)[1]
    for i in range(num_points_0 - 1):
        for j in range(num_points_1 - 1):
            v00 = i * num_points_1 + j
            v01 = i * num_points_1 + j + 1
            v11 = (i + 1) * num_points_1 + j + 1
            v10 = (i + 1) * num_points_1 + j
            faces.extend([(v00, v10, v01), (v01, v10, v11)])
    faces = np.array(faces)
    vertices = np.array(vertices)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    #mesh.export("output_trimesh.stl",  file_type="stl_ascii")
    d=0
    dec = P(0 + d)[1] - P(0 + 0)[1]*1.8
    mesh = mesh.apply_translation([0,dec-78.1/2,0])
    mesh.export(f"{output_name}.stl")


def GenerateFrame(distance):
    fig = go.Figure()

    # COMPUTATION OF THE RESULT

    X, THETA = np.meshgrid(x_values, thetas)
    R = THETA * 0

    for x_idx, x in enumerate(x_values):
        for θ_idx, θ in enumerate(thetas):
            R[θ_idx, x_idx] = Outil_shape_to_d(x, θ, distance)[1]
    X = X
    Y = R * np.cos(THETA)
    Z = R * np.sin(THETA)
    S = X*0

    mid_idx = np.floor(len(thetas)/2)
    for θ_idx in range(0,len(thetas)):
        θ_idxopp = int((θ_idx + mid_idx) % len(thetas))

        P1 = np.array([X[θ_idx,:],Y[θ_idx,:],Z[θ_idx,:]])
        P2 = np.array([X[θ_idxopp,:],Y[θ_idxopp,:],Z[θ_idxopp,:]])
        dist = np.sum((P2 - P1)**2,axis=0)**0.5
        S[θ_idx,:] = dist


    for x_idx, x in enumerate(x_values):
        xis = np.zeros_like(thetas) + x
        yis = R[:, x_idx] * np.cos(thetas)
        zis = R[:, x_idx] * np.sin(thetas)
        fig.add_trace(
            go.Scatter3d(x=xis, y=yis, z=zis, opacity=0.5, mode="lines", line=dict(color="rgba(0,0,0,0.5)", width=1),name=""))
    for θ_idx, θ in enumerate(thetas):
        xis = x_values
        yis = R[θ_idx, :] * np.cos(θ)
        zis = R[θ_idx, :] * np.sin(θ)

    # Plot of the tool
    fig.add_trace(go.Surface(x=X, y=Y, z=Z,surfacecolor=S, opacity=0.6, colorscale="RdBu"))
    linescolor = "rgba(0,0,0,1)"
    fig.add_trace(go.Scatter3d(x=xis, y=yis, z=zis, opacity=0.5, mode="lines",
                               line=dict(color=linescolor, width=2), name=""))

    #Plot Cavity in position
    xs_cavity = np.arange(-distance, LongueurDemiCavité * 2 - distance, deltaD)
    X_cavity, THETA = np.meshgrid(xs_cavity, thetas)
    R_cavity = X_cavity * 0
    for x_idx, x in enumerate(xs_cavity):
        for θ_idx, θ in enumerate(thetas): #values should be equal... could copy dims... or broadcast...
            R_cavity[θ_idx, x_idx] = P(x + distance)[1]
            X_cavity[θ_idx, x_idx] = Pm(x, distance)[0]

    delta_y = Pm(0, distance)[1] - P(0 + distance)[1]
    X_cavity = X_cavity
    Y_cavity = R_cavity * np.cos(THETA) - delta_y
    Z_cavity = R_cavity * np.sin(THETA)

    """fig.add_trace(go.Scatter3d(x=np.transpose(X_cavity).flatten(),
                               y=np.transpose(Y_cavity).flatten(),
                               z=np.transpose(Z_cavity).flatten(),
                               mode="lines", marker=dict(color="rgba(0,0,0,0.2)", size=2)))"""

    #Plot of the cavity
    linecolor = "rgba(255,255,255,0.2)"
    for x_idx, x in enumerate(xs_cavity):
        fig.add_trace(go.Scatter3d(x=X_cavity[:, x_idx], y=Y_cavity[:, x_idx], z=Z_cavity[:, x_idx], opacity=0.5,
                                   mode="lines", line=dict(color=linecolor, width=1),name=""))
    for θ_idx, θ in enumerate(thetas):
        fig.add_trace(go.Scatter3d(x=X_cavity[θ_idx, :], y=Y_cavity[θ_idx, :], z=Z_cavity[θ_idx, :], opacity=0.5,
                                   mode="lines", line=dict(color=linecolor, width=1),name=""))

    layout = go.Layout(
        scene=dict(
            aspectmode='data'
        ))
    fig.update_layout(layout)


    #fig.show()
    return fig.data


if PLOT:
    if GENERATE_ANIMATION == False:
        data = GenerateFrame(-deltaX_end)
    else:
        data=GenerateFrame(0)
    fig = go.Figure(data=data,
                    layout=go.Layout(
                        xaxis=dict(range=[0, 5], autorange=False),
                        yaxis=dict(range=[0, 5], autorange=False),
                        title="Start Title",
                        updatemenus=[dict(
                            type="buttons",
                            buttons=[dict(label="Play",
                                          method="animate",
                                          args=[None])])]
                    ),)
    style = dict(
        backgroundcolor="rgba(0,0,0,0.1)",
        gridcolor="rgba(0,0,0,0.1)",
        showbackground=False,
        zerolinecolor="rgba(0,0,0,0.1)")
    fig.update_layout(scene=dict(xaxis=style, yaxis=style, zaxis=style))


# Set orthographic projection and camera view (along z axis)
fig.update_layout(
    scene=dict(
        camera=dict(
            projection=dict(type='orthographic'),
            eye=dict(x=0, y=0, z=-1)  # or try z=-2 for opposite direction
        )
    )
)

if GENERATE_ANIMATION:
    frames=[]
    for d in np.arange(0,deltaX_end,2):
        frames.append(go.Frame(data=GenerateFrame(d)))
    fig.frames = frames

if PLOT:
    fig.update_scenes(aspectmode="data")
    fig.update_layout(template="plotly_dark")
    fig.show()

if SAVEPLOT:
    fig.write_html("export.html")

if SAVE_PNG_nobg:

    fig.update_layout(
        template="simple_white",
        paper_bgcolor="rgba(0,0,0,0)",  # transparent canvas
        plot_bgcolor="rgba(0,0,0,0)",  # transparent plotting region
        margin=dict(l=0, r=0, t=0, b=0),  # no padding

    )

    # Also disable any legend or annotations if present
    fig.update_layout(showlegend=False)

    fig.write_image(
        f"{output_name}.png",
        format="png",
        width=2048,
        height=2048,
        scale=3)