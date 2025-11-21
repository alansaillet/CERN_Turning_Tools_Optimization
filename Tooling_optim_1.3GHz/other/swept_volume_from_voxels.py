import numpy as np
import plotly.graph_objs as go
import trimesh as trimesh

LongueurDemiCavité = 144 / 2
LongueurDemiCavité = 413.8 / 2  # 1.3GHz complète
LongueurOutil = LongueurDemiCavité * 2

def P_centré(x):
    if x < -LongueurDemiCavité:
        return [x, np.inf]
    elif -LongueurDemiCavité <= x <= -57:
        return [x, 39.05]
    elif -57 < x <= -48.807:
        return [x, 51.85 - np.sqrt((1 - (x - (-57)) ** 2 / 9 ** 2) * (12.8) ** 2)]
    elif -48.807 < x <= -40.006:
        return [x, 46.553 + (x - (-48.807)) * np.tan(72.275 * np.pi / 180)]
    elif -40.006 < x <= 0:
        return [x, 61.3 + np.sqrt(42 ** 2 - (x - 0) ** 2)]
    else:
        return [x, P_centré(-x)[1]]

deltaD = 4
deltaθ = np.pi/21

x_values = np.array(list(np.arange(-LongueurDemiCavité, LongueurDemiCavité, deltaD)))
rs = [P_centré(x)[1] for x in x_values]

thetas = np.arange(-np.pi, np.pi, deltaθ)
thetas = [*thetas, thetas[0]]

R,THETA = np.meshgrid(rs, thetas)
X,_ = np.meshgrid(x_values, thetas)
Y = R*np.cos(THETA)
Z = R*np.sin(THETA)


#High Ø
Rout = np.amax(rs) + 10
x_new = X[:,-1].reshape(-1, 1)
y_new = (Rout*np.cos(thetas)).reshape(-1, 1)
z_new = (Rout*np.sin(thetas)).reshape(-1, 1)
X = np.hstack((X, x_new))
Y = np.hstack((Y, y_new))
Z = np.hstack((Z, z_new))

#back in X-
x_new = X[:,0].reshape(-1, 1)
y_new = Y[:,-1].reshape(-1, 1)
z_new = Z[:,-1].reshape(-1, 1)
X = np.hstack((X, x_new))
Y = np.hstack((Y, y_new))
Z = np.hstack((Z, z_new))

#back in X-
x_new = X[:,0].reshape(-1, 1)
y_new = Y[:,0].reshape(-1, 1)
z_new = Z[:,0].reshape(-1, 1)
X = np.hstack((X, x_new))
Y = np.hstack((Y, y_new))
Z = np.hstack((Z, z_new))

print(np.shape(X))

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
 # ------------------
from gpytoolbox import read_mesh
from gpytoolbox.copyleft import swept_volume
# Read sample mesh
# v, f = gpytoolbox.read_mesh("test/unit_tests_data/bunny_oded.obj")
v = vertices
f = faces
# Translation vectors to make Catmull-Rom spline
translation_0 = np.array([0,0,0])
translation_1 = np.array([100,0,-1])
translations = [translation_0,translation_1]

"""fig = go.Figure()
fig.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=0.6, colorscale="RdBu"))
fig.show()
exit()"""

# Call swept volume function
u,g = swept_volume(v,f,translations=translations,eps=1,
verbose=True,align_rotations_with_velocity=False)

mesh = trimesh.Trimesh(vertices=u, faces=g)
mesh.export("output_swept_volume.stl")