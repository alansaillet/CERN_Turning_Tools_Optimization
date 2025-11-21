import numpy as np
import plotly.graph_objs as go
import scipy.spatial.transform
import trimesh as trimesh

LongueurDemiCavité = 144/2
LongueurDemiCavité = 413.8/2
LongueurOutil = LongueurDemiCavité

deltaD = 0.2
deltaθ = np.pi/20

deltaD = 2
deltaθ = np.pi/41

deltaD = 4
deltaθ = np.pi/21

EXPORT = False
PLOT = True
GENERATE_ANIMATION = False

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

def Workpiece(x,theta):
    r = P_centré(x)[1]
    return [x,r*np.cos(theta),r*np.sin(theta)]

class FiveAxisTool:
    def __init__(self):
        xs_build = np.linspace(0,100,100)
        thetas_build = np.linspace(0,2*np.pi,20)
        r_build = 200

        X_build, THETA_build = np.meshgrid(xs_build,thetas_build)
        Y_build = r_build*np.cos(THETA_build)
        Z_build = r_build*np.sin(THETA_build)

        self.Points = np.transpose([X_build,Y_build,Z_build])
        self.P = np.array([0,0,0])#x_idx, theta_idx,1, components
        self.current_Rotation = scipy.spatial.transform.Rotation.from_rotvec([0,0,1e-10])

    def moveTo(self,P1):
        V  = P1 - self.P
        self.moveOf(V)

    def moveOf(self,V):
        self.Points += V

    def Rotate(self,axis=None,rotation=None,inverse=False):
        if rotation is None and axis is not None:
            rotation = scipy.spatial.transform.Rotation.from_rotvec(axis)

        #Prepare
        shape = np.shape(self.Points)
        size = np.prod(shape[:-1])
        Points = np.reshape(self.Points.copy(),(size,3))
        #Apply
        Points = rotation.apply(Points,inverse=inverse)
        #Post
        Points = np.reshape(Points,shape)
        #Save
        self.current_Rotation = scipy.spatial.transform.Rotation.concatenate([self.current_Rotation,rotation])
        self.Points = Points

    def CancelRotation(self):
        self.Rotate(rotation=self.current_Rotation, inverse=False)

tool = FiveAxisTool()

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=tool.Points[:,:,0].flatten(),y=tool.Points[:,:,1].flatten(),z=tool.Points[:,:,2].flatten(),mode="lines+markers"))
tool.Rotate(axis=np.array([0,0,0.3]))
fig.add_trace(go.Scatter3d(x=tool.Points[:,:,0].flatten(),y=tool.Points[:,:,1].flatten(),z=tool.Points[:,:,2].flatten(),mode="lines+markers"))
tool.Rotate(axis=np.array([0.3,0,0]))
fig.add_trace(go.Scatter3d(x=tool.Points[:,:,0].flatten(),y=tool.Points[:,:,1].flatten(),z=tool.Points[:,:,2].flatten(),mode="lines+markers"))

tool.CancelRotation()
"""tool.Rotate(axis=np.array([0.3,0,0]))
fig.add_trace(go.Scatter3d(x=tool.Points[:,:,0].flatten(),y=tool.Points[:,:,1].flatten(),z=tool.Points[:,:,2].flatten(),mode="lines+markers"))"""


fig.show()
