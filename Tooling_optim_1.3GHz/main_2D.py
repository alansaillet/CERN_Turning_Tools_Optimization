import numpy as np
import matplotlib.pyplot as plt

LongueurCavité = 72
LongueurOutil = LongueurOutil

d = 1

def P_centré(x):
    if x < -LongueurCavité:
        return [x, 60]
    elif -LongueurCavité <= x <= -57:
        return [x, 39.05]
    elif -57 < x <= -48.807:
        return [x, 51.85 - np.sqrt((1 - (x - (-57))**2/9**2)*(12.8)**2)]
    elif -48.807 < x <= -40.006:
        return [x, 46.553 + (x - (-48.807))*np.tan(72.275 * np.pi / 180)]
    elif -40.006 < x <= 0:
        return [x, 61.3 + np.sqrt(42**2 - (x - 0)**2)]
    else:
        return [x, P_centré(-x)[1]]

def P(x):
    return [x, P_centré(x - LongueurCavité)[1]]

def P_low(x):
    return [x, -P(x)[1]]

def Pm(x, d):
    return [x, P(x + d)[1] - P(0 + d)[1] + P(0 + 0)[1]]

def Pm_low(x, d):
    return [x, P_low(x + d)[1] + P_low(0 + d)[1] - P_low(0 + 0)[1]]

def Outil(x):
    return [x, np.min([Pm(x, d)[1] for d in np.arange(0, LongueurCavité + 2,d)])]

def Outil_low(x):
    return [x, np.max([Pm_low(x, d)[1] for d in np.arange(0, LongueurCavité + 2,d)])]

x_values = np.arange(-200, 1, d)
tools_data = np.array([Outil(x) for x in x_values])
tools_data_low = np.array([Outil_low(x) for x in x_values])

"""plt.plot(tools_data[:, 0], tools_data[:, 1], label='Tool')
plt.plot(tools_data_low[:, 0], tools_data_low[:, 1], label='Low Tool')
plt.show()


print(tools_data)
import pandas as pd
df = pd.DataFrame(data=tools_data)
df.to_csv("up")
df = pd.DataFrame(data=tools_data_low)
df.to_csv("down")
print(df)
exit()
"""
def plot_tools(k):
    plt.plot(tools_data[:, 0], tools_data[:, 1], label='Tool')
    plt.plot(tools_data_low[:, 0], tools_data_low[:, 1], label='Low Tool')

    plt.plot([Pm(x, k)[0] for x in np.arange(-k, 2*LongueurCavité - k, 1)],
             [Pm(x, k)[1] for x in np.arange(-k, 2*LongueurCavité - k, 1)], label='Tool')
    plt.plot([Pm_low(x, k)[0] for x in np.arange(-k, 2*LongueurCavité - k, 1)],
             [Pm_low(x, k)[1] for x in np.arange(-k, 2*LongueurCavité - k, 1)], label='Low Tool')
    plt.xlabel('x')
    plt.ylabel('Tool Height')
    plt.legend()
    plt.show()

k_value = float(input("Enter a value for k (0 to 72): "))
plot_tools(k_value)