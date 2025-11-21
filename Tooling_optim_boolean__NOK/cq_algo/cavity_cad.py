import cadquery as cq
import numpy as np


def profile_cavity_1_3GHz__Rmach(x):
    cavity_length = 144
    cavity_halflength = cavity_length / 2

    def profile_cavity_1_3GHz__centered(x):
        if x < -cavity_halflength:
            return np.nan
        elif -cavity_halflength <= x <= -57:
            return 39.05
        elif -57 < x <= -48.807:
            return 51.85 - np.sqrt((1 - (x - (-57)) ** 2 / 9 ** 2) * (12.8) ** 2)
        elif -48.807 < x <= -40.006:
            return 46.553 + (x - (-48.807)) * np.tan(72.275 * np.pi / 180)
        elif -40.006 < x <= 0:
            return 61.3 + np.sqrt(42 ** 2 - (x - 0) ** 2)
        else:
            return profile_cavity_1_3GHz__centered(-x)

    return profile_cavity_1_3GHz__centered(x + cavity_halflength)


# Generate profile points
outer_r = 150

x_values = np.linspace(0, -144, 100)
profile_points = [(x, profile_cavity_1_3GHz__Rmach(x)) for x in x_values if
                  not np.isnan(profile_cavity_1_3GHz__Rmach(x))]

print(profile_points)
print(profile_points[-1][-1])
profile_points.append((x_values[-1], outer_r))
profile_points.append((x_values[0], outer_r))
profile_points.append((x_values[0], profile_points[-3][-1]))
print(profile_points)
# Create a wire from the profile
wire = cq.Workplane("XY").polyline(profile_points).close()

# Revolve the profile to create the 3D shape
cavity = wire.revolve(360, (0, 0, 0), (1, 0, 0))

# Export the cavity as a STEP file
cq.exporters.export(cavity, "cavity_1_3GHz.step")

print("STEP file exported: cavity_1_3GHz.step")
