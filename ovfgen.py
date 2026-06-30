import numpy as np

NX, NY, NZ = 4096, 4096, 1
DX, DY, DZ = 1.5e-6, 1.5e-6, 1e-5

AMPLITUDE = 1.0
SIGMA = 3e-4
LAMBDA0 = 5e-5      # CHANGE THIS

k = 2 * np.pi / LAMBDA0

x = (np.arange(NX) - (NX - 1) / 2) * DX
y = (np.arange(NY) - (NY - 1) / 2) * DY

sinx = np.sin(k * x)
cosx = np.cos(k * x)


def write_file(filename, carrier):
    with open(filename, "w") as f:
        f.write("# OOMMF OVF 2.0\n")
        f.write("# Segment count: 1\n")
        f.write("# Begin: Segment\n")
        f.write("# Begin: Header\n")
        f.write(f"# xnodes: {NX}\n")
        f.write(f"# ynodes: {NY}\n")
        f.write(f"# znodes: {NZ}\n")
        f.write(f"# xstepsize: {DX}\n")
        f.write(f"# ystepsize: {DY}\n")
        f.write(f"# zstepsize: {DZ}\n")
        f.write("# valuedim: 3\n")
        f.write("# End: Header\n")
        f.write("# Begin: Data text\n")

        for yy in y:
            envelope = np.exp(-(x**2 + yy**2) / (2 * SIGMA**2))
            row = AMPLITUDE * envelope * carrier

            for value in row:
                f.write(f"{value:.6e} 0.000000e+00 0.000000e+00\n")

        f.write("# End: Data text\n")
        f.write("# End: Segment\n")


write_file("bprofile_sin.ovf", sinx)
write_file("bprofile_cos.ovf", cosx)

print("Done.")
