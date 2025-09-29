import numpy as np
import colour
from agx_emulsion.config import SPECTRAL_SHAPE

# Get D65 spectral distribution aligned to the project shape
sd = colour.SDS_ILLUMINANTS['D65'].copy().align(SPECTRAL_SHAPE)

# Save as CSV: wavelength, value
np.savetxt("D65.csv",
           np.column_stack([sd.wavelengths, sd.values]),
           delimiter=",", fmt="%.6f")
