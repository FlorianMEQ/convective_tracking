# convective_tracking

Hierarchical 3D detection and tracking of convective structures (MesoNH model output, synthetic satellite brightness temperature).

Structures are detected on several **nested levels**, then tracked in time level by level:

```
mcs                  (root)     mesoscale convective system, detected on (BT)
└─ convective_object            3D convective objects (WT)
   └─ core                      3D convective cores (WT)
```

## Installation

Dependencies:

```bash
pip install numpy scipy scikit-image pandas xarray netcdf4
```
