# VisualisorAccessor
Visualising xarray DataArrays. You also need MaskAccessor.

You can install all other required packages and VisualisorAccessor by 
`pip install -e .`

You can import it by
`import visacc`

Every xarray.DataArray (assume its name is `cube`) that you make after importing visacc contains a property named `visualize`

You can use the properties or functions of VisualisorAccessor by `cube.visualize.<desired property or function>`

Check out the example notebook for detailed example.

I will not answer questions regarding this package on other platforms. Please raise issue here on github if you have one.

Citation: Annala, L., Eskelinen, M. A., Hämäläinen, J., Riihinen, A., and Pölönen, I.: PRACTICAL APPROACH FOR HYPERSPECTRAL IMAGE PROCESSING IN PYTHON, Int. Arch. Photogramm. Remote Sens. Spatial Inf. Sci., XLII-3, 45-52, https://doi.org/10.5194/isprs-archives-XLII-3-45-2018, 2018. 
