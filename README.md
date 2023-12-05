# PyFFD

**PyFFD** is an open-source Free Form Deformation (FFD) library.


## Code
### Quick start

**(1)** Define a FEM mesh :
```python
n = np.array([[x0, y0], [x1, y1], ...])
```

**(2)** Define the elements size and degree for the B-Spline box in each direction :
```python
# Element size per directions
size = [size_x, size_y]
# global setting
size = 0.1 # example

# Element degree per directions
deg = [deg_x, deg_y]
# global setting
deg = 3 # example
```

**(3)** Create the B-Spline box python object : 
```python
sbox = pf.FFD(n, size, deg)
```

**(4)** Compute the FFD opperator $R_s$ :
```python
Rs = sbox.OperatorFFD()
```
