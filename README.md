# PyFFD

**PyFFD** is an open-source Free Form Deformation (FFD) library.


## Quick start

(1) Define a FEM mesh :
```python
n = np.array([[x0, y0], [x1, y1], ...])
```

(2) Define the number of element for the B-Spline box :
```python
ne_s = np.array([number_x,number_y])
```

(3) Create the B-Spline box python object : 
```python
sbox = pf.FFD(n, ne_s, 3, delta=0.1)
```

(4) Compute the FFD opperator :
```python
Rs = ms.OperatorFFD()
```