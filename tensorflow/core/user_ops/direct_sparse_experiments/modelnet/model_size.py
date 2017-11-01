import math as math
b = 36
ti = 8
td = 4
r1 = 256
d1 = 0.01
r2 = 128
d2 = 0.03
r3 = 64
d3 = 0.05
d4 = 0.4

r4 = 32
r5 = 16
r6=8

#backprop + forward pass
sc1 = (ti + td + td) * b * 8 * math.pow(r1,3)* d1 * 3
sc2 = (ti + td + td) * b * 16 * math.pow(r2,3) * d2 * 3
sc3 = (ti + td + td) * b * 24 * math.pow(r3,3) * d3 *3 
sp1 = td * b * 8 * math.pow(r1,3)* d1 +  (ti + td) * b * 16 * math.pow(r2,3) * d2
sp2 = td * b * 16 * math.pow(r2,3) * d2 + (ti + td) * b * 24 * math.pow(r3,3) * d3
sp3 = td * b * 24 * math.pow(r3,3) * d3 + (ti + td) * b * 32 * pow(r4,3) * d4
s2d = td * b * 24 * math.pow(r3,3) * d3 + td *  b * 32 * pow(r4,3)

sparse_size = (sc1 + sc2 + sc3 + sp1 + sp2 + sp3 + s2d) / pow(10,9)
print("sparse layers size [GB]: ", sparse_size)

#backprop + forward pass
dc1 = (td + td) * b * 32 * pow(r4,3) * 3
dc2 = (td + td) * b * 40 * pow(r5,3) * 3
dc3 = (td + td) * b * 48 * pow(r6,3) * 3
pd1 = td * b * 32 * pow(r4,3) + td * b * 40 * pow(r5,3)
pd2 = td * b * 40 * pow(r5,3) + td * b * 48 * pow(r6,3)

dense_size = (dc1 + dc2 + dc3 + pd1 + pd2) / float(pow(10,9))
print("dense layers size [GB]: ", dense_size)
print("total size [GB]: ", dense_size + sparse_size)

