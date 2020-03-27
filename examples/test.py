from mumax5 import World, Grid, anisotropyField

w = World(cellsize=(1e-9,1e-9,1e-9))

magnet = w.addFerromagnet(name='magnet', grid=Grid((10,10,1)))

f = anisotropyField(magnet)

print(f.get())