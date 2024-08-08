import pyGEM as gem

path = "dem/srtm_bigtujunga_30m.tif"
gem.display(path)

asp = gem.aspect(path)
gem.display(asp)

shd = gem.hillshade(path)
gem.display(shd, colormap='binary_r')

shd = gem.hillshade(path)
gem.display(shd, colormap='binary_r')

slope = gem.slope(path)
gem.display(slope, colormap='binary_r')

shd = gem.hillshade(path)
gem.display(shd, colormap='binary_r')

tpi = gem.tpi(path, ksize=21)
gem.display(tpi, colormap='binary_r')

relief = gem.relief(path, ksize=21)
gem.display(relief, colormap='jet')
