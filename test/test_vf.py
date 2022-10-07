import polykriging as pk

path = utility.choose_file(titl="Directory for file GeometryFeatures file (.geo)")
geomFeatures = pk.pk_load(path).to_numpy()