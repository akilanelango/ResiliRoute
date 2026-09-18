import osmnx as ox
# Fetch the driving network for a known flood-prone area
graph = ox.graph_from_place("New Orleans, Louisiana", network_type="drive")
ox.plot_graph(graph)