import matplotlib.pyplot as plt

def method():
    # Create a new graph
    G = nx.Graph()

    # Add two nodes
    G.add_node(1)
    G.add_node(2)

    # Add an edge between the nodes
    G.add_edge(1, 2)

    # Draw the graph
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(G, pos, with_labels=True)  # draw the graph

    # Display the graph
    plt.show()

    # The output is the plotted graph
    output = G

    return output

# Call the method for validation
if __name__ == "__main__":
    result = method()
    print(result)