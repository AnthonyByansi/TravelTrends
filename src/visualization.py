# creating graphs and charts to visualize the results of the machine learning analysis


import matplotlib.pyplot as plt

def create_graphs(patterns, trends):
    # Plot the patterns
    plt.scatter(patterns[:, 0], patterns[:, 1], c=patterns)
    plt.title('Patterns in Travel Industry')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.savefig('results/patterns.png')
    plt.close()
    
    # Plot the trends
    plt.plot(trends)
    plt.title('Trends in Travel Industry')
    plt.xlabel('Year')
    plt.ylabel('Trend Value')
    plt.savefig('results/trends.png')
    plt.close()
