import tensorflow as tf
import numpy as np

def choose_random_centroids(samples, n_clusters):
    # Step 0: Initialisation: Select `n_clusters` number of random points
    #get number of data set points
    n_samples = tf.shape(samples)[0]
    #create an arrow with all indices from 0 to n_samples in an arbitrary order
    random_indices = tf.random_shuffle(tf.range(0, n_samples))
    begin = [0,]
    size = [n_clusters,]
    size[0] = n_clusters
    centroid_indices = tf.slice(random_indices, begin, size)
    initial_centroids = tf.gather(samples, centroid_indices)
    return initial_centroids

def create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed):
    np.random.seed(seed)
    slices = []
    centroids = []
    # Create samples for each cluster
    for i in range(n_clusters):
        samples = tf.random_normal((n_samples_per_cluster, n_features), mean=0.0, stddev=5.0, dtype=tf.float32, seed=seed, name="cluster_{}".format(i))
        current_centroid = (np.random.random((1, n_features)) * embiggen_factor) - (embiggen_factor/2)
        centroids.append(current_centroid)
        samples += current_centroid
        slices.append(samples)
    # Create a big "samples" dataset
    samples = tf.concat(slices, 0, name='samples')
    centroids = tf.concat(centroids, 0, name='centroids')
    return centroids, samples

def plot_clusters(centroids,partitions):
    import matplotlib.pyplot as plt
    # Plot out the different clusters
    # Choose a different colour for each cluster
    n_clusters=len(partitions)
    print("partitions",partitions)
    print("LEEEEEEEEEEE",n_clusters)
    colour = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
    print("HOLOOOOOOOOO",colour)
    colours=["blue","green","red"]
    for i in range(n_clusters):
        # Grab just the samples fpr the given cluster and plot them out with a new colour
        samples =partitions[i]
        print("SAMPLEEEEEEEEEEEEEE",samples)
        centroid=centroids[i]
        print("CENTROID",centroid)
        print(samples[:, 0])
        print(samples[:, 1])
        print("coloR",colour[i])
        plt.scatter(samples[:, 0], samples[:, 1], c=colour[i])
        # Also plot centroid
        plt.plot(centroid[0], centroid[1], markersize=35, marker="x", color='k', mew=10)
        plt.plot(centroid[0], centroid[1], markersize=30, marker="x", color='m', mew=5)
    plt.show()

n_features = 2
n_clusters = 3
n_samples_per_cluster = 3
seed = 700
embiggen_factor = 70

np.random.seed(seed)

#data
data_centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)

#choose centroids randomly
centroids = choose_random_centroids(samples, n_clusters)

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    sample_values = session.run(samples)
    data_centroids=session.run(data_centroids)
    #initialice cluster data
    _,centroid8,initial_indices,_=session.run([model, samples,centroids,initial_indices])

    partitions_values = session.run(tf.dynamic_partition(sample_values, indice, n_clusters))

    plot_clusters(centroids,partitions_values)
