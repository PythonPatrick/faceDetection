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

def assign_to_nearest(samples, centroids):
    # Finds the nearest centroid for each sample
    expanded_vectors = tf.cast(tf.expand_dims(samples, 0),tf.float64)
    expanded_centroids = tf.cast(tf.expand_dims(centroids, 1),tf.float64)
    distances = tf.reduce_sum(
                    tf.square(
                        tf.subtract(expanded_vectors, expanded_centroids)
                    )
                , 2)

    min=tf.argmin(distances, 0)
    x,y,z=tf.unique_with_counts(min)

    def missing():
        given, b = tf.unique(min)
        clusters=tf.constant(list(range(centroids.shape[0])),tf.int64)
        return tf.setdiff1d(clusters, given).out


    result=tf.cond((tf.shape(x)<centroids.shape[0])[0],
                    lambda: min,
                    lambda: min)

    #missingClusters=tf.sets.set_difference(x, tf.argmin(distances, 0))
    return missing()

def update_centroids(samples, nearest_indices, n_clusters):
    # Updates the centroid to be the mean of all samples associated with it.
    nearest_indices = tf.to_int32(nearest_indices)
    partitions = tf.dynamic_partition(samples, nearest_indices, n_clusters)
    new_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions], 0)
    return new_centroids

def condition(indices,centroids,i): return i < 0

def kmneansItertions(samples,initial_indices,initial_centroids,n_clusters):
    indices=tf.Variable(tf.to_int64(initial_indices))
    centroids=tf.Variable(initial_centroids)

    def body(indices, centroids, i):
        ind = assign_to_nearest(samples, centroids)
        cen = update_centroids(samples, ind, n_clusters)
        tf.print(i.eval,[i])
        return ind, cen, i + 1

    i=tf.constant(0,tf.int32)
    return tf.while_loop(condition, body, [indices,centroids,i])

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
        print("coloR",colours[i])
        plt.scatter(samples[:, 0], samples[:, 1], c=colours[i])
        # Also plot centroid
        plt.plot(centroid[0], centroid[1], markersize=35, marker="x", color='k', mew=10)
        plt.plot(centroid[0], centroid[1], markersize=30, marker="x", color='m', mew=5)
    plt.show()


if __name__ == '__main__':
    n_features = 2
    n_clusters = 3
    n_samples_per_cluster = 5
    seed = 700
    embiggen_factor = 70

    np.random.seed(seed)

    #data
    data_centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)

    #choose centroids randomly
    centroids = choose_random_centroids(samples, n_clusters)

    #actualize the clusters
    initial_indices = assign_to_nearest(samples, centroids)

    #initialice variables with initial values
    centroids_variable=tf.Variable(centroids)
    indices_variable=tf.Variable(initial_indices)

    assign_indices=tf.assign(indices_variable,assign_to_nearest(samples, centroids_variable))
    assign_centroid=tf.assign(centroids_variable, update_centroids(samples, indices_variable, n_clusters))


    model = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(model)
        #initialice data and clusters
        sample_values, data_centroids = session.run([samples,data_centroids])
        #initialice cluster data
        _,samples2,centroid8,initial_indices=session.run([model, samples,centroids,initial_indices])
        print("SAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAM",samples2,initial_indices,centroid8)
        session.run(model)
        for step in range(10):
            #Assignment step:
            #Update step:
            ind, cent=session.run([assign_indices, assign_centroid])
            print(ind,cent)

        #nearest_indices=session.run(nearest_indices)
        centroids=session.run(centroids_variable)
        indice = session.run(indices_variable)
        print("yeeeeeeeeeeeeeeee",initial_indices,samples2)
        partitions_values = session.run(tf.dynamic_partition(samples2, indice, n_clusters))
        partitions_values2 = session.run(tf.dynamic_partition(samples2, initial_indices, n_clusters))

        plot_clusters(centroids,partitions_values)
        plot_clusters(centroid8, partitions_values2)