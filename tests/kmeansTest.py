import tensorflow as tf
from venv.main.unsupervised.Kmeans import kmeans as kmean


class kmeansTest(tf.test.TestCase):

    def testNearestAssignation(self):
        with self.test_session():
            samples=tf.constant([[1,20],[3,23],[5,27],[7,8],[9,10],[11,12],[13,1],[15,2],[17,3]])
            centroids=tf.constant([[0,20],[9,10],[20,2]])
            x = kmean.assign_to_nearest(samples,centroids)
            self.assertAllEqual(x.eval(), [0,0,0,1,1,1,2,2,2])

    def testNearestAssignation2(self):
        with self.test_session():
            samples=tf.constant([[1.,20.],[3.,31.],[5.,27.],[7.,8.],[9.,10.],[11.,12.],[13.,1.],[15.,2.],[17.,3.]])
            centroids=tf.Variable([[3.,-26.],[9.,1.],[1.,2.]])
            self.run(tf.global_variables_initializer())
            x = kmean.assign_to_nearest(samples,centroids)
            print("esooooooooooo",x.eval(),[samples[0]])
            o = tf.scatter_update(centroids, x, [samples[0]])
            print(o.eval())
            a=tf.constant([0,0,0,1,1,1,2,2,2],tf.int64)
            g=tf.concat([x, a], 0)
            a_vecs = tf.unstack(centroids, axis=0)
            print(a_vecs[1].eval())
            def function(x):
                del a_vecs[x]
                return a_vecs
            tf.map_fn(lambda a: function(a),x)
            a_new = tf.stack(a_vecs, 0)

            print(a_new.eval())
            self.assertAllEqual(x.eval(), [0,0,0,1,1,1,2,2,2])


    def testCentroidUpdate(self):
        with self.test_session():
            samples=tf.constant([[1.,20.],[3.,31.],[5.,27.],[7.,8.],[9.,10.],[11.,12.],[13.,1.],[15.,2.],[17.,3.]])
            nearest_indices=tf.constant([0,0,0,1,1,1,2,2,2])
            x = kmean.update_centroids(samples,nearest_indices,3)
            print(x.eval())
            self.assertAllEqual(x.eval(), [[3.,26.],[9.,10.],[15.,2.]])

    def testCentroidUpdate2(self):
        with self.test_session():
            samples=tf.constant([[1.,20.],[3.,31.],[5.,27.],[7.,8.],[9.,10.],[11.,12.],[13.,1.],[15.,2.],[17.,3.]])
            nearest_indices=tf.constant([1, 1, 0, 1, 1, 1, 2, 2, 2])
            x = kmean.update_centroids(samples,nearest_indices,3)
            print(x.eval())
            self.assertAllEqual(x.eval(), [[3.,26.],[9.,10.],[15.,2.]])


    def testKMeansIterations(self):
        with self.test_session():
            samples = tf.constant([[1., 20.], [3., 31.], [5., 27.], [7., 8.], [9., 10.], [11., 12.], [13., 1.], [15., 2.], [17., 3.]])
            nearest_indices = tf.constant([1, 1, 0, 1, 1, 1, 2, 2, 2])
            initial_centroids= tf.constant([[3.,-26.],[9.,1.],[1.,2.]])
            x,y,z = kmean.kmneansItertions(samples, nearest_indices, initial_centroids,3)
            tf.global_variables_initializer().run()
            print(x.eval(),y.eval(),z.eval())
            self.assertAllEqual(x.eval(), [0,0,0,1,1,1,2,2,2])


if __name__ == '__main__':
    tf.test.main()