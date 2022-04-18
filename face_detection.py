import tensorflow as tf
'''
def mtcnn_fun(img, min_size, factor, thresholds):
    with open('./mtcnn.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef.FromString(f.read())

    with tf.device('/cpu:0'):
        prob, landmarks, box = tf.compat.v1.import_graph_def(graph_def,
            input_map={
                'input:0': img,
                'min_size:0': min_size,
                'thresholds:0': thresholds,
                'factor:0': factor
            },
            return_elements=[
                'prob:0',
                'landmarks:0',
                'box:0']
            , name='')
    #print(box, prob, landmarks)
    return box, prob, landmarks
'''

def mtcnn_detector():
    def mtcnn_fun(img, min_size, factor, thresholds):
        with open('./mtcnn.pb', 'rb') as f:
            graph_def = tf.compat.v1.GraphDef.FromString(f.read())

        with tf.device('/cpu:0'):
            prob, landmarks, box = tf.compat.v1.import_graph_def(graph_def,
                                                                 input_map={
                                                                     'input:0': img,
                                                                     'min_size:0': min_size,
                                                                     'thresholds:0': thresholds,
                                                                     'factor:0': factor},
                                                                 return_elements=[
                                                                     'prob:0',
                                                                     'landmarks:0',
                                                                     'box:0']
                                                                 , name='')
        # print(box, prob, landmarks)
        return box, prob, landmarks

    mtcnn_fun = tf.compat.v1.wrap_function(mtcnn_fun, [
        tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.float32),
        tf.TensorSpec(shape=[3], dtype=tf.float32)
    ])
    return mtcnn_fun