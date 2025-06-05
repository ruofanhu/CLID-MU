import tensorflow as tf
tfrecord_path = '/content/data'

feature_description = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/class/label': tf.io.FixedLenFeature([1], tf.int64),
    'image/class/fine_label': tf.io.FixedLenFeature([1], tf.int64),
}
def parse_tfrecord(example_proto):
    # Parse the input tf.train.Example proto
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    # Decode the image
    image = tf.io.decode_raw(parsed_features['image/encoded'], tf.uint8)
    image = tf.reshape(image, [32, 32, 3])  # Assuming 32x32 RGB images
    
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]  
    # image = tf.cast(image, tf.float32)   # Normalize to [0, 1]   
 
    # Extract labels
    label = tf.cast(parsed_features['image/class/label'], tf.int32)
    fine_label = tf.cast(parsed_features['image/class/fine_label'], tf.int32)
    
    return image, label, fine_label
# Path to the TFRecord file
tfrecord_path = 'data'

# Load the dataset
dataset = tf.data.TFRecordDataset(tfrecord_path)

# Parse the dataset
parsed_dataset = dataset.map(parse_tfrecord)


