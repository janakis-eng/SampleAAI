import tensorflow as tf

def build_resnet(input_shape, num_classes):
    base = tf.keras.applications.ResNet50(
        include_top=False,
        input_shape=input_shape,
        pooling='avg',
        weights='imagenet'
    )
    base.trainable = False

    model = tf.keras.Sequential([
        base,
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    return model