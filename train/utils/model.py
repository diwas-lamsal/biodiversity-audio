import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet as BaseModel
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras import layers, losses

def create_model(cfg):
    inputs = layers.Input(shape=cfg.img_shape, dtype=tf.float32)
    x = tf.image.grayscale_to_rgb(inputs)
    x = layers.Lambda(preprocess_input, name="preprocess_input")(x)
    base_model = BaseModel(include_top=False, weights=cfg.base_model_weights, pooling="avg")
    x = base_model(x, training=False)
    x = layers.Dropout(cfg.dropout, name="top_dropout")(x)
    outputs = layers.Dense(cfg.n_label, name="logits")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=cfg.model_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr),
        loss=losses.BinaryCrossentropy(from_logits=True, label_smoothing=cfg.label_smoothing),
        metrics=['acc']
    )
    return model



