import tensorflow as tf
import numpy as np
import tensorflow_io as tfio
import keras
import keras_cv

from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks
from .model import create_model
from tqdm.keras import TqdmCallback

AUTOTUNE = tf.data.AUTOTUNE

def show_img_stats(img):
    if isinstance(img, tf.Tensor):
        print((img.shape, img.dtype, img.numpy().min(), img.numpy().max()))
    elif isinstance(img, np.array):
        print((img.shape, img.dtype, img.min(), img.max()))
    else:
        print(f"unexpected type: {type(img)}")


def read_image(img_shape, channels, path_img):
    img_data = tf.io.read_file(path_img)
    img = tf.io.decode_jpeg(img_data, channels=channels)
    img = tf.reshape(img, img_shape)
    img = tf.cast(img, tf.float32)
    return img

def decode_label(n_label, label):
    return tf.one_hot(label, depth=n_label)


class RandomRowMask(keras_cv.layers.BaseImageAugmentationLayer):
    def __init__(self, param=10, num_mask=1, **kwargs):
        super().__init__(**kwargs)
        self.param = param
        self.num_mask = num_mask

    def augment_image(self, image, transformation=None, **kwargs):
        image_shape = tf.shape(image)
        num = self._random_generator.random_uniform((), 1, self.num_mask, dtype=tf.int32)
        for _ in tf.range(num):
            image = tfio.audio.time_mask(tf.squeeze(image), param=self.param)
            image = tf.reshape(image, shape=image_shape)
        return image
    
    
class RandomColumnMask(keras_cv.layers.BaseImageAugmentationLayer):
    def __init__(self, param=10, num_mask=1, **kwargs):
        super().__init__(**kwargs)
        self.param = param
        self.num_mask = num_mask

    def augment_image(self, image, transformation=None, **kwargs):
        image_shape = tf.shape(image)
        num = self._random_generator.random_uniform((), 1, self.num_mask, dtype=tf.int32)
        for _ in tf.range(num):        
            image = tfio.audio.freq_mask(tf.squeeze(image), param=self.param)
            image = tf.reshape(image, shape=image_shape)
        return image


def get_data_augmenter():
    augmenter = keras_cv.layers.Augmenter(
        layers=[
            keras_cv.layers.RandomBrightness(factor=0.2),
            keras.layers.RandomContrast(factor=0.2),
            keras_cv.layers.GridMask(ratio_factor=(0.05, 0.10)),
            keras_cv.layers.RandomGaussianBlur(kernel_size=2, factor=0.1),
            RandomRowMask(10, 3),
            RandomColumnMask(40, 2)
        ]
    )
    return augmenter


def augment_image(aug_proba, img, augmenter):
    if tf.random.uniform([]) <= aug_proba:
        img = augmenter(img)
    return img


def create_dataset(cfg, data, include_label=True, repeat=False, shuffle=False, augment=False, prefetch=False, batch_size=None):
    slices = data["path_img"].values
    read_func = read_image
    aug_func = augment_image
    augmenter = get_data_augmenter()
    if include_label:
        slices = slices, decode_label(cfg.n_label, data[cfg.label].values)
        read_func = lambda path_img, label: (read_image(cfg.img_shape, cfg.channels, path_img), label)
        aug_func = lambda img, label: (augment_image(cfg.aug_proba, img, augmenter), label)
    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.map(read_func, num_parallel_calls=AUTOTUNE)
    if repeat: ds = ds.repeat()
    if shuffle: ds = ds.shuffle(buffer_size=cfg.shuffle_size)
    if augment: ds = ds.map(aug_func, num_parallel_calls=AUTOTUNE)
    if batch_size: ds = ds.batch(batch_size)
    if prefetch: ds = ds.prefetch(AUTOTUNE)
    return ds


def create_training_dataset(cfg, data):
    return create_dataset(
        cfg,
        data,
        include_label=True,
        repeat=True,
        shuffle=True,
        augment=True,
        prefetch=True,
        batch_size=cfg.batch_size,
    )

def create_validation_dataset(cfg, data):
    return create_dataset(
        cfg,
        data,
        include_label=True,
        repeat=False,
        shuffle=False,
        augment=False,
        prefetch=True,
        batch_size=cfg.valid_batch_size,
    )

# TODO: Modify the train_test_split to be based on file name instead
def get_train_test_split(cfg, data):
    train_df, valid_df = train_test_split(data, test_size=cfg.test_size, stratify=data[cfg.label])
    print(f"Split: {len(train_df)} vs {len(valid_df)}")
    return train_df, valid_df

def get_strategy():
    return tf.distribute.MirroredStrategy()

def get_callbacks(cfg, filepath):
    """Get callbacks"""
    cbs = [
        callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor=cfg.monitor,
            mode=cfg.monitor_mode,
            verbose=1,
            save_best_only=True,
            save_weights_only=True            
        ),
        callbacks.EarlyStopping(
            monitor=cfg.monitor,
            mode=cfg.monitor_mode,
            verbose=1,
            patience=cfg.patience,
            restore_best_weights=False,
        ),
        TqdmCallback(verbose=2),
    ]
    return cbs


def show_history(history):
    """Show history"""
    history_frame = pd.DataFrame(history.history)
    history_frame.index = pd.RangeIndex(1, len(history_frame) + 1, name="epoch")
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    history_frame.loc[:, ['loss', 'val_loss']].plot(ax=ax[0], title='loss')
    history_frame.loc[:, ['acc', 'val_acc']].plot(ax=ax[1], title='acc')
    plt.tight_layout()
    plt.show()
    
    
def compute_oof(cfg, model, valid_ds, valid_df):
    """Compute OOF"""
    oof_pred = model.predict(valid_ds, verbose=False)
    oof_pred = pd.DataFrame(tf.nn.sigmoid(oof_pred).numpy(), index=valid_df.index)
    oof = pd.concat({"y_true": valid_df[cfg.label], "y_pred": oof_pred}, axis=1)
    return oof    



def run_training(cfg, train_df, valid_df, model_name):
    """Run training"""
    # prepare dataset
    train_ds = create_training_dataset(cfg, train_df)
    valid_ds = create_validation_dataset(cfg, valid_df)
    # create model
    tf.keras.backend.clear_session()
    strategy = get_strategy()
    with strategy.scope():
        model = create_model(cfg)
    # fit
    steps_per_epoch = cfg.steps_per_epoch
    print("steps_per_epoch:", steps_per_epoch)
    path_weight = f"./working/weights_{model_name}.h5"
    print("path_weights:", path_weight)
    hist = model.fit(
        train_ds,
        epochs=cfg.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_ds,
        callbacks=get_callbacks(cfg, path_weight),
        verbose=cfg.fit_verbose
    )
    # restore
    model.load_weights(path_weight)
    oof = compute_oof(cfg, model, valid_ds, valid_df)
    return hist, oof