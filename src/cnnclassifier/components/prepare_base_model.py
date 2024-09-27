import tensorflow as tf
from cnnclassifier.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all,
                            freez_till, learning_rate):
        if freeze_all:
            model.trainable = False  # More efficient than looping each layer
        elif (freez_till is not None) and (freez_till > 0):
            for layer in model.layers[:-freez_till]:
                layer.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        predictions = tf.keras.layers.Dense(
            units=classes, activation='softmax')(flatten_in)

        full_model = tf.keras.models.Model(inputs=model.input, 
                                           outputs=predictions)
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=['accuracy']
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freez_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(
            path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        if not isinstance(path, Path):
            path = Path(path)  # Ensure path is a Path object
        path = path.with_suffix(".keras")  # Ensure the model saved as .keras
        model.save(str(path))  # Save the model as a string path
        logging.info(f"Model saved at: {path}")
