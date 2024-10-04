import numpy as np
import tensorflow as tf
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.class_labels = {
            0: 'Early Blight Disease',
            1: 'Late Blight Disease',
            2: 'Healthy'
        }

    def predict(self):
        # Load the model
        try:
            model = tf.keras.models.load_model(
                os.path.join("artifacts", "training", "model.keras"))
        except Exception as e:
            return [{"error": str(e)}]

        # Preprocess the input image
        imagename = self.filename
        test_image = tf.keras.preprocessing.image.load_img(
            imagename, target_size=(224, 224))
        test_image = tf.keras.preprocessing.image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image /= 255.0  # Normalize if needed

        # Predict the class
        result = np.argmax(model.predict(test_image), axis=1)

        # Map the result to class labels
        prediction = self.class_labels.get(result[0], 'Unknown')

        return [{"image shows": prediction}]

# import numpy as np
# from tensorflow.keras.models import load_model  # type: ignore
# from tensorflow.keras.preprocessing import image  # type: ignore
# import os


# class PredictionPipeline:
#     def __init__(self, filename):
#         self.filename = filename

#     def predict(self):
#         # Load the model
#         model = load_model(os.path.join(
#             "artifacts", "training", "model.keras"))

#         # Load and preprocess the image
#         imagename = self.filename
#         test_image = image.load_img(imagename, target_size=(224, 224))
#         test_image = image.img_to_array(test_image)
#         test_image = np.expand_dims(test_image, axis=0)

#         # Predict the class
#         result = np.argmax(model.predict(test_image), axis=1)
#         print(result)

#         # Define the class mapping
#         class_labels = {0: 'Early Blight', 1: 'Late Blight', 2: 'Healthy'}

#         # Return the prediction
#         prediction = class_labels.get(result[0], "Unknown")
#         return [{"image": prediction}]
