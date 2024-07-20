import tensorflow as tf
import numpy as np

class Scan_sage:
    def __init__(self):
        self.model = self.load_model()
        print("model made")

    def load_model(self):
        try:
            # Load your trained TensorFlow/Keras model
            print("inside the load model")
            model = tf.keras.models.load_model('pneumonia_detection_model.h5')  # Adjust path accordingly
            print("model is created")
            return model
        except FileNotFoundError:
            raise FileNotFoundError("Model file not found. Please check the path.")

    def preprocess_image(self, image_path):
        # Example method to preprocess an image
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image / 255.0  # Normalize pixel values
        return image

    def process_image(self, image_path):
        # Example method to process an X-ray image
        image = self.preprocess_image(image_path)
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make predictions using the loaded model
        predictions = self.model.predict(image)

        # Example: Interpret model predictions
        if predictions[0] > 0.5:
            return "Positive result: Detected disease"
        else:
            return "Negative result: No disease detected"

# Example usage:
if __name__ == "__main__":
    scan = Scan_sage()
    image_path = 'path_to_your_image.jpg'  # Replace with your X-ray image path
    result = scan.process_image(image_path)
    print("Result:", result)