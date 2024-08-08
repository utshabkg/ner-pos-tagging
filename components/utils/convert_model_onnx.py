import tensorflow as tf
import tf2onnx


def convert_model_to_onnx(model_path, onnx_model_path):
    model = tf.keras.models.load_model(model_path)
    spec = (tf.TensorSpec(model.inputs[0].shape, tf.float32, name="input"),)

    output_path = onnx_model_path
    model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=13)

    with open(output_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    print(f"Model has been converted to ONNX and saved at {onnx_model_path}")


if __name__ == "__main__":
    model_path = "../../notebooks/models_evaluation/models/base_model.h5"
    onnx_model_path = "../../notebooks/models_evaluation/models/base_model.onnx"
    convert_model_to_onnx(model_path, onnx_model_path)
