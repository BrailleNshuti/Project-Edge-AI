"""Converts both saved Keras models to TFLite and smoke-tests each one.

The smoke test (load with tf.lite.Interpreter + run one dummy inference)
catches converter/runtime opset mismatches -- e.g. "Didn't find op for
builtin opcode 'FULLY_CONNECTED' version 'X'" -- immediately, in Python,
before the model ever touches the Android app. That specific error means
the .tflite file was produced by a newer TFLite converter than the runtime
(the app's TFLite dependency version) understands; keeping the training
TensorFlow version and the app's org.tensorflow:tensorflow-lite version
aligned (see android_app README) avoids it entirely.

Usage:
  python convert_to_tflite.py [--model age_gender|expression|both]
"""
import argparse
import pathlib

import numpy as np
import tensorflow as tf

ROOT = pathlib.Path(__file__).resolve().parent
MODELS_DIR = ROOT / "saved_models"
TFLITE_DIR = ROOT / "tflite"

IMG_SIZE = 96

MODEL_SPECS = {
    "age_gender": {"keras_path": MODELS_DIR / "age_gender_model.keras", "tflite_name": "age_gender_model.tflite"},
    "expression": {"keras_path": MODELS_DIR / "expression_model.keras", "tflite_name": "expression_model.tflite"},
}


def convert_and_smoke_test(keras_path: pathlib.Path, tflite_name: str) -> None:
    if not keras_path.exists():
        raise FileNotFoundError(f"{keras_path} not found -- train the model first.")

    model = tf.keras.models.load_model(keras_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic-range quantization: smaller + faster on edge
    tflite_bytes = converter.convert()

    TFLITE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TFLITE_DIR / tflite_name
    out_path.write_bytes(tflite_bytes)
    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote {out_path} ({size_kb:.1f} KB)")

    # --- smoke test: reload + run one dummy inference ---
    interpreter = tf.lite.Interpreter(model_path=str(out_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    dummy_input = np.zeros(input_details[0]["shape"], dtype=input_details[0]["dtype"])
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()
    outputs = [interpreter.get_tensor(od["index"]) for od in output_details]

    print(f"  Smoke test OK. Output shapes: {[o.shape for o in outputs]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["age_gender", "expression", "both"], default="both")
    args = parser.parse_args()
    names = ["age_gender", "expression"] if args.model == "both" else [args.model]
    for name in names:
        spec = MODEL_SPECS[name]
        convert_and_smoke_test(spec["keras_path"], spec["tflite_name"])
