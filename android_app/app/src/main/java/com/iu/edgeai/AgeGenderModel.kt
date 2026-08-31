package com.iu.edgeai

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter

/** Wraps age_gender_model.tflite: two heads sharing one MobileNetV2 backbone. */
class AgeGenderModel(context: Context) {
    private val interpreter: Interpreter = Interpreter(ImageUtils.loadModelFile(context, "age_gender_model.tflite"))

    data class Result(val age: Float, val genderLabel: String, val latencyMs: Long)

    fun run(faceBitmap: Bitmap): Result {
        val input = ImageUtils.bitmapToMobileNetInput(faceBitmap)
        val out0 = Array(1) { FloatArray(1) }
        val out1 = Array(1) { FloatArray(1) }
        val outputs = mapOf<Int, Any>(0 to out0, 1 to out1)

        val start = System.nanoTime()
        interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
        val latencyMs = (System.nanoTime() - start) / 1_000_000

        val v0 = out0[0][0]
        val v1 = out1[0][0]
        // The gender head is a sigmoid (always in [0,1]); the age head is an
        // unbounded regression (real faces predict well above 1.0). Whichever
        // value falls in [0,1] is gender -- this avoids depending on the
        // converter preserving Keras output-layer order.
        val (genderRaw, age) = if (v0 in 0f..1f) v0 to v1 else v1 to v0
        val genderLabel = if (genderRaw < 0.5f) "Male" else "Female"
        return Result(age = age, genderLabel = genderLabel, latencyMs = latencyMs)
    }

    fun close() = interpreter.close()
}
