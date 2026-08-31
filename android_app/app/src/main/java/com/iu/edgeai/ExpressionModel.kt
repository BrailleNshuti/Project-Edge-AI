package com.iu.edgeai

import android.content.Context
import android.graphics.Bitmap
import org.json.JSONArray
import org.tensorflow.lite.Interpreter

/** Wraps expression_model.tflite: softmax over the classes listed in emotion_labels.json. */
class ExpressionModel(context: Context) {
    private val interpreter: Interpreter = Interpreter(ImageUtils.loadModelFile(context, "expression_model.tflite"))
    val labels: List<String> = run {
        val json = context.assets.open("emotion_labels.json").bufferedReader().use { it.readText() }
        val arr = JSONArray(json)
        (0 until arr.length()).map { arr.getString(it) }
    }

    data class Result(val label: String, val confidence: Float, val latencyMs: Long)

    fun run(faceBitmap: Bitmap): Result {
        val input = ImageUtils.bitmapToMobileNetInput(faceBitmap)
        val output = Array(1) { FloatArray(labels.size) }

        val start = System.nanoTime()
        interpreter.run(input, output)
        val latencyMs = (System.nanoTime() - start) / 1_000_000

        var bestIdx = 0
        for (i in labels.indices) if (output[0][i] > output[0][bestIdx]) bestIdx = i
        return Result(label = labels[bestIdx], confidence = output[0][bestIdx], latencyMs = latencyMs)
    }

    fun close() = interpreter.close()
}
