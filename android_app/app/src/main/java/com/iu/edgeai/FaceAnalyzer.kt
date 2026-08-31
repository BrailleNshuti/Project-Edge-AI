package com.iu.edgeai

import android.content.Context
import android.graphics.Bitmap
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions

private const val FACE_INPUT_SIZE = 96

/** Combines on-device face detection with the two custom TFLite classifiers. Blocking -- call off the main thread. */
class FaceAnalyzer(context: Context) {
    private val detector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .build()
    )
    private val ageGenderModel = AgeGenderModel(context)
    private val expressionModel = ExpressionModel(context)

    data class AnalysisResult(
        val faceFound: Boolean,
        val age: Float = 0f,
        val genderLabel: String = "",
        val expressionLabel: String = "",
        val faceDetectMs: Long = 0,
        val ageGenderMs: Long = 0,
        val expressionMs: Long = 0,
    )

    fun analyze(bitmap: Bitmap): AnalysisResult {
        val image = InputImage.fromBitmap(bitmap, 0)
        val detectStart = System.nanoTime()
        val faces = Tasks.await(detector.process(image))
        val faceDetectMs = (System.nanoTime() - detectStart) / 1_000_000

        val face = faces.maxByOrNull { it.boundingBox.width().toLong() * it.boundingBox.height() }
            ?: return AnalysisResult(faceFound = false, faceDetectMs = faceDetectMs)

        val faceCrop = ImageUtils.cropAndResize(bitmap, face.boundingBox, FACE_INPUT_SIZE)
        val ageGender = ageGenderModel.run(faceCrop)
        val expression = expressionModel.run(faceCrop)

        return AnalysisResult(
            faceFound = true,
            age = ageGender.age,
            genderLabel = ageGender.genderLabel,
            expressionLabel = expression.label,
            faceDetectMs = faceDetectMs,
            ageGenderMs = ageGender.latencyMs,
            expressionMs = expression.latencyMs,
        )
    }

    fun close() {
        detector.close()
        ageGenderModel.close()
        expressionModel.close()
    }
}
