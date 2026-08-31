package com.iu.edgeai

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.iu.edgeai.databinding.ActivityMainBinding
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var analyzer: FaceAnalyzer
    private val executor = Executors.newSingleThreadExecutor()

    private val takePhoto = registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap ->
        if (bitmap != null) onPhotoReady(bitmap)
    }

    private val pickPhoto = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let { loadBitmap(it)?.let { bmp -> onPhotoReady(bmp) } }
    }

    private val requestCameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) takePhoto.launch(null) else {
            Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        analyzer = FaceAnalyzer(this)

        binding.btnTakePhoto.setOnClickListener {
            requestCameraPermission.launch(android.Manifest.permission.CAMERA)
        }
        binding.btnPickPhoto.setOnClickListener { pickPhoto.launch("image/*") }
        binding.btnBenchmark.setOnClickListener { startActivity(Intent(this, BenchmarkActivity::class.java)) }
        binding.btnEvaluate.setOnClickListener { startActivity(Intent(this, EvaluateActivity::class.java)) }
    }

    private fun loadBitmap(uri: Uri): Bitmap? =
        ImageUtils.loadBitmapWithCorrectOrientation(contentResolver, uri)

    private fun onPhotoReady(bitmap: Bitmap) {
        binding.imagePreview.setImageBitmap(bitmap)
        binding.resultText.text = "Analyzing..."
        executor.execute {
            val result = analyzer.analyze(bitmap)
            runOnUiThread { showResult(result) }
        }
    }

    private fun showResult(result: FaceAnalyzer.AnalysisResult) {
        binding.resultText.text = if (!result.faceFound) {
            "No face detected."
        } else {
            """
            Age: %.1f
            Gender: %s
            Expression: %s

            Latency -- detect: %dms  age/gender: %dms  expression: %dms
            """.trimIndent().format(
                result.age, result.genderLabel, result.expressionLabel,
                result.faceDetectMs, result.ageGenderMs, result.expressionMs,
            )
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        analyzer.close()
        executor.shutdown()
    }
}
