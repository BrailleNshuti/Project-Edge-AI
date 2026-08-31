package com.iu.edgeai

import android.graphics.Bitmap
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.iu.edgeai.databinding.ActivityBenchmarkBinding
import java.util.concurrent.Executors

/** Measures average on-device inference latency -- feeds the "edge" column of the report's performance table. */
class BenchmarkActivity : AppCompatActivity() {
    private lateinit var binding: ActivityBenchmarkBinding
    private lateinit var analyzer: FaceAnalyzer
    private val executor = Executors.newSingleThreadExecutor()
    private var pickedBitmap: Bitmap? = null

    private val pickPhoto = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let {
            pickedBitmap = ImageUtils.loadBitmapWithCorrectOrientation(contentResolver, it)
            binding.btnRunBenchmark.isEnabled = true
            binding.benchmarkResult.text = "Photo loaded. Ready to benchmark."
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityBenchmarkBinding.inflate(layoutInflater)
        setContentView(binding.root)
        analyzer = FaceAnalyzer(this)

        binding.btnPickForBenchmark.setOnClickListener { pickPhoto.launch("image/*") }
        binding.btnRunBenchmark.setOnClickListener { runBenchmark() }
    }

    private fun runBenchmark(passes: Int = 50) {
        val bitmap = pickedBitmap ?: return
        binding.benchmarkResult.text = "Running $passes passes..."
        binding.btnRunBenchmark.isEnabled = false
        executor.execute {
            analyzer.analyze(bitmap) // warm-up pass, excluded from the average

            var faceMs = 0L
            var ageGenderMs = 0L
            var expressionMs = 0L
            var found = 0
            repeat(passes) {
                val r = analyzer.analyze(bitmap)
                if (r.faceFound) {
                    found++
                    faceMs += r.faceDetectMs
                    ageGenderMs += r.ageGenderMs
                    expressionMs += r.expressionMs
                }
            }

            val text = if (found == 0) {
                "No face detected in the picked photo -- pick a clearer one."
            } else {
                """
                Passes: $passes (face found in $found)
                Avg face detection:      %.1f ms
                Avg age/gender inference: %.1f ms
                Avg expression inference: %.1f ms
                Avg total pipeline:       %.1f ms
                """.trimIndent().format(
                    faceMs.toDouble() / found,
                    ageGenderMs.toDouble() / found,
                    expressionMs.toDouble() / found,
                    (faceMs + ageGenderMs + expressionMs).toDouble() / found,
                )
            }
            runOnUiThread {
                binding.benchmarkResult.text = text
                binding.btnRunBenchmark.isEnabled = true
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        analyzer.close()
        executor.shutdown()
    }
}
