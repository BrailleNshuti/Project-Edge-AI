package com.iu.edgeai

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.documentfile.provider.DocumentFile
import com.iu.edgeai.databinding.ActivityEvaluateBinding
import java.util.concurrent.Executors

private const val ELDERLY_AGE_THRESHOLD = 60f

/**
 * Scores the 20 self-collected evaluation photos against evaluation/LABELING_TEMPLATE.csv
 * (placed as labels.csv next to the photos) and reports per-subgroup accuracy, matching the
 * brief's 50/50 adult-elderly / male-female / happy-sad evaluation requirement.
 */
class EvaluateActivity : AppCompatActivity() {
    private lateinit var binding: ActivityEvaluateBinding
    private lateinit var analyzer: FaceAnalyzer
    private val executor = Executors.newSingleThreadExecutor()
    private var folderUri: Uri? = null

    private data class GroundTruth(val ageGroup: String, val gender: String, val expression: String)
    private data class Row(val filename: String, val gt: GroundTruth, val predAge: String, val predGender: String, val predExpr: String)

    private val pickFolder = registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
        if (uri != null) {
            contentResolver.takePersistableUriPermission(
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION,
            )
            folderUri = uri
            binding.btnRunEvaluation.isEnabled = true
            binding.evaluateResult.text = "Folder selected. Ready to evaluate."
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityEvaluateBinding.inflate(layoutInflater)
        setContentView(binding.root)
        analyzer = FaceAnalyzer(this)

        binding.btnPickFolder.setOnClickListener { pickFolder.launch(null) }
        binding.btnRunEvaluation.setOnClickListener { runEvaluation() }
    }

    private fun runEvaluation() {
        val uri = folderUri ?: return
        binding.evaluateResult.text = "Running evaluation..."
        binding.btnRunEvaluation.isEnabled = false
        executor.execute {
            val root = DocumentFile.fromTreeUri(this, uri)
            val files = root?.listFiles()
            if (root == null || files == null) return@execute showError("Could not open folder.")

            val labelsFile = files.firstOrNull { it.name == "labels.csv" }
                ?: return@execute showError("labels.csv not found in the selected folder. See evaluation/LABELING_TEMPLATE.csv.")

            val groundTruth = mutableMapOf<String, GroundTruth>()
            contentResolver.openInputStream(labelsFile.uri)?.bufferedReader()?.useLines { lines ->
                lines.drop(1).forEach { line ->
                    if (line.isBlank()) return@forEach
                    val parts = line.split(",").map { it.trim() }
                    if (parts.size >= 4) groundTruth[parts[0]] = GroundTruth(parts[1], parts[2], parts[3])
                }
            }

            val rows = mutableListOf<Row>()
            for (file in files) {
                val name = file.name ?: continue
                val gt = groundTruth[name] ?: continue
                val bitmap = ImageUtils.loadBitmapWithCorrectOrientation(contentResolver, file.uri) ?: continue
                val result = analyzer.analyze(bitmap)
                if (!result.faceFound) {
                    rows.add(Row(name, gt, "no-face", "no-face", "no-face"))
                    continue
                }
                val predAgeGroup = if (result.age >= ELDERLY_AGE_THRESHOLD) "elderly" else "adult"
                rows.add(Row(name, gt, predAgeGroup, result.genderLabel, result.expressionLabel))
            }

            if (rows.isEmpty()) return@execute showError("No images in the folder matched a row in labels.csv.")

            writeResultsCsv(root, rows)

            val summary = buildString {
                appendLine("Evaluated ${rows.size} of ${groundTruth.size} labeled images.\n")
                appendSubgroupAccuracy("Age group", rows, { it.gt.ageGroup }) { it.predAge.equals(it.gt.ageGroup, true) }
                appendSubgroupAccuracy("Gender", rows, { it.gt.gender }) { it.predGender.equals(it.gt.gender, true) }
                appendSubgroupAccuracy("Expression", rows, { it.gt.expression }) { it.predExpr.equals(it.gt.expression, true) }
                append("\nFull per-image results written to results.csv in the selected folder.")
            }

            runOnUiThread {
                binding.evaluateResult.text = summary
                binding.btnRunEvaluation.isEnabled = true
            }
        }
    }

    private fun StringBuilder.appendSubgroupAccuracy(
        label: String,
        rows: List<Row>,
        groupOf: (Row) -> String,
        matches: (Row) -> Boolean,
    ) {
        val overall = rows.count(matches).toDouble() / rows.size * 100.0
        appendLine("$label accuracy (overall: %.1f%%)".format(overall))
        rows.groupBy(groupOf).toSortedMap().forEach { (group, groupRows) ->
            val acc = groupRows.count(matches).toDouble() / groupRows.size * 100.0
            appendLine("  $group: %.1f%% (n=${groupRows.size})".format(acc))
        }
        appendLine()
    }

    private fun writeResultsCsv(root: DocumentFile, rows: List<Row>) {
        val csv = StringBuilder("filename,gt_age_group,pred_age_group,gt_gender,pred_gender,gt_expression,pred_expression\n")
        for (r in rows) {
            csv.append("${r.filename},${r.gt.ageGroup},${r.predAge},${r.gt.gender},${r.predGender},${r.gt.expression},${r.predExpr}\n")
        }
        root.findFile("results.csv")?.delete()
        root.createFile("text/csv", "results.csv")?.let { f ->
            contentResolver.openOutputStream(f.uri)?.use { it.write(csv.toString().toByteArray()) }
        }
    }

    private fun showError(message: String) {
        runOnUiThread {
            binding.evaluateResult.text = message
            binding.btnRunEvaluation.isEnabled = true
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        analyzer.close()
        executor.shutdown()
    }
}
