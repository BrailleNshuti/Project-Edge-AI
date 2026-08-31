package com.iu.edgeai

import android.content.ContentResolver
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Rect
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

object ImageUtils {

    /** Memory-maps a .tflite file straight out of assets/ -- avoids depending on tensorflow-lite-support. */
    fun loadModelFile(context: Context, filename: String): MappedByteBuffer {
        val afd = context.assets.openFd(filename)
        FileInputStream(afd.fileDescriptor).use { input ->
            return input.channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
        }
    }

    /**
     * Decodes [uri] and rotates/flips it to match its EXIF orientation tag.
     * Gallery photos and files picked via Storage Access Framework are commonly
     * stored with landscape pixel data plus a rotation tag rather than
     * pre-rotated pixels -- BitmapFactory alone ignores that tag, which silently
     * feeds the face detector sideways/upside-down photos. Use this instead of
     * a bare BitmapFactory.decodeStream() for any photo coming from a Uri.
     */
    fun loadBitmapWithCorrectOrientation(resolver: ContentResolver, uri: Uri): Bitmap? {
        val bitmap = resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) } ?: return null
        val orientation = resolver.openInputStream(uri)?.use {
            ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        } ?: ExifInterface.ORIENTATION_NORMAL

        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
            else -> return bitmap
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    /** Crops [rect] out of [bitmap] (clamped to bounds) and scales it to size x size. */
    fun cropAndResize(bitmap: Bitmap, rect: Rect, size: Int): Bitmap {
        val left = rect.left.coerceIn(0, bitmap.width - 1)
        val top = rect.top.coerceIn(0, bitmap.height - 1)
        val right = rect.right.coerceIn(left + 1, bitmap.width)
        val bottom = rect.bottom.coerceIn(top + 1, bitmap.height)
        val cropped = Bitmap.createBitmap(bitmap, left, top, right - left, bottom - top)
        return Bitmap.createScaledBitmap(cropped, size, size, true)
    }

    /**
     * Converts a size x size ARGB bitmap into a float32 NHWC ByteBuffer scaled to [-1, 1],
     * matching tf.keras.applications.mobilenet_v2.preprocess_input used during training.
     */
    fun bitmapToMobileNetInput(bitmap: Bitmap): ByteBuffer {
        val size = bitmap.width
        val buffer = ByteBuffer.allocateDirect(4 * size * size * 3)
        buffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(size * size)
        bitmap.getPixels(pixels, 0, size, 0, 0, size, size)
        for (pixel in pixels) {
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF
            buffer.putFloat((r / 127.5f) - 1f)
            buffer.putFloat((g / 127.5f) - 1f)
            buffer.putFloat((b / 127.5f) - 1f)
        }
        buffer.rewind()
        return buffer
    }
}
