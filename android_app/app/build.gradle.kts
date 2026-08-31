plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.iu.edgeai"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.iu.edgeai"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        viewBinding = true
    }
    // TFLite files in assets/ must not be compressed at build time, or the
    // interpreter cannot mmap() them directly on-device.
    androidResources {
        noCompress += "tflite"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // On-device face detection (crops the face before classification)
    implementation("com.google.mlkit:face-detection:16.1.7")

    // On-device inference for our two custom models
    implementation("org.tensorflow:tensorflow-lite:2.17.0")

    // Folder access for Evaluate mode (labels.csv + 20 photos -> results.csv)
    implementation("androidx.documentfile:documentfile:1.0.1")

    // Reads EXIF orientation so gallery/folder photos aren't fed to the face
    // detector sideways (phones store many photos rotated + tagged, not
    // pre-rotated pixel data).
    implementation("androidx.exifinterface:exifinterface:1.3.7")

    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.8.4")
}
