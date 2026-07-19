package com.faceswaplocal.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.faceswaplocal.app.ui.FaceSwapRoute
import com.faceswaplocal.app.ui.theme.FaceSwapLocalTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            FaceSwapLocalTheme {
                FaceSwapRoute()
            }
        }
    }
}

