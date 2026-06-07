package com.example.spamdetector

import android.content.Intent
import android.os.Bundle
import android.provider.Settings
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        val layout = android.widget.LinearLayout(this)
        layout.orientation = android.widget.LinearLayout.VERTICAL
        layout.setPadding(50, 50, 50, 50)

        val titleText = TextView(this)
        titleText.text = "Spam Guard Active 🛡️"
        titleText.textSize = 24f
        titleText.setPadding(0, 0, 0, 50)
        
        val descText = TextView(this)
        descText.text = "This app silently monitors incoming messages (WhatsApp, SMS) in the background.\n\nIf a message is predicted as spam by your ML model, it will automatically alert you!"
        descText.textSize = 16f
        descText.setPadding(0, 0, 0, 50)

        val permissionButton = Button(this)
        permissionButton.text = "Enable Notification Access"
        permissionButton.setOnClickListener {
            val intent = Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS)
            startActivity(intent)
        }

        layout.addView(titleText)
        layout.addView(descText)
        layout.addView(permissionButton)

        setContentView(layout)
    }
}
