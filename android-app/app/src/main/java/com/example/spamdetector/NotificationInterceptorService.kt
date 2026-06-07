package com.example.spamdetector

import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import android.util.Log
import android.widget.Toast
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import android.os.Handler
import android.os.Looper

class NotificationInterceptorService : NotificationListenerService() {

    private val TAG = "SpamInterceptor"
    // CHANGE THIS IP TO YOUR COMPUTER'S LOCAL IP ADDRESS BEFORE COMMITTING AND PUSHING!
    // Example: "http://192.168.1.100:8000/analyze"
    private val BACKEND_URL = "http://192.168.1.100:8000/analyze"

    override fun onNotificationPosted(sbn: StatusBarNotification) {
        val packageName = sbn.packageName
        
        if (packageName == "com.whatsapp" || packageName.contains("messaging") || packageName.contains("mms")) {
            val extras = sbn.notification.extras
            val text = extras.getCharSequence("android.text")?.toString()
            val title = extras.getString("android.title")
            
            if (text != null) {
                Log.d(TAG, "Intercepted message from $title: $text")
                analyzeMessageWithAI(text, title)
            }
        }
    }

    private fun analyzeMessageWithAI(messageText: String, senderTitle: String?) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val url = URL(BACKEND_URL)
                val connection = url.openConnection() as HttpURLConnection
                connection.requestMethod = "POST"
                connection.setRequestProperty("Content-Type", "application/json")
                connection.doOutput = true

                val jsonInputString = JSONObject().apply {
                    put("text", messageText)
                    put("app_source", "android_notification")
                }.toString()

                connection.outputStream.use { os ->
                    val input = jsonInputString.toByteArray(Charsets.UTF_8)
                    os.write(input, 0, input.size)
                }

                if (connection.responseCode == HttpURLConnection.HTTP_OK) {
                    val responseString = connection.inputStream.bufferedReader().use { it.readText() }
                    val jsonResponse = JSONObject(responseString)
                    
                    val isSpam = jsonResponse.getBoolean("is_spam")
                    
                    if (isSpam) {
                        Log.w(TAG, "🚨 SPAM DETECTED: $messageText")
                        showSpamAlert(senderTitle)
                    }
                }
                connection.disconnect()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to connect to AI Backend: ${e.message}")
            }
        }
    }

    private fun showSpamAlert(sender: String?) {
        Handler(Looper.getMainLooper()).post {
            Toast.makeText(applicationContext, "🚨 SPAM ALERT: Message from $sender flagged as FRAUD!", Toast.LENGTH_LONG).show()
        }
    }
}
