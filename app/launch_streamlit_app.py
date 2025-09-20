from pyngrok import ngrok
import subprocess
import os
import time
from google.colab import userdata

# Terminate open tunnels if any
print("Terminating open ngrok tunnels...")
ngrok.kill()

# Set ngrok authtoken from Colab secrets
# You need to add your NGROK_AUTH_TOKEN to Colab secrets
# Go to the '🔑' icon in the left sidebar, click 'Add new secret',
# and add a secret named 'NGROK_AUTH_TOKEN' with your ngrok authtoken as the value.
try:
    ngrok_token = userdata.get('NGROK_AUTH_TOKEN')
    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)
        print("Ngrok authtoken set from Colab secrets.")
    else:
        print("NGROK_AUTH_TOKEN not found in Colab secrets. Please add it.")
        # Fallback to direct input if preferred, but secrets are recommended
        # ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN") # Replace with your actual token if not using secrets
except Exception as e:
    print(f"Error retrieving NGROK_AUTH_TOKEN from Colab secrets: {e}")
    print("Please ensure you have added NGROK_AUTH_TOKEN to Colab secrets.")


# Set up ngrok tunnel
port = 8501
print(f"Establishing ngrok tunnel for port {port}...")
try:
    public_url = ngrok.connect(port).public_url
    print(f"Ngrok tunnel established at: {public_url}")

    # Run the Streamlit app in the background
    print("Starting Streamlit app...")
    process = subprocess.Popen(["streamlit", "run", "app.py"])

    try:
        # Keep the cell alive while the Streamlit app is running
        while process.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Streamlit app interrupted. Terminating...")
        process.terminate()
        ngrok.kill()
        print("Ngrok tunnel terminated.")
except Exception as e:
    print(f"Failed to establish ngrok tunnel: {e}")
    print("Please check your ngrok authtoken and ensure it is correctly set.")