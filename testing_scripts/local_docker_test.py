import websocket
import json
import base64

def run_test():
    """Connects to the local Docker container and sends a test audio payload."""
    # This is a sample base64 string of a silent WAV file.
    # In a real test, you'd encode one of your test audio files.
    silent_audio_b64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAAABkYXRhAAAAAA=="
    
    ws_url = "ws://localhost:8000/listen"
    print(f"Connecting to {ws_url}...")
    
    try:
        ws = websocket.create_connection(ws_url)
        print("Connection successful. Sending test audio...")
        
        # This payload mimics what Exotel would send
        payload = {"audio": silent_audio_b64}
        ws.send(json.dumps(payload))
        
        print("Payload sent. Waiting for response...")
        response = ws.recv()
        print("Received response from agent:")
        print(json.loads(response))
        
        ws.close()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_test()
