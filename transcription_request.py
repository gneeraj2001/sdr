import requests
import base64
import json

def transcribe_audio(audio_file_path, api_key):
    """Send audio file directly to the AudioTranscriptionChainlet"""
    try:
        # Read and encode audio file
        with open(audio_file_path, 'rb') as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Prepare request payload
        payload = {
            "audio_data": audio_base64,
            "config": {
                "model_size": "base",  # Options: tiny, base, small, medium, large
                "language": None,      # Optional: specify language code
                "task": "transcribe"   # Options: transcribe, translate
            }
        }
        
        # Make request to the transcription chainlet
        endpoint = ""
        
        print(f"Sending request to transcription service...")
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Api-Key {api_key}"},
            json=payload
        )
        
        # Pretty print response
        result = response.json()
        print("\nTranscription Result:")
        print(json.dumps(result, indent=2))
        
        return result
        
    except FileNotFoundError:
        print(f"Error: Audio file not found at {audio_file_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {str(e)}")
    except json.JSONDecodeError:
        print("Error: Invalid JSON response")
        print("Raw response:", response.text)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    # Replace these with your values
    AUDIO_FILE = "sample.wav"  # Your audio file
    API_KEY = ""  # Your API key
    
    transcribe_audio(AUDIO_FILE, API_KEY) 
