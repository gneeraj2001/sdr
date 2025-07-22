import requests
import base64
import json

def send_audio_request(audio_file_path, api_key, endpoint_url):
    """Send audio file to Baseten endpoint"""
    try:
        # Read and encode audio file
        with open(audio_file_path, 'rb') as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Prepare request payload
        payload = {
            "audio_data": audio_base64,
            "config": {}
        }
        
        # Make request
        print(f"Sending request to {endpoint_url}")
        response = requests.post(
            endpoint_url,
            headers={"Authorization": f"Api-Key {api_key}"},
            json=payload
        )
        
        # Pretty print response
        result = response.json()
        print("\nResponse:")
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
    AUDIO_FILE = "sample.wav"  # Path to your audio file
    API_KEY = "BpjRmzHc.Vv3stAS1sTO5Bjzq3fu60zOVUeIpWpZ7"  # Your Baseten API key
    ENDPOINT = "https://chain-yqvv7eq8.api.baseten.co/development/run_remote"  # Your endpoint URL
    
    send_audio_request(AUDIO_FILE, API_KEY, ENDPOINT) 