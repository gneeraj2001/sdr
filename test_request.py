import base64
import asyncio
from sdr_assistant import SDRAssistant, AudioInput, chains

def prepare_audio_request(audio_file_path):
    """Prepare audio data exactly as it would be sent in an HTTP request"""
    with open(audio_file_path, 'rb') as f:
        audio_bytes = f.read()
    
    # Convert to base64 - exactly as it would be sent in the request
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # Create the same payload structure as the HTTP request
    return AudioInput(
        audio_data=audio_base64,
        config={}  # You can add configuration here if needed
    )

async def test_local_request(audio_file_path):
    """Test the chain with the same format as HTTP request"""
    try:
        # Prepare request data
        request_data = prepare_audio_request(audio_file_path)
        
        # Initialize chain
        chain = SDRAssistant()
        
        print(f"\nProcessing audio file: {audio_file_path}")
        
        # Run chain - this is what would happen on the server
        result = await chain.run_remote(request_data)
        
        # Print results
        print("\nResults (this is what you'll get in the HTTP response):")
        print("\nTranscription:")
        print(result.transcription.text)
        print("\nAnalysis:")
        print(f"- Sentiment: {result.analysis.sentiment}")
        print(f"- Intents: {result.analysis.intents}")
        print(f"- Key Topics: {result.analysis.key_topics}")
        print(f"- Summary: {result.analysis.summary}")
        print("\nGenerated Email:")
        print(f"Subject: {result.email.subject}")
        print(f"Body: {result.email.body}")
        
    except FileNotFoundError:
        print(f"Error: Audio file not found at {audio_file_path}")
    except Exception as e:
        print(f"Error processing request: {str(e)}")

if __name__ == "__main__":
    # Test with your audio file
    with chains.run_local():
        audio_path = "your_audio.wav"  # Replace with your audio file path
        asyncio.run(test_local_request(audio_path)) 