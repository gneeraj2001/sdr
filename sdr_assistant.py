import asyncio
import truss_chains as chains
import pydantic
from typing import Dict, List
import numpy as np
import io
import soundfile as sf
import base64

# Data Models
class AudioInput(pydantic.BaseModel):
    """Input format for the SDR assistant"""
    audio_data: str  # base64 encoded audio data
    config: Dict[str, str | int | float | bool] = {}  # Optional configuration with specific types

    class Config:
        arbitrary_types_allowed = True

class TranscriptionResult(pydantic.BaseModel):
    text: str
    language: str
    duration: float

class ConversationAnalysis(pydantic.BaseModel):
    sentiment: Dict[str, str | float]  # Allow both string and float values
    intents: List[str]
    key_topics: List[str]
    summary: str

class EmailContent(pydantic.BaseModel):
    subject: str
    body: str

class SDRResponse(pydantic.BaseModel):
    """Complete response from the SDR assistant"""
    transcription: TranscriptionResult
    analysis: ConversationAnalysis
    email: EmailContent

class AudioTranscriptionChainlet(chains.ChainletBase):
    """Transcribes audio using Whisper"""
    
    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(
            pip_requirements=[
                "openai-whisper",
                "torch",
                "numpy",
                "soundfile"
            ]
        ),
        compute=chains.Compute(gpu="T4")
    )

    async def run_remote(self, audio_input: AudioInput) -> TranscriptionResult:
        import whisper
        
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_input.audio_data)
            
            # Convert bytes to numpy array
            audio_io = io.BytesIO(audio_bytes)
            audio, sr = sf.read(audio_io)
            
            # Ensure float32
            audio = audio.astype(np.float32)
            if len(audio.shape) > 1:  # Convert stereo to mono
                audio = audio.mean(axis=1).astype(np.float32)
            
            # Initialize Whisper model
            model = whisper.load_model("base")
            
            # Process audio and get transcription
            result = model.transcribe(audio)
            
            return TranscriptionResult(
                text=result["text"],
                language=result["language"],
                duration=float(len(audio) / sr)
            )
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            raise

class ConversationAnalysisChainlet(chains.ChainletBase):
    """Analyzes conversation content"""
    
    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(
            pip_requirements=[
                "transformers",
                "torch",
                "accelerate"
            ]
        ),
        compute=chains.Compute(gpu="T4")
    )

    async def run_remote(self, transcript: TranscriptionResult) -> ConversationAnalysis:
        from transformers import pipeline
        
        # Initialize analyzers
        sentiment_analyzer = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Analyze text
        sentiment = sentiment_analyzer(transcript.text)[0]
        
        # Generate summary
        summary = summarizer(transcript.text, max_length=130, min_length=30)[0]["summary_text"]
        
        # Extract key topics (simplified version)
        topics = ["pricing", "features"] if "price" in transcript.text.lower() else ["general inquiry"]
        
        # Determine intent (simplified version)
        intents = []
        if "how" in transcript.text.lower(): intents.append("information_request")
        if "buy" in transcript.text.lower() or "purchase" in transcript.text.lower(): intents.append("purchase_intent")
        if not intents: intents.append("general_inquiry")
        
        return ConversationAnalysis(
            sentiment={
                "score": float(sentiment["score"]),  # Ensure it's float
                "label": str(sentiment["label"])     # Ensure it's string
            },
            intents=intents,
            key_topics=topics,
            summary=summary
        )

class EmailGenerationChainlet(chains.ChainletBase):
    """Generates follow-up emails"""
    
    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(
            pip_requirements=[
                "transformers",
                "torch",
                "accelerate"
            ]
        ),
        compute=chains.Compute(gpu="T4")
    )

    async def run_remote(self, analysis: ConversationAnalysis) -> EmailContent:
        from transformers import pipeline
        
        # Initialize text generator
        generator = pipeline("text-generation", model="gpt2")
        
        # Create prompt based on analysis
        prompt = f"""Write a follow-up email.
        Topics discussed: {', '.join(analysis.key_topics)}
        Customer sentiment: {analysis.sentiment['label']}
        Summary: {analysis.summary}
        Intent: {', '.join(analysis.intents)}
        """
        
        # Generate email content
        generated = generator(prompt, max_length=200, num_return_sequences=1)[0]["generated_text"]
        
        # Extract subject and body (simplified)
        lines = generated.split('\n')
        subject = lines[0] if lines else "Follow-up from our conversation"
        body = '\n'.join(lines[1:]) if len(lines) > 1 else generated
        
        return EmailContent(
            subject=subject,
            body=body
        )

@chains.mark_entrypoint
class SDRAssistant(chains.ChainletBase):
    """Main entrypoint for the SDR assistant"""
    
    def __init__(
        self,
        transcriber=chains.depends(AudioTranscriptionChainlet),
        analyzer=chains.depends(ConversationAnalysisChainlet),
        email_generator=chains.depends(EmailGenerationChainlet)
    ):
        self._transcriber = transcriber
        self._analyzer = analyzer
        self._email_generator = email_generator

    async def run_remote(self, audio_input: AudioInput) -> SDRResponse:
        # 1. Transcribe audio
        transcript = await self._transcriber.run_remote(audio_input)
        
        # 2. Analyze conversation
        analysis = await self._analyzer.run_remote(transcript)
        
        # 3. Generate email
        email = await self._email_generator.run_remote(analysis)
        
        # Return complete response
        return SDRResponse(
            transcription=transcript,
            analysis=analysis,
            email=email
        )

def create_test_audio():
    """Create a simple test audio file (a 3-second sine wave)"""
    # Generate a 3-second sine wave at 440 Hz
    sample_rate = 16000
    duration = 3
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # Convert to float32
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format='WAV', subtype='FLOAT')  # Specify float format
    buffer.seek(0)
    
    return buffer.read()

# Example of how to use it with requests:
"""
import requests
import base64

def send_audio_to_sdr(audio_file_path, api_key):
    # Read audio file
    with open(audio_file_path, 'rb') as f:
        audio_bytes = f.read()
    
    # Convert to base64
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # Prepare request
    payload = {
        "audio_data": audio_base64,
        "config": {}  # Optional configuration
    }
    
    # Send request
    response = requests.post(
        "https://chain-xxxxx.api.baseten.co/development/run_remote",
        headers={"Authorization": f"Api-Key {api_key}"},
        json=payload
    )
    
    return response.json()
"""

if __name__ == "__main__":
    # Test the chain locally
    with chains.run_local():
        try:
            # Option 1: Test with generated sine wave
            print("\nTesting with generated audio...")
            audio_data = create_test_audio()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Option 2: Test with real audio file (uncomment to use)
            """
            print("\nTesting with real audio file...")
            with open("your_audio.wav", "rb") as f:
                audio_data = f.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            """
            
            # Create input in the same format as HTTP requests
            test_input = AudioInput(
                audio_data=audio_base64,
                config={}  # Optional configuration
            )
            
            print("Initializing chain...")
            chain = SDRAssistant()
            
            print("Running chain...")
            result = asyncio.run(chain.run_remote(test_input))
            
            print("\nResults:")
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
            
        except Exception as e:
            print(f"Error during testing: {str(e)}")
            raise 