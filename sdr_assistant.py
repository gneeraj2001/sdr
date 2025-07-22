import asyncio
import truss_chains as chains
import pydantic
from typing import Dict, List
import base64

class AudioInput(pydantic.BaseModel):
    """Input format for the SDR assistant"""
    audio_data: str  # base64 encoded audio data
    config: Dict[str, str | int | float | bool] = {}

    class Config:
        arbitrary_types_allowed = True

class TranscriptionResult(pydantic.BaseModel):
    text: str
    language: str
    duration: float

class ConversationAnalysis(pydantic.BaseModel):
    sentiment: Dict[str, str | float]
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
                "openai-whisper==20231117",
                "torch==2.0.1",
                "numpy==1.24.3",
                "soundfile==0.12.1"
            ]
        ),
        compute=chains.Compute(
            gpu="T4",
            memory="16Gi"
        )
    )

    async def run_remote(self, audio_input: AudioInput) -> TranscriptionResult:
        # Note: imports are inside run_remote, just like in Baseten's example
        import whisper
        import numpy as np
        import soundfile as sf
        import io

        # Initialize model (done each time, like in their example)
        model = whisper.load_model("base")
        
        # Process audio
        audio_bytes = base64.b64decode(audio_input.audio_data)
        audio_io = io.BytesIO(audio_bytes)
        audio, sr = sf.read(audio_io)
        audio = audio.astype(np.float32)
        
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1).astype(np.float32)
        
        # Get transcription
        result = model.transcribe(audio)
        
        return TranscriptionResult(
            text=result["text"],
            language=result["language"],
            duration=float(len(audio) / sr)
        )

class ConversationAnalysisChainlet(chains.ChainletBase):
    """Analyzes conversation content"""
    
    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(
            pip_requirements=[
                "transformers==4.30.0",
                "torch==2.0.1"
            ]
        ),
        compute=chains.Compute(
            gpu="T4",
            memory="8Gi"
        )
    )

    async def run_remote(self, transcript: TranscriptionResult) -> ConversationAnalysis:
        from transformers import pipeline
        
        # Initialize models (done each time, like in their example)
        sentiment_analyzer = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Analyze
        sentiment = sentiment_analyzer(transcript.text)[0]
        summary = summarizer(transcript.text, max_length=130, min_length=30)[0]["summary_text"]
        
        # Simple rule-based analysis
        topics = ["pricing", "features"] if "price" in transcript.text.lower() else ["general inquiry"]
        intents = []
        if "how" in transcript.text.lower(): intents.append("information_request")
        if "buy" in transcript.text.lower() or "purchase" in transcript.text.lower(): intents.append("purchase_intent")
        if not intents: intents.append("general_inquiry")
        
        return ConversationAnalysis(
            sentiment={"score": float(sentiment["score"]), "label": str(sentiment["label"])},
            intents=intents,
            key_topics=topics,
            summary=summary
        )

class EmailGenerationChainlet(chains.ChainletBase):
    """Generates follow-up emails"""
    
    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(
            pip_requirements=[
                "transformers==4.30.0",
                "torch==2.0.1"
            ]
        ),
        compute=chains.Compute(
            gpu="T4",
            memory="8Gi"
        )
    )

    async def run_remote(self, analysis: ConversationAnalysis) -> EmailContent:
        from transformers import pipeline
        
        # Initialize model (done each time, like in their example)
        generator = pipeline("text-generation", model="gpt2")
        
        # Create prompt
        prompt = f"""Write a follow-up email.
        Topics discussed: {', '.join(analysis.key_topics)}
        Customer sentiment: {analysis.sentiment['label']}
        Summary: {analysis.summary}
        Intent: {', '.join(analysis.intents)}
        """
        
        # Generate
        generated = generator(prompt, max_length=200, num_return_sequences=1)[0]["generated_text"]
        
        # Format
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

    async def run_remote(self, audio_data: bytes) -> SDRResponse:
        # Create input with base64 audio
        audio_input = AudioInput(audio_data=base64.b64encode(audio_data).decode('utf-8'))
        
        # Process pipeline
        transcript = await self._transcriber.run_remote(audio_input)
        analysis = await self._analyzer.run_remote(transcript)
        email = await self._email_generator.run_remote(analysis)
        
        return SDRResponse(
            transcription=transcript,
            analysis=analysis,
            email=email
        )

if __name__ == "__main__":
    with chains.run_local():
        # Create test audio (simple sine wave)
        import numpy as np
        import soundfile as sf
        import io
        
        print("Creating test audio...")
        sample_rate = 16000
        duration = 3
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format='WAV', subtype='FLOAT')
        audio_data = buffer.getvalue()
        
        print("Running chain...")
        chain = SDRAssistant()
        result = asyncio.run(chain.run_remote(audio_data))
        
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