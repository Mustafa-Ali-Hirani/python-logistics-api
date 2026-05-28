# voice_logistics_agent.py
import os
from gtts import gTTS
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Mock logistics database
shipment_database = {
    "bol-99281-x": {
        "status": "Cleared Customs",
        "location": "Karachi Port (Off-site bonded facility)",
        "carrier": "Atlantic Ocean Lines",
        "eta": "Delivered to DPD warehouse"
    }
}

# ==========================================
# 1. TEXT-TO-SPEECH (TTS) ENGINE
# ==========================================
def text_to_speech(text: str, output_filename: str):
    """Converts a string of text into an MP3 voice file."""
    print(f"[TTS] Synthesizing speech to: '{output_filename}'...")
    
    # We use gTTS (Google Text-to-Speech) for a completely free local loop
    tts = gTTS(text=text, lang="en", tld="com")
    tts.save(output_filename)
    print(f"✓ Audio file successfully written.")

# ==========================================
# 2. SPEECH-TO-TEXT (STT) ENGINE (Whisper V3)
# ==========================================
def speech_to_text(audio_filepath: str) -> str:
    """Sends an audio file to Groq's Whisper API and returns the transcription."""
    print(f"\n[STT] Sending '{audio_filepath}' to Groq Whisper-Large-V3...")
    
    if not os.path.exists(audio_filepath):
        raise FileNotFoundError(f"Audio file {audio_filepath} not found.")
        
    with open(audio_filepath, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(audio_filepath, file.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
            temperature=0.0
        )
        
    print(f"✓ Transcribed Text: \"{transcription.text}\"")
    return transcription.text.strip()

# ==========================================
# 3. LLM ENTITY EXTRACTION & DATABASE LOOKUP
# ==========================================
def query_shipment_status(transcription_text: str) -> str:
    """Uses LLM to extract the BOL number and checks our mock database."""
    print(f"\n[Brain] Extracting BOL number from text using LLM...")
    
    prompt = f"""
    Analyze this transcribed customer query and extract any Bill of Lading (BOL) shipment number.
    Format the output strictly as: bol-XXXXX-X (lowercase, hyphenated).
    If no BOL is found, output 'NONE'. Do not include any other text or punctuation.
    
    Query: "{transcription_text}"
    """
    
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    # Standardize output (e.g. bol-99281-x)
    bol_number = completion.choices[0].message.content.strip().lower()
    print(f" -> Extracted BOL Key: '{bol_number}'")
    
    if bol_number in shipment_database:
        data = shipment_database[bol_number]
        response_text = (
            f"I found shipment {bol_number.upper()}. "
            f"The status is currently {data['status']} at the {data['location']}. "
            f"The carrier is {data['carrier']}."
        )
    else:
        response_text = (
            "I received your inquiry, but I could not locate that specific Bill of Lading "
            "number in our active tracking database. Please check your document and try again."
        )
        
    print(f"✓ Prepared Agent Response: \"{response_text}\"")
    return response_text

# ==========================================
# 4. EXECUTING THE SELF-CONTAINED VOICE LOOP
# ==========================================
def run_voice_loop():
    print("====================================================")
    print("       LAUNCHING LOGISTICS VOICE AGENT LOOP          ")
    print("====================================================")
    
    # Step 1: Simulate a customer speaking (Generate "customer_query.mp3")
    customer_speech_text = "Hi, I am looking for the status of shipment BOL 99281 X."
    print(f"[Simulated Customer Speaks]: \"{customer_speech_text}\"")
    text_to_speech(customer_speech_text, "customer_query.mp3")
    
    # Step 2: Use Whisper STT to transcribe the generated audio file
    transcribed_query = speech_to_text("customer_query.mp3")
    
    # Step 3: Parse query and find database answer
    agent_response_text = query_shipment_status(transcribed_query)
    
    # Step 4: Use TTS to synthesize the final agent vocal response
    text_to_speech(agent_response_text, "agent_response.mp3")
    
    print("\n====================================================")
    print("✓ SUCCESS: Voice loop complete!")
    print("Files written to your workspace:")
    print(" - 'customer_query.mp3' (Simulated customer voice)")
    print(" - 'agent_response.mp3' (Live synthesized agent response)")
    print("====================================================")

if __name__ == "__main__":
    run_voice_loop()