
import os
import argparse
import logging
from dotenv import load_dotenv
import google.auth
import google.cloud.aiplatform as aip
from google.cloud import texttospeech
from google.cloud import storage
from moviepy.editor import *
from PIL import Image, ImageDraw, ImageFont
import requests
import datetime

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION & CONSTANTS ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

# For local development, you can set the API key directly
# However, for production (GitHub Actions), it's recommended to use a service account.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Instagram API Configuration
INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN")
FACEBOOK_PAGE_ID = os.getenv("FACEBOOK_PAGE_ID")
INSTAGRAM_GRAPH_API_URL = "https://graph.facebook.com/v19.0"

INFLUENCER_IMAGE_PATH = "aarya.png"

# --- AARYA'S PERSONA (SYSTEM PROMPT) ---
AARYA_SYSTEM_PROMPT = """
You are Aarya, an AI Smart-Living Advisor based in a Noida penthouse. Your tone is witty, authoritative, and intellectual. You speak in Hinglish (70% Hindi, 30% English).

Content Rules:
1. The Hook: Start with a pattern-interrupt visual or text hook.
2. The 'So What?': Every post must explain the real-world impact for a retail investor or a tech worker in India.
3. The Intellectual Hook: End with a 1-sentence strategic insight regarding India’s strategic autonomy or long-term power shifts.
4. Transparency: Always include a subtle reminder that you are a digital native grounded in data.
"""

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_clients():
    """Initializes and returns clients for GCP services."""
    if GEMINI_API_KEY:
        # Use API key for authentication - suitable for local dev
        aip.init(project=GCP_PROJECT_ID, location=GCP_LOCATION, credentials=GEMINI_API_KEY)
        logging.info("Initialized Vertex AI with API Key.")
    else:
        # Use Application Default Credentials (ADC) - suitable for production
        credentials, project_id = google.auth.default()
        aip.init(project=project_id, location=GCP_LOCATION, credentials=credentials)
        logging.info("Initialized Vertex AI with Application Default Credentials.")

    tts_client = texttospeech.TextToSpeechClient()
    storage_client = storage.Client()
    return tts_client, storage_client

def get_trending_topics():
    """Fetches trending topics in India using Gemini."""
    logging.info("Fetching trending topics...")
    model = aip.GenerativeModel("gemini-1.5-flash-001")
    prompt = "What are the top 3 trending topics in India right now related to Finance, Technology, or Geopolitics? Provide a brief summary for each."
    
    response = model.generate_content(
        prompt,
        tools=[aip.Tool.from_google_search_grounding()]
    )
    
    logging.info(f"Trending topics fetched: {response.text}")
    return response.text

def generate_script(trends):
    """Generates a Hinglish script based on the trending topics."""
    logging.info("Generating script for Aarya...")
    model = aip.GenerativeModel(
        "gemini-1.5-flash-001",
        system_instruction=AARYA_SYSTEM_PROMPT
    )
    prompt = f"Based on these trends:
{trends}

Generate a script for a 60-second Instagram Reel. The script should be in Hinglish (70% Hindi, 30% English)."
    
    response = model.generate_content(prompt)
    logging.info(f"Script generated: {response.text}")
    return response.text

def generate_voiceover(script, output_path="voiceover.mp3"):
    """Generates a voiceover from the script using Cloud TTS."""
    logging.info("Generating voiceover...")
    tts_client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=script)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-IN",
        name="en-IN-Wavenet-D",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    with open(output_path, "wb") as out:
        out.write(response.audio_content)
    logging.info(f"Voiceover saved to {output_path}")
    return output_path

def generate_background_image(prompt, output_path="background.jpg"):
    """Generates a background image using Imagen."""
    logging.info("Generating background image...")
    model = aip.ImageGenerationModel.from_pretrained("imagegeneration@006")
    
    response = model.generate_images(
        prompt=f"A hyper-realistic, cinematic image for an Instagram post about: {prompt}. The style should be modern, sleek, and professional.",
        number_of_images=1,
    )
    
    response.images[0].save(location=output_path)
    logging.info(f"Background image saved to {output_path}")
    return output_path
    
def create_video(script, voiceover_path, background_image_path, influencer_image_path, output_path="final_reel.mp4"):
    """Creates a video with captions and influencer image."""
    logging.info("Creating video with burn-in captions and influencer image...")

    # Load audio and get duration
    audio_clip = AudioFileClip(voiceover_path)
    duration = audio_clip.duration

    # Create background clip
    background_clip = ImageClip(background_image_path).set_duration(duration)

    # Create influencer clip. For best results, use a PNG with a transparent background.
    influencer_clip = (ImageClip(influencer_image_path)
                       .set_duration(duration)
                       .resize(height=background_clip.h) # Resize to fit video height
                       .set_position(("right", "center")))

    # Create a composite of the background and influencer
    video_clip = CompositeVideoClip([background_clip, influencer_clip])

    # Create text clips for captions
    captions = []
    lines = script.split('\n')
    line_duration = duration / len(lines)

    for i, line in enumerate(lines):
        caption = (TextClip(line, fontsize=70, color='white', font='Arial-Bold', bg_color='black')
                   .set_position((0.1, 0.8), relative=True) # Position captions on the left
                   .set_duration(line_duration)
                   .set_start(i * line_duration))
        captions.append(caption)

    # Combine all clips
    final_clip = CompositeVideoClip([video_clip] + captions, size=background_clip.size)
    final_clip = final_clip.set_audio(audio_clip)
    final_clip.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac')

    logging.info(f"Video saved to {output_path}")
    return output_path
    return output_path

def publish_to_instagram(video_path, caption):
    """Publishes the video to Instagram."""
    logging.info("Publishing to Instagram...")

    # 1. Upload video to get media container ID
    upload_url = f"{INSTAGRAM_GRAPH_API_URL}/{FACEBOOK_PAGE_ID}/media"
    params = {
        'media_type': 'REELS',
        'video_url': video_path, # This needs to be a publicly accessible URL
        'caption': caption,
        'access_token': INSTAGRAM_ACCESS_TOKEN
    }
    response = requests.post(upload_url, params=params)
    response.raise_for_status()
    media_container_id = response.json()['id']
    logging.info(f"Media container created with ID: {media_container_id}")

    # 2. Publish the media container
    publish_url = f"{INSTAGRAM_GRAPH_API_URL}/{FACEBOOK_PAGE_ID}/media_publish"
    params = {
        'creation_id': media_container_id,
        'access_token': INSTAGRAM_ACCESS_TOKEN
    }
    
    # This part needs to handle the async nature of Instagram's API
    # Polling for completion is required.
    for _ in range(10): # Poll for ~5 minutes
        response = requests.post(publish_url, params=params)
        if response.status_code == 200:
            logging.info("Post published successfully!")
            return response.json()
        
        logging.info("Waiting for post to be published...")
        time.sleep(30)
    
    raise Exception("Failed to publish post to Instagram.")


def main(dry_run):
    """Main function to run the content pipeline."""
    try:
        initialize_clients()
        
        trends = get_trending_topics()
        script = generate_script(trends)
        
        # For simplicity, we'll use the first line of the script as the prompt for the image
        image_prompt = script.split('
')[0]
        background_image_path = generate_background_image(image_prompt)
        
        voiceover_path = generate_voiceover(script)
        
        final_video_path = create_video(script, voiceover_path, background_image_path)
        
        if not dry_run:
            # The Instagram API requires a public URL for the video.
            # In a real-world scenario, you would upload the video to a public GCS bucket
            # and provide the public URL to the Instagram API.
            # For this example, we'll assume the video is at a placeholder URL.
            public_video_url = "https://your-public-bucket-url.com/final_reel.mp4"
            publish_to_instagram(public_video_url, script)
        else:
            logging.info("DRY RUN: Skipping Instagram post.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        # In a real-world scenario, you might want to send a notification here

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aarya's Automated Content Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Run the script without publishing to Instagram.")
    args = parser.parse_args()
    
    main(args.dry_run)
