# download_test_image.py
import urllib.request
import os

def download_sample_image():
    # A public, direct URL to an image of a dented blue shipping container
    image_url = "https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?q=80&w=600&auto=format&fit=crop"
    output_filename = "damaged_cargo.jpg"
    
    print(f"[Download] Downloading sample damaged cargo photo from Unsplash...")
    try:
        urllib.request.urlretrieve(image_url, output_filename)
        print(f"✓ Success! Saved test image to '{output_filename}' in your workspace.")
    except Exception as e:
        print(f"[Error] Failed to download sample image: {e}")

if __name__ == "__main__":
    download_sample_image()