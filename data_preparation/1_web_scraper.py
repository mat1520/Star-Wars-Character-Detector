import os
import time
import random
import requests
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler

class StarWarsImageScraper:
    def __init__(self, output_dir="dataset_raw"):
        self.output_dir = Path(output_dir)
        self.characters = [
            "Darth Vader",
            "Luke Skywalker",
            "Yoda",
            "R2-D2",
            "C-3PO",
            "Chewbacca",
            "Han Solo",
            "Leia Organa"
        ]
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
        ]
        
    def get_random_user_agent(self):
        return random.choice(self.user_agents)
    
    def create_character_directory(self, character):
        """Create directory for character images if it doesn't exist."""
        char_dir = self.output_dir / character.replace(" ", "_")
        char_dir.mkdir(parents=True, exist_ok=True)
        return char_dir
    
    def download_images(self, character, num_images=150):
        """Download images for a character using multiple crawlers."""
        print(f"\nSearching for images of {character}...")
        char_dir = self.create_character_directory(character)
        
        # Try Google first
        google_crawler = GoogleImageCrawler(
            downloader_threads=4,
            storage={'root_dir': str(char_dir)}
        )
        google_crawler.crawl(
            keyword=f"{character} Star Wars character",
            max_num=num_images,
            file_idx_offset=0
        )
        
        # If we don't have enough images, try Bing
        downloaded = len(list(char_dir.glob('*.jpg')))
        if downloaded < num_images:
            remaining = num_images - downloaded
            bing_crawler = BingImageCrawler(
                downloader_threads=4,
                storage={'root_dir': str(char_dir)}
            )
            bing_crawler.crawl(
                keyword=f"{character} Star Wars character",
                max_num=remaining,
                file_idx_offset=downloaded
            )
        
        # Count final number of images
        downloaded = len(list(char_dir.glob('*.jpg')))
        print(f"Downloaded {downloaded} images for {character}")
        return downloaded
    
    def run(self):
        """Run the scraper for all characters."""
        print("Starting Star Wars character image collection...")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        total_downloaded = 0
        for character in self.characters:
            downloaded = self.download_images(character)
            total_downloaded += downloaded
            
            # Add a delay between characters
            if character != self.characters[-1]:  # Don't wait after the last character
                wait_time = random.uniform(5, 10)
                print(f"\nWaiting {wait_time:.1f} seconds before next character...")
                time.sleep(wait_time)
        
        print(f"\nDownload complete! Total images downloaded: {total_downloaded}")
        print(f"Images saved in: {self.output_dir.absolute()}")

if __name__ == "__main__":
    scraper = StarWarsImageScraper()
    scraper.run() 