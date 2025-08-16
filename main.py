import time
import json
import torch
import requests
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertModel
from PIL import Image
from io import BytesIO
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from googletrans import Translator
from fastapi.middleware.cors import CORSMiddleware
# FastAPI initialization
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.vgg16_features = vgg16.features
        
        self.encoder = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512 * 7 * 7),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        with torch.no_grad():
            x = self.vgg16_features(x)
        x = torch.flatten(x, start_dim=1)
        latent_rep = self.encoder(x)
        reconstructed = self.decoder(latent_rep)
        return latent_rep, reconstructed


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return outputs.pooler_output


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, num_heads=4, ff_size=512, num_layers=2):
        super(TransformerEncoder, self).__init__() 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=num_heads, dim_feedforward=ff_size
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.transformer_encoder(x)
        return x


class BotDetectionModel(nn.Module):
    def __init__(self):
        super(BotDetectionModel, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.transformer_encoder = TransformerEncoder(input_size=256 + 768)
        self.dense_layer = nn.Sequential(
            nn.Linear(256 + 768 + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, images, input_ids, attention_mask, followers_count, following_count):
        image_features, _ = self.image_encoder(images)
        text_embeddings = self.text_encoder(input_ids, attention_mask)
        combined_features = torch.cat((image_features, text_embeddings), dim=1)
        transformer_output = self.transformer_encoder(combined_features.unsqueeze(1))
        combined_with_counts = torch.cat((transformer_output.squeeze(1), followers_count, following_count), dim=1)
        output = self.dense_layer(combined_with_counts)
        return output
    
# Model and image transformation setup
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BotDetectionModel()  # Assuming you already defined this
checkpoint = torch.load('final_bot_detection_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


# FastAPI Input Model
class BotDetectionInput(BaseModel):
    profile_image_url: str
    background_image_url: str
    media_urls: list[str]
    post_content: str
    followers_count: int
    following_count: int

# Function to load image from URL
def load_image_from_url(url, transform):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        if transform:
            img = transform(img)
        return img
    except:
        return torch.zeros(3, 224, 224)
def auto_translate(text, target_language='en'):
    translator = Translator()
    translated_text = translator.translate(text, dest=target_language).text
    return translated_text
# Selenium Extraction Function
def extract_data_from_account(account_name):
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Enable headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Initialize WebDriver
    driver = webdriver.Chrome(options=chrome_options)

    try:
        # After login, navigate to the account page
        driver.get(f'https://x.com/{account_name}')
        time.sleep(4)
        login_button = driver.find_element(By.CSS_SELECTOR, '[data-testid="login"]')
        login_button.click()

        # Wait for 4 seconds
        time.sleep(5)
        # Find and fill the phone number input field
        phone_input = driver.find_element(By.CSS_SELECTOR, 'input.r-30o5oe')  # Adjust selector as needed
        phone_input.send_keys('9392759282')

        # Submit phone number
        phone_input.send_keys(Keys.RETURN)

        # Allow time for the next page to load
        time.sleep(5)

        # Find and fill the password input field
        password_input = driver.find_element(By.NAME, 'password')  # Adjust selector as needed
        password_input.send_keys('Klenin45')

        # Submit password
        password_input.send_keys(Keys.RETURN)

        # Allow time for the login to complete
        time.sleep(5)
        # Extract URLs and counts
        try:
            # Get all profile image elements
            profile_image_elements = driver.find_elements(By.CSS_SELECTOR, 'img[src^="https://pbs.twimg.com/profile_images/"]')

            # If at least one profile image exists, pick the last one (usually the highest resolution)
            profile_image = profile_image_elements[-1].get_attribute('src') if profile_image_elements else ""

            # Get all background image elements
            background_image_elements = driver.find_elements(By.CSS_SELECTOR, 'img[src^="https://pbs.twimg.com/profile_banners/"]')

            # Pick the first background image if available
            background_image = background_image_elements[0].get_attribute('src') if background_image_elements else ""

        except Exception as e:
            print(f"Error extracting profile images: {e}")
            profile_image, background_image = "", ""

        # Extract follower and following counts
        script_element = driver.find_element(By.XPATH, '//script[@data-testid="UserProfileSchema-test"]')
        json_content = script_element.get_attribute('textContent')
        data = json.loads(json_content)

        interaction_stats = data['mainEntity']['interactionStatistic']
        followers_count = next(stat['userInteractionCount'] for stat in interaction_stats if stat['name'] == 'Follows')
        following_count = next(stat['userInteractionCount'] for stat in interaction_stats if stat['name'] == 'Friends')

        # Extract post content (latest tweets)
        post_content = ""
        tweet_elements = driver.find_elements(By.XPATH, '//div[@data-testid="tweetText"]')[:2]
        for tweet_element in tweet_elements:
            post_content += auto_translate(tweet_element.text)

        # Extract media URLs (up to 4 images)
        unique_image_urls = set()
        max_scrolls = 10
        while len(unique_image_urls) < 4 and max_scrolls > 0:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            images = driver.find_elements(By.CSS_SELECTOR, 'img[src^="https://pbs.twimg.com/media/"]')
            new_image_urls = {img.get_attribute('src') for img in images}
            unique_image_urls.update(new_image_urls)
            max_scrolls -= 1

        media_urls = list(unique_image_urls)[:4]
        if not media_urls:
            media_urls=[""]*4
        print("The Media Update: ",media_urls)

        return {
            "profile_image_url": profile_image,
            "background_image_url": background_image,
            "followers_count": followers_count,
            "following_count": following_count,
            "post_content": post_content,
            "media_urls": media_urls
        }

    except Exception as e:
        print(f"Error extracting data: {e}")
        return {}

    finally:
        driver.quit()


# FastAPI Endpoint for bot detection
@app.post("/detect-bot/")
async def detect_bot(account_name: str):
    # Step 1: Extract data using Selenium
    data = extract_data_from_account(account_name)
    print("data========",data)
  
    
    if not data:
        raise HTTPException(status_code=400, detail="Failed to extract data for the provided account.")

    # Step 2: Convert the data to BotDetectionInput
    input_data = BotDetectionInput(
        profile_image_url=data["profile_image_url"],
        background_image_url=data["background_image_url"],
        followers_count=data["followers_count"],
        following_count=data["following_count"],
        post_content=data["post_content"],
        media_urls=data["media_urls"]
    )

    # Step 3: Prepare data for model inference
    profile_image = load_image_from_url(input_data.profile_image_url, transform)
    background_image = load_image_from_url(input_data.background_image_url, transform)
    media_images = torch.stack([load_image_from_url(url, transform) for url in input_data.media_urls], dim=0).mean(0)
    
    images = torch.stack([profile_image, background_image, media_images], dim=0).mean(0).unsqueeze(0)

    encoding = tokenizer(input_data.post_content, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    followers_count = torch.tensor([[input_data.followers_count]], dtype=torch.float32)
    following_count = torch.tensor([[input_data.following_count]], dtype=torch.float32)

    # Step 4: Predict with the model
    with torch.no_grad():
        prediction = model(images, input_ids, attention_mask, followers_count, following_count)
        bot_prob = round(prediction.item(), 7)

        if bot_prob < 0.73 or input_data.followers_count >= 10e5:
            result = {
                "bot_probability": bot_prob,
                "status": "Human",
                "message": "This account appears to be operated by a HUMAN. No issues detected."
            }
        else:
            result = {
                "bot_probability": bot_prob,
                "status": "Bot",
                "message": "This account appears to be a BOT. Proceed with caution."
            }
        return result
