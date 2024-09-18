import discord
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image

# Modeli yükle
model = tf.keras.models.load_model('model_keras/')

# Etiketleri tanımlayın
labels = ['az', 'orta', 'çok']

# Öneri metinleri
suggestions = {
    'az': 'Arabanız kalabilir.',
    'orta': 'Arabanızı daha az kullanın.',
    'çok': 'Toplu taşıma kullanın.'
}

# Discord botu için ayar
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

def process_image(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(image_bytes):
    image = process_image(image_bytes)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    label = labels[predicted_class]
    confidence = predictions[0][predicted_class]
    return label, confidence

@client.event
async def on_ready():
    print(f'Bot olarak giriş yapıldı: {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.attachments:
        for attachment in message.attachments:
            if attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_bytes = await attachment.read()
                label, confidence = predict(image_bytes)
                suggestion = suggestions[label]
                response = (f'Tahmin edilen etiket: {label}, Güven skoru: {confidence:.2f}\n'
                            f'Öneri: {suggestion}')
                await message.channel.send(response)
                await message.channel.send(file=discord.File(BytesIO(image_bytes), filename=attachment.filename))

# Bot tokenınızı buraya ekleyin
TOKEN = 'MTIxMjQ1OTc2MjI1NjQ1MzY0Mg.GQuWXc.f8_QHZpm1Ei8-PTvbmjb9SP4UOkCfYEUQMfwdw'
client.run(TOKEN)
