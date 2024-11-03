import os
import logging
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.cluster import KMeans
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Токен вашего бота
TOKEN = "Your_token"

# Пути к папкам с мемами
MEME_FOLDERS = {
    (0, 18): "memes/age_0_18",
    (18, 30): "memes/age_18_30",
    (30, 40): "memes/age_30_40",
    (40, 100): "memes/age_40_plus",
}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Отправь мне фото, и я пришлю тебе подходящий мем!"
    )

def get_greeting(gender, age):
    if gender == "Man":
        if age < 18:
            return "Привет, юный джентльмен!"
        elif 18 <= age < 30:
            return "Здравствуй, молодой человек!"
        elif 30 <= age < 40:
            return "Приветствую, уважаемый мужчина!"
        else:
            return "Добрый день, опытный джентльмен!"
    else:
        if age < 18:
            return "Привет, юная леди!"
        elif 18 <= age < 30:
            return "Здравствуй, молодая девушка!"
        elif 30 <= age < 40:
            return "Приветствую, уважаемая женщина!"
        else:
            return "Добрый день, опытная леди!"

def get_age_category(age):
    if age < 18:
        return (0, 18)
    elif 18 <= age < 30:
        return (18, 30)
    elif 30 <= age < 40:
        return (30, 40)
    else:
        return (40, 100)

def get_dominant_color(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(image)
    return kmeans.cluster_centers_[0].astype(int)

def find_closest_meme(folder, target_color):
    closest_meme = None
    min_distance = float("inf")
    for filename in os.listdir(folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            meme_path = os.path.join(folder, filename)
            meme_image = cv2.imread(meme_path)
            meme_color = get_dominant_color(meme_image)
            distance = np.linalg.norm(target_color - meme_color)
            if distance < min_distance:
                min_distance = distance
                closest_meme = meme_path
    return closest_meme

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Получение фото
        photo_file = await update.message.photo[-1].get_file()
        photo_array = np.asarray(
            bytearray(await photo_file.download_as_bytearray()), dtype=np.uint8
        )
        image = cv2.imdecode(photo_array, -1)

        # Анализ лица
        result = DeepFace.analyze(
            image, actions=["age", "gender"], enforce_detection=True
        )

        if len(result) > 1:
            await update.message.reply_text(
                "На фото обнаружено несколько лиц. Пожалуйста, отправьте фото с одним человеком."
            )
            return

        age = result[0]["age"]
        gender = result[0]["dominant_gender"]

        # Формирование приветствия
        greeting = get_greeting(gender, age)
        await update.message.reply_text(greeting)

        # Определение возрастной категории
        age_category = get_age_category(age)

        # Анализ цветовой гаммы
        dominant_color = get_dominant_color(image)

        # Выбор подходящего мема
        meme_folder = MEME_FOLDERS[age_category]
        meme_path = find_closest_meme(meme_folder, dominant_color)

        # Отправка мема
        if meme_path:
            with open(meme_path, "rb") as meme:
                await update.message.reply_photo(meme)
        else:
            await update.message.reply_text(
                "К сожалению, не удалось найти подходящий мем."
            )

    except Exception as e:
        logger.error(f"Error processing photo: {str(e)}")
        await update.message.reply_text(
            "Произошла ошибка при обработке фото. Убедитесь, что на фото четко видно лицо."
        )

def main():
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    application.run_polling()

if __name__ == "__main__":
    main()
