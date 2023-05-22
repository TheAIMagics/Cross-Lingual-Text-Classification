from src.pipeline.training_pipeline import TrainPipeline
from src.pipeline.prediction_pipeline import SinglePrediction
from pprint import PrettyPrinter

pp = PrettyPrinter()

# obj = TrainPipeline()
# obj.run_pipeline()

obj = SinglePrediction()



english_text = "Yes! Awesome soy cap, scone, and atmosphere. Nice place to hang out & read, and free WiFi with no login procedure."

german_translation = "Ja! Tolle Sojakappe, Scone und Atmosphäre. Schöner Ort zum Abhängen und Lesen und kostenloses WLAN ohne Anmeldeprozedur."

pp.pprint(obj.predict([english_text, german_translation]))