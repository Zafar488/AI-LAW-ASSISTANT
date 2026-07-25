import unittest

from app.chatbot import LegalChatbot
from app.language import detect_language


class LanguageDetectionTests(unittest.TestCase):
    def setUp(self):
        self.bot = object.__new__(LegalChatbot)

    def test_english(self):
        self.assertEqual(detect_language("I was robbed near my home"), "en")

    def test_urdu(self):
        self.assertEqual(detect_language("میرے قانونی حقوق کیا ہیں؟"), "ur")

    def test_hindi(self):
        self.assertEqual(detect_language("मेरे कानूनी अधिकार क्या हैं?"), "hi")

    def test_pashto(self):
        self.assertEqual(detect_language("که غلا وشي، کوم جرم واقع شوی دی؟"), "ps")

    def test_chatbot_uses_shared_detector(self):
        self.assertEqual(self.bot.detect_language("مجھے ضمانت کیسے مل سکتی ہے؟"), "ur")

    def test_empty_input_uses_fallback(self):
        self.assertEqual(detect_language("", fallback="hi"), "hi")


if __name__ == "__main__":
    unittest.main()

