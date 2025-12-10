import joblib
import jieba
import re


class SimpleAviationTester:
    def __init__(self):
        self.vectorizer = None
        self.aspect_classifier = None
        self.sentiment_classifier = None
        self.aspect_mlb = None
        self.load_models()

    def load_models(self):
        """load model"""
        try:
            self.vectorizer = joblib.load('Aviation_vectorizer.pkl')
            self.aspect_classifier = joblib.load('Aviation_aspect_classifier.pkl')
            self.sentiment_classifier = joblib.load('Aviation_sentiment_classifier.pkl')
            self.aspect_mlb = joblib.load('Aspect_mlb.pkl')
            print("Loading models done")
        except Exception as e:
            print(f"Loading failed: {e}")
            exit(1)

    def preprocess_text(self, text):
        """pre-processing text"""
        words = jieba.lcut(text)
        words = [word for word in words if word.strip() and not re.match(r'[^\w\s]', word)]
        return ' '.join(words)

    def predict(self, text):
        """predicate"""
        try:
            # pre-process
            processed_text = self.preprocess_text(text)
            X = self.vectorizer.transform([processed_text])

            # predicate aspects
            aspect_predictions = self.aspect_classifier.predict(X)
            predicted_aspects = self.aspect_mlb.inverse_transform(aspect_predictions)
            aspects = list(predicted_aspects[0]) if len(predicted_aspects) > 0 else ["service"]

            # predicate emotion
            sentiment_pred = self.sentiment_classifier.predict(X)[0]
            sentiment_proba = self.sentiment_classifier.predict_proba(X)[0]
            confidence = max(sentiment_proba)

            # build result
            sentiments = {aspect: sentiment_pred for aspect in aspects}

            return {
                "text": text,
                "aspects": aspects,
                "sentiments": sentiments,
                "overall_sentiment": sentiment_pred,
                "confidence": round(confidence, 3)
            }

        except Exception as e:
            print(f"predicate failed: {e}")
            return {
                "text": text,
                "aspects": ["service"],
                "sentiments": {"service": "neutral"},
                "overall_sentiment": "neutral",
                "confidence": 0.0,
                "error": str(e)
            }

    def display_result(self, result):
        """display result"""
        print(f"\nAnalyze result:")
        print(f"text: {result['text']}")
        print(f"aspects: {result['aspects']}")
        print(f"sentiment: {result['overall_sentiment']}")
        print(f"confidence: {result['confidence']}")

        if 'error' in result:
            print(f"error: {result['error']}")


def main():
    print("=" * 30)
    print("Test model demo")

    tester = SimpleAviationTester()

    while True:
        user_input = input("\nInput your text in Chinese (input 'quit' or '退出' to end): ").strip()

        if user_input.lower() in ['quit', '退出']:
            break

        if not user_input:
            continue

        result = tester.predict(user_input)
        tester.display_result(result)


if __name__ == "__main__":
    main()