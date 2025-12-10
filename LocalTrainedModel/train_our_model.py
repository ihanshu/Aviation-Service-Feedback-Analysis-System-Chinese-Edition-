import joblib
import jieba
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import warnings

warnings.filterwarnings('ignore')


class RuleBasedModelTrainer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=1, max_df=0.8)
        self.aspect_classifier = None
        self.sentiment_classifier = None
        self.aspect_mlb = None
        self.aspects = ['booking', 'payment', 'checkin', 'boarding', 'flight', 'service', 'price', 'refund',
                        'customer_service', 'baggage', 'seating']

    def create_intensive_training_data(self):
        """Create balanced training data"""
        training_samples = []

        # 1. Basic aspect keywords training - balanced positive and negative
        aspect_keywords = {
            "booking": ["预订", "订票", "预约", "下单", "买票", "购票"],
            "payment": ["支付", "付款", "收费", "价钱", "价格", "付费"],
            "checkin": ["值机", "办理登机", "值机手续", "柜台"],
            "boarding": ["登机", "登机口", "登机过程", "上飞机"],
            "flight": ["航班", "飞行", "延误", "取消", "准点", "准时", "飞机"],
            "service": ["服务", "态度", "空姐", "乘务员", "服务员", "工作人员", "餐食"],
            "price": ["价格", "价钱", "贵", "便宜", "性价比", "收费", "费用"],
            "refund": ["退款", "退票", "退钱", "改签"],
            "customer_service": ["客服", "客户服务", "咨询", "电话", "热线"],
            "baggage": ["行李", "托运", "行李箱", "手提行李"],
            "seating": ["座位", "选座", "座椅", "坐位"]
        }

        # Create balanced samples for each aspect
        for aspect, keywords in aspect_keywords.items():
            for keyword in keywords:
                # Positive sentiment samples
                training_samples.append({
                    "text": f"{keyword}很好",
                    "aspects": [aspect],
                    "sentiments": {aspect: "positive"},
                    "overall_sentiment": "positive"
                })
                training_samples.append({
                    "text": f"{keyword}不错",
                    "aspects": [aspect],
                    "sentiments": {aspect: "positive"},
                    "overall_sentiment": "positive"
                })
                training_samples.append({
                    "text": f"{keyword}很棒",
                    "aspects": [aspect],
                    "sentiments": {aspect: "positive"},
                    "overall_sentiment": "positive"
                })

                # Negative sentiment samples
                training_samples.append({
                    "text": f"{keyword}很差",
                    "aspects": [aspect],
                    "sentiments": {aspect: "negative"},
                    "overall_sentiment": "negative"
                })
                training_samples.append({
                    "text": f"{keyword}不好",
                    "aspects": [aspect],
                    "sentiments": {aspect: "negative"},
                    "overall_sentiment": "negative"
                })

        # 2. Core sentiment vocabulary training
        # Positive words
        positive_words = ["好", "很好", "非常好", "不错", "很棒", "优秀", "满意", "牛逼", "厉害", "赞"]
        for word in positive_words:
            training_samples.append({
                "text": word,
                "aspects": ["service"],
                "sentiments": {"service": "positive"},
                "overall_sentiment": "positive"
            })

        # Negative words
        negative_words = ["差", "很差", "糟糕", "差劲", "烂", "垃圾", "失望", "不行", "不好"]
        for word in negative_words:
            training_samples.append({
                "text": word,
                "aspects": ["service"],
                "sentiments": {"service": "negative"},
                "overall_sentiment": "negative"
            })

        # 3. Specific scenarios
        scenarios = [
            # Negative scenarios
            ("价格贵", "price", "negative"),
            ("价格很贵", "price", "negative"),
            ("飞机延误", "flight", "negative"),
            ("服务差", "service", "negative"),
            # Positive scenarios
            ("价格便宜", "price", "positive"),
            ("服务很好", "service", "positive"),
            ("航班准点", "flight", "positive"),
        ]

        for text, aspect, sentiment in scenarios:
            training_samples.append({
                "text": text,
                "aspects": [aspect],
                "sentiments": {aspect: sentiment},
                "overall_sentiment": sentiment
            })

        print(f"Generated {len(training_samples)} training samples")
        return training_samples

    def preprocess_text(self, text):
        """Text preprocessing"""
        words = jieba.lcut(text)
        words = [word for word in words if word.strip()
                 and not re.match(r'[^\w\s]', word)
                 and len(word) > 0]
        return ' '.join(words)

    def train_aspect_classifier(self, texts, aspects_list):
        """Train aspect classifier"""
        print("Training aspect classifier...")

        self.aspect_mlb = MultiLabelBinarizer(classes=self.aspects)
        y_aspects = self.aspect_mlb.fit_transform(aspects_list)

        processed_texts = [self.preprocess_text(text) for text in texts]
        X = self.vectorizer.fit_transform(processed_texts)

        print(f"Feature matrix shape: {X.shape}")

        base_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42
        )

        self.aspect_classifier = MultiOutputClassifier(base_classifier)
        self.aspect_classifier.fit(X, y_aspects)

        joblib.dump(self.aspect_mlb, 'Aspect_mlb.pkl')
        print("Aspect classifier training completed")
        return self.aspect_classifier

    def train_sentiment_classifier(self, texts, overall_sentiments):
        """Train sentiment classifier"""
        print("Training sentiment classifier...")

        processed_texts = [self.preprocess_text(text) for text in texts]
        X = self.vectorizer.transform(processed_texts)

        # Use SVC, which performs better for text classification
        self.sentiment_classifier = SVC(
            kernel='linear',
            probability=True,
            random_state=42
        )
        self.sentiment_classifier.fit(X, overall_sentiments)

        print("Sentiment classifier training completed")
        return self.sentiment_classifier

    def train_complete_model(self):
        """Train complete model"""
        print("Starting aviation domain sentiment analysis model training...")

        training_data = self.create_intensive_training_data()

        texts = [sample['text'] for sample in training_data]
        aspects_list = [sample['aspects'] for sample in training_data]
        overall_sentiments = [sample['overall_sentiment'] for sample in training_data]

        self.train_aspect_classifier(texts, aspects_list)
        self.train_sentiment_classifier(texts, overall_sentiments)

        self.save_model()
        self.evaluate_model(training_data)

        return self

    def save_model(self):
        """Save model"""
        joblib.dump(self.vectorizer, 'Aviation_vectorizer.pkl')
        joblib.dump(self.aspect_classifier, 'Aviation_aspect_classifier.pkl')
        joblib.dump(self.sentiment_classifier, 'Aviation_sentiment_classifier.pkl')
        print("Model saved successfully")

    def load_model(self):
        """Load model"""
        try:
            self.vectorizer = joblib.load('Aviation_vectorizer.pkl')
            self.aspect_classifier = joblib.load('Aviation_aspect_classifier.pkl')
            self.sentiment_classifier = joblib.load('Aviation_sentiment_classifier.pkl')
            self.aspect_mlb = joblib.load('Aspect_mlb.pkl')
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Model loading failed: {e}")
            return False

    def predict(self, text):
        """Prediction method - simplified version"""
        try:
            # First use rule-based system for sentiment judgment
            rule_sentiment = self._rule_based_sentiment(text)

            # Preprocess text
            processed_text = self.preprocess_text(text)
            X = self.vectorizer.transform([processed_text])

            # Predict aspects
            aspect_predictions = self.aspect_classifier.predict(X)
            predicted_aspects = self.aspect_mlb.inverse_transform(aspect_predictions)
            aspects = list(predicted_aspects[0]) if len(predicted_aspects) > 0 else []

            # If no aspects detected, use keyword matching
            if not aspects:
                aspects = self._keyword_based_aspects(text)

            # Ensure at least one aspect (default to service)
            if not aspects:
                aspects = ["service"]

            # If rule system has clear judgment, prioritize rules
            if rule_sentiment != "neutral":
                final_sentiment = rule_sentiment
                confidence = 0.9
            else:
                # Use model to predict sentiment
                sentiment_proba = self.sentiment_classifier.predict_proba(X)[0]
                predicted_sentiment = self.sentiment_classifier.predict(X)[0]
                confidence = max(sentiment_proba)
                final_sentiment = predicted_sentiment

            # Build results
            sentiments = {aspect: final_sentiment for aspect in aspects}

            return {
                "aspects": aspects,
                "sentiments": sentiments,
                "overall_sentiment": final_sentiment,
                "confidence": round(confidence, 3),
                "_method": "hybrid"
            }

        except Exception as e:
            print(f"Prediction failed: {e}")
            return self._rule_based_fallback(text)

    def _keyword_based_aspects(self, text):
        """Keyword-based aspect recognition"""
        aspect_keywords = {
            "booking": ["预订", "订票", "预约", "下单", "买票", "购票"],
            "payment": ["支付", "付款", "收费", "价钱", "价格", "付费"],
            "checkin": ["值机", "办理登机", "值机手续", "柜台"],
            "boarding": ["登机", "登机口", "登机过程", "上飞机"],
            "flight": ["航班", "飞行", "延误", "取消", "准点", "准时", "飞机"],
            "service": ["服务", "态度", "空姐", "乘务员", "服务员", "工作人员", "餐食"],
            "price": ["价格", "价钱", "贵", "便宜", "性价比", "收费", "费用"],
            "refund": ["退款", "退票", "退钱", "改签"],
            "customer_service": ["客服", "客户服务", "咨询", "电话", "热线"],
            "baggage": ["行李", "托运", "行李箱", "手提行李"],
            "seating": ["座位", "选座", "座椅", "坐位"]
        }

        aspects = []
        for aspect, keywords in aspect_keywords.items():
            if any(keyword in text for keyword in keywords):
                aspects.append(aspect)

        return aspects

    def _rule_based_sentiment(self, text):
        """Rule-based sentiment analysis - fixed logic"""
        # Positive vocabulary
        positive_words = ["好", "很好", "非常好", "不错", "很棒", "优秀", "满意",
                          "牛逼", "厉害", "赞", "完美", "出色", "专业", "周到"]

        # Negative vocabulary
        negative_words = ["差", "很差", "糟糕", "差劲", "烂", "垃圾", "失望", "废物",
                          "不行", "不好", "贵", "很贵", "太贵", "延误", "取消"]

        # Negation structures
        negation_words = ["不", "没", "没有", "无"]

        # Check positive words
        for word in positive_words:
            if word in text:
                # Check if there are negation words before the positive word
                has_negation = False
                for neg in negation_words:
                    if neg in text and text.find(neg) < text.find(word):
                        has_negation = True
                        break
                if not has_negation:
                    return "positive"

        # Check negative words
        for word in negative_words:
            if word in text:
                return "negative"

        return "neutral"

    def _rule_based_fallback(self, text):
        """Pure rule-based fallback prediction"""
        sentiment = self._rule_based_sentiment(text)
        aspects = self._keyword_based_aspects(text)

        if not aspects:
            aspects = ["service"]

        sentiments = {aspect: sentiment for aspect in aspects}

        return {
            "aspects": aspects,
            "sentiments": sentiments,
            "overall_sentiment": sentiment,
            "confidence": 0.8,
            "_method": "rules_only"
        }

    def evaluate_model(self, training_data):
        """Evaluate model performance"""
        print("\n=== Model Evaluation ===")

        test_cases = [
            ("好", ["service"], "positive"),
            ("牛逼", ["service"], "positive"),
            ("价格很贵", ["price"], "negative"),
            ("服务很好", ["service"], "positive"),
            ("飞机延误", ["flight"], "negative"),
            ("不错", ["service"], "positive"),
            ("座位很好", ["seating"], "positive"),
        ]

        aspect_correct = 0
        sentiment_correct = 0

        print("Key sample test results:")
        for text, expected_aspects, expected_sentiment in test_cases:
            result = self.predict(text)
            predicted_aspects = result['aspects']
            predicted_sentiment = result['overall_sentiment']

            aspect_status = "✓" if set(predicted_aspects) == set(expected_aspects) else "✗"
            sentiment_status = "✓" if predicted_sentiment == expected_sentiment else "✗"

            method = result.get('_method', 'unknown')
            print(f"  {aspect_status}{sentiment_status} '{text}'")
            print(f"     Aspects: {predicted_aspects} (Expected: {expected_aspects})")
            print(f"     Sentiment: {predicted_sentiment} (Expected: {expected_sentiment}) [{method}]")

            if set(predicted_aspects) == set(expected_aspects):
                aspect_correct += 1
            if predicted_sentiment == expected_sentiment:
                sentiment_correct += 1

        aspect_accuracy = aspect_correct / len(test_cases)
        sentiment_accuracy = sentiment_correct / len(test_cases)

        print(f"\nAspect recognition accuracy: {aspect_accuracy:.1%}")
        print(f"Sentiment classification accuracy: {sentiment_accuracy:.1%}")

        return aspect_accuracy, sentiment_accuracy


if __name__ == "__main__":
    print("Starting model training...")
    trainer = RuleBasedModelTrainer()
    trainer.train_complete_model()