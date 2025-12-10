from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import os
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import joblib
import numpy as np
from RuleMatch.rule_match import RuleMatcher
app = Flask(__name__)
CORS(app)


class LLMAssistant:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load LLM model - using GPU"""
        try:
            print("Loading LLM model to GPU...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            if torch.cuda.is_available():
                print("CUDA detected, model will load to GPU")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                print("CUDA not detected, model will load to CPU")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True
                )

            device = next(self.model.parameters()).device
            print(f"LLM model loaded successfully, running on: {device}")

        except Exception as e:
            print(f"LLM model loading failed: {e}")
            try:
                print("Attempting simplified loading...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
                device = next(self.model.parameters()).device
                print(f"LLM model simplified loading successful, running on: {device}")
            except Exception as e2:
                print(f"LLM model complete loading failed: {e2}")
                self.model = None

    def analyze_with_llm(self, feedback_text):
        """Use LLM to analyze feedback - improved prompt"""
        if self.model is None:
            return None

        try:
            prompt = f"""请严格分析以下航空服务反馈，只识别明确提到的服务方面和情感倾向。

    服务方面定义：
    - booking: 预订、订票、预约、下单
    - payment: 支付、付款、收费、价格  
    - checkin: 值机、办理登机
    - boarding: 登机、登机口、上飞机
    - flight: 航班、飞行、延误、取消、飞机
    - service: 服务、态度、空姐、乘务员
    - price: 价格、价钱、贵、便宜
    - refund: 退款、退票、改签
    - customer_service: 客服、客户服务、投诉
    - baggage: 行李、托运
    - seating: 座位、选座、座椅

    情感倾向：positive(正面)、negative(负面)、neutral(中性)

    反馈内容："{feedback_text}"

    要求：
    1. 只输出JSON格式，不要其他任何文字
    2. 只包含明确提到的方面，不要推测
    3. 情感判断要基于文本证据

    输出格式：
    {{"aspects": ["aspect1"], "sentiments": {{"aspect1": "sentiment"}}, "overall_sentiment": "sentiment"}}

    JSON输出："""

            # Get the device where the model is located
            device = next(self.model.parameters()).device

            # Tokenize and move to same device as model
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=150,  # Reduce generation length
                    do_sample=True,
                    temperature=0.1,  # Lower temperature, more deterministic output
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=inputs["attention_mask"],
                    eos_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove prompt part, keep only generated content
            if "JSON输出：" in response:
                response = response.split("JSON输出：")[1].strip()

            json_str = self._extract_json(response)

            if json_str:
                result = json.loads(json_str)
                print(f"LLM analysis result: {result}")
                return result
            else:
                print("LLM failed to return valid JSON")
                print(f"Cleaned response: {response}")
                return None

        except Exception as e:
            print(f"LLM analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_json(self, text):
        """Extract JSON string from text - enhanced version"""
        try:
            # Method 1: Directly find JSON structure
            json_pattern = r'\{[^{}]*"[^"]*"[^{}]*\{[^{}]*"[^"]*"[^{}]*:[^}]*\}[^{}]*\}'
            matches = re.findall(json_pattern, text, re.DOTALL)

            if matches:
                for match in matches:
                    try:
                        # Try to parse found JSON
                        result = json.loads(match)
                        # Verify required fields
                        if all(key in result for key in ["aspects", "sentiments", "overall_sentiment"]):
                            print(f"Extracted valid JSON from response: {result}")
                            return match
                    except:
                        continue

            # Method 2: Find JSON in code blocks
            code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            code_matches = re.findall(code_block_pattern, text, re.DOTALL)
            if code_matches:
                for match in code_matches:
                    try:
                        result = json.loads(match)
                        if all(key in result for key in ["aspects", "sentiments", "overall_sentiment"]):
                            print(f"Extracted valid JSON from code block: {result}")
                            return match
                    except:
                        continue

            # Method 3: Find last {...} structure
            last_brace = text.rfind('{')
            last_end_brace = text.rfind('}')
            if last_brace != -1 and last_end_brace > last_brace:
                json_str = text[last_brace:last_end_brace + 1]
                try:
                    result = json.loads(json_str)
                    if all(key in result for key in ["aspects", "sentiments", "overall_sentiment"]):
                        print(f"Extracted valid JSON from end: {result}")
                        return json_str
                except:
                    pass

            print(f"Unable to extract valid JSON from response")
            print(f"Original response content: {text}")
            return None

        except Exception as e:
            print(f"JSON extraction failed: {e}")
            return None


class LocalModelAnalyzer:
    def __init__(self, model_dir='LocalTrainedModel'):
        """Initialize locally trained model"""
        try:
            self.vectorizer = joblib.load(f'{model_dir}\\Aviation_vectorizer.pkl')
            self.aspect_classifier = joblib.load(f'{model_dir}\\Aviation_aspect_classifier.pkl')
            self.sentiment_classifier = joblib.load(f'{model_dir}\\Aviation_sentiment_classifier.pkl')
            self.aspect_mlb = joblib.load(f'{model_dir}\\Aspect_mlb.pkl')
            self.models_loaded = True
            print("Local model loaded successfully")
        except Exception as e:
            print(f"Local model loading failed: {e}")
            self.models_loaded = False

    def analyze_with_local_model(self, feedback_text):
        """Use locally trained model for analysis"""
        if not self.models_loaded:
            return None, 0.0

        try:
            # Text vectorization
            text_vector = self.vectorizer.transform([feedback_text])

            # Aspect classification
            aspect_proba = self.aspect_classifier.predict_proba(text_vector)
            aspect_predictions = self.aspect_classifier.predict(text_vector)

            # Get confidence
            max_confidence = np.max(aspect_proba)

            # Decode aspect labels
            aspect_labels = self.aspect_mlb.inverse_transform(aspect_predictions)

            # Sentiment analysis
            sentiment_prediction = self.sentiment_classifier.predict(text_vector)[0]
            sentiment_proba = self.sentiment_classifier.predict_proba(text_vector)[0]
            sentiment_confidence = np.max(sentiment_proba)

            # Build result
            aspects = []
            sentiments = {}

            if aspect_labels and len(aspect_labels[0]) > 0:
                for aspect in aspect_labels[0]:
                    aspects.append(aspect)
                    sentiments[aspect] = sentiment_prediction
            else:
                # If no specific aspect detected, default to service
                aspects = ["service"]
                sentiments = {"service": sentiment_prediction}

            # Overall sentiment uses sentiment analysis result
            overall_sentiment = sentiment_prediction

            result = {
                "aspects": aspects,
                "sentiments": sentiments,
                "overall_sentiment": overall_sentiment
            }

            # Return result and confidence (take minimum of aspect and sentiment confidence)
            confidence = min(max_confidence, sentiment_confidence)

            print(f"Local model analysis result: {result}, confidence: {confidence}")
            return result, confidence

        except Exception as e:
            print(f"Local model analysis failed: {e}")
            return None, 0.0


class FeedbackSystem:
    def __init__(self, storage_file="templates/feedback_data.json"):
        self.storage_file = storage_file
        self.feedback_data = self.load_feedback_data()

        # Initialize LLM assistant using relative path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "DeepSeekModel", "snapshots", "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562")
        self.llm_assistant = LLMAssistant(model_path)

        # Initialize local model
        self.local_analyzer = LocalModelAnalyzer()

        # Initialize rule matcher
        self.rule_matcher = RuleMatcher()

        # Define service aspects
        self.service_aspects = self.rule_matcher.service_aspects

    def load_feedback_data(self):
        """Load historical feedback data"""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
            return []
        except Exception as e:
            print(f"Data loading failed: {e}")
            return []

    def save_feedback_data(self):
        """Save feedback data to file"""
        try:
            # Ensure templates directory exists
            os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Data saving failed: {e}")

    def analyze_feedback(self, feedback_text):
        """Integrated feedback analysis using three models"""
        if not feedback_text or not isinstance(feedback_text, str):
            return self._get_default_result()

        print(f"Analysis text: {feedback_text}")
        text_length = len(feedback_text)

        # 1. Use local model when text length less than 5 characters or local model confidence low
        if text_length < 5:
            print("Text length less than 5 characters, using local model")
            local_result, confidence = self.local_analyzer.analyze_with_local_model(feedback_text)
            if local_result and confidence >= 0.6:
                local_result["_analysis_method"] = "local_model"
                local_result["_confidence"] = confidence
                return local_result
            else:
                print("Local model confidence low, using rule system")
                return self._analyze_with_rules_fallback(feedback_text)

        # 2. Use rule matching method when text length greater than or equal to 5 characters
        print("Text length greater than or equal to 5 characters, using rule system")
        rule_based_result = self.rule_matcher.analyze_with_rules(feedback_text)
        print(f"Rule system result: {rule_based_result}")

        # 3. Check rule system confidence, use LLM fallback when confidence low
        rule_confidence = self.rule_matcher.calculate_rule_confidence(rule_based_result, feedback_text)
        print(f"Rule system confidence: {rule_confidence}")

        if rule_confidence < 0.6 and self.rule_matcher.need_llm_fallback(rule_based_result, feedback_text):
            print("Rule system confidence low, attempting LLM fallback")
            llm_result = self.llm_assistant.analyze_with_llm(feedback_text)
            if llm_result and self._validate_llm_result(llm_result):
                print("Using LLM analysis result")
                llm_result["_analysis_method"] = "llm"
                llm_result["_confidence"] = 0.8  # LLM default higher confidence
                return llm_result

        # If LLM also fails or confidence sufficient, use rule result
        rule_based_result["_analysis_method"] = "rules"
        rule_based_result["_confidence"] = rule_confidence
        return rule_based_result

    def _analyze_with_rules_fallback(self, feedback_text):
        """Rule system fallback analysis"""
        rule_result = self.rule_matcher.analyze_with_rules(feedback_text)
        rule_result["_analysis_method"] = "rules_fallback"
        rule_result["_confidence"] = self.rule_matcher.calculate_rule_confidence(rule_result, feedback_text)
        return rule_result

    def _get_default_result(self):
        """Get default result"""
        return {
            "aspects": ["service"],
            "sentiments": {"service": "neutral"},
            "overall_sentiment": "neutral",
            "_analysis_method": "default",
            "_confidence": 0.0
        }

    def _validate_llm_result(self, llm_result):
        """Verify if LLM returned result is valid"""
        try:
            # Check required fields
            if not all(key in llm_result for key in ["aspects", "sentiments", "overall_sentiment"]):
                return False

            # Check if aspects is a list
            if not isinstance(llm_result["aspects"], list):
                return False

            # Check if sentiments is a dictionary
            if not isinstance(llm_result["sentiments"], dict):
                return False

            # Check if sentiment values are valid
            valid_sentiments = ["positive", "negative", "neutral"]
            for aspect, sentiment in llm_result["sentiments"].items():
                if sentiment not in valid_sentiments:
                    return False

            if llm_result["overall_sentiment"] not in valid_sentiments:
                return False

            return True

        except Exception as e:
            print(f"LLM result validation failed: {e}")
            return False

    def add_feedback(self, user_feedback):
        """Add feedback"""
        try:
            analysis_result = self.analyze_feedback(user_feedback)

            feedback_record = {
                "id": len(self.feedback_data) + 1,
                "timestamp": datetime.now().isoformat(),
                "raw_feedback": user_feedback,
                "analysis": analysis_result
            }

            self.feedback_data.append(feedback_record)
            self.save_feedback_data()

            print(f"Final analysis result: {analysis_result}")
            return analysis_result
        except Exception as e:
            print(f"Feedback addition failed: {e}")
            return {
                "aspects": [],
                "sentiments": {},
                "overall_sentiment": "neutral"
            }

    def get_statistics(self):
        """Get statistics - calculate overall data based on statistics of each service aspect"""
        try:
            if not self.feedback_data:
                return {
                    "total_feedbacks": 0,
                    "overall_sentiments": {
                        "positive": {"count": 0, "percentage": 0},
                        "negative": {"count": 0, "percentage": 0},
                        "neutral": {"count": 0, "percentage": 0}
                    },
                    "aspect_stats": {},
                    "recent_feedbacks": []
                }

            # Initialize statistics for each service aspect
            aspect_stats = {}
            for aspect in self.service_aspects:
                aspect_stats[aspect] = {"positive": 0, "negative": 0, "neutral": 0, "total": 0}

            # Count sentiment distribution for each service aspect across all feedback
            for feedback in self.feedback_data:
                analysis = feedback.get("analysis", {})
                aspects = analysis.get("aspects", [])
                sentiments = analysis.get("sentiments", {})

                for aspect in aspects:
                    if aspect in aspect_stats:
                        if isinstance(sentiments, dict):
                            sentiment = sentiments.get(aspect, "neutral")
                        else:
                            sentiment = "neutral"

                        if sentiment in aspect_stats[aspect]:
                            aspect_stats[aspect][sentiment] += 1
                        aspect_stats[aspect]["total"] += 1

            # Calculate percentages for each service aspect
            aspect_stats_percentage = {}
            total_aspect_feedbacks = 0  # Total feedback count for all service aspects (deduplicated count)
            overall_sentiments = {"positive": 0, "negative": 0, "neutral": 0}

            for aspect, stats in aspect_stats.items():
                if stats["total"] > 0:
                    # Calculate percentage for this service aspect
                    aspect_stats_percentage[aspect] = {
                        "positive": round((stats["positive"] / stats["total"]) * 100, 1),
                        "negative": round((stats["negative"] / stats["total"]) * 100, 1),
                        "neutral": round((stats["neutral"] / stats["total"]) * 100, 1),
                        "total": stats["total"]
                    }

                    # Accumulate sentiment counts from each service aspect to calculate overall statistics
                    overall_sentiments["positive"] += stats["positive"]
                    overall_sentiments["negative"] += stats["negative"]
                    overall_sentiments["neutral"] += stats["neutral"]
                    total_aspect_feedbacks += stats["total"]

            # Calculate overall percentages based on statistics from each service aspect
            overall_stats = {}
            if total_aspect_feedbacks > 0:
                for sentiment, count in overall_sentiments.items():
                    percentage = (count / total_aspect_feedbacks) * 100
                    overall_stats[sentiment] = {
                        "count": count,
                        "percentage": round(percentage, 1)
                    }
            else:
                # If no service aspect feedback, use default values
                overall_stats = {
                    "positive": {"count": 0, "percentage": 0},
                    "negative": {"count": 0, "percentage": 0},
                    "neutral": {"count": 0, "percentage": 0}
                }

            return {
                "total_feedbacks": len(self.feedback_data),
                "overall_sentiments": overall_stats,
                "aspect_stats": aspect_stats_percentage
            }

        except Exception as e:
            print(f"Getting statistics failed: {e}")
            return {
                "error": f"Getting statistics failed: {str(e)}"
            }


# Initialize system
feedback_system = FeedbackSystem()


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400

        feedback_text = data.get('feedback', '').strip()

        if not feedback_text:
            return jsonify({"error": "Feedback content cannot be empty"}), 400

        analysis_result = feedback_system.add_feedback(feedback_text)

        return jsonify({
            "success": True,
            "message": "Thank you for your feedback! Your opinion is very important to us.",
            "analysis": analysis_result
        })

    except Exception as e:
        return jsonify({"error": f"Error processing feedback: {str(e)}"}), 500


@app.route('/get_statistics', methods=['GET'])
def get_statistics():
    """Get statistics"""
    try:
        stats = feedback_system.get_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": f"Error getting statistics: {str(e)}"}), 500


if __name__ == '__main__':
    # Ensure templates directory exists
    if not os.path.exists('templates'):
        os.makedirs('templates')

    # Ensure data file exists
    if not os.path.exists('templates/feedback_data.json'):
        with open('templates/feedback_data.json', 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)

    app.run(debug=True, port=5000)