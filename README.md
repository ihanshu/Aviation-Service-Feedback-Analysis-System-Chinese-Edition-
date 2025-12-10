# Aviation Service Feedback Analysis System (Chinese Edition)

## Project Introduction

This is a Flask-based Chinese intelligent analysis system for aviation service feedback, integrating three analysis methods: 
rule matching, local machine learning models, and large language models (LLM). 
It can automatically identify service aspects and sentiment tendencies in feedback.

The system provides customer feedback collection and analysis service for flight-booking agent services. 
It receives a user's feedback and returns to the user the statistical summary of feedback received so far, 
showing percentages of positive and negative feedback on different aspects of the flight-booking service.


## System Architecture
- **Architecture Design**: Multi-model integrated analysis framework
- **Algorithm Development**: Rule matching + Machine learning + Large language models
- **System Optimization**: Performance tuning and user experience improvement

### Core Modules

1. **Rule Matching Engine (RuleMatcher)** - Rule-based analysis using keywords and sentiment dictionaries
2. **Local Model Analysis (LocalModelAnalyzer)** - Using pre-trained machine learning models
3. **Large Language Model Analysis (LLMAssistant)** - Intelligent analysis based on DeepSeek model
4. **Feedback Management System (FeedbackSystem)** - Analysis scheduler integrating three models

### Analysis Workflow

```
Text Input → Length Judgment → Model Selection → Result Return
     ↓
<6 chars: Local Model → Confidence<0.6 → Rule System
     ↓
≥6 chars: Rule System → Confidence<0.6 → LLM Fallback
```

## Features

### Multi-Model Integration

- **Intelligent Routing**: Automatically selects the optimal analysis model based on text length and confidence
- **Fallback Mechanism**: Automatically enables LLM analysis when rule system confidence is low
- **Service Classification**: Automatically categorizes as "service" when no specific aspect is identified

### Service Aspect Identification

Supports identification of 11 aviation service aspects:

- `booking` - `payment` - `checkin` - `seating`
- `boarding` - `flight` - `service` - `baggage`
- `price` - `refund` - `customer_service`

### Sentiment Analysis

- **Sentiment Tendency**: positive, negative, neutral
- **Sarcasm Detection**: Intelligently detects irony and sarcastic expressions

## Installation & Deployment

### Environment Requirements

```bash
Flask==3.1.2
jieba==0.42.1
joblib==1.5.2
scikit-learn==1.7.2
torch==2.9.0+cu126
torchvision==0.24.0+cu126
transformers==4.57.1
```

### API Endpoints

#### Submit Feedback

**POST** `/submit_feedback`

```json
{
  "feedback": "航班延误了3小时，服务态度很好但餐食一般"
}
```

Response Example:
```json
{
  "success": true,
  "message": "感谢您的反馈！您的意见对我们非常重要。",
  "analysis": {
    "aspects": ["flight", "service", "service"],
    "sentiments": {
      "flight": "negative",
      "service": "positive",
      "service": "neutral"
    },
    "overall_sentiment": "neutral",
    "_analysis_method": "rules",
    "_confidence": 0.85
  }
}
```

#### Get Statistics

**GET** `/get_statistics`

Response Example:
```json
{
  "total_feedbacks": 150,
  "overall_sentiments": {
    "positive": {"count": 80, "percentage": 53.3},
    "negative": {"count": 45, "percentage": 30.0},
    "neutral": {"count": 25, "percentage": 16.7}
  },
  "aspect_stats": {
    "service": {
      "positive": 65.2,
      "negative": 20.5,
      "neutral": 14.3,
      "total": 112
    }
  }
}
```

## Project Structure

```
aviation-feedback-system/
├── app.py                         # Main application
├── RuleMatch/
│   └── rule_match.py              # Rule matching engine
├── LocalTrainedModel/             # Local machine learning models
├── DeepSeekModel/                 # Large language model
├── templates/
│   ├── index.html                 # Frontend page
│   └── feedback_data.json         # Feedback data storage
├── requirements.txt               # Dependencies list
└── README.md                      # Project documentation
```

### Model Path Configuration

Modify model paths in `app.py`:

```python
# Local model path
model_dir = 'LocalTrainedModel'

# DeepSeek model path (relative path)
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "DeepSeekModel", "snapshots", "model-folder")
```

### Analysis Threshold Configuration

```python
# Text length threshold
TEXT_LENGTH_THRESHOLD = 6

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.6

# LLM default confidence
LLM_DEFAULT_CONFIDENCE = 0.8
```

## Usage Examples

### Simple Feedback

```python
Input: "服务很好"
Output: {"aspects": ["service"], "sentiments": {"service": "positive"}}
```

### Complex Feedback

```python
Input: "航班延误了2小时，但空乘服务很周到，餐食也不错"
Output: {
  "aspects": ["flight", "service", "service"],
  "sentiments": {
    "flight": "negative", 
    "service": "positive",
    "service": "positive"
  },
  "overall_sentiment": "positive"
}
```

### Sarcastic Feedback

```python
Input: "要是托运能像收费一样快就好了"
Output: {
  "aspects": ["service"],
  "sentiments": {"service": "negative"},
  "overall_sentiment": "negative"
}
```

## Technical Highlights

### Intelligent Model Scheduling

- **Adaptive Selection**: Automatically selects the best analysis model based on text characteristics
- **Confidence Assessment**: Multi-dimensional evaluation of analysis result reliability
- **Performance Optimization**: Uses lightweight models for short texts, deep analysis for long texts

### Advanced Language Processing

- **Chinese Word Segmentation Optimization**: Custom dictionary for aviation domain
- **Irony Detection**: Pattern-based sarcastic expression identification

### Common Issues

1. **Model Loading Failed**
   - Check if model file paths are correct
   - Verify model file integrity

2. **Inaccurate Analysis Results**
   - Check if rule dictionaries need updating
   - Verify LLM model is loaded correctly

3. **Service Startup Failed**
   - Check if port 5000 is occupied
   - Confirm all dependencies are properly installed

## Configuration

**Install Dependencies**
```bash
pip install -r requirements.txt
```

**Prepare Model Files**
```bash
# Local Machine Learning Model
LocalTrainedModel/
├── Aviation_vectorizer.pkl
├── Aviation_aspect_classifier.pkl
├── Aviation_sentiment_classifier.pkl
└── Aspect_mlb.pkl

# DeepSeek LLM
DeepSeekModel/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562/
(We use DeepSeek-R1-Distill-Qwen-1.5B in the system because of the limitation of PC hardware)
```

**Start Service**
```bash
python app.py
```

**Start Using the System**
```
Open browser: http://localhost:5000
```
