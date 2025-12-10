import jieba
import re


class RuleMatcher:
    def __init__(self):
        # Define service aspects
        self.service_aspects = [
            "booking", "payment", "checkin", "boarding",
            "flight", "service", "price", "refund",
            "customer_service", "baggage", "seating"
        ]

        # Initialize jieba tokenizer
        self.init_jieba()

        # Aspect detection keywords
        self.aspect_keywords = {
            "booking": ["预订", "订票", "预约", "下单", "买票", "购票", "预定", "票务", "官网", "APP", "网站"],
            "payment": ["支付", "付款", "收费", "价钱", "付费", "价格", "支付宝", "微信支付", "信用卡", "银行卡",
                        "转账"],
            "checkin": ["值机", "办理登机", "值机手续", "checkin", "柜台", "自助值机", "网上值机", "值机台"],
            "boarding": ["登机", "登机口", "登机过程", "上飞机", "boarding", "登机牌", "登机时间", "登机广播"],
            "flight": ["航班", "飞行", "延误", "取消", "准点", "准时", "飞机", "航空", "起飞", "降落", "航程",
                       "飞行时间", "机长", "飞行员", "飞行技术", "平稳", "颠簸", "气流"],
            "service": ["服务", "态度", "帮助", "空姐", "乘务员", "服务员", "工作人员", "空乘", "地勤", "机组成员",
                        "餐"],
            "price": ["价格", "价钱", "贵", "便宜", "性价比", "收费", "费用", "票价", "机票价格", "折扣", "优惠"],
            "refund": ["退款", "退票", "退钱", "退货", "改签", "退改", "退票费", "手续费", "改期"],
            "customer_service": ["客服", "客户服务", "咨询", "电话", "热线", "支持", "投诉", "售后", "回复", "解答"],
            "baggage": ["行李", "托运", "行李箱", "手提行李", "行李额", "行李托运", "行李提取", "转盘", "超重",
                        "行李费"],
            "seating": ["座位", "选座", "座椅", "坐位", "腿部空间", "座位舒适", "靠窗", "靠走道", "座位间距", "前排",
                        "后排"]
        }

        # Extended sentiment dictionary
        self.positive_words = {
            "strong": [
                "很好", "非常好", "特别棒", "极好", "完美", "优秀", "满意", "点赞", "牛逼", "超赞",
                "太棒了", "完美无缺", "无可挑剔", "一流", "顶尖", "出色", "卓越", "棒极了", "绝了",
                "惊喜", "惊艳", "物超所值", "强烈推荐", "十分满意", "非常满意", "极其满意", "点赞",
                "专业", "周到", "贴心", "热情", "耐心", "细致", "快捷", "高效", "舒适", "宽敞",
                "无可挑剔", "完美体验", "超出预期", "令人惊叹", "五星好评", "强烈推荐", "物超所值",
                "宾至如归", "服务周到", "态度亲切", "技术娴熟", "安全可靠", "准时准点", "干净整洁", "无微不至"
                                                                                                    "设施完善",
                "体验极佳", "值得推荐", "下次还选", "印象深刻", "非常专业", "超级满意", "安利",
                "奈斯"
            ],
            "medium": [
                "好", "不错", "可以", "还行", "舒适", "方便", "快捷", "满意", "愉快", "顺利",
                "靠谱", "稳定", "干净", "整洁", "周到", "贴心", "专业", "高效", "及时", "合理",
                "舒适", "宽敞", "明亮", "温暖", "凉爽", "安静", "平稳", "安全", "可靠", "准时",
                "标准", "规范", "正规", "有序", "清晰", "明确", "简单", "容易", "便捷", "快速",
                "舒服", "便宜", "快"
            ],
            "weak": [
                "一般", "还好", "过得去", "还行", "马马虎虎", "勉强", "将就", "凑合", "尚可",
                "普通", "平常", "常规", "基本", "正常", "达标", "合格", "符合预期", "中规中矩",
                "不算差", "还可以", "说得过去", "能接受", "不太差", "基本满意", "没有大问题"
            ]
        }

        self.neutral_words = {
            "medium": [
                "一般", "普通", "平常", "常规", "标准", "正常", "基本", "还行", "尚可",
                "中规中矩", "马马虎虎", "过得去", "勉强", "将就", "凑合", "不算好也不算差",
                "没什么特别", "没什么印象", "普普通通", "平平常常", "没什么感觉", "就那么回事",
                "没什么亮点", "没什么缺点", "不好不坏", "中等水平", "平均水平", "符合标准",
                "达到预期", "基本达标", "合格水平", "正常水平", "标准服务", "常规操作"
            ],
            "weak": [
                "略微", "稍微", "有点", "有些", "还算", "基本", "大致", "总体", "整体",
                "基本上", "大体上", "总体上", "大致上", "差不多", "接近", "类似", "相当",
                "相对", "比较", "较为", "算是", "视为", "当作", "看成"
            ]
        }

        self.negative_words = {
            "strong": [
                "很差", "非常差", "特别差", "极差", "糟糕", "垃圾", "废物", "失望", "屎", "操蛋",
                "坑爹", "骗钱", "垃圾服务", "差劲", "烂透了", "恶心", "愤怒", "气愤", "火大",
                "操", "妈的", "鸡巴", "去死", "倒闭", "投诉", "差评", "再也不坐", "拉黑", "不敢恭维",
                "坑人", "欺诈", "骗子", "黑心", "无耻", "卑鄙", "恶心死了", "气死", "想骂人",
                "史上最差", "一生黑", "千万别选", "后悔", "想退票", "浪费时间", "浪费钱", "坑爹",
                "折磨", "受罪", "煎熬", "崩溃", "绝望", "愤怒至极", "极度不满", "令人发指", "byd",
                "无法忍受", "难以接受", "太差了", "糟透了", "烂到极点", "服务恶劣", "态度恶劣",
                "技术差劲", "管理混乱", "系统垃圾", "体验极差", "再也不会", "永久拉黑", "拉胯", "拉跨"
            ],
            "medium": [
                "差", "不好", "不行", "难受", "不舒服", "麻烦", "复杂", "混乱", "拥挤", "吵闹",
                "脏乱", "延误", "取消", "慢", "久等", "不便", "困难", "问题", "故障", "错误",
                "失望", "不满意", "不愉快", "糟糕", "差劲", "烂", "坑", "贵", "昂贵", "不合理",
                "不专业", "不周到", "冷漠", "粗暴", "不耐烦", "拖延", "低效", "狭窄", "拥挤",
                "陈旧", "破旧", "脏污", "异味", "嘈杂", "闷热", "寒冷", "颠簸", "不稳", "贵",
                "脏", "臭", "丑", "难过", "难受", "悲伤", "什么玩意", "难吃", "不好吃", "差劲"
            ],
            "weak": [
                "一般", "普通", "马马虎虎", "勉强", "有待提高", "需要改进", "不足", "欠缺", "不够好",
                "不太满意", "有点失望", "略有不足", "小问题", "小瑕疵", "美中不足", "可以更好", "不太"
                                                                                                "需要加强", "有待完善",
                "不够完美", "略有遗憾", "稍有不足", "差强人意", "偏贵"
            ]
        }

        # Negation words and degree adverbs
        self.negation_words = ["不", "没", "没有", "无", "未", "别", "莫", "勿", "非", "不是", "不会", "不能", "不太",
                               "不够", "不怎么"]
        # Transition words
        self.transition_words = ["但是", "可是", "然而", "不过", "却", "但", "只是", "唯独", "除了", "除了...之外"]

    def init_jieba(self):
        """Initialize tokenizer dictionary"""
        custom_words = [
            '值机', '登机', '托运', '行李', '航班', '延误', '取消', '改签',
            '退票', '客服', '空姐', '乘务员', '座位', '选座', '餐食', '准点',
            '安检', '候机', '登机口', '摆渡车', '头等舱', '经济舱', '商务舱',
            '行李额', '手提行李', '托运行李', '航班动态', '机票', '机场大巴',
            '航站楼', '免税店', '贵宾厅', '休息室', '机上娱乐', '毛毯', '枕头',
            '饮料', '餐点', '飞机餐', '洗手间', '卫生间', '安全演示', '救生衣',
            '氧气面罩', '安全带', '颠簸', '气流', '转机', '经停', '直飞', 'WIFI',
            'Wi-Fi'
        ]
        for word in custom_words:
            jieba.add_word(word)

    def analyze_with_rules(self, feedback_text):
        """Analyze feedback using rule-based system"""
        if not feedback_text or not isinstance(feedback_text, str):
            return self._get_default_result()

        print(f"Rule system analyzing text: {feedback_text}")

        sentences = self.split_sentences_intelligently(feedback_text, self.transition_words)
        aspect_sentiments = {}

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            print(f"Analyzing sentence: {sentence}")

            # Find aspects involved in the sentence
            for aspect, keywords in self.aspect_keywords.items():
                for kw in keywords:
                    if kw in sentence:
                        sentiment = self.analyze_aspect_sentiment(
                            sentence, kw,
                            self.positive_words, self.neutral_words, self.negative_words,
                            self.negation_words
                        )
                        print(f"  Aspect '{aspect}' keyword '{kw}' sentiment: {sentiment}")

                        # If same aspect appears multiple times, keep the stronger sentiment
                        if aspect in aspect_sentiments:
                            new_strength = self.get_sentiment_strength(sentiment)
                            old_strength = self.get_sentiment_strength(aspect_sentiments[aspect])
                            if new_strength > old_strength:
                                aspect_sentiments[aspect] = sentiment
                        else:
                            aspect_sentiments[aspect] = sentiment

        # If no specific aspects detected, use overall sentiment analysis and default to service
        if not aspect_sentiments:
            print("No specific aspects detected, using overall sentiment analysis and defaulting to service category")
            overall = self.analyze_overall_sentiment(
                feedback_text, self.positive_words, self.neutral_words, self.negative_words, self.negation_words
            )
            return {
                "aspects": ["service"],
                "sentiments": {"service": overall},
                "overall_sentiment": overall
            }

        # Calculate overall sentiment
        overall = self.calculate_overall_sentiment(aspect_sentiments)

        print(f"Final analysis result: aspects={list(aspect_sentiments.keys())}, sentiments={aspect_sentiments}, overall={overall}")

        return {
            "aspects": list(aspect_sentiments.keys()),
            "sentiments": aspect_sentiments,
            "overall_sentiment": overall
        }

    def _get_default_result(self):
        """Get default result"""
        return {
            "aspects": ["service"],
            "sentiments": {"service": "neutral"},
            "overall_sentiment": "neutral"
        }

    def _detect_sarcasm_and_irony(self, sentence):
        """Detect sarcasm and irony expressions - complete version"""
        irony_patterns = [
            # Clear irony markers
            r"真是谢谢.*(高效|快速|优秀|出色|完美|丰盛|周到|专业|贴心|舒适|便捷|便宜|实惠)",
            r"可真是.*(丰盛|豪华|舒适|宽敞|便捷|快速|高效|优秀|出色|完美|周到)",
            r"太.*了.*(感谢|满意|惊喜|开心|感动|荣幸)",
            r"真是.*呢.*(高效|快速|优秀|出色|完美|丰盛|周到|专业)",
            r"让人.*(印象深刻|难忘|感动|惊喜|大开眼界)",
            r"感谢.*让.*体验.*(高效|快速|优秀|出色|完美)",

            # Quotation irony (words in quotes indicating opposite meaning)
            r"['\"](高效|快速|优秀|出色|完美|丰盛|周到|舒适|便捷|便宜|实惠|豪华|专业|贴心)['\"]",

            # Contrast irony
            r"要是.*像.*那么.*就好",
            r"要是.*有.*就好"
            r"要是.*像.*就好"
            r"速度.*像.*收钱.*快",
            r"效率.*像.*收费.*高",
            r"服务.*像.*价格.*好",
            r".*像.*一样.*快",
            r".*像.*一样.*好",

            # Degree adverbs + negative facts
            r"(简直|实在|确实|绝对|真的).*(太|很|特别|非常).*(慢|差|少|贵|乱|久|难|麻烦|糟糕|差劲)",
            r"(真是|真的是).*(太|很|特别|非常).*(贴心|周到|专业|高效|快速).*但是",

            # Internet irony and colloquial expressions
            r"(绝了|牛逼|奈斯|棒呆了).*但是",
            r"真是.*谢谢.*您.*嘞",
            r"我.*谢谢.*您",
            r"这可真是.*了",
            r"好一个.*",

            # Expectation vs reality contrast
            r"本以为.*结果",
            r"期待.*却",
            r"想象.*现实",
            r"以为.*实际上",

            # Minimizing expression irony
            r"就.*一包.*一瓶.*一个",
            r"只有.*和",
            r"仅仅.*就",

            # Time contrast irony
            r"两小时.*两分钟",
            r"半天.*一分钟",
            r"很久.*很快",

            # Price service contrast
            r"钱.*收.*快.*服务.*慢",
            r"收费.*高效.*服务.*低效"
        ]

        # Check each pattern
        for pattern in irony_patterns:
            if re.search(pattern, sentence):
                print(f"Detected sarcastic expression: '{sentence}' -> matched pattern: {pattern}")
                return True

        # Additional word-level detection
        irony_words = [
            "惊喜", "惊喜不断", "印象深刻", "难忘", "感动", "开心",
            "满意", "荣幸", "谢谢", "感谢", "太棒了", "完美"
        ]

        # If contains these words and context has negative facts, judge as irony
        has_irony_word = any(word in sentence for word in irony_words)
        if has_irony_word:
            negative_facts = [
                "排队", "等待", "延误", "取消", "慢", "差", "少", "贵",
                "乱", "久", "难", "麻烦", "糟糕", "差劲", "垃圾", "烂",
                "一包", "一瓶", "一点点", "几乎没", "基本没有"
            ]
            has_negative_fact = any(fact in sentence for fact in negative_facts)
            if has_negative_fact:
                print(f"Detected word-level irony: '{sentence}'")
                return True

        return False

    def _analyze_ironic_sentiment(self, sentence, keyword, positive_words, neutral_words, negative_words,
                                  negation_words):
        """Analyze real sentiment of ironic sentences - complete version"""
        print(f"  Analyzing ironic sentence: {sentence}")

        # Irony intensity detection
        irony_intensity = 0

        # Strong irony patterns
        strong_irony_patterns = [
            r"真是谢谢.*",
            r"可真是.*",
            r"太.*了.*感谢",
            r".*像.*收钱.*快",
            r"两小时.*两分钟"
        ]

        for pattern in strong_irony_patterns:
            if re.search(pattern, sentence):
                irony_intensity += 2

        # Specific negative fact detection
        negative_facts = {
            # Time related
            "排队": 2, "等待": 2, "延误": 3, "取消": 3, "慢": 2, "久": 2,
            "小时": 3, "半天": 2, "很久": 2, "长时间": 2,

            # Quality related
            "差": 2, "糟糕": 3, "差劲": 3, "垃圾": 3, "烂": 3,
            "难吃": 2, "难喝": 2, "难受": 2, "不舒服": 2,

            # Quantity related
            "少": 2, "一包": 2, "一瓶": 2, "一点点": 2, "几乎没": 2,
            "基本没有": 2, "只有": 2, "仅仅": 2,

            # Price related
            "贵": 2, "昂贵": 2, "收费": 2, "花钱": 2,

            # Service problems
            "麻烦": 2, "复杂": 2, "混乱": 2, "乱": 2
        }

        for fact, weight in negative_facts.items():
            if fact in sentence:
                irony_intensity += weight

        # If strong irony evidence, directly return negative
        if irony_intensity >= 3:
            print(f"  Strong irony evidence, intensity: {irony_intensity}, directly judging as negative")
            return "negative"

        # Medium intensity irony, combine with traditional sentiment analysis
        elif irony_intensity >= 1:
            print(f"  Medium irony evidence, intensity: {irony_intensity}, combining with sentiment analysis")

            # Perform traditional sentiment analysis but bias towards negative
            pos_score, neu_score, neg_score = 0, 0, 0

            # Analyze positive words (in irony these may indicate negative)
            for level, words in positive_words.items():
                for w in words:
                    if w in sentence:
                        weight = self.get_intensity_weight(level)
                        # In ironic context, positive words may strengthen negative sentiment
                        neg_score += weight * 0.5  # Positive words contribute negative score in irony

            # Analyze negative words
            for level, words in negative_words.items():
                for w in words:
                    if w in sentence:
                        weight = self.get_intensity_weight(level)
                        neg_score += weight

            # Analyze neutral words
            for level, words in neutral_words.items():
                for w in words:
                    if w in sentence:
                        weight = self.get_intensity_weight(level)
                        neu_score += weight

            # In ironic context, overall bias towards negative
            neg_score += irony_intensity * 0.5

            print(f"  Ironic context sentiment scores - positive: {pos_score}, neutral: {neu_score}, negative: {neg_score}")

            if neg_score > max(pos_score, neu_score):
                return "negative"
            else:
                return "negative"  # Default negative in ironic context

        else:
            # Weak or no irony, use traditional analysis
            print(f"  Weak irony evidence, using traditional analysis")
            return self.analyze_aspect_sentiment(sentence, keyword, positive_words, neutral_words, negative_words,
                                                 negation_words)

    def split_sentences_intelligently(self, text, transition_words):
        """Intelligently split sentences - supports multi-aspect comments without punctuation"""
        # First split by punctuation marks
        sentences = re.split(r'[，。！!.？?；;、]', text)

        # For each split sentence, check if it contains transition words that need further splitting
        final_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if contains transition words
            has_transition = False
            for transition in transition_words:
                if transition in sentence:
                    has_transition = True
                    # Further split by transition word
                    parts = sentence.split(transition)
                    for part in parts:
                        part = part.strip()
                        if part:
                            final_sentences.append(part)
                    break

            # If no transition words, add directly
            if not has_transition:
                final_sentences.append(sentence)

        # If still no split (completely no punctuation and transition words), try splitting by keywords
        if len(final_sentences) == 1 and len(final_sentences[0]) > 10:
            long_sentence = final_sentences[0]
            # Use tokenization to find possible boundaries
            words = jieba.lcut(long_sentence)
            print(f"Long sentence tokenization result: {words}")

            # Find emotion words as split points
            emotion_words = ["很好", "很棒", "不错", "满意", "很差", "糟糕", "差劲", "不好", "不满意"]
            split_points = []

            for i, word in enumerate(words):
                if word in emotion_words and i > 0 and i < len(words) - 1:
                    split_points.append(i)

            # If split points exist, re-split the sentence
            if split_points:
                final_sentences = []
                start = 0
                for point in split_points:
                    segment = ''.join(words[start:point + 1])
                    if segment.strip():
                        final_sentences.append(segment.strip())
                    start = point + 1

                # Add last segment
                if start < len(words):
                    segment = ''.join(words[start:])
                    if segment.strip():
                        final_sentences.append(segment.strip())

        print(f"Intelligent split result: {final_sentences}")
        return final_sentences

    def analyze_aspect_sentiment(self, sentence, keyword, positive_words, neutral_words, negative_words,
                                 negation_words):
        """Analyze sentiment of a specific aspect in a single sentence - integrated with complete sarcasm detection"""

        # First detect if it's a sarcastic expression
        if self._detect_sarcasm_and_irony(sentence):
            return self._analyze_ironic_sentiment(sentence, keyword, positive_words, neutral_words, negative_words,
                                                  negation_words)

        # Original sentiment analysis logic
        pos_score, neu_score, neg_score = 0, 0, 0

        # Analyze positive words
        for level, words in positive_words.items():
            for w in words:
                if w in sentence:
                    word_pos = sentence.find(w)
                    context_before = sentence[max(0, word_pos - 5):word_pos]
                    has_negation = any(n in context_before for n in negation_words)

                    weight = self.get_intensity_weight(level)
                    if has_negation:
                        neg_score += 0.8 * weight
                    else:
                        pos_score += weight

        # Analyze neutral words
        for level, words in neutral_words.items():
            for w in words:
                if w in sentence:
                    word_pos = sentence.find(w)
                    context_before = sentence[max(0, word_pos - 5):word_pos]
                    has_negation = any(n in context_before for n in negation_words)

                    weight = self.get_intensity_weight(level)
                    if has_negation:
                        pos_score += 0.3 * weight
                    else:
                        neu_score += weight

        # Analyze negative words
        for level, words in negative_words.items():
            for w in words:
                if w in sentence:
                    word_pos = sentence.find(w)
                    context_before = sentence[max(0, word_pos - 5):word_pos]
                    has_negation = any(n in context_before for n in negation_words)

                    weight = self.get_intensity_weight(level)
                    if has_negation:
                        pos_score += 0.8 * weight
                    else:
                        neg_score += weight

        print(f"  Sentiment scores - positive: {pos_score}, neutral: {neu_score}, negative: {neg_score}")

        # Determine final sentiment
        max_score = max(pos_score, neu_score, neg_score)

        if max_score == pos_score and pos_score > neg_score + 0.2 and pos_score > neu_score + 0.2:
            return "positive"
        elif max_score == neg_score and neg_score > pos_score + 0.2 and neg_score > neu_score + 0.2:
            return "negative"
        elif max_score == neu_score and neu_score > pos_score - 0.1 and neu_score > neg_score - 0.1:
            return "neutral"
        elif abs(pos_score - neg_score) < 0.3 and neu_score > 0:
            return "neutral"
        else:
            if pos_score > neg_score:
                return "positive"
            elif neg_score > pos_score:
                return "negative"
            else:
                return "neutral"

    def analyze_overall_sentiment(self, text, pos_words, neu_words, neg_words, negation_words):
        """Analyze overall text sentiment"""
        pos_score, neu_score, neg_score = 0, 0, 0

        # Calculate positive score
        for level, words in pos_words.items():
            for w in words:
                if w in text:
                    word_pos = text.find(w)
                    context_before = text[max(0, word_pos - 5):word_pos]
                    has_negation = any(n in context_before for n in negation_words)

                    weight = self.get_intensity_weight(level)
                    if has_negation:
                        neg_score += weight * 0.8
                    else:
                        pos_score += weight

        # Calculate neutral score
        for level, words in neu_words.items():
            for w in words:
                if w in text:
                    word_pos = text.find(w)
                    context_before = text[max(0, word_pos - 5):word_pos]
                    has_negation = any(n in context_before for n in negation_words)

                    weight = self.get_intensity_weight(level)
                    if has_negation:
                        pos_score += weight * 0.3
                    else:
                        neu_score += weight

        # Calculate negative score
        for level, words in neg_words.items():
            for w in words:
                if w in text:
                    word_pos = text.find(w)
                    context_before = text[max(0, word_pos - 5):word_pos]
                    has_negation = any(n in context_before for n in negation_words)

                    weight = self.get_intensity_weight(level)
                    if has_negation:
                        pos_score += weight * 0.8
                    else:
                        neg_score += weight

        print(f"Overall sentiment scores - positive: {pos_score}, neutral: {neu_score}, negative: {neg_score}")

        # Determine overall sentiment
        max_score = max(pos_score, neu_score, neg_score)

        if max_score == pos_score and pos_score > neg_score + 0.2:
            return "positive"
        elif max_score == neg_score and neg_score > pos_score + 0.2:
            return "negative"
        else:
            return "neutral"

    def get_intensity_weight(self, level):
        """Get sentiment intensity weight"""
        weights = {
            "strong": 3,
            "medium": 2,
            "weak": 1
        }
        return weights.get(level, 1)

    def get_sentiment_strength(self, sentiment):
        """Get sentiment strength value"""
        strength_map = {
            "positive": 3,
            "negative": 3,
            "neutral": 2  # Increase neutral strength to make it easier to retain
        }
        return strength_map.get(sentiment, 1)

    def calculate_overall_sentiment(self, aspect_sentiments):
        """Calculate overall sentiment based on aspect sentiments"""
        if not aspect_sentiments:
            return "neutral"

        pos_count = sum(1 for s in aspect_sentiments.values() if s == "positive")
        neg_count = sum(1 for s in aspect_sentiments.values() if s == "negative")
        neu_count = sum(1 for s in aspect_sentiments.values() if s == "neutral")

        total = len(aspect_sentiments)

        # If neutral is majority, overall is neutral
        if neu_count > pos_count and neu_count > neg_count:
            return "neutral"
        elif pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"

    def calculate_rule_confidence(self, rule_result, feedback_text):
        """Calculate confidence of rule-based system"""
        confidence = 0.5  # Base confidence

        # Increase confidence based on number of detected aspects
        aspects_count = len(rule_result.get("aspects", []))
        if aspects_count > 0:
            confidence += min(aspects_count * 0.1, 0.3)

        # Increase confidence based on sentiment strength
        sentiments = rule_result.get("sentiments", {})
        strong_sentiments = 0
        for sentiment in sentiments.values():
            if sentiment in ["positive", "negative"]:
                strong_sentiments += 1

        if strong_sentiments > 0:
            confidence += min(strong_sentiments * 0.1, 0.2)

        # Text length factor
        text_length = len(feedback_text)
        if text_length > 20:
            confidence += 0.1

        return min(confidence, 1.0)

    def need_llm_fallback(self, rule_result, feedback_text):
        """Extremely strict LLM trigger conditions"""
        aspects_count = len(rule_result.get("aspects", []))
        text_length = len(feedback_text)

        # Rule 1: Very long complex text (>80 characters) and rule system identifies few aspects
        if text_length > 80 and aspects_count <= 1:
            print("Trigger condition: Very long complex text")
            return True

        # Rule 2: Contains obvious sarcasm or irony, and rule system cannot handle it
        irony_indicators = ["真是谢谢", "可真是", "太.*了", "真是.*呢", "惊喜", "印象深刻"]
        has_irony = any(re.search(pattern, feedback_text) for pattern in irony_indicators)
        if has_irony and text_length > 30:
            # Check if rule system identified the sarcasm
            sentiments = rule_result.get("sentiments", {})
            if not any(sentiment == "negative" for sentiment in sentiments.values()):
                print("Trigger condition: Sarcastic expression not recognized by rule system")
                return True

        # Rule 3: Multiple complex negation structures
        complex_negation = [
            r"不是说.*不好.*只是", r"很难说.*满意", r"并没有.*但是",
            r"不算.*但", r"不至于.*但"
        ]
        if any(re.search(pattern, feedback_text) for pattern in complex_negation):
            print("Trigger condition: Complex negation structure")
            return True

        # Rule 4: Rule system completely fails (no aspects identified) and text is long
        if aspects_count == 0 and text_length > 20:
            print("Trigger condition: Rule system completely failed")
            return True

        # Rule 5: Extreme sentiment conflict (equal positive and negative counts and both >=2)
        sentiments = rule_result.get("sentiments", {})
        if len(sentiments) >= 4:
            pos_count = sum(1 for s in sentiments.values() if s == "positive")
            neg_count = sum(1 for s in sentiments.values() if s == "negative")
            if pos_count == neg_count and pos_count >= 2:
                print("Trigger condition: Extreme sentiment conflict")
                return True

        return False