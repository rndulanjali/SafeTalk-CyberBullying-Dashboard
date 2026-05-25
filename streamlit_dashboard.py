"""
CYBERBULLYING DETECTION DASHBOARD
SafeNet Analytics - Interactive Streamlit Application

Run this with: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pickle
import re

# Page configuration
st.set_page_config(
    page_title="Cyberbullying Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    .alert-high {
        background: #fff5f5;
        border-left: 4px solid #e53e3e;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background: #fffaf0;
        border-left: 4px solid #ed8936;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .alert-low {
        background: #fffff0;
        border-left: 4px solid #ecc94b;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .safe-content {
        background: #f0fff4;
        border-left: 4px solid #48bb78;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Simple text preprocessing function
def preprocess_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    text = re.sub(r'@\w+', '<user>', text)
    text = re.sub(r'http\S+|www\S+', '<url>', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s!?.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Simulated ML model (you can replace this with your actual model)
def predict_cyberbullying(text):
    """
    Simulated prediction function
    Replace this with your actual trained model
    """
    # Keywords that indicate cyberbullying
    bullying_keywords = ['stupid', 'idiot', 'dumb', 'hate', 'kill', 'die', 
                         'loser', 'worthless', 'ugly', 'fat', 'nobody']
    
    text_lower = text.lower()
    keyword_count = sum(1 for word in bullying_keywords if word in text_lower)
    
    # Simple scoring (replace with your model)
    if keyword_count >= 2:
        confidence = min(0.95, 0.65 + (keyword_count * 0.1))
        prediction = 1
    elif keyword_count == 1:
        confidence = 0.55 + (len(text.split()) * 0.01)
        prediction = 1 if confidence > 0.6 else 0
    else:
        confidence = max(0.15, 0.35 - (len(text.split()) * 0.01))
        prediction = 0
    
    # Get triggered words
    triggered_words = [word for word in bullying_keywords if word in text_lower]
    
    return prediction, confidence, triggered_words

# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ Cyberbullying Detection Dashboard</h1>
    <p>SafeNet Analytics - AI-Powered Content Moderation System</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/000000/security-shield-green.png", width=100)
    st.title("Navigation")
    
    page = st.radio(
        "Select Page:",
        ["🏠 Live Detection", "📊 Analytics Dashboard", "⚙️ System Info", "📚 About"]
    )
    
    st.markdown("---")
    st.markdown("### Quick Stats")
    st.metric("Model Accuracy", "81.0%")
    st.metric("Precision", "96.5%")
    st.metric("Analyses Today", len(st.session_state.analysis_history))

# ==============================================================================
# PAGE 1: LIVE DETECTION
# ==============================================================================
if page == "🏠 Live Detection":
    st.header("🔍 Real-Time Cyberbullying Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Text to Analyze")
        
        # Text input options
        input_method = st.radio("Input method:", ["Type text", "Try examples"])
        
        if input_method == "Type text":
            user_text = st.text_area(
                "Enter the text you want to analyze:",
                height=150,
                placeholder="Type or paste the text here..."
            )
        else:
            example_texts = {
                "Severe Cyberbullying": "You're so stupid, nobody likes you and you should just kill yourself",
                "Moderate Harassment": "lol you're such a loser haha everyone thinks you're dumb",
                "Borderline": "I disagree with you, that's a terrible idea",
                "Safe Content": "Great job on the presentation today! Well done!",
                "Neutral Comment": "I think we should consider other options for this project"
            }
            
            selected_example = st.selectbox("Choose an example:", list(example_texts.keys()))
            user_text = example_texts[selected_example]
            st.text_area("Selected text:", user_text, height=100)
        
        analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
        
        if analyze_button and user_text:
            with st.spinner("Analyzing content..."):
                # Preprocess
                cleaned_text = preprocess_text(user_text)
                
                # Predict
                prediction, confidence, triggered_words = predict_cyberbullying(user_text)
                
                # Store in history
                st.session_state.analysis_history.append({
                    'timestamp': datetime.now(),
                    'text': user_text[:100],
                    'prediction': prediction,
                    'confidence': confidence
                })
                
                # Display results
                st.markdown("---")
                st.subheader("📋 Analysis Results")
                
                if prediction == 1:
                    if confidence >= 0.8:
                        st.markdown(f"""
                        <div class="alert-high">
                            <h3>⚠️ HIGH RISK - Likely Cyberbullying</h3>
                            <p><strong>Confidence: {confidence*100:.1f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                        risk_color = "red"
                    elif confidence >= 0.6:
                        st.markdown(f"""
                        <div class="alert-medium">
                            <h3>⚠️ MODERATE RISK - Possible Cyberbullying</h3>
                            <p><strong>Confidence: {confidence*100:.1f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                        risk_color = "orange"
                    else:
                        st.markdown(f"""
                        <div class="alert-low">
                            <h3>⚠️ LOW RISK - Review Recommended</h3>
                            <p><strong>Confidence: {confidence*100:.1f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                        risk_color = "yellow"
                else:
                    st.markdown(f"""
                    <div class="safe-content">
                        <h3>✅ SAFE CONTENT - Not Cyberbullying</h3>
                        <p><strong>Confidence: {(1-confidence)*100:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    risk_color = "green"
                
                # Confidence meter
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = confidence * 100 if prediction == 1 else (1-confidence) * 100,
                    title = {'text': "Confidence Level"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': risk_color},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "lightyellow"},
                            {'range': [75, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 80
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
                
                # Triggered words
                if triggered_words:
                    st.markdown("### 🎯 Flagged Terms")
                    st.warning(f"Detected words: **{', '.join(triggered_words)}**")
                
                # Recommendation
                st.markdown("### 💡 Recommendation")
                if prediction == 1 and confidence >= 0.8:
                    st.error("**Action Required:** This content should be reviewed by a human moderator immediately. Consider removing content and warning the user.")
                elif prediction == 1 and confidence >= 0.6:
                    st.warning("**Manual Review:** This content requires human judgment. Consider the full context before taking action.")
                elif prediction == 1:
                    st.info("**Monitor:** Content is borderline. Keep an eye on this user's future posts.")
                else:
                    st.success("**No Action Needed:** Content appears safe. No moderation required.")
    
    with col2:
        st.subheader("📊 Analysis Metrics")
        
        # System performance metrics
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Value': [81.0, 96.5, 80.2, 87.6]
        }
        metrics_df = pd.DataFrame(metrics_data)
        
        fig = px.bar(
            metrics_df, 
            x='Metric', 
            y='Value',
            color='Value',
            color_continuous_scale='RdYlGn',
            text='Value'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            showlegend=False,
            height=300,
            yaxis_range=[0, 100]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Recent analysis history
        st.subheader("📜 Recent Analyses")
        if st.session_state.analysis_history:
            recent = st.session_state.analysis_history[-5:][::-1]
            for i, item in enumerate(recent, 1):
                status = "🔴" if item['prediction'] == 1 else "🟢"
                st.markdown(f"""
                **{status} {item['timestamp'].strftime('%H:%M:%S')}**  
                *{item['text'][:50]}...*  
                Confidence: {item['confidence']*100:.1f}%
                """)
                st.markdown("---")
        else:
            st.info("No analyses yet. Try analyzing some text!")

# ==============================================================================
# PAGE 2: ANALYTICS DASHBOARD
# ==============================================================================
elif page == "📊 Analytics Dashboard":
    st.header("📊 System Analytics & Performance")
    
    # Performance metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2 style="color: #667eea;">81.0%</h2>
            <p>Overall Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2 style="color: #764ba2;">96.5%</h2>
            <p>Precision</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2 style="color: #f093fb;">80.2%</h2>
            <p>Recall</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h2 style="color: #4facfe;">87.6%</h2>
            <p>F1-Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Confusion Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔲 Confusion Matrix")
        
        cm_data = pd.DataFrame({
            'Not CB (Predicted)': [1300, 1548],
            'CB (Predicted)': [231, 6287]
        }, index=['Not CB (Actual)', 'CB (Actual)'])
        
        fig = px.imshow(
            cm_data,
            text_auto=True,
            color_continuous_scale='RdYlGn_r',
            labels=dict(x="Predicted", y="Actual", color="Count")
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("**False Positives:** 231 | **False Negatives:** 1,548")
    
    with col2:
        st.subheader("🥧 Dataset Distribution")
        
        class_data = pd.DataFrame({
            'Category': ['Religion', 'Age', 'Gender', 'Ethnicity', 'Not CB', 'Other CB'],
            'Count': [7995, 7988, 7875, 7955, 7657, 7358]
        })
        
        fig = px.pie(
            class_data,
            values='Count',
            names='Category',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Top indicators
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Top Cyberbullying Indicators")
        
        bullying_words = pd.DataFrame({
            'Word': ['rape', 'dumb', 'bullies', 'nigger', 'feminazi', 
                    'idiot', 'muslims', 'idiots', 'gay', 'bitch'],
            'Weight': [8.38, 8.26, 8.10, 8.10, 7.87, 6.69, 6.40, 6.37, 6.06, 5.91]
        })
        
        fig = px.bar(
            bullying_words,
            y='Word',
            x='Weight',
            orientation='h',
            color='Weight',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📉 Top Safe Content Indicators")
        
        safe_words = pd.DataFrame({
            'Word': ['mkr', 'daesh', 'class', 'mosul', 'bullying',
                    'college', 'yesallwomen', 'kat and', 'andre', 'user also'],
            'Weight': [5.39, 3.43, 3.42, 3.40, 3.23, 2.70, 2.45, 2.38, 2.35, 2.17]
        })
        
        fig = px.bar(
            safe_words,
            y='Word',
            x='Weight',
            orientation='h',
            color='Weight',
            color_continuous_scale='Greens'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE 3: SYSTEM INFO
# ==============================================================================
elif page == "⚙️ System Info":
    st.header("⚙️ System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Model Details")
        st.markdown("""
        **Algorithm:** Logistic Regression  
        **Feature Extraction:** TF-IDF  
        **Vocabulary Size:** 5,000 features  
        **Training Samples:** 37,462 tweets  
        **Test Samples:** 9,366 tweets  
        **Training Time:** ~2 minutes  
        **Inference Time:** <100ms per text
        """)
        
        st.markdown("---")
        
        st.subheader("📊 Dataset Information")
        st.markdown("""
        **Total Samples:** 47,692 tweets  
        **Categories:** 6 types  
        **Language:** English  
        **Source:** Twitter/X  
        **Class Balance:** 16-17% per category  
        **Processing:** Cleaned, anonymized
        """)
    
    with col2:
        st.subheader("🎯 Performance Targets")
        
        targets = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Target': [75, 90, 75, 80],
            'Achieved': [81.0, 96.5, 80.2, 87.6]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Target',
            x=targets['Metric'],
            y=targets['Target'],
            marker_color='lightgray'
        ))
        fig.add_trace(go.Bar(
            name='Achieved',
            x=targets['Metric'],
            y=targets['Achieved'],
            marker_color='#667eea'
        ))
        fig.update_layout(height=300, barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("✅ All performance targets exceeded!")
        
        st.markdown("---")
        
        st.subheader("⚠️ Known Limitations")
        st.warning("""
        - Struggles with sarcasm and irony
        - May miss subtle harassment
        - Limited to English text only
        - Cannot process images or videos
        - Context-dependent cases need review
        """)
    
    st.markdown("---")
    
    st.subheader("🔧 Model Configuration")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.code("""
# Logistic Regression Settings
max_iter = 1000
class_weight = 'balanced'
C = 1.0 (regularization)
random_state = 42
solver = 'lbfgs'
        """, language='python')
    
    with config_col2:
        st.code("""
# TF-IDF Vectorizer Settings
max_features = 5000
ngram_range = (1, 2)
min_df = 5
max_df = 0.8
analyzer = 'word'
        """, language='python')

# ==============================================================================
# PAGE 4: ABOUT
# ==============================================================================
else:  # About page
    st.header("📚 About This System")
    
    st.markdown("""
    ## 🛡️ Cyberbullying Detection System
    
    ### Overview
    This system uses **Artificial Intelligence** and **Natural Language Processing** to detect 
    potentially harmful content in social media posts. 
    It was developed as part of the 
    **SafeNet Analytics** project to promote digital safety and responsible online behavior.
    
    ### How It Works
    1. **Text Input:** User submits text for analysis
    2. **Preprocessing:** Text is cleaned and standardized
    3. **Feature Extraction:** TF-IDF converts text to numerical features
    4. **Classification:** Logistic Regression model predicts cyberbullying
    5. **Results:** System provides prediction with confidence score
    
    ### Key Features
    - ✅ Real-time text analysis
    - ✅ Confidence scoring
    - ✅ Interpretable results (shows which words triggered the flag)
    - ✅ Human-in-the-loop design
    - ✅ Transparent decision-making
    
    ### Ethical Considerations
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.error("""
        **⚠️ Potential Harms:**
        - False positives silence legitimate speech
        - False negatives allow harm to continue
        - Bias against certain language patterns
        - Over-reliance on automation
        """)
    
    with col2:
        st.success("""
        **✅ Mitigation Strategies:**
        - Use as flagging tool, not enforcement
        - Require human review for decisions
        - Regular bias audits
        - Transparent explanations
        - User appeals process
        """)
    
    st.markdown("---")
    
    st.subheader("🎓 Academic Context")
    st.info("""
    This system was developed as part of an academic project exploring the intersection of:
    - **Data Science** - Machine learning and NLP techniques
    - **Human-Computer Interaction** - User-centered interface design
    - **Ethics** - Responsible AI deployment and bias mitigation
    
    **Institution:** SafeNet Analytics Research Project  
    **Date:** May 2026  
    **Model:** Logistic Regression with TF-IDF  
    **Dataset:** 47,692 labeled tweets
    """)
    
    st.markdown("---")
    
    st.subheader("📖 References & Resources")
    st.markdown("""
    - **Scikit-learn Documentation:** [scikit-learn.org](https://scikit-learn.org)
    - **Streamlit Documentation:** [streamlit.io](https://streamlit.io)
    - **Ethical AI Guidelines:** Partnership on AI, ACM Code of Ethics
    """)
    
    st.markdown("---")
    
    st.subheader("💬 Feedback & Contact")
    st.markdown("""
    This is an educational prototype. For questions or feedback about this system:
    - Report issues through your institution
    - Refer to the project documentation
    - Consult with the SafeNet Analytics team
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <p><strong>SafeNet Analytics Cyberbullying Detection System v1.0</strong></p>
    <p>Built with Streamlit | Powered by Machine Learning | Designed for Human Oversight</p>
    <p><em>This is an educational prototype. Always use human judgment for final decisions.</em></p>
</div>
""", unsafe_allow_html=True)
