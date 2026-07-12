"""
KGF 2 Movie Review Analysis App
--------------------------------
An end-to-end NLP dashboard built with Streamlit + Hugging Face Transformers.
 
Tasks demonstrated:
- Sentiment Analysis (Positive / Negative)
- Machine Translation (English -> Spanish)
- Text Summarization (condense long reviews)
- Model Evaluation (Accuracy, F1, BLEU)
 
Design goals:
- Every model call is wrapped in try/except so a failure never crashes the
  app or shows a raw traceback — the user always gets a readable message.
- Models are loaded once and cached (st.cache_resource) so buttons respond
  quickly after the first click.
- Reviews are loaded directly from the bundled kgf2_reviews.csv file.
"""
 
import streamlit as st
import pandas as pd
 
# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="KGF 2 Movie Review Analysis",
    page_icon="🎬",
    layout="wide"
)
 
st.title("🎬 KGF 2 Movie Review Analysis with LLMs / GenAI")
st.write(
    "An interactive NLP dashboard that analyzes **KGF 2** movie reviews using "
    "Hugging Face Transformer pipelines — sentiment analysis, translation, "
    "question answering, summarization, and model evaluation."
)
 
# ------------------------------------------------------------------
# Built-in sample dataset (used if no file is uploaded)
# ------------------------------------------------------------------
DEFAULT_DATA = pd.DataFrame({
    "Review": [
        "KGF 2 is an amazing movie with powerful action and excellent performance.",
        "The storyline was weak and the pacing felt too slow in the second half.",
        "Yash's performance as Rocky Bhai is outstanding, truly a mass entertainer.",
        "The background score and cinematography are breathtaking throughout the film.",
        "I found the plot predictable and the dialogues a bit over the top.",
        "A visual spectacle with high-octane action sequences and a gripping climax.",
        "The movie drags in places, and some characters feel underdeveloped.",
        "One of the best pan-India films, the direction and scale are massive.",
        "Too much violence and not enough substance in the narrative.",
        "An entertaining watch overall, though the runtime could have been trimmed."
    ],
    "Class": [
        "POSITIVE", "NEGATIVE", "POSITIVE", "POSITIVE", "NEGATIVE",
        "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE"
    ]
})
 
# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.title("KGF 2 Review Analyzer")
st.sidebar.info(
    """
    **Pipelines used:**
    - Sentiment Analysis
    - Translation (EN → ES)
    - Summarization
    - Evaluation (Accuracy, F1, BLEU)
    """
)
 
import os
 
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kgf2_reviews.csv")
 
try:
    df = pd.read_csv(CSV_PATH, sep=";")
    if "Review" not in df.columns or "Class" not in df.columns:
        st.sidebar.warning(
            "kgf2_reviews.csv must contain 'Review' and 'Class' columns. "
            "Using the built-in sample dataset instead."
        )
        df = DEFAULT_DATA.copy()
except Exception:
    df = DEFAULT_DATA.copy()
 
reviews = df["Review"].tolist()
real_labels = df["Class"].tolist()
 
st.subheader("📁 Dataset Preview")
st.dataframe(df, use_container_width=True)
st.caption(f"Total reviews: {len(df)}")
 
st.markdown("---")
 
# ------------------------------------------------------------------
# Cached model loaders
# Each loader is wrapped in try/except and returns None on failure so
# the calling code can show a friendly message instead of crashing.
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading sentiment analysis model...")
def load_sentiment_model():
    from transformers import pipeline
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
 
 
@st.cache_resource(show_spinner="Loading translation model...")
def load_translation_model():
    from transformers import pipeline
    return pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")
 
 
@st.cache_resource(show_spinner="Loading summarization model...")
def load_summarization_model():
    from transformers import pipeline
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
 
 
def safe_load(loader_fn, label):
    """Load a model safely; show a friendly error instead of crashing."""
    try:
        return loader_fn()
    except Exception as e:
        st.error(
            f"⚠️ Could not load the {label} model right now. "
            f"This is usually a temporary download/network issue — please try again in a moment."
        )
        with st.expander("Technical details"):
            st.code(str(e))
        return None
 
 
# ------------------------------------------------------------------
# Tabs for each NLP task
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["😊 Sentiment Analysis", "🌐 Translation", "📝 Summarization", "📊 Model Evaluation"]
)
 
# ------------------------- TAB 1: SENTIMENT ------------------------
with tab1:
    st.header("Sentiment Analysis")
    st.write("Classify a KGF 2 review as **Positive** or **Negative**.")
 
    default_review = reviews[0] if reviews else "KGF 2 is an amazing movie with powerful action."
    user_review = st.text_area(
        "Enter a review (or edit the default one below):",
        value=default_review,
        height=100,
        key="sentiment_input"
    )
 
    if st.button("Analyze Sentiment", key="btn_sentiment"):
        if not user_review.strip():
            st.warning("Please enter some review text first.")
        else:
            classifier = safe_load(load_sentiment_model, "sentiment analysis")
            if classifier is not None:
                try:
                    result = classifier(user_review)[0]
                    label = result["label"]
                    score = result["score"]
                    emoji = "😊" if label == "POSITIVE" else "😞"
                    st.success(f"{emoji} **Sentiment: {label}**  (confidence: {score * 100:.2f}%)")
                    st.progress(min(max(score, 0.0), 1.0))
                except Exception as e:
                    st.error("Something went wrong while analyzing sentiment. Please try again.")
                    with st.expander("Technical details"):
                        st.code(str(e))
 
    st.markdown("##### Batch analysis on the full dataset")
    if st.button("Run Sentiment Analysis on All Reviews", key="btn_sentiment_batch"):
        classifier = safe_load(load_sentiment_model, "sentiment analysis")
        if classifier is not None:
            try:
                predictions = classifier(reviews)
                results_df = pd.DataFrame({
                    "Review": reviews,
                    "Actual": real_labels,
                    "Predicted": [p["label"] for p in predictions],
                    "Confidence": [f"{p['score'] * 100:.2f}%" for p in predictions]
                })
                st.dataframe(results_df, use_container_width=True)
                st.session_state["predicted_labels"] = predictions
            except Exception as e:
                st.error("Batch sentiment analysis failed. Please try again.")
                with st.expander("Technical details"):
                    st.code(str(e))
 
# ------------------------- TAB 2: TRANSLATION -----------------------
with tab2:
    st.header("Machine Translation (English → Spanish)")
    text_to_translate = st.text_area(
        "Enter English text to translate:",
        value=reviews[0] if reviews else "KGF 2 is an amazing movie.",
        height=100,
        key="translation_input"
    )
 
    if st.button("Translate to Spanish", key="btn_translate"):
        if not text_to_translate.strip():
            st.warning("Please enter some text to translate.")
        else:
            translator = safe_load(load_translation_model, "translation")
            if translator is not None:
                try:
                    translation = translator(text_to_translate)[0]["translation_text"]
                    st.success("**Translated Text (Spanish):**")
                    st.write(translation)
                except Exception as e:
                    st.error("Translation failed. Please try again with a shorter text.")
                    with st.expander("Technical details"):
                        st.code(str(e))
 
# ------------------------- TAB 3: SUMMARIZATION -----------------------
with tab3:
    st.header("Text Summarization")
 
    long_text_default = " ".join(reviews) if reviews else \
        "KGF 2 is an amazing movie with powerful action and excellent performance."
 
    text_to_summarize = st.text_area(
        "Text to summarize:",
        value=long_text_default,
        height=180,
        key="summary_input"
    )
 
    if st.button("Summarize", key="btn_summarize"):
        if not text_to_summarize.strip():
            st.warning("Please enter some text to summarize.")
        elif len(text_to_summarize.split()) < 15:
            st.warning("Please provide a slightly longer text (at least ~15 words) for a meaningful summary.")
        else:
            summarizer = safe_load(load_summarization_model, "summarization")
            if summarizer is not None:
                try:
                    word_count = len(text_to_summarize.split())
                    max_len = max(20, min(120, word_count // 2))
                    min_len = max(10, max_len // 3)
                    summary = summarizer(
                        text_to_summarize,
                        max_length=max_len,
                        min_length=min_len,
                        do_sample=False
                    )[0]["summary_text"]
                    st.success("**Summary:**")
                    st.write(summary)
                except Exception as e:
                    st.error("Summarization failed. Please try again with a different text.")
                    with st.expander("Technical details"):
                        st.code(str(e))
 
# ------------------------- TAB 4: MODEL EVALUATION ---------------------
with tab4:
    st.header("Model Evaluation")
 
    st.markdown("##### Sentiment Model — Accuracy & F1 Score")
    if st.button("Evaluate Sentiment Model", key="btn_eval_sentiment"):
        classifier = safe_load(load_sentiment_model, "sentiment analysis")
        if classifier is not None:
            try:
                predicted_labels = classifier(reviews)
                references = [1 if label == "POSITIVE" else 0 for label in real_labels]
                predictions = [1 if p["label"] == "POSITIVE" else 0 for p in predicted_labels]
 
                try:
                    import evaluate
                    accuracy_metric = evaluate.load("accuracy")
                    f1_metric = evaluate.load("f1")
                    accuracy_result = accuracy_metric.compute(references=references, predictions=predictions)["accuracy"]
                    f1_result = f1_metric.compute(references=references, predictions=predictions)["f1"]
                except Exception:
                    # Fallback to scikit-learn if the `evaluate` library / its
                    # remote metric scripts are unavailable (e.g. offline).
                    from sklearn.metrics import accuracy_score, f1_score
                    accuracy_result = accuracy_score(references, predictions)
                    f1_result = f1_score(references, predictions, zero_division=0)
 
                col1, col2 = st.columns(2)
                col1.metric("Accuracy", f"{accuracy_result * 100:.2f}%")
                col2.metric("F1 Score", f"{f1_result:.3f}")
            except Exception as e:
                st.error("Evaluation failed. Please try again.")
                with st.expander("Technical details"):
                    st.code(str(e))
 
    st.markdown("---")
    st.markdown("##### Translation Quality — BLEU Score")
    st.write("Provide an English sentence and its correct Spanish reference translation "
              "to see how close the model's translation is (BLEU score).")
 
    col1, col2 = st.columns(2)
    with col1:
        bleu_source = st.text_input(
            "English sentence:",
            value="KGF 2 is an amazing movie.",
            key="bleu_source"
        )
    with col2:
        bleu_reference = st.text_input(
            "Reference Spanish translation:",
            value="KGF 2 es una película increíble.",
            key="bleu_reference"
        )
 
    if st.button("Evaluate Translation (BLEU)", key="btn_eval_bleu"):
        if not bleu_source.strip() or not bleu_reference.strip():
            st.warning("Please fill in both the source sentence and the reference translation.")
        else:
            translator = safe_load(load_translation_model, "translation")
            if translator is not None:
                try:
                    prediction_text = translator(bleu_source)[0]["translation_text"]
                    st.write(f"**Model Translation:** {prediction_text}")
 
                    try:
                        import evaluate
                        bleu_metric = evaluate.load("bleu")
                        bleu_result = bleu_metric.compute(
                            predictions=[prediction_text],
                            references=[[bleu_reference]]
                        )
                        st.metric("BLEU Score", f"{bleu_result['bleu']:.3f}")
                    except Exception:
                        # Fallback to sacrebleu / nltk-free simple BLEU if
                        # the `evaluate` library can't fetch its metric script.
                        try:
                            import sacrebleu
                            bleu = sacrebleu.corpus_bleu([prediction_text], [[bleu_reference]])
                            st.metric("BLEU Score", f"{bleu.score / 100:.3f}")
                        except Exception:
                            st.info(
                                "BLEU scoring libraries are unavailable right now, "
                                "but here is the model's translation above for manual comparison."
                            )
                except Exception as e:
                    st.error("Translation evaluation failed. Please try again.")
                    with st.expander("Technical details"):
                        st.code(str(e))
 
st.markdown("---")
st.caption("KGF 2 Movie Review Analysis | Built with Streamlit and Hugging Face Transformers")
 
