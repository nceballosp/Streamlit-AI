"""
Laboratorio NLP con Groq
- Tokenización (tiktoken + Hugging Face) con IDs y tokens coloreados
- Embeddings (sentence-transformers) y similitud coseno entre frases
- Generación de texto con modelos GPT de Groq (varias estrategias)

Ejecutar:  streamlit run main.py
"""
import html
import os

import numpy as np
import pandas as pd
import streamlit as st
import tiktoken
from groq import Groq

st.set_page_config(page_title="Laboratorio NLP + Groq", layout="wide")

# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------
PALETTE = ["#FFB3BA", "#FFDFBA", "#FFF3A8", "#BAFFC9", "#BAE1FF", "#E0BBE4"]

TIKTOKEN_MODELS = {
    "o200k_base (GPT-4o / gpt-oss)": "o200k_base",
    "cl100k_base (GPT-4 / GPT-3.5)": "cl100k_base",
    "p50k_base (Codex)": "p50k_base",
    "r50k_base (GPT-3)": "r50k_base",
}
HF_TOKENIZERS = {
    "BERT multilingual (WordPiece)": "bert-base-multilingual-cased",
    "XLM-RoBERTa (SentencePiece)": "xlm-roberta-base",
}
EMBEDDING_MODELS = {
    "MiniLM L6 (inglés, rápido)": "sentence-transformers/all-MiniLM-L6-v2",
    "MiniLM L12 multilingüe": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "E5 small multilingüe": "intfloat/multilingual-e5-small",
    "MPNet base (inglés)": "sentence-transformers/all-mpnet-base-v2",
}
DEFAULT_GPT_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]

DECODING_STRATEGIES = {
    "Greedy (temp 0)": {"temperature": 0.0},
    "Baja temperatura (0.3)": {"temperature": 0.3},
    "Equilibrada (0.7, top_p 0.9)": {"temperature": 0.7, "top_p": 0.9},
    "Creativa (1.2, top_p 0.95)": {"temperature": 1.2, "top_p": 0.95},
}
PROMPT_STYLES = ["Zero-shot", "Few-shot", "Chain-of-thought", "Con rol (system)"]


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def colored_tokens_html(tokens: list[str]) -> str:
    spans = []
    for i, tok in enumerate(tokens):
        safe = html.escape(tok).replace(" ", "·").replace("\n", "↵") or "∅"
        color = PALETTE[i % len(PALETTE)]
        spans.append(
            f'<span style="background:{color};color:#111;padding:2px 4px;'
            f'margin:1px;border-radius:4px;font-family:monospace;'
            f'display:inline-block">{safe}</span>'
        )
    return "".join(spans)


@st.cache_resource(show_spinner=False)
def load_hf_tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


@st.cache_resource(show_spinner="Cargando modelo de embeddings...")
def load_embedder(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


@st.cache_data(show_spinner=False)
def list_groq_gpt_models(api_key: str) -> list[str]:
    try:
        ids = [m.id for m in Groq(api_key=api_key).models.list().data]
        gpt = sorted(i for i in ids if "gpt" in i.lower())
        return gpt or DEFAULT_GPT_MODELS
    except Exception:
        return DEFAULT_GPT_MODELS


def tokenize(text: str, kind: str, name: str) -> tuple[list[int], list[str]]:
    if kind == "tiktoken":
        enc = tiktoken.get_encoding(name)
        ids = enc.encode(text)
        toks = [
            enc.decode_single_token_bytes(i).decode("utf-8", errors="replace")
            for i in ids
        ]
        return ids, toks
    tok = load_hf_tokenizer(name)
    ids = tok.encode(text, add_special_tokens=False)
    return ids, tok.convert_ids_to_tokens(ids)


def build_messages(style: str, task: str) -> list[dict]:
    if style == "Zero-shot":
        return [{"role": "user", "content": task}]
    if style == "Few-shot":
        return [
            {"role": "system", "content": "Responde siguiendo el estilo de los ejemplos."},
            {"role": "user", "content": "Escribe una frase sobre el mar."},
            {"role": "assistant", "content": "El mar guarda en silencio todos los secretos del horizonte."},
            {"role": "user", "content": "Escribe una frase sobre la montaña."},
            {"role": "assistant", "content": "La montaña respira despacio mientras las nubes se sientan a su lado."},
            {"role": "user", "content": task},
        ]
    if style == "Chain-of-thought":
        return [{"role": "user", "content": f"{task}\n\nPiensa paso a paso y al final da la respuesta."}]
    return [
        {"role": "system", "content": "Eres un escritor experto, claro y conciso. Responde en español."},
        {"role": "user", "content": task},
    ]


def generate(client: Groq, model: str, messages: list[dict], stream: bool, **params):
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, stream=stream, **params
        )
        if stream:
            for chunk in resp:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        else:
            yield resp.choices[0].message.content or ""
    except Exception as e:
        yield f"⚠️ Error: {e}"


# ----------------------------------------------------------------------------
# Sidebar: API key
# ----------------------------------------------------------------------------
st.title("Laboratorio NLP con Groq")
with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Se lee aquí o desde la variable de entorno GROQ_API_KEY.",
    )
    gpt_models = list_groq_gpt_models(api_key) if api_key else DEFAULT_GPT_MODELS
    gpt_model = st.selectbox("Modelo GPT (Groq)", gpt_models)

tab_tok, tab_emb, tab_gen = st.tabs(["🔤 Tokenización", "📐 Embeddings", "✍️ Generación"])

# ----------------------------------------------------------------------------
# Tab 1: Tokenización
# ----------------------------------------------------------------------------
with tab_tok:
    text = st.text_area(
        "Texto", "Los modelos de lenguaje dividen el texto en tokens. ¡Hola, Medellín!", height=100
    )
    options = {**{k: ("tiktoken", v) for k, v in TIKTOKEN_MODELS.items()},
               **{k: ("hf", v) for k, v in HF_TOKENIZERS.items()}}
    chosen = st.multiselect(
        "Tokenizadores", list(options), default=list(TIKTOKEN_MODELS)[:2] + list(HF_TOKENIZERS)[:1]
    )
    for label in chosen:
        kind, name = options[label]
        try:
            ids, toks = tokenize(text, kind, name)
        except Exception as e:
            st.error(f"{label}: {e}")
            continue
        st.subheader(f"{label} — {len(ids)} tokens")
        st.markdown(colored_tokens_html(toks), unsafe_allow_html=True)
        with st.expander("Token IDs"):
            st.dataframe(
                pd.DataFrame({"pos": range(len(ids)), "token": toks, "token_id": ids}),
                use_container_width=True,
                hide_index=True,
            )

# ----------------------------------------------------------------------------
# Tab 2: Embeddings + similitud coseno
# ----------------------------------------------------------------------------
with tab_emb:
    st.caption("Groq no ofrece endpoint de embeddings; se calculan localmente con sentence-transformers.")
    sentences_raw = st.text_area(
        "Frases (una por línea)",
        "El gato duerme en el sofá.\nUn felino descansa sobre el sillón.\n"
        "La bolsa de valores cayó hoy.\nMe encanta patinar en el parque.",
        height=140,
    )
    emb_choice = st.multiselect(
        "Modelos de embedding", list(EMBEDDING_MODELS), default=list(EMBEDDING_MODELS)[1:3]
    )
    if st.button("Calcular similitud coseno"):
        sentences = [s.strip() for s in sentences_raw.splitlines() if s.strip()]
        if len(sentences) < 2:
            st.warning("Escribe al menos 2 frases.")
        for label in emb_choice if len(sentences) >= 2 else []:
            try:
                model = load_embedder(EMBEDDING_MODELS[label])
                prefix = "query: " if "e5" in EMBEDDING_MODELS[label] else ""
                vecs = model.encode([prefix + s for s in sentences], normalize_embeddings=True)
                sim = np.clip(vecs @ vecs.T, -1, 1)  # normalizados -> coseno = producto punto
                df = pd.DataFrame(sim, index=sentences, columns=[f"F{i+1}" for i in range(len(sentences))])
                st.subheader(f"{label} — dim {vecs.shape[1]}")
                st.dataframe(df.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1).format("{:.3f}"),
                             use_container_width=True)
            except Exception as e:
                st.error(f"{label}: {e}")

# ----------------------------------------------------------------------------
# Tab 3: Generación de texto
# ----------------------------------------------------------------------------
with tab_gen:
    task = st.text_area("Tarea / prompt", "Escribe un párrafo corto sobre el futuro de la ciencia de datos.", height=100)
    c1, c2, c3 = st.columns(3)
    style = c1.selectbox("Proceso de prompting", PROMPT_STYLES)
    strategies = c2.multiselect("Estrategias de decodificación", list(DECODING_STRATEGIES),
                                default=list(DECODING_STRATEGIES)[:3])
    max_tokens = c3.number_input("max_completion_tokens", 128, 8192, 1024, step=128,
                                 help="Los modelos gpt-oss razonan; deja margen.")
    use_stream = st.checkbox("Streaming", value=True)
    seed = st.number_input("Seed (0 = aleatoria)", 0, 999999, 0)

    if st.button("Generar"):
        if not api_key:
            st.error("Ingresa tu Groq API Key en la barra lateral.")
        elif not strategies:
            st.warning("Elige al menos una estrategia.")
        else:
            client = Groq(api_key=api_key)
            messages = build_messages(style, task)
            cols = st.columns(len(strategies))
            for col, name in zip(cols, strategies):
                params = dict(DECODING_STRATEGIES[name], max_completion_tokens=int(max_tokens))
                if seed:
                    params["seed"] = int(seed)
                with col:
                    st.markdown(f"**{name}**")
                    st.write_stream(generate(client, gpt_model, messages, use_stream, **params))
