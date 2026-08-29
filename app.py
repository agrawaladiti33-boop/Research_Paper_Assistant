import streamlit as st
import os
from document_processor import process_pdf
from vector_manager import VectorManager
from llm_manager import LLMManager

# ==========================================
# 1. Page Configuration & Initialization
# ==========================================
st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize Backend Managers in Session State to persist them
if "vector_manager" not in st.session_state:
    st.session_state.vector_manager = VectorManager()
    
if "llm_manager" not in st.session_state:
    st.session_state.llm_manager = LLMManager()

vector_manager = st.session_state.vector_manager
llm_manager = st.session_state.llm_manager

# ==========================================
# 2. Sidebar: Multi-Paper Upload & Management
# ==========================================
with st.sidebar:
    st.header("📚 Document Management")
    
    # Accept multiple files
    uploaded_files = st.file_uploader(
        "Upload research papers",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Uploaded Papers"):
            with st.spinner("Processing documents..."):
                for uploaded_file in uploaded_files:
                    # We use the filename as the paper name
                    paper_name = uploaded_file.name
                    
                    # Only process if not already in DB
                    existing_papers = vector_manager.get_uploaded_papers()
                    if paper_name not in existing_papers:
                        st.write(f"Processing: {paper_name}")
                        chunks, num_pages = process_pdf(uploaded_file)
                        vector_manager.add_chunks(chunks, paper_name)
                        st.success(f"Added {paper_name} ({num_pages} pages)")
                    else:
                        st.info(f"{paper_name} is already processed.")
                        
    st.divider()
    
    st.subheader("Current Library")
    available_papers = vector_manager.get_uploaded_papers()
    
    if not available_papers:
        st.write("No papers uploaded yet.")
    else:
        for paper in available_papers:
            st.write(f"- {paper}")

# ==========================================
# 3. Main Body Setup
# ==========================================
st.title("📚 Research Paper Assistant")
st.write("Upload research papers in the sidebar and interact with them here.")

# We will add tabs for different features
tab1, tab2, tab3 = st.tabs(["Agent Chat", "Document Analysis", "Compare Papers"])

with tab1:
    st.header("💬 Agent Chat")
    st.write("Ask anything about your uploaded papers. The agent remembers the conversation.")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    available_papers = vector_manager.get_uploaded_papers()

    if not available_papers:
        st.warning("⚠️ Please upload and process a paper from the sidebar first.")
    else:
        # Paper filter (optional): let user pick which paper to chat with
        chat_paper_options = ["All Papers"] + available_papers
        selected_chat_paper = st.selectbox(
            "Chat with:",
            chat_paper_options,
            key="chat_paper_select"
        )

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about the paper..."):

            # Show user message
            with st.chat_message("user"):
                st.markdown(prompt)
            # Retrieve relevant context from ChromaDB BEFORE appending to history
            paper_filter = None if selected_chat_paper == "All Papers" else selected_chat_paper
            retrieved_chunks = vector_manager.query(prompt, paper_name=paper_filter, n_results=5)

            context = "\n\n".join(
                f"[{chunk['paper_name']} | Page {chunk['page']}]\n{chunk['text']}"
                for chunk in retrieved_chunks
            )

            # Build the system message with fresh context
            system_message = {
                "role": "system",
                "content": (
                    "You are a helpful research paper assistant. "
                    "Answer the user's question using ONLY the provided context from the research papers. "
                    "If the answer is not in the context, say so clearly. "
                    "Always cite the paper name and page number when referencing information.\n\n"
                    f"Research Paper Context:\n-----------------------\n{context}\n-----------------------"
                )
            }

            # Build messages: system + history (history does NOT yet include current user message)
            llm_messages = [system_message] + st.session_state.messages + [{"role": "user", "content": prompt}]

            # Now append user message to history for display
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = llm_manager.ask_groq_conversational(llm_messages)
                st.markdown(answer)

            # Store assistant response in history
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Show source pages
            if retrieved_chunks:
                source_info = sorted(set(
                    f"{c['paper_name']} — Page {c['page']}" for c in retrieved_chunks
                ))
                with st.expander("📄 Sources"):
                    for s in source_info:
                        st.write(f"- {s}")

        # Button to clear chat history
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()

with tab2:
    st.header("📊 Document Analysis")
    st.write("Select a paper and generate AI-powered analysis.")

    available_papers = vector_manager.get_uploaded_papers()

    if not available_papers:
        st.warning("⚠️ Please upload and process a paper from the sidebar first.")
    else:
        selected_paper = st.selectbox(
            "Select a paper to analyze:",
            available_papers,
            key="analysis_paper_select"
        )

        # Clear stale analysis result when a different paper is selected
        if st.session_state.get("last_analyzed_paper") != selected_paper:
            st.session_state["analysis_result"] = ""
            st.session_state["last_analyzed_paper"] = selected_paper

        st.divider()

        col1, col2, col3, col4 = st.columns(4)

        # ---- Summary ----
        with col1:
            if st.button("📄 Summary", use_container_width=True):
                with st.spinner("Generating summary..."):
                    query = "abstract research problem motivation proposed method methodology experiments results conclusion"
                    chunks = vector_manager.query(query, paper_name=selected_paper, n_results=10)
                    context = "\n\n".join(c["text"] for c in chunks)
                    result = llm_manager.generate_analysis(context, "summary")
                st.session_state["analysis_result"] = result

        # ---- Key Contributions ----
        with col2:
            if st.button("🔑 Contributions", use_container_width=True):
                with st.spinner("Identifying contributions..."):
                    query = "main contributions innovations proposed method important improvements novelty"
                    chunks = vector_manager.query(query, paper_name=selected_paper, n_results=5)
                    context = "\n\n".join(c["text"] for c in chunks)
                    result = llm_manager.generate_analysis(context, "contributions")
                st.session_state["analysis_result"] = result

        # ---- Limitations ----
        with col3:
            if st.button("⚠️ Limitations", use_container_width=True):
                with st.spinner("Analyzing limitations..."):
                    query = "limitations weaknesses constraints disadvantages problems future challenges"
                    chunks = vector_manager.query(query, paper_name=selected_paper, n_results=5)
                    context = "\n\n".join(c["text"] for c in chunks)
                    result = llm_manager.generate_analysis(context, "limitations")
                st.session_state["analysis_result"] = result

        # ---- Future Work ----
        with col4:
            if st.button("🔮 Future Work", use_container_width=True):
                with st.spinner("Finding future directions..."):
                    query = "future work future research directions open problems improvements extensions"
                    chunks = vector_manager.query(query, paper_name=selected_paper, n_results=5)
                    context = "\n\n".join(c["text"] for c in chunks)
                    result = llm_manager.generate_analysis(context, "future_work")
                st.session_state["analysis_result"] = result

        # Display the latest analysis result
        if "analysis_result" in st.session_state and st.session_state["analysis_result"]:
            st.divider()
            st.markdown(st.session_state["analysis_result"])

with tab3:
    st.header("⚖️ Compare Papers")
    st.write("Select two or more papers to compare them side-by-side.")

    available_papers = vector_manager.get_uploaded_papers()

    if len(available_papers) < 2:
        st.warning("⚠️ You need at least **2 papers** in your library to use this feature. Upload more papers from the sidebar.")
    else:
        selected_papers_for_comparison = st.multiselect(
            "Select papers to compare (choose 2 or more):",
            available_papers,
            default=available_papers[:2] if len(available_papers) >= 2 else []
        )

        comparison_question = st.text_area(
            "What would you like to compare? (optional)",
            placeholder="e.g. How do the methodologies differ? Which paper has better results?\n\nLeave blank to get an automatic side-by-side summary of all selected papers.",
            height=100
        )
        st.caption("💡 Leave the field blank to auto-generate a structured summary of both papers.")

        if st.button("⚖️ Compare Papers", use_container_width=True):
            if len(selected_papers_for_comparison) < 2:
                st.warning("Please select at least 2 papers to compare.")
            else:
                is_auto_summary = not comparison_question.strip()
                spinner_msg = (
                    f"Generating summary for {len(selected_papers_for_comparison)} papers..."
                    if is_auto_summary
                    else f"Comparing {len(selected_papers_for_comparison)} papers..."
                )
                with st.spinner(spinner_msg):
                    summary_query = (
                        "abstract research problem motivation proposed method methodology experiments results conclusion limitations future work"
                        if is_auto_summary
                        else comparison_question
                    )
                    paper_chunks_dict = vector_manager.query_multiple_papers(
                        query_text=summary_query,
                        paper_names=selected_papers_for_comparison,
                        n_results_per_paper=8 if is_auto_summary else 5
                    )
                    comparison_result = llm_manager.compare_papers(
                        paper_chunks_dict=paper_chunks_dict,
                        question=comparison_question if not is_auto_summary else None
                    )

                st.divider()
                st.subheader("📊 Summary" if is_auto_summary else "📊 Comparison Result")
                st.markdown(comparison_result)

                # Show sources per paper
                with st.expander("📄 Sources Used"):
                    for paper, chunks in paper_chunks_dict.items():
                        pages = sorted(set(c["page"] for c in chunks))
                        st.write(f"**{paper}**: Pages {', '.join(str(p) for p in pages)}")