import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Streamlit app title
st.title("Phrase Identification System")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

# Service categories and phrases
service_categories = {
    "Roadside Assistance Services": [
        "Emergency towing service",
        "Battery jump-start",
        "Flat tire repair",
        "Fuel delivery service",
        "Lockout assistance",
        "Winching and extraction",
        "Accident recovery services",
        "24/7 roadside support",
        "Vehicle breakdown support",
        "Roadside mechanic assistance"
    ],
    "ROP (Roadside On-the-Spot Repairs) Services": [
        "Minor mechanical repairs",
        "Emergency battery replacement",
        "Mobile tire replacement",
        "Emergency fuel refill",
        "Electrical system repairs",
        "Cooling system top-up",
        "Quick engine diagnostics",
        "Temporary leak fixing",
        "On-site brake adjustments",
        "Fuse replacement"
    ],
    "Motor Policy Renewal Phrases/Services": [
        "Policy renewal reminder",
        "Insurance premium payment",
        "Online policy renewal",
        "No Claim Bonus (NCB) benefits",
        "Coverage enhancement options",
        "Add-on coverage selection",
        "Policy document download",
        "Renewal confirmation receipt",
        "Expiry date notification",
        "Premium calculator tool"
    ]
}

# Chat Prompt Template
prompt_template = ChatPromptTemplate.from_template(
    """
    Identify which of the following services are relevant to the given paragraph.
    Paragraph: {context}
    Services: {services}
    Provide the matched services along with their respective categories.
    Respond only in this exact format:
    Category: [Category Name]
    Matched Services: [list of matched services]
    """
)

# Initialize Embedding Model and Vector Store
def vector_embedding(paragraph, phrases):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    combined_data = [paragraph] + phrases
    vectors = FAISS.from_texts(combined_data, embeddings)
    return vectors

# Input field for paragraph
paragraph_input = st.text_area("Enter the paragraph:")

# Button to start processing
if st.button("Identify Relevant Services"):
    if paragraph_input:
        all_phrases = [phrase for phrases in service_categories.values() for phrase in phrases]
        vectors = vector_embedding(paragraph_input, all_phrases)
        retriever = vectors.as_retriever()

        document_chain = prompt_template.format(
            context=paragraph_input, 
            services="; ".join(all_phrases)
        )

        start = time.process_time()
        response = llm.invoke(document_chain)
        end = time.process_time()

        # Extract content from the AIMessage object
        response_text = response.content.strip() if hasattr(response, 'content') else str(response).strip()

        # Parsing the response
        matched_services = {}
        for category, services in service_categories.items():
            matched_services[category] = [service for service in services if service in response_text]

        # Display results
        for category, services in matched_services.items():
            if services:
                st.write(f"**{category}:** {', '.join(services)}")

        if not any(matched_services.values()):
            st.write("No relevant services matched the paragraph.")

        st.write(f"Response Time: {end - start:.2f} seconds")

    else:
        st.error("Please provide a paragraph.")
