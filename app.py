"""
FFAR Partnership Research Agent — Streamlit web app.

Run locally with:
    streamlit run app.py

The app provides a simple UI for the team to research organizations
and download structured Business Profiles.
"""

import os
import streamlit as st
from agent import build_agent_pipeline, research_organization, profile_to_word


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FFAR Partnership Research Agent",
    page_icon="🌾",
    layout="centered"
)


# ============================================================
# HEADER
# ============================================================

st.title("🌾 FFAR Partnership Research Agent")
st.markdown(
    "Enter an organization name to generate a structured Business Profile "
    "following the FFAR template."
)
st.divider()


# ============================================================
# API KEYS (from environment)
# ============================================================

# In local dev, we set these before running.
# On Hugging Face Spaces, they come from Space Secrets.
if "ANTHROPIC_API_KEY" not in os.environ or "TAVILY_API_KEY" not in os.environ:
    st.error(
        "⚠️ Missing API keys. Please set ANTHROPIC_API_KEY and TAVILY_API_KEY "
        "as environment variables before running."
    )
    st.stop()


# ============================================================
# INITIALIZE AGENT (cached — runs only once per session)
# ============================================================

@st.cache_resource
def get_agent():
    """Load the agent pipeline once and reuse it across runs."""
    return build_agent_pipeline()


with st.spinner("Loading agent (first run takes ~1 minute)..."):
    rag_agent, structured_llm = get_agent()


# ============================================================
# USER INPUT
# ============================================================

organization = st.text_input(
    "Organization Name",
    placeholder="e.g., Yara International, Nutrien, Bayer Crop Science"
)

research_button = st.button("🔍 Research Organization", type="primary")


# ============================================================
# RESEARCH PROCESS
# ============================================================

if research_button:
    if not organization.strip():
        st.warning("Please enter an organization name.")
    else:
        with st.spinner(f"Researching {organization}... (this takes 1-3 minutes)"):
            try:
                # Run the research
                profile = research_organization(
                    organization,
                    rag_agent,
                    structured_llm
                )
                
                # Save to Word
                safe_name = organization.replace(" ", "_").replace("/", "_")
                output_file = f"{safe_name}_profile.docx"
                profile_to_word(profile, output_file)
                
                # Success message
                st.success(f"✅ Business Profile ready for {profile.key_facts.organization_name}")
                
                # Display preview on screen
                st.divider()
                st.subheader("📄 Preview")
                
                st.markdown(f"### {profile.key_facts.organization_name}")
                
                # Key facts as a table
                st.markdown("#### Key Facts")
                key_facts_data = {
                    "Field": [
                        "Headquarters", "Structure", "Fiscal Year End",
                        "Geographic Focus", "Core Mission", "Business Offerings"
                    ],
                    "Value": [
                        profile.key_facts.headquarters,
                        profile.key_facts.structure,
                        profile.key_facts.fiscal_year_end,
                        profile.key_facts.geographic_focus,
                        profile.key_facts.core_mission,
                        profile.key_facts.business_offerings,
                    ]
                }
                st.table(key_facts_data)
                
                # Goals
                st.markdown("#### Publicly Stated Goals & Commitments")
                for goal in profile.publicly_stated_goals:
                    st.markdown(f"- {goal}")
                
                # Partnerships
                st.markdown("#### Existing R&D Partnerships")
                for partnership in profile.existing_partnerships:
                    st.markdown(f"- {partnership}")
                
                # FFAR Opportunity Analysis
                st.markdown("#### FFAR Opportunity Analysis")
                st.markdown(profile.ffar_opportunity_analysis)
                
                # Download button
                st.divider()
                with open(output_file, "rb") as f:
                    st.download_button(
                        label="📥 Download Word Document",
                        data=f.read(),
                        file_name=output_file,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                
            except Exception as e:
                st.error(f"❌ Error during research: {e}")
                st.info("Try again with a different organization name.")


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "Built with LangChain + Claude for the FFAR Partnerships team. "
    "Output is a first-pass draft — always verify facts before external use."
)