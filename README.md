# JalDrishya: FloatChat ⚓

**FloatChat** is an intelligent, conversational data explorer for ARGO oceanographic data, built for the Smart India Hackathon. It uses a hybrid AI system to allow users to ask complex questions in natural language and receive instant, interactive visualizations.

---

### ## Key Features

- **Conversational Queries**: Ask questions in plain English like `"Plot the temperature vs depth graph of January month"`.
- **Hybrid AI System**: Utilizes a fast, local Llama model for simple queries and automatically escalates to the powerful Google Gemini for complex requests, ensuring both speed and accuracy.
- **"Rules-First" Guardrails**: For common queries (like maps by city), a high-precision rule-based parser is used first to guarantee 100% accuracy and instant response.
- **Interactive Visualizations**: 
  - **Interactive Maps**: Not only displays float locations but allows users to draw a selection on the map to filter data in real-time.
  - **Attractive Plots**: Generates custom-themed, professional plots for depth profiles and other analyses.

---

### ## Tech Stack

- **Framework**: Streamlit
- **AI Models**:
  - Google Gemini 2.5 Flash (Cloud-based)
  - Llama 3.1 8B (Local, via Ollama)
- **Data Visualization**: Plotly, Folium
- **Data Handling**: Pandas, SQLite
- **Language**: Python

---

### ## Setup and Installation

**1. Clone the Repository:**
```bash
git clone [https://github.com/Amarshah07/JalDrishya.git](https://github.com/Amarshah07/JalDrishya.git)
cd JalDrishya
```

**2. Install Dependencies:**
Make sure you have Python 3.9+ installed.
```bash
pip install -r requirements.txt
```

**3. Set Up Local AI (Ollama):**
- Download and install [Ollama](https://ollama.com/).
- Pull the Llama 3.1 model:
  ```bash
  ollama pull llama3.1:8b
  ```

**4. Configure API Key (IMPORTANT):**
- Create a folder: `mkdir .streamlit`
- Create a secrets file: `touch .streamlit/secrets.toml`
- Add your Gemini API key to the `secrets.toml` file:
  ```toml
  # .streamlit/secrets.toml
  GEMINI_API_KEY = "YOUR_API_KEY_HERE"
  ```

**5. Place Your Database:**
- Ensure your `argo_data.db` file is inside the `data/` directory.

**6. Run the App:**
```bash
streamlit run app.py
```