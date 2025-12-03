# TaskLoop AI 🚀
### Intelligent Code Review & Automated Mentorship Agent

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![Gemini](https://img.shields.io/badge/AI-Gemini%202.0-8E75B2?logo=google&logoColor=white)
![MongoDB](https://img.shields.io/badge/Database-MongoDB%20Atlas-47A248?logo=mongodb&logoColor=white)
![Netlify](https://img.shields.io/badge/Frontend-Netlify-00C7B7?logo=netlify&logoColor=white)

**TaskLoop AI** is a full-stack EdTech platform designed to automate the code review feedback loop. Unlike standard chatbots, it acts as a structured mentorship agent: it reviews student submissions, calculates technical scores, and—most importantly—**autonomously generates unique follow-up tasks** to address specific coding weaknesses.

---

## 🌟 Live Demo
- **Frontend (UI):** [https://sid12taskloop-ai.netlify.app/]
- **Backend (API):** [https://siddhesh1207-taskloopai.hf.space]

---

## 🔥 Key Features

### 🎓 For Teachers (Admin)
- **Classroom Management:** Secure authentication to create and manage private class environments.
- **Task Creation:** Define specific coding assignments with descriptions and learning goals.
- **Pending Reviews Dashboard:** View student submissions that need human oversight (DHI - Dignity, Honesty, Integrity scores).

### 👨‍💻 For Students
- **Scoped Login:** Access assignments specific to your teacher's classroom.
- **AI Code Review:** Get instant feedback on code submissions (Text, File, or GitHub Link).
- **Automated Mentorship:** The AI doesn't just grade; it creates a **new, custom assignment challenge** based on your mistakes to help you improve immediately.

---

## 🏗️ Architecture

The application follows a decoupled **Multi-Tenant Architecture**:

1.  **Frontend:** Pure HTML/JS/CSS hosted on **Netlify**. It connects to the backend via REST API.
2.  **Backend:** A **FastAPI** service hosted on **Hugging Face Spaces (Dockerized)**.
3.  **Database:** **MongoDB Atlas** (Cloud) stores users, tasks, reviews, and history.
4.  **AI Engine:** **Google Gemini 2.0 Flash** via **LangChain** handles the logic for code analysis and task generation.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10
- **Framework:** FastAPI
- **AI/LLM:** Google Gemini 2.0 Flash, LangChain
- **Database:** MongoDB Atlas (PyMongo)
- **Security:** Passlib (PBKDF2 Hashing), JWT-style API Keys
- **Deployment:** Docker, Hugging Face Spaces, Netlify

---
