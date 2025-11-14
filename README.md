
# Edmond Chong's 1st Project

### 👉 Try Demo: [https://edmondsproject3-rosebupmvbyfrwxz5u2dxk.streamlit.app/](https://edmondproject1st-22x4h2wlhiwogmsj2uylul.streamlit.app/)

🖼️ Use this for testing demo >>> [Images_to_test_demo](./Images_to_test_demo)  

### 🤝 Full project is private to prevent unauthorized copying, but Happy to share upon "Recruiter Request"  
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*** The Streamlit demo may go idle after a period of inactivity — click “Yes, get this app back up” to restart it. Please note that it may take a short while to reload.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
📖 Project Overview: **🏭 Wafer Defect Classification for Semiconductor Manufacturing SMAI**. 

This project focuses on detecting wafer map defects using AI to support Smart Manufacturing AI (SMAI) applications in semiconductor fabs.
It classifies wafer maps into 6 defect types and uses XAI to highlight defect-critical regions for engineering diagnosis.

This system mirrors real-world use cases such as yield analysis, defect clustering, and inline inspection.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🔬 Key Approaches

EfficientNet-B0 Transfer Learning

Used pretrained EfficientNet-B0 to classify wafer defects with 87.01% accuracy, optimizing for both accuracy and inference speed.

6-Class Wafer Defect Taxonomy

Detected industry-inspired wafer map patterns:
Center, Edge-Loc, Edge-Ring, Loc, Scratch, None

Explainable AI (XAI)

Integrated Grad-CAM to visualize defect-driving regions, helping engineers understand failure mechanisms.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
⚙️ Tech Stack

Python

PyTorch

EfficientNet-B0 (Transfer Learning)

OpenCV

Grad-CAM (XAI)

Streamlit

Docker / AWS / Kubernetes

CI/CD (GitHub Actions)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
✨ Features

Built a 6-class wafer defect classifier using EfficientNet-B0.

Achieved 87.01% accuracy on the WM811K wafer map dataset.

Applied Grad-CAM to highlight defect regions for interpretability.

Clean and interactive Streamlit UI for quick testing.

Deployed on cloud infrastructure (Streamlit + Docker + AWS EC2).

Kubernetes + CI/CD support for production readiness.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
📌 Notes for Recruiters

The GitHub repo contains only the demo app code to protect model IP and avoid unauthorized copying.

The full training pipeline (data preprocessing, augmentation, training scripts, experiments, deployment stack, and evaluation) is kept private, but I am happy to share it upon recruiter request.

This project demonstrates end-to-end AI engineering skills, including:
→ data processing → model training → evaluation → explainability → deployment.


