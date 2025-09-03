# BME i9400: Special Topics in Machine Learning
## Fall 2025 Course Syllabus

### _Note: all information provided in this syllabus is subject to change at anytime at the instructor's discretion._


### Course Information
- **Course Title:** BME i9400 — Special Topics in Machine Learning (Graduate)
- **Department:** Biomedical Engineering, City College of New York
- **Credits:** 3
- **Schedule:** Mondays & Wednesday 11 AM - 12:15 PM
- **Location:** Steinman Hall, C-51 (Cellar Level)
- **Duration:** 14 weeks (28 lectures)
- **Modality:** In-person, workshop-style sessions
- **Platform:** JupyterHub on AWS (http://54.162.215.33)

### Instructor Information
- **Instructor:** Jacek Dmochowski, PhD
- **Office:** Steinman Hall, Room 460
- **Email:** jdmochowski@ccny.cuny.edu
- **Office Hours:** [TBD]

### Course Description
This graduate-level course provides hands-on experience with machine learning techniques applied to biomedical engineering problems. Students will master fundamental ML concepts, implement neural networks and transformers, work with multimodal biomedical data, and critically evaluate AI systems for healthcare applications. The course emphasizes practical implementation through interactive coding sessions using Python and modern ML frameworks.
The use of generative AI and large language models (LLMs) is permitted and encouraged; however, students must validate and critically interpret AI-generated outputs. The course culminates in a capstone project where students apply learned techniques to a biomedical problem of their choice.

### Learning Outcomes
By the end of this course, students will be able to:
1. **Formulate** biomedical problems as regression/classification tasks and choose suitable models
2. **Train** models via gradient-based optimization; understand losses, regularization, and evaluation
3. **Implement** and **analyze** CNNs, sequence models, and transformer-based LLM workflows
4. **Integrate** text, images, and time-series in simplified multimodal pipelines
5. **Critically evaluate** fairness, explainability, privacy, and deployment constraints in biomedicine
6. **Deliver** a reproducible mini-project with appropriate metrics, visualizations, and limitations

### Prerequisites
- Linear algebra and probability/statistics foundation
- Basic Python programming experience
- No prior ML experience required

### Required Technology
- Laptop with web browser (Chrome/Firefox recommended)
- GitHub account for accessing course materials
- JupyterHub access (provided on first day)

### Assessment & Grading

| Component | Weight | Description                                                                   |
|-----------|--------|-------------------------------------------------------------------------------|
| **In-Class Micro-deliverables** | 20% | Weekly small artifacts (plots, metrics, code snippets) completed during class |
| **Homework Assignments** | 10% | Take-home exercises extending lab work                                        |
| **Midterm Exam** | 30% | Practical exam covering lectures 1-13                                         |
| **Final Project** | 40% | Capstone with implementation, presentation, and report                        |
| **Total** | 100% |                                                                               |

#### Micro-deliverable Rubric (3-point scale)
- **2 points (Complete):** Code runs, meets specification
- **3 points (Insightful):** Correct + interpretation/diagnostics
- **4 points (Exceptional):** Clean visualizations, robust error analysis

#### Final Project Components
- Proposal & dataset selection (5%)
- Implementation & documentation (25%)
- Final presentation & model card (10%)
- Dates and deliverables will be provided in class

### Course Schedule

#### **Weeks 1-3: Foundations** (6 lectures)
Building core mathematical and evaluation literacy before modern architectures.

| Lec | Date   | Topic                                     | Key Concepts                                     |
|-----|--------|-------------------------------------------|--------------------------------------------------|
| 1   | Week 1 | Python Crash Course                       | Numpy, Pandas, Matplotlib                        |
| 2   | Week 1 | Probability & Distributions Refresher     | Bayes' rule, sensitivity/specificity, prevalence |
| 3   | Week 2 | Linear Algebra for ML                     | Vectors/matrices, PCA on gene expression         |
| 4   | Week 2 | Least Squares Regression                  | Normal equations, geometric view, Gaussian noise |
| 5   | Week 3 | Optimization & Training Basics            | GD/SGD, learning rates, convergence              |
| 6   | Week 3 | Regression vs Classification + Evaluation | ROC/PRC, thresholds, AUC                         |

#### **Weeks 4-5: Core ML in Biomedical Context** (4 lectures)
Fundamental models with biomedical applications.

| Lec | Date   | Topic | Key Concepts |
|-----|--------|-------|--------------|
| 7   | Week 4 | Logistic Regression & Regularization | L1/L2, overfitting control |
| 8   | Week 4 | Convolutional Neural Networks | Conv/pool, medical image analysis |
| 9   | Week 5 | Neural Networks | MLP, nonlinearity, capacity |
| 10  | Week 5 | Sequence Models | RNN/LSTM/GRU for ECG/EEG |

#### **Weeks 6-8: Generative AI Fundamentals** (6 lectures)
Deep understanding of transformers and LLMs with biomedical NLP.

| Lec | Date   | Topic | Key Concepts         |
|-----|--------|-------|----------------------|
| 11  | Week 6 | From RNNs to Transformers | Attention motivation |
| 12  | Week 6 | Transformer Internals I | Embeddings & positional encodings |
| 13  | Week 7 | Transformer Internals II | Self-attention & multi-head |
| 14  | Week 7 |  **MIDTERM**     |                      |
| 15  | Week 8 | Training an LLM | Tokenization         |
| 16  | Week 8 | Inference & Prompt Engineering | Decoding, PubMed summarization |

#### **Weeks 9-10: Multimodal Biomedical AI** (3 lectures)
Combining text, images, and time-series data.

| Lec | Date    | Topic | Key Concepts |
|-----|---------|-------|--------------|
| 17  | Week 9  | Text + Images | CLIP-style pairing, zero-shot demos |
| 18  | Week 9  | Text + Time Series | Simple fusion strategies |
| 19  | Week 10 | Self-Supervised & Generative Multimodal | Masked reconstruction |

#### **Weeks 10-11: Translation, Ethics, and Safety** (3 lectures)
Critical evaluation of AI in healthcare.

| Lec | Date    | Topic | Key Concepts |
|-----|---------|-------|--------------|
| 20  | Week 10 | Bias & Fairness | Audit & mitigation strategies |
| 21  | Week 11 | Explainability | SHAP/Grad-CAM, trust boundaries |
| 22  | Week 11 | Privacy & Federated Learning | Toy federation simulation |

#### **Weeks 12-14: Capstone Sprint** (6 lectures)
Guided final project development.

| Lec | Date    | Topic | Deliverable |
|-----|---------|-------|-------------|
| 23  | Week 12 | Ideation & Dataset | 1-page proposal |
| 24  | Week 12 | Prototype I | Working baseline |
| 25  | Week 13 | Prototype II | LLM/multimodal integration |
| 26  | Week 13 | Peer Review & Critique | Peer review forms |
| 27  | Week 14 | Presentations & Reflection | Final presentation |
| 28  | Week 14 | Presentations & Reflection | Final presentation |

### Course Policies

#### Attendance Policy
- Workshop-style format requires active participation
- One unexcused absence permitted: please do not email the instructor with an explanation _after_ a missed class under any circumstances.

#### Late Work Policy
- Micro-deliverables: Must be completed in class (no late submission)
- Homework: 10% penalty per day late
- Project milestones: 20% penalty per day late
- Medical/emergency exceptions considered case-by-case

#### Academic Integrity
- Individual work must be your own
- Pair programming allowed for in-class work with attribution
- LLM assistance permitted with citation of prompts
- Plagiarism or unauthorized collaboration results in course failure

#### Use of AI Tools
- GitHub Copilot, ChatGPT, and similar tools allowed as assistants and educational tools
- Policy: cite your prompts used and validate outputs manually

### Required Materials
- **Textbook:** None required; materials provided via JupyterHub
- **Software:** Python 3.11, NumPy, Pandas, Matplotlib, scikit-learn, PyTorch, HuggingFace (all pre-installed)
- **Datasets:** Public, de-identified biomedical datasets will be provided as needed

### Technical Requirements
- All computation runs on CPU-only cloud instances
- No GPU required
- No local installation needed
- Internet connection required for JupyterHub access

### Communication
- Course announcements via JupyterHub landing page
- Questions during class encouraged
- Email response within 48 hours on weekdays

### Resources & Support
- **JupyterHub:** http://54.162.215.33
- **Course Repository:** https://github.com/BME-i9400/BME-i9400
- **Python Documentation:** https://docs.python.org/3.11/
- **PyTorch Tutorials:** https://pytorch.org/tutorials/


### Learning Philosophy
This course emphasizes:
- **Hands-on learning** through immediate implementation
- **Ethical considerations** in AI deployment
- **Collaborative problem-solving** in workshop format


---
*Last updated: August 2025*
