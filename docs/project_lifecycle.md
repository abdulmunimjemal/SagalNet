# Project Lifecycle Guide

## 1. Project Setup
- **Environment**: Python 3.9+ with Virtual Environment.
- **Dependencies**: Listed in `requirements.txt`.
- **Structure**:
    - `src/`: Source code (data, models).
    - `data/`: Dataset storage.
    - `notebooks/`: Exploration and prototyping.
    - `tests/`: Automated tests.

## 2. Development Workflow
1.  **Exploration**: Use Jupyter notebooks in `notebooks/` to inspect data and prototype ideas.
2.  **Implementation**: Move stable code to `src/` modules.
3.  **Training**: Use `run.py train` to train the model.
4.  **Evaluation**: Use `run.py predict` or evaluation scripts to verify performance.

## 3. Version Control
- Commit often.
- Use `.gitignore` to exclude large data files and virtual environments.

## 4. Documentation
- Maintain `README.md` for general info.
- Use `docs/` for detailed technical documentation.
