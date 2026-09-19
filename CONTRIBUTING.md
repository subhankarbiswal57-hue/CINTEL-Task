# Contributing Guidelines

Thank you for your interest in contributing to the **Titanic Survival Prediction & Model Comparison Pipeline**!

## Development Workflow

1. **Fork or Clone the Repository**:
   ```bash
   git clone https://github.com/subhankarbiswal57-hue/CINTEL-Task.git
   cd CINTEL-Task
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Unit Tests**:
   ```bash
   pytest tests
   ```
   Ensure all unit tests pass before committing or proposing changes.

5. **Code Quality Standards**:
   - Follow PEP 8 guidelines for formatting.
   - Maintain strict separation between feature engineering, validation, and testing to prevent data leakage.
   - All contributions must include corresponding unit tests when introducing new functionality.

## Reporting Issues
If you encounter any bugs or data inconsistencies, please open an Issue with steps to reproduce.
