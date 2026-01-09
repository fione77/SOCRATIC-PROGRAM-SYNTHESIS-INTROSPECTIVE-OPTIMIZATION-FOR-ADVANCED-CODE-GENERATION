Socratic Program Synthesis 
This project is written in Python and uses external depenencies listed in 'requirements.txt'

## 1. Clone the repository:

Open your terminal and run: 
'''
git clone <https://github.com/fione77/Socratic-Program-Synthesis.git>

cd <REPO_FOLDER_NAME>
'''

## 2. (Recommended) Create a virtual environment 
'''
python -m venv venv

venv\Scripts\activate
'''

## 3. Install dependencies
'''
pip install -r requirements.txt
'''

## 4. Create an .env file in project root directory
- This project uses the 3.3 70B Instruct Model from OpenRouter

## 5. Run the program

To get the debate and generate code files (for socratic):
'''
python socratic_gen.py
'''

to get the generated code file for direct:
'''
python direct_generation.py
'''

to run the evaluation for both the direct and socratic generation: 
'''
python eval.py
'''
