import os
from gui import start_gui
from Face_Recog import load_known_encodings

# Constants
DATA_DIRECTORY_NAME = 'people'

# Main function
def main():
    # Load known encodings
    people_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', DATA_DIRECTORY_NAME))
    known_encodings, known_filenames = load_known_encodings(people_directory)
    
    # Start GUI
    start_gui(known_encodings, known_filenames, people_directory)

if __name__ == "__main__":
    main()