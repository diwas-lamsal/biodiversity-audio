import os
import subprocess
import shlex
import pickle

def check_audio_files(directory, log_file):
    problem_files = []
    with open(log_file, 'w') as log:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.mp3'):
                    file_path = os.path.join(root, file)
                    command = f"python3 -c \"import librosa; librosa.load('{file_path}', sr=32000, mono=False)\""
                    result = subprocess.run(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout_output = result.stdout.decode()
                    stderr_output = result.stderr.decode()
                    if stdout_output or stderr_output:
                        output = stdout_output + stderr_output
                        log.write(f"Problems in file {file_path}:\n{output}\n")
                        print(f"Problems in file {file_path}:\n{output}")
                        problem_files.append(file_path)
    return problem_files

def save_problem_files(problem_files, pickle_file):
    """Save the list of problem files to a pickle file."""
    with open(pickle_file, 'wb') as pf:
        pickle.dump(problem_files, pf)
    print(f"Saved problem files to {pickle_file}")

if __name__ == "__main__":
    log_file = 'audio_processing_issues.log'
    directory = './sounds' # Path to the main audio directory containing all the species data
    pickle_file = 'problem_files.pkl'
    problem_files = check_audio_files(directory, log_file)
    save_problem_files(problem_files, pickle_file)
