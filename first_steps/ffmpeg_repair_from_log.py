import os
import subprocess


def load_problem_files_from_log(log_file):
    """Load paths of problematic files from the log file."""
    problem_files = []
    with open(log_file, 'r') as file:
        for line in file:
            if line.startswith("Problems in file"):
                file_path = line.split(':')[0].strip()[17:]
                problem_files.append(file_path)
    print(problem_files)
    return problem_files


def reencode_files(problem_files):
    """Re-encode the specified audio files."""
    for file_path in problem_files:
        temp_file_path = file_path + '.temp.mp3'  # Temp file to avoid in-place writing issues
        reencode_cmd = f"ffmpeg -i '{file_path}' -codec:a libmp3lame -qscale:a 2 '{temp_file_path}'"
        subprocess.run(reencode_cmd, shell=True)
        # Move the temp file back to the original file path to replace it
        os.replace(temp_file_path, file_path)
        print(f"Re-encoded and replaced {file_path}")


if __name__ == "__main__":
    log_file = 'audio_processing_issues.log'
    problem_files = load_problem_files_from_log(log_file)
    if problem_files:
        reencode_files(problem_files)
    else:
        print("No problematic files found.")
