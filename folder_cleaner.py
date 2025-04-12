#!/usr/bin/env python3
import os
import shutil
import argparse
from typing import List, Union, Optional

def clear_folder(folder_path: str, keep_gitkeep: bool = True) -> int:
    """
    Clear all files in a folder, optionally preserving .gitkeep files.
    
    Args:
        folder_path: Path to the folder to clear
        keep_gitkeep: Whether to preserve .gitkeep files
        
    Returns:
        Number of files removed
    """
    if not os.path.exists(folder_path):
        print(f"Warning: Folder '{folder_path}' does not exist")
        return 0
    
    if not os.path.isdir(folder_path):
        print(f"Warning: '{folder_path}' is not a directory")
        return 0
    
    count = 0
    for filename in os.listdir(folder_path):
        # Skip .gitkeep files if requested
        if keep_gitkeep and filename == ".gitkeep":
            continue
            
        file_path = os.path.join(folder_path, filename)
        
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                count += 1
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                count += 1
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    
    return count

def clear_input_folder(keep_gitkeep: bool = True) -> int:
    """
    Clear all files in the input folder.
    
    Args:
        keep_gitkeep: Whether to preserve .gitkeep files
        
    Returns:
        Number of files removed
    """
    folder_path = os.path.join(os.getcwd(), "input")
    count = clear_folder(folder_path, keep_gitkeep)
    print(f"Cleared {count} files/folders from input folder")
    return count

def clear_output_folder(keep_gitkeep: bool = True) -> int:
    """
    Clear all files in the output folder.
    
    Args:
        keep_gitkeep: Whether to preserve .gitkeep files
        
    Returns:
        Number of files removed
    """
    folder_path = os.path.join(os.getcwd(), "output")
    count = clear_folder(folder_path, keep_gitkeep)
    print(f"Cleared {count} files/folders from output folder")
    return count

def clear_batch_jobs_folder(keep_gitkeep: bool = True) -> int:
    """
    Clear all files in the batch_jobs folder.
    
    Args:
        keep_gitkeep: Whether to preserve .gitkeep files
        
    Returns:
        Number of files removed
    """
    folder_path = os.path.join(os.getcwd(), "batch_jobs")
    count = clear_folder(folder_path, keep_gitkeep)
    print(f"Cleared {count} files/folders from batch_jobs folder")
    return count

def clear_folders(folders: List[str], keep_gitkeep: bool = True) -> int:
    """
    Clear multiple folders.
    
    Args:
        folders: List of folder names to clear ('input', 'output', 'batch_jobs')
        keep_gitkeep: Whether to preserve .gitkeep files
        
    Returns:
        Total number of files removed
    """
    total_count = 0
    
    for folder in folders:
        if folder.lower() == 'input':
            total_count += clear_input_folder(keep_gitkeep)
        elif folder.lower() == 'output':
            total_count += clear_output_folder(keep_gitkeep)
        elif folder.lower() == 'batch_jobs':
            total_count += clear_batch_jobs_folder(keep_gitkeep)
        else:
            print(f"Warning: Unknown folder '{folder}'")
    
    return total_count

def clear_all_folders(keep_gitkeep: bool = True) -> int:
    """
    Clear all folders (input, output, batch_jobs).
    
    Args:
        keep_gitkeep: Whether to preserve .gitkeep files
        
    Returns:
        Total number of files removed
    """
    print("Clearing all folders...")
    total_count = 0
    total_count += clear_input_folder(keep_gitkeep)
    total_count += clear_output_folder(keep_gitkeep)
    total_count += clear_batch_jobs_folder(keep_gitkeep)
    print(f"Cleared a total of {total_count} files/folders")
    return total_count

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Clear input, output, and batch_jobs folders")
    
    # Add folder selection arguments
    parser.add_argument("--input", action="store_true", help="Clear the input folder")
    parser.add_argument("--output", action="store_true", help="Clear the output folder")
    parser.add_argument("--batch-jobs", action="store_true", help="Clear the batch_jobs folder")
    parser.add_argument("--all", action="store_true", help="Clear all folders")
    
    # Add option to delete .gitkeep files
    parser.add_argument("--delete-gitkeep", action="store_true", 
                        help="Also delete .gitkeep files (default: preserve them)")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    # Determine whether to keep .gitkeep files
    keep_gitkeep = not args.delete_gitkeep
    
    # Check if any folder was specified
    if not (args.input or args.output or args.batch_jobs or args.all):
        print("Error: No folders specified to clear")
        print("Use --input, --output, --batch-jobs, or --all to specify which folders to clear")
        return
    
    # Clear specified folders
    if args.all:
        clear_all_folders(keep_gitkeep)
    else:
        folders_to_clear = []
        if args.input:
            folders_to_clear.append('input')
        if args.output:
            folders_to_clear.append('output')
        if args.batch_jobs:
            folders_to_clear.append('batch_jobs')
        
        clear_folders(folders_to_clear, keep_gitkeep)

if __name__ == "__main__":
    main()
