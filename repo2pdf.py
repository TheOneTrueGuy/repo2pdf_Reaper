"""
Script to convert GitHub repository content to PDF format.

Dependencies:
 pip install PyMuPDF requests PyGithub pillow
"""

import fitz  # PyMuPDF
import os
import json
import requests
from github import Github
from datetime import datetime
from io import BytesIO
from PIL import Image
import argparse
import sys
from urllib.parse import urlparse

def get_file_content(git_repo, path):
    """
    Get file content from GitHub repository.
    Returns:
    - For text files: decoded text content
    - For binary files: raw bytes
    - None: if file cannot be retrieved or should be skipped
    """
    # Extensions to skip (model weights, large binary files, etc.)
    skip_extensions = {
        '.gguf',  # LLaMA model weights
        '.bin',   # Generic binary files
        '.pth',   # PyTorch weights
        '.onnx',  # ONNX model files
        '.h5',    # Keras/HDF5 weights
        '.pkl',   # Pickle files
        '.model', # Generic model files
        '.weights', # Generic weight files
        '.pt',    # PyTorch weights
        '.ckpt',  # Checkpoint files
        '.safetensors', # Safe tensors format
        '.data',  # Generic data files
        '.gz',    # Compressed files
        '.zip',   # Zip archives
        '.tar',   # Tar archives
        '.7z',    # 7zip archives
        '.rar',   # RAR archives
        '.dll',   # Dynamic libraries
        '.so',    # Shared objects
        '.dylib', # Dynamic libraries (Mac)
        '.exe',   # Executables
    }
    
    # Skip files with these extensions
    if any(path.lower().endswith(ext) for ext in skip_extensions):
        print(f"Skipping binary/model file: {path}")
        return None
        
    try:
        file = git_repo.get_contents(path)
        # Check if the file is likely binary based on its extension
        binary_extensions = {'.ico', '.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff'}
        is_binary = any(path.lower().endswith(ext) for ext in binary_extensions)
        
        if is_binary:
            return file.decoded_content  # Return raw bytes for binary files
        elif file.encoding == 'base64':
            return file.decoded_content.decode('utf-8')  # Decode as text for text files
        return None
    except Exception as e:
        print(f"Could not get content for {path}: {e}")
        return None

def add_file_to_pdf(git_repo, pdf, path, page_width, page_height):
    """
    Add file content or image to the PDF.
    """
    page = pdf.new_page(width=page_width, height=page_height)
    
    # Add the file path at the top of the page
    page.insert_text((50, 30), f"File: {path}", fontsize=8)
    
    content = get_file_content(git_repo, path)
    
    if content is None:
        # Add a note for skipped files
        if any(path.lower().endswith(ext) for ext in get_file_content.__defaults__[0]):  # Access the skip_extensions set
            note = "Note: Binary/model weight file - skipped for PDF generation"
            page.insert_text((50, 100), note, fontsize=10, color=(0.7, 0.7, 0.7))  # Gray text
            return
    # Handle binary (image) files
    if content and path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff', '.ico')):
        try:
            img_data = BytesIO(content)  # Use the binary content directly
            
            try:
                # Use PIL to open and verify the image
                with Image.open(img_data) as img:
                    # For ICO files, take the largest size available
                    if path.lower().endswith('.ico') and hasattr(img, 'ico'):
                        sizes = img.info.get('sizes', [(img.size[0], img.size[1])])
                        max_size = max(sizes, key=lambda x: x[0] * x[1])
                        img = img.resize(max_size)
                    
                    # Convert to RGB, handling RGBA and palette modes
                    if img.mode in ('RGBA', 'P', 'LA'):
                        # Create white background for transparent images
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        if img.mode in ('RGBA', 'LA'):
                            background.paste(img, mask=img.split()[-1])
                            img = background
                        else:
                            img = img.convert('RGB')
                    
                    # Save to a new BytesIO object in JPEG format
                    img_bytes = BytesIO()
                    img.save(img_bytes, format='JPEG', quality=95)
                    img_bytes.seek(0)
                    
                    # Calculate scaling to fit the page while maintaining aspect ratio
                    img_width, img_height = img.size
                    page_rect = fitz.Rect(50, 70, page_width-50, page_height-50)
                    available_width = page_rect.width
                    available_height = page_rect.height
                    
                    # Define minimum and maximum dimensions (in points)
                    MIN_DISPLAY_SIZE = 100  # Minimum width or height
                    MAX_DISPLAY_SIZE = min(available_width, available_height)
                    
                    # Calculate initial scale based on available space
                    scale_width = available_width / img_width
                    scale_height = available_height / img_height
                    scale = min(scale_width, scale_height)
                    
                    # Calculate dimensions after initial scaling
                    new_width = img_width * scale
                    new_height = img_height * scale
                    
                    # Check if image is too small and adjust scale if necessary
                    if new_width < MIN_DISPLAY_SIZE and new_height < MIN_DISPLAY_SIZE:
                        # Scale up to reach minimum size while maintaining aspect ratio
                        scale_up = MIN_DISPLAY_SIZE / min(new_width, new_height)
                        new_width *= scale_up
                        new_height *= scale_up
                    
                    # Ensure we don't exceed maximum size
                    if new_width > MAX_DISPLAY_SIZE or new_height > MAX_DISPLAY_SIZE:
                        scale_down = MAX_DISPLAY_SIZE / max(new_width, new_height)
                        new_width *= scale_down
                        new_height *= scale_down
                    
                    # Center the image on the page
                    x_offset = (page_width - new_width) / 2
                    y_offset = 70  # Leave space for the file path at the top
                    
                    # Create a new rect for the scaled and centered image
                    img_rect = fitz.Rect(x_offset, y_offset, 
                                       x_offset + new_width, 
                                       y_offset + new_height)
                    
                    # Insert the image
                    page.insert_image(img_rect, stream=img_bytes.getvalue())
                    
                    # Add image dimensions and format as a caption
                    caption = f"Original dimensions: {img_width}x{img_height} pixels | Format: {img.format}"
                    page.insert_text((50, y_offset + new_height + 20), caption, fontsize=8)
                    
            except Exception as e:
                error_msg = f"Failed to process image {path}: {str(e)}"
                print(error_msg)
                page.insert_text((50, 100), error_msg, fontsize=10, color=(1, 0, 0))  # Red error text
                
        except Exception as e:
            error_msg = f"Failed to download image {path}: {str(e)}"
            print(error_msg)
            page.insert_text((50, 100), error_msg, fontsize=10, color=(1, 0, 0))  # Red error text
    elif content:
        frame = fitz.Rect(50, 50, page_width-50, page_height-50)
        page.insert_textbox(frame, content, fontsize=8, align=0)

def repo_to_pdf(git_repo, target_path=None):
    """
    Convert repository structure to a PDF file including file content and images.
    """
    pdf = fitz.open()
    page_width, page_height = 595, 842  # A4 size in points
    
    # Statistics tracking
    stats = {
        'processed': {
            'text_files': [],
            'images': [],
        },
        'skipped': {
            'binary_files': [],
            'model_files': [],
            'archives': [],
            'executables': [],
            'other': []
        }
    }
    
    def categorize_skipped_file(path):
        """Categorize skipped files for the summary"""
        ext = path.lower()
        if ext.endswith(('.gguf', '.pth', '.onnx', '.h5', '.weights', '.pt', '.ckpt', '.safetensors')):
            return 'model_files'
        elif ext.endswith(('.zip', '.tar', '.7z', '.rar', '.gz')):
            return 'archives'
        elif ext.endswith(('.exe', '.dll', '.so', '.dylib')):
            return 'executables'
        elif ext.endswith(('.bin', '.pkl', '.data')):
            return 'binary_files'
        return 'other'

    def add_summary_page():
        """Add a summary page at the start of the PDF"""
        page = pdf.new_page(width=page_width, height=page_height)
        y = 50  # Starting y position
        
        # Add title
        page.insert_text((50, y), "Repository Summary", fontsize=16)
        y += 40
        
        # Add repository info
        page.insert_text((50, y), f"Repository: {git_repo.full_name}", fontsize=12)
        y += 20
        page.insert_text((50, y), f"Target path: {target_path if target_path else 'Root'}", fontsize=12)
        y += 20
        page.insert_text((50, y), f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=12)
        y += 40
        
        # Add statistics
        total_processed = len(stats['processed']['text_files']) + len(stats['processed']['images'])
        total_skipped = sum(len(files) for files in stats['skipped'].values())
        
        page.insert_text((50, y), "Statistics:", fontsize=14)
        y += 30
        page.insert_text((70, y), f"Total files processed: {total_processed}", fontsize=12)
        y += 20
        page.insert_text((90, y), f"Text files: {len(stats['processed']['text_files'])}", fontsize=10)
        y += 15
        page.insert_text((90, y), f"Images: {len(stats['processed']['images'])}", fontsize=10)
        y += 25
        page.insert_text((70, y), f"Total files skipped: {total_skipped}", fontsize=12)
        y += 20
        
        # List skipped files by category
        for category, files in stats['skipped'].items():
            if files:  # Only show categories that have files
                page.insert_text((90, y), f"{category.replace('_', ' ').title()}: {len(files)}", fontsize=10)
                y += 15
                for file in sorted(files)[:5]:  # Show first 5 files of each category
                    page.insert_text((110, y), f"- {file}", fontsize=9)
                    y += 12
                if len(files) > 5:
                    page.insert_text((110, y), f"... and {len(files) - 5} more", fontsize=9, color=(0.5, 0.5, 0.5))
                    y += 20
        
        # Add a note about skipped files
        if total_skipped > 0:
            y = min(y + 20, page_height - 100)  # Ensure we don't write off the page
            note = ("Note: Binary files, model weights, and archives are skipped to keep the PDF focused "
                   "on readable content. Their paths are preserved for reference.")
            page.insert_textbox(fitz.Rect(50, y, page_width-50, y+50), note, fontsize=10, color=(0.5, 0.5, 0.5))
    
    def process_path(path):
        try:
            if target_path and not path.startswith(target_path):
                return

            print(f"Processing path: {path}")
            contents = git_repo.get_contents(path)
            
            # If contents is a list, it's a directory
            if isinstance(contents, list):
                print(f"Directory found: {path}")
                for content in contents:
                    if content.type == "dir":
                        print(f"Entering directory: {content.path}")
                        process_path(content.path)
                    else:
                        print(f"Processing file: {content.path}")
                        # Get content before adding page to determine if it should be skipped
                        content_result = get_file_content(git_repo, content.path)
                        
                        # Track file in statistics
                        if content_result is None:
                            category = categorize_skipped_file(content.path)
                            stats['skipped'][category].append(content.path)
                        elif content.path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff', '.ico')):
                            stats['processed']['images'].append(content.path)
                        else:
                            stats['processed']['text_files'].append(content.path)
                        
                        add_file_to_pdf(git_repo, pdf, content.path, page_width, page_height)
            else:
                # Single file
                print(f"Processing single file: {path}")
                content_result = get_file_content(git_repo, path)
                
                # Track file in statistics
                if content_result is None:
                    category = categorize_skipped_file(path)
                    stats['skipped'][category].append(path)
                elif path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff', '.ico')):
                    stats['processed']['images'].append(path)
                else:
                    stats['processed']['text_files'].append(path)
                
                add_file_to_pdf(git_repo, pdf, path, page_width, page_height)
                
        except Exception as e:
            print(f"Error processing path {path}: {str(e)}")
            stats['skipped']['other'].append(f"{path} (Error: {str(e)})")

    try:
        # Start processing from root or target path
        initial_path = target_path if target_path else ""
        print(f"Starting PDF generation from path: {initial_path}")
        process_path(initial_path)
        
        # Add summary page at the beginning
        add_summary_page()
        
        # Move summary page to the front
        pdf.move_page(pdf.page_count-1, 0)
        
        if len(stats['processed']['text_files']) + len(stats['processed']['images']) == 0:
            print("Warning: No files were processed!")
            # Add a warning page
            page = pdf.new_page(width=page_width, height=page_height)
            page.insert_textbox(
                fitz.Rect(50, 50, page_width-50, page_height-50),
                "No files were processed. This could be due to:\n\n" +
                "1. Invalid GitHub token\n" +
                "2. Repository access issues\n" +
                "3. Empty repository or invalid target path",
                fontsize=12,
                align=1
            )
        
        return pdf
    except Exception as e:
        print(f"Error in PDF generation: {str(e)}")
        # Create an error page
        pdf = fitz.open()
        page = pdf.new_page(width=page_width, height=page_height)
        page.insert_textbox(
            fitz.Rect(50, 50, page_width-50, page_height-50),
            f"Error generating PDF:\n\n{str(e)}",
            fontsize=12,
            align=1
        )
        return pdf

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Convert a GitHub repository to PDF, including text files and images.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python repo2pdf.py --repo ggerganov/llama.cpp
  python repo2pdf.py --repo https://github.com/ggerganov/llama.cpp.git
  python repo2pdf.py --repo ggerganov/llama.cpp --path src/examples
  python repo2pdf.py --repo ggerganov/llama.cpp --token YOUR_GITHUB_TOKEN
        '''
    )
    
    parser.add_argument('--repo', required=True,
                      help='Repository address (e.g., "owner/repo" or full GitHub URL)')
    parser.add_argument('--path', 
                      help='Target path within repository (optional)')
    parser.add_argument('--token',
                      help='GitHub access token (optional, but recommended for private repos)')
    parser.add_argument('--output',
                      help='Output PDF filename (optional, will be auto-generated if not provided)')
    
    args = parser.parse_args()
    
    # Process repository address
    if 'github.com' in args.repo:
        # Extract owner/repo from URL
        parsed_url = urlparse(args.repo)
        path_parts = parsed_url.path.strip('/').split('/')
        if path_parts[-1].endswith('.git'):
            path_parts[-1] = path_parts[-1][:-4]
        repo_name = '/'.join(path_parts[-2:])
    else:
        # Assume format is owner/repo
        repo_name = args.repo
    
    # Initialize GitHub client
    try:
        g = Github(args.token) if args.token else Github()
        user = g.get_user()
        print(f"Successfully authenticated as: {user.login}")
    except Exception as e:
        print(f"Failed to authenticate with GitHub: {str(e)}")
        sys.exit(1)

    # Get the repository
    try:
        print(f"Attempting to access repository: {repo_name}")
        repo = g.get_repo(repo_name)
        print(f"Successfully accessed repository: {repo.full_name}")
    except Exception as e:
        print(f"Failed to access repository: {str(e)}")
        sys.exit(1)

    # Generate the PDF
    print("Starting PDF generation...")
    pdf = repo_to_pdf(repo, args.path)

    # Generate output filename if not provided
    if not args.output:
        args.output = f'repository_structure_{repo_name.replace("/", "_")}{"_" + args.path.replace("/", "_") if args.path else ""}.pdf'

    # Save the PDF
    pdf.save(args.output)
    print(f"Repository structure with file content has been saved to {args.output}")

if __name__ == '__main__':
    main()

"""
Basic Usage Examples:
--------------------
# Basic usage with owner/repo format:
python repo2pdf.py --repo ggerganov/llama.cpp

# Using full GitHub URL:
python repo2pdf.py --repo https://github.com/ggerganov/llama.cpp.git

# Specifying a target path within the repo:
python repo2pdf.py --repo ggerganov/llama.cpp --path src/examples

# Using a GitHub token (recommended for private repos):
python repo2pdf.py --repo ggerganov/llama.cpp --token YOUR_GITHUB_TOKEN

# Custom output filename:
python repo2pdf.py --repo ggerganov/llama.cpp --output custom_name.pdf

For more options:
python repo2pdf.py --help
"""