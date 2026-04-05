#!/usr/bin/env python3
"""
Debug utility for LaTeX compilation issues
"""
import sys
import os
import subprocess
import tempfile
from pathlib import Path

def check_pdflatex():
    """Check if pdflatex is installed and accessible"""
    print("=" * 60)
    print("Checking pdflatex installation...")
    print("=" * 60)
    
    try:
        result = subprocess.run(['pdflatex', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"[OK] pdflatex found: {version_line}")
            return True
        else:
            print("[ERROR] pdflatex command failed")
            return False
    except FileNotFoundError:
        print("[ERROR] pdflatex not found in PATH")
        print("\nTo install pdflatex:")
        print("  macOS:   brew install --cask mactex-no-gui")
        print("  Ubuntu:  sudo apt-get install texlive-latex-base")
        print("  Windows: Install MiKTeX from https://miktex.org/")
        return False

def test_simple_latex():
    """Test compilation of a simple LaTeX document"""
    print("\n" + "=" * 60)
    print("Testing simple LaTeX compilation...")
    print("=" * 60)
    
    simple_doc = r"""
\documentclass{article}
\begin{document}
Hello, World!
\end{document}
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False) as f:
        f.write(simple_doc)
        tex_file = f.name
    
    try:
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', tex_file],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.path.dirname(tex_file)
        )
        
        pdf_file = tex_file.replace('.tex', '.pdf')
        
        if os.path.exists(pdf_file):
            print("[OK] Simple LaTeX document compiled successfully")
            return True
        else:
            print("[ERROR] PDF was not created")
            print("\nLast 10 lines of output:")
            print(result.stdout.split('\n')[-10:])
            return False
    except subprocess.TimeoutExpired:
        print("[ERROR] Compilation timed out")
        return False
    except Exception as e:
        print(f"[ERROR] Error during compilation: {e}")
        return False
    finally:
        # Clean up
        for ext in ['.tex', '.log', '.aux', '.pdf']:
            try:
                os.unlink(tex_file.replace('.tex', ext))
            except:
                pass

def analyze_tex_file(tex_file_path):
    """Analyze a specific .tex file"""
    print("\n" + "=" * 60)
    print(f"Analyzing: {tex_file_path}")
    print("=" * 60)
    
    if not os.path.exists(tex_file_path):
        print(f"[ERROR] File not found: {tex_file_path}")
        return False
    
    with open(tex_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"\nFile size: {len(content)} characters")
    print(f"Number of lines: {len(content.split(chr(10)))}")
    
    # Check for required elements
    required = [
        (r'\documentclass', 'Document class'),
        (r'\begin{document}', 'Begin document'),
        (r'\end{document}', 'End document'),
    ]
    
    print("\nRequired LaTeX elements:")
    all_present = True
    for pattern, name in required:
        if pattern in content:
            print(f"  [OK] {name}")
        else:
            print(f"  [ERROR] {name} MISSING")
            all_present = False
    
    if not all_present:
        print("\n[ERROR] Document is missing required LaTeX elements")
        return False
    
    # Try to compile
    print("\nAttempting compilation...")
    
    try:
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', tex_file_path],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=os.path.dirname(tex_file_path) or '.'
        )
        
        pdf_file = tex_file_path.replace('.tex', '.pdf')
        
        if os.path.exists(pdf_file):
            size = os.path.getsize(pdf_file)
            print(f"[OK] PDF created successfully ({size} bytes)")
            print(f"   Location: {pdf_file}")
            return True
        else:
            print("[ERROR] PDF was not created")
            
            # Extract errors from log
            output = result.stdout + result.stderr
            error_lines = []
            for line in output.split('\n'):
                if any(keyword in line.lower() for keyword in ['error', 'undefined', 'missing', '!']):
                    error_lines.append(line)
            
            if error_lines:
                print("\nPossible errors found:")
                for line in error_lines[:20]:  # Show first 20 error lines
                    print(f"  {line}")
            
            # Check log file
            log_file = tex_file_path.replace('.tex', '.log')
            if os.path.exists(log_file):
                print(f"\nFull log available at: {log_file}")
            
            return False
    except subprocess.TimeoutExpired:
        print("[ERROR] Compilation timed out (>60s)")
        return False
    except Exception as e:
        print(f"[ERROR] Error during compilation: {e}")
        return False

def main():
    print("LaTeX Compilation Debugger")
    print("=" * 60)
    
    # Check pdflatex
    if not check_pdflatex():
        print("\n[WARN] Cannot continue without pdflatex")
        sys.exit(1)
    
    # Test simple compilation
    if not test_simple_latex():
        print("\n[WARN] Simple LaTeX compilation failed")
        print("This indicates a problem with your LaTeX installation")
        sys.exit(1)
    
    # If file path provided, analyze it
    if len(sys.argv) > 1:
        tex_file = sys.argv[1]
        analyze_tex_file(tex_file)
    else:
        # Look for recent logs
        print("\n" + "=" * 60)
        print("Looking for recent LaTeX files in logs/...")
        print("=" * 60)
        
        logs_dir = Path('logs')
        if logs_dir.exists():
            tex_files = sorted(logs_dir.glob('**/*.tex'), key=os.path.getmtime, reverse=True)
            
            if tex_files:
                print(f"\nFound {len(tex_files)} .tex files")
                print("\nMost recent files:")
                for i, tex_file in enumerate(tex_files[:5], 1):
                    age_sec = os.path.getmtime(tex_file)
                    import time
                    age_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(age_sec))
                    print(f"  {i}. {tex_file.name} ({age_str})")
                
                print("\nTo analyze a specific file, run:")
                print(f"  python debug_latex.py {tex_files[0]}")
            else:
                print("No .tex files found in logs/")
        else:
            print("logs/ directory not found")
    
    print("\n" + "=" * 60)
    print("Debugging complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
