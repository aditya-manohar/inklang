from flask import Flask, render_template, request, jsonify, send_from_directory
import subprocess
import tempfile
import os
import sys
import pathlib
import uuid
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

APP_ROOT = pathlib.Path(__file__).parent.resolve()
INKLANG_PY = str(APP_ROOT / "inklang.py")
UPLOAD_FOLDER = APP_ROOT / "uploads"  # Changed to lowercase 'uploads' for consistency with ml.py
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route("/", methods=["GET"])
def index():
    logging.debug("Serving index page")
    examples = []
    ex_dir = APP_ROOT / "examples"
    if ex_dir.exists():
        for p in sorted(ex_dir.iterdir()):
            if p.suffix == ".inkl":
                examples.append(p.name)
    return render_template("index.html", examples=examples)

@app.route('/docs')
def docs():
    return render_template('doc.html')

@app.route("/run", methods=["POST"])
def run_code():
    logging.debug("Received /run request")
    data = request.json or {}
    code = data.get("code", "")
    if not code:
        logging.error("No code provided in request")
        return jsonify({"ok": False, "error": "No code provided"}), 400

    tmp_dir = str(UPLOAD_FOLDER)
    fname = f"ink_{uuid.uuid4().hex}.inkl"
    fpath = os.path.join(tmp_dir, fname)
    
    try:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(code)
        logging.debug(f"Wrote code to {fpath}")
    except Exception as e:
        logging.error(f"Failed to write temp file: {str(e)}")
        return jsonify({"ok": False, "error": f"Failed to write temp file: {str(e)}"}), 500
    
    cmd = [sys.executable, INKLANG_PY, fpath]
    pkl_path = None
    try:
        logging.debug(f"Executing command: {cmd}")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120, text=True)
        out = proc.stdout
        err = proc.stderr
        combined = ""
        if out:
            combined += out
        if err:
            combined += ("\n[stderr]\n" + err)
        if proc.returncode != 0:
            logging.error(f"Subprocess failed with code {proc.returncode}: {combined}")
            return jsonify({"ok": False, "error": f"Execution failed: {combined}"}), 500
        logging.debug(f"Subprocess output: {combined}")
        
        for line in code.splitlines():
            if line.strip().startswith("savemodel"):
                parts = line.split()
                if len(parts) >= 4 and parts[2] == "to":
                    pkl_path = parts[3].strip('"')
                    full_pkl_path = os.path.join(UPLOAD_FOLDER, pkl_path)
                    if not os.path.exists(full_pkl_path):
                        pkl_path = None
                        logging.warning(f"PKL file not found: {full_pkl_path}")
                    else:
                        logging.debug(f"PKL file found: {full_pkl_path}")
        
        try:
            os.remove(fpath)
            logging.debug(f"Removed temp file: {fpath}")
        except Exception as e:
            logging.warning(f"Failed to remove temp file: {str(e)}")
        return jsonify({"ok": True, "output": combined, "pkl_path": pkl_path})
    except subprocess.TimeoutExpired:
        logging.error("Subprocess timed out")
        try:
            os.remove(fpath)
        except Exception:
            pass
        return jsonify({"ok": False, "error": "Execution timed out (limit 120s)."}), 500
    except Exception as e:
        logging.error(f"Subprocess error: {str(e)}")
        try:
            os.remove(fpath)
        except Exception:
            pass
        return jsonify({"ok": False, "error": f"Subprocess error: {str(e)}\n{combined}"}), 500
    finally:
        # Clear stale .pkl files older than 1 hour
        for f in os.listdir(UPLOAD_FOLDER):
            if f.endswith('.pkl'):
                f_path = os.path.join(UPLOAD_FOLDER, f)
                if os.path.getmtime(f_path) < time.time() - 3600:
                    try:
                        os.remove(f_path)
                        logging.debug(f"Removed stale PKL file: {f_path}")
                    except Exception as e:
                        logging.warning(f"Failed to remove stale PKL file {f_path}: {str(e)}")

@app.route("/examples/<name>")
def get_example(name):
    logging.debug(f"Fetching example: {name}")
    p = APP_ROOT / "examples" / name
    if p.exists():
        return p.read_text(encoding="utf-8")
    logging.error(f"Example not found: {name}")
    return jsonify({"error": "Example not found"}), 404

@app.route("/download/<path:filename>")
def download_file(filename):
    logging.debug(f"Downloading file: {filename}")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)
    logging.error(f"File not found: {file_path}")
    return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    import time
    logging.debug("Starting Flask server")
    app.run(debug=True, host="0.0.0.0", port=5000)