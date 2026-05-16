from flask import Flask, request, jsonify
import os
import pandas as pd
from typing import Dict, List, Any

app = Flask(__name__)


class SourceCodeResponse:
    def __init__(self, data: List[Dict], total_files: int, message: str):
        self.data = data
        self.total_files = total_files
        self.message = message

    def to_dict(self):
        return {
            'data': self.data,
            'total_files': self.total_files,
            'message': self.message
        }


@app.route('/api/source/get-source-code', methods=['POST'])
def get_source_code_endpoint():
    """
    Flask endpoint that wraps your get_source_code function
    """
    try:
        # Get JSON data from request
        request_data = request.get_json()

        if not request_data or 'directory' not in request_data:
            return jsonify({
                'error': 'Missing required field: directory'
            }), 400

        directory = request_data['directory']

        # Validate directory exists
        if not os.path.exists(directory):
            return jsonify({
                'error': f"Directory '{directory}' not found"
            }), 404

        # Call your existing function
        df = get_source_code(directory)

        # Convert DataFrame to list of dictionaries for JSON response
        data = df.to_dict('records')

        response = SourceCodeResponse(
            data=data,
            total_files=len(data),
            message=f"Successfully processed {len(data)} files"
        )

        return jsonify(response.to_dict()), 200

    except Exception as e:
        return jsonify({
            'error': f"Error processing directory: {str(e)}"
        }), 500


def get_source_code(directory):
    """
    Fixed version that properly includes database and services folders
    """
    print(f"🔍 Processing directory: {os.path.abspath(directory)}")

    df = pd.DataFrame(columns=['filepath', 'text'])
    processed_count = 0

    for root, dirs, files in os.walk(directory):
        # Fix the skipping logic - only skip if the path STARTS with these patterns
        # and make sure we're not accidentally skipping our project folders
        relative_root = os.path.relpath(root, directory)

        # Skip only specific dependency directories, not our project folders
        skip_patterns = ['.venv', 'venv', 'node_modules', '__pycache__', '.git', 'analyzer', 'analyzer_app']
        should_skip = any(
            relative_root.startswith(pattern) or f'/{pattern}/' in relative_root
            for pattern in skip_patterns
        )

        if should_skip:
            continue

        print(f"📁 Processing directory: {relative_root}")

        for file in files:
            # Include more file types and make sure we get Python files
            if file.endswith(('.py', '.html', '.css', '.js', '.jsx', '.ts', '.tsx', '.txt')) or file == 'Procfile':
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory)

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        raw_text = f.read()

                    print(f'✅ Processing: {relative_path} - Size: {len(raw_text)} chars')

                    # Use pd.concat instead of deprecated _append
                    new_row = pd.DataFrame([{'filepath': relative_path, 'text': raw_text}])
                    df = pd.concat([df, new_row], ignore_index=True)
                    processed_count += 1

                except Exception as file_error:
                    print(f"❌ Could not read file {relative_path}: {str(file_error)}")
                    continue
            else:
                print(f"⏭️  Skipping file: {file} (not a target file type)")

    print(f"\n✅ Processed {processed_count} files total")
    return df


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8010)