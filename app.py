from flask import Flask, render_template, request, jsonify
import subprocess
import os
import signal

app = Flask(__name__)

# Variable to store the YOLOv5 process
yolov5_process = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start-model', methods=['POST'])
def start_model():
    global yolov5_process
    try:
        # Get command-line arguments from the request
        args = request.json.get('args', '')
        # Split the arguments string into a list
        args_list = args.split()
        # Command to start the YOLOv5 model with arguments
        command = ['python', '.\detect.py'] + args_list
        yolov5_process = subprocess.Popen(command)
        return jsonify({'status': 'success', 'message': 'YOLOv7 model started successfully!'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop-model', methods=['POST'])
def stop_model():
    global yolov5_process
    try:
        if yolov5_process is not None:
            os.kill(yolov5_process.pid, signal.SIGTERM)
            yolov5_process = None
            return jsonify({'status': 'success', 'message': 'YOLOv7 model stopped successfully!'})
        else:
            return jsonify({'status': 'error', 'message': 'YOLOv7 model is not running!'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/run-ml-model', methods=['POST'])
def run_ml_model():
    try:
        print("Ml model Started\n")
        # Command to run the ML model script and capture its output
        result = subprocess.run(['python', '.\mlproject\mlgproject.py'], capture_output=True, text=True)
        return jsonify({'status': 'success', 'output': result.stdout})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
