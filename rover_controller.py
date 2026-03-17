#!/usr/bin/env python3
"""
Rover Controller with External Text Detection and HTTP Image Server
Sends frames to external server to detect "TRASH" and approaches based on coordinates
Includes manual control mode with adjustable wheel speed and servo enable/disable
Live video streaming without page reload
Non-blocking detection using threading and callbacks
"""

import RPi.GPIO as GPIO
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from gpiozero import Servo
import signal
import sys
import threading
import socket
import warnings
import subprocess
import os
from threading import Lock, Thread
import http.server
import socketserver
import io
from http import HTTPStatus
import requests
import json
from collections import deque
from enum import Enum
import urllib.parse
import base64
from concurrent.futures import ThreadPoolExecutor
import queue

# Suppress warnings
warnings.filterwarnings("ignore")

# Try different import methods for PWM fallback
try:
    from gpiozero.pins.pwm import PWMSoftwareFallback
    PINS_FACTORY = PWMSoftwareFallback
    print("Using PWMSoftwareFallback from gpiozero.pins.pwm")
except ImportError:
    try:
        from gpiozero.pins import SoftwarePWM
        PINS_FACTORY = SoftwarePWM
        print("Using SoftwarePWM from gpiozero.pins")
    except ImportError:
        PINS_FACTORY = None
        print("Using default pin factory (may have jitter)")

# --- Configuration ---
# Use BCM pin numbers
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Pin Definitions
# Motor Pins
MOTOR_A_EN = 18
MOTOR_A_IN1 = 22
MOTOR_A_IN2 = 23
MOTOR_B_EN = 13
MOTOR_B_IN3 = 24
MOTOR_B_IN4 = 25

# Ultrasonic Pins
ULTRA_TRIG = 17
ULTRA_ECHO = 27

# Trash find LED
MAIN_LED = 16

# IR Sensor Pins
IR_LEFT = 5
IR_RIGHT = 6
IR_BACK = 12

# Servo Pins
SERVO_BASE = 26
SERVO_SHOULDER = 19
SERVO_ELBOW = 20
SERVO_LID = 21

# External Detection Server Configuration
DETECTION_SERVER_URLS = [
    "http://10.119.139.123:3000/ocr/uppercase-center",  # Primary server
    "http://10.30.60.238:3000/ocr/uppercase-center"  # Fallback server
]
DETECTION_TIMEOUT = 2.0  # Timeout for detection requests
TARGET_WORD = "TRASH"  # Word to search for
DETECTION_COOLDOWN = 2  # Minimum time between detection requests (seconds)
MAX_DETECTION_QUEUE_SIZE = 2  # Maximum number of pending detection requests

# HTTP Image Server Configuration
HTTP_PORT = 9090

# Navigation Constants
SAFE_DISTANCE = 40  # cm
CAUTION_DISTANCE = 30  # cm
DANGER_DISTANCE = 25  # cm
TURN_SPEED = 30
BASE_SPEED = 35
SLOW_SPEED = 20
APPROACH_SPEED = 25  # Speed when approaching target
MIN_CONFIDENCE = 70.0  # Minimum confidence to consider detection valid

# Manual Control Speed (adjustable via web interface)
MANUAL_SPEED = 30  # Default manual control speed

# Servo Control Flag (adjustable via web interface)
SERVOS_ENABLED = True  # Set to False to disable servo control

# --- Operation Mode Enum ---
class OperationMode(Enum):
    AUTO_SEARCH = "AUTO_SEARCH"
    MANUAL_CONTROL = "MANUAL_CONTROL"

# --- State Enum ---
class RoverState(Enum):
    SEARCHING = "SEARCHING"
    APPROACHING = "APPROACHING"
    PICKUP = "PICKUP"
    AVOIDING = "AVOIDING"
    ERROR = "ERROR"
    MANUAL = "MANUAL"

# --- Detection Data Class ---
class DetectionData:
    def __init__(self):
        self.detected = False
        self.word = None
        self.center_x = None
        self.center_y = None
        self.bbox = None
        self.confidence = 0
        self.timestamp = 0
        self.frame_center = 320  # Assuming 640x480 frame
        
    def update_from_response(self, response_json):
        """Update detection data from server response"""
        try:
            self.detected = response_json.get('detected', False) and response_json.get('locationFound', False)
            if self.detected:
                self.word = response_json.get('word', '')
                center = response_json.get('center', {})
                self.center_x = center.get('x')
                self.center_y = center.get('y')
                self.bbox = response_json.get('bbox', {})
                self.confidence = response_json.get('confidence', 0)
                self.timestamp = time.time()
                return True
        except Exception as e:
            print(f"Error parsing detection response: {e}")
        return False
    
    def is_valid(self, max_age=1.0):
        """Check if detection is still valid (not too old)"""
        return (self.detected and 
                self.word == TARGET_WORD and 
                self.confidence >= MIN_CONFIDENCE and
                time.time() - self.timestamp < max_age)

# --- Non-blocking Detection Manager ---
class DetectionManager:
    """Manages non-blocking detection requests using threading"""
    
    def __init__(self, server_urls, timeout=DETECTION_TIMEOUT):
        if isinstance(server_urls, str):
            self.server_urls = [server_urls]
        else:
            self.server_urls = [url for url in server_urls if url]

        if not self.server_urls:
            raise ValueError("At least one detection server URL is required")

        self.active_server_index = 0
        self.server_lock = Lock()
        self.timeout = timeout
        self.session = requests.Session()
        self.detection_queue = queue.Queue(maxsize=MAX_DETECTION_QUEUE_SIZE)
        self.result_queue = queue.Queue()
        self.latest_detection = DetectionData()
        self.lock = Lock()
        self.running = True
        self.last_request_time = 0
        self.cooldown = DETECTION_COOLDOWN
        
        # Start worker threads
        self.worker_threads = []
        for i in range(2):  # 2 worker threads
            thread = Thread(target=self._worker_loop, daemon=True)
            thread.start()
            self.worker_threads.append(thread)
        
        # Start result processor
        self.result_thread = Thread(target=self._result_loop, daemon=True)
        self.result_thread.start()
        
        print(f"🔍 Detection manager started with {len(self.worker_threads)} workers")
        print(f"🌐 Detection servers: {', '.join(self.server_urls)}")
    
    def _worker_loop(self):
        """Worker thread that processes detection requests"""
        while self.running:
            try:
                # Get frame from queue with timeout
                frame_data = self.detection_queue.get(timeout=0.1)
                if frame_data is None:
                    continue
                
                frame, timestamp = frame_data
                
                # Send to server
                result = self._detect_sync(frame)
                
                # Put result in result queue
                if result:
                    self.result_queue.put((result, timestamp))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")
    
    def _result_loop(self):
        """Process detection results"""
        while self.running:
            try:
                result, timestamp = self.result_queue.get(timeout=0.1)
                
                # Create detection object
                detection = DetectionData()
                if detection.update_from_response(result):
                    with self.lock:
                        # Only update if this result is newer than current
                        if detection.timestamp > self.latest_detection.timestamp:
                            self.latest_detection = detection
                            
                            # Add to global detection history
                            if detection.detected and detection.word == TARGET_WORD:
                                detection_history.append(detection)
                                print(f"✅ Detection received: {detection.word} ({detection.confidence:.1f}%)")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Result processor error: {e}")
    
    def _detect_sync(self, frame):
        """Synchronous detection request"""
        try:
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                return None

            files = {'image': ('frame.jpg', jpeg.tobytes(), 'image/jpeg')}

            with self.server_lock:
                start_index = self.active_server_index

            server_count = len(self.server_urls)
            for offset in range(server_count):
                index = (start_index + offset) % server_count
                server_url = self.server_urls[index]

                try:
                    response = self.session.post(
                        server_url,
                        files=files,
                        timeout=self.timeout
                    )

                    if response.status_code == 200:
                        with self.server_lock:
                            previous_index = self.active_server_index
                            self.active_server_index = index

                        if index != previous_index:
                            print(f"🔁 Switched detection server to: {server_url}")

                        return response.json()

                    print(f"Server error from {server_url}: {response.status_code}")

                except requests.exceptions.Timeout:
                    print(f"Detection request timeout: {server_url}")
                except requests.exceptions.ConnectionError:
                    print(f"Cannot connect to detection server: {server_url}")
                except Exception as e:
                    print(f"Detection error on {server_url}: {e}")

            return None

        except Exception as e:
            print(f"Detection error: {e}")
            return None
    
    def request_detection(self, frame):
        """Request detection (non-blocking)"""
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_request_time < self.cooldown:
            return False
        
        # Check if queue is full
        if self.detection_queue.full():
            return False
        
        try:
            # Add to queue
            self.detection_queue.put_nowait((frame.copy(), current_time))
            self.last_request_time = current_time
            return True
        except queue.Full:
            return False
    
    def get_latest_detection(self):
        """Get the latest detection result"""
        with self.lock:
            return self.latest_detection
    
    def stop(self):
        """Stop the detection manager"""
        self.running = False
        # Clear queues
        while not self.detection_queue.empty():
            try:
                self.detection_queue.get_nowait()
            except queue.Empty:
                break
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

# --- HTTP Image Server Class ---
class ImageServerHandler(http.server.SimpleHTTPRequestHandler):
    """Handler for HTTP requests that serves the latest frame with control interface"""
    
    def __init__(self, *args, **kwargs):
        self.frame_buffer = kwargs.pop('frame_buffer', None)
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        global SERVOS_ENABLED 
        global MANUAL_SPEED
        global operation_mode
        
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        query = urllib.parse.parse_qs(parsed_path.query)
        
        if path == '/':
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            
            html = self._generate_html()
            self.wfile.write(html.encode())
            
        elif path == '/stream':
            """Serve MJPEG stream"""
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            
            try:
                while True:
                    if self.frame_buffer and self.frame_buffer.get_frame() is not None:
                        frame = self.frame_buffer.get_frame()
                        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        if ret:
                            frame_data = jpeg.tobytes()
                            self.wfile.write(b'--frame\r\n')
                            self.wfile.write(b'Content-Type: image/jpeg\r\n')
                            self.wfile.write(b'Content-Length: ' + str(len(frame_data)).encode() + b'\r\n')
                            self.wfile.write(b'\r\n')
                            self.wfile.write(frame_data)
                            self.wfile.write(b'\r\n')
                    time.sleep(0.05)  # ~20 fps
            except (BrokenPipeError, ConnectionResetError):
                # Client disconnected
                pass
                
        elif path == '/image':
            self._serve_image()
            
        elif path == '/status':
            self._serve_status()
            
        elif path == '/toggle_servos':
            """Endpoint to toggle servos on/off"""
            global SERVOS_ENABLED
            SERVOS_ENABLED = not SERVOS_ENABLED
            self._send_json_response({'servos_enabled': SERVOS_ENABLED})
            
        elif path == '/set_speed':
            """Endpoint to set manual control speed"""
            try:
                speed = int(query.get('speed', [30])[0])
                MANUAL_SPEED = max(0, min(100, speed))
                self._send_json_response({'speed': MANUAL_SPEED})
            except:
                self._send_json_response({'error': 'Invalid speed'}, 400)
                
        elif path == '/set_mode':
            """Endpoint to set operation mode (auto/manual)"""
            try:
                mode = query.get('mode', ['AUTO'])[0].upper()
                global operation_mode
                if mode == 'AUTO':
                    operation_mode = OperationMode.AUTO_SEARCH
                    global_state['state'] = RoverState.SEARCHING.value
                elif mode == 'MANUAL':
                    operation_mode = OperationMode.MANUAL_CONTROL
                    global_state['state'] = RoverState.MANUAL.value
                    stop_motors()  # Stop motors when switching to manual
                self._send_json_response({'mode': operation_mode.value})
            except:
                self._send_json_response({'error': 'Invalid mode'}, 400)
                
        elif path == '/manual_control':
            """Endpoint for manual control commands"""
            if operation_mode != OperationMode.MANUAL_CONTROL:
                self._send_json_response({'error': 'Not in manual mode'}, 400)
                return
                
            try:
                cmd = query.get('cmd', ['stop'])[0].lower()
                self._handle_manual_command(cmd)
                self._send_json_response({'command': cmd, 'speed': MANUAL_SPEED})
            except Exception as e:
                self._send_json_response({'error': str(e)}, 400)
                
        elif path == '/get_config':
            """Get current configuration"""
            self._send_json_response({
                'servos_enabled': SERVOS_ENABLED,
                'manual_speed': MANUAL_SPEED,
                'mode': operation_mode.value,
                'target_word': TARGET_WORD
            })
                
        else:
            self.send_error(HTTPStatus.NOT_FOUND)
    
    def do_POST(self):
        """Handle POST requests for form submissions"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        
        if self.path == '/manual_control':
            # Handle JSON POST for manual control
            try:
                data = json.loads(post_data)
                cmd = data.get('command', 'stop')
                if operation_mode == OperationMode.MANUAL_CONTROL:
                    self._handle_manual_command(cmd)
                    self._send_json_response({'command': cmd, 'speed': MANUAL_SPEED})
                else:
                    self._send_json_response({'error': 'Not in manual mode'}, 400)
            except:
                self._send_json_response({'error': 'Invalid request'}, 400)
        
        elif self.path == '/set_config':
            # Handle configuration updates
            try:
                data = json.loads(post_data)
                
                if 'servos_enabled' in data:
                    SERVOS_ENABLED = bool(data['servos_enabled'])
                if 'manual_speed' in data:
                    MANUAL_SPEED = max(0, min(100, int(data['manual_speed'])))
                if 'mode' in data:
                    mode = data['mode'].upper()
                    if mode == 'AUTO':
                        operation_mode = OperationMode.AUTO_SEARCH
                        global_state['state'] = RoverState.SEARCHING.value
                    elif mode == 'MANUAL':
                        operation_mode = OperationMode.MANUAL_CONTROL
                        global_state['state'] = RoverState.MANUAL.value
                        stop_motors()
                
                self._send_json_response({
                    'servos_enabled': SERVOS_ENABLED,
                    'manual_speed': MANUAL_SPEED,
                    'mode': operation_mode.value
                })
            except Exception as e:
                self._send_json_response({'error': str(e)}, 400)
    
    def _generate_html(self):
        """Generate the HTML control interface with live streaming"""
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Rover Control Center - Live Stream</title>
    <meta charset="UTF-8">
    <style>
        body {{ 
            background: #1a1a1a; 
            color: #fff; 
            font-family: 'Segoe UI', Arial; 
            margin: 0;
            padding: 20px;
        }}
        .container {{ 
            max-width: 1400px; 
            margin: 0 auto;
        }}
        h1, h2 {{ 
            color: #00ff00;
            text-align: center;
            margin-top: 0;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
        }}
        .video-section {{
            background: #333;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .video-container {{
            position: relative;
            width: 100%;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
        }}
        #videoFeed {{
            width: 100%;
            height: auto;
            display: block;
            image-rendering: crisp-edges;
        }}
        .video-overlay {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.7);
            color: #00ff00;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 14px;
            pointer-events: none;
        }}
        .control-section {{
            background: #333;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .info-panel {{
            background: #444;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }}
        .status {{
            color: #00ff00;
            font-weight: bold;
        }}
        .target {{ color: #ffff00; }}
        .servo-enabled {{ color: #00ff00; }}
        .servo-disabled {{ color: #ff0000; }}
        table {{ 
            width: 100%;
            border-collapse: collapse;
        }}
        td {{ 
            padding: 8px;
            border-bottom: 1px solid #555;
        }}
        .control-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 15px;
        }}
        .control-btn {{
            background: #4CAF50;
            color: white;
            border: none;
            padding: 15px;
            font-size: 16px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            font-weight: bold;
        }}
        .control-btn:hover {{
            background: #45a049;
            transform: scale(1.05);
        }}
        .control-btn:active {{
            transform: scale(0.95);
        }}
        .control-btn.stop {{
            background: #f44336;
        }}
        .control-btn.stop:hover {{
            background: #da190b;
        }}
        .control-btn:disabled {{
            background: #666;
            cursor: not-allowed;
            transform: none;
        }}
        .slider-container {{
            margin: 15px 0;
            padding: 10px;
            background: #444;
            border-radius: 8px;
        }}
        .slider {{
            width: 100%;
            margin: 10px 0;
            -webkit-appearance: none;
            height: 10px;
            border-radius: 5px;
            background: #555;
            outline: none;
        }}
        .slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #4CAF50;
            cursor: pointer;
        }}
        .mode-selector {{
            display: flex;
            gap: 20px;
            justify-content: center;
            margin: 15px 0;
        }}
        .mode-btn {{
            padding: 12px 30px;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .mode-btn.auto {{
            background: #2196F3;
            color: white;
        }}
        .mode-btn.manual {{
            background: #FF9800;
            color: white;
        }}
        .mode-btn.active {{
            transform: scale(1.05);
            box-shadow: 0 0 15px currentColor;
        }}
        .mode-btn.auto.active {{
            background: #0b7dda;
        }}
        .mode-btn.manual.active {{
            background: #e68900;
        }}
        .speed-value {{
            font-size: 24px;
            font-weight: bold;
            color: #00ff00;
            text-align: center;
        }}
        .manual-controls {{
            display: {'block' if operation_mode == OperationMode.MANUAL_CONTROL else 'none'};
        }}
        .keyboard-hint {{
            background: #555;
            padding: 8px;
            border-radius: 5px;
            font-size: 12px;
            text-align: center;
            margin-top: 10px;
            color: #aaa;
        }}
        .keyboard-hint kbd {{
            background: #333;
            padding: 2px 6px;
            border-radius: 3px;
            color: #fff;
        }}
        .notification {{
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            border-radius: 5px;
            background: #4CAF50;
            color: white;
            opacity: 0;
            transition: opacity 0.3s;
            z-index: 1000;
        }}
        .notification.error {{
            background: #f44336;
        }}
        .notification.show {{
            opacity: 1;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-top: 10px;
        }}
        .stat-card {{
            background: #555;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }}
        .stat-label {{
            font-size: 12px;
            color: #aaa;
        }}
        .stat-value {{
            font-size: 20px;
            font-weight: bold;
            color: #00ff00;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Rover Control Center - Live Stream</h1>
        
        <div class="notification" id="notification"></div>
        
        <div class="grid">
            <!-- Video Feed Section -->
            <div class="video-section">
                <h2>📹 Live Camera Feed</h2>
                <div class="video-container">
                    <img id="videoFeed" src="/stream" alt="Live Video Stream">
                    <div class="video-overlay" id="videoOverlay">
                        Mode: <span id="overlayMode">AUTO</span> | 
                        State: <span id="overlayState">SEARCHING</span>
                    </div>
                </div>
                
                <!-- Quick Stats -->
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Detection</div>
                        <div class="stat-value" id="statDetection">❌</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Confidence</div>
                        <div class="stat-value" id="statConfidence">0%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Distance</div>
                        <div class="stat-value" id="statDistance">- cm</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Servos</div>
                        <div class="stat-value" id="statServos">ON</div>
                    </div>
                </div>
            </div>
            
            <!-- Control Section -->
            <div class="control-section">
                <h2>🎮 Control Panel</h2>
                
                <!-- Mode Selection -->
                <div class="info-panel">
                    <h3>🎯 Operation Mode</h3>
                    <div class="mode-selector">
                        <button class="mode-btn auto" id="modeAuto" onclick="setMode('AUTO')">Auto Search</button>
                        <button class="mode-btn manual" id="modeManual" onclick="setMode('MANUAL')">Manual Control</button>
                    </div>
                </div>
                
                <!-- Status Panel -->
                <div class="info-panel">
                    <h3>📊 System Status</h3>
                    <table>
                        <tr><td>Current Mode:</td><td class="status" id="statusMode">AUTO_SEARCH</td></tr>
                        <tr><td>State:</td><td class="status" id="statusState">SEARCHING</td></tr>
                        <tr><td>Target:</td><td class="target">{TARGET_WORD}</td></tr>
                        <tr><td>Detection:</td><td class="status" id="statusDetected">❌ Searching</td></tr>
                        <tr><td>Confidence:</td><td class="status" id="statusConfidence">0%</td></tr>
                        <tr><td>Distance:</td><td class="status" id="statusDistance">- cm</td></tr>
                        <tr><td>Servos:</td><td class="servo-enabled" id="statusServos">ENABLED</td></tr>
                        <tr><td>Manual Speed:</td><td class="status" id="statusSpeed">30%</td></tr>
                    </table>
                </div>
                
                <!-- Speed Control -->
                <div class="info-panel">
                    <h3>⚡ Speed Control</h3>
                    <div class="slider-container">
                        <div class="speed-value" id="speedDisplay">30%</div>
                        <input type="range" class="slider" id="speedSlider" min="0" max="100" value="30"
                               oninput="updateSpeedDisplay(this.value)" onchange="setSpeed(this.value)">
                        <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                            <span>0%</span>
                            <span>50%</span>
                            <span>100%</span>
                        </div>
                    </div>
                </div>
                
                <!-- Manual Controls -->
                <div class="info-panel manual-controls" id="manualControls">
                    <h3>🎮 Drive Controls</h3>
                    <div class="control-grid">
                        <button class="control-btn" onclick="sendCommand('forward')">⬆️ Forward</button>
                        <button class="control-btn" onclick="sendCommand('left')">⬅️ Left</button>
                        <button class="control-btn" onclick="sendCommand('stop')">⏹️ Stop</button>
                        <button class="control-btn" onclick="sendCommand('right')">➡️ Right</button>
                        <button class="control-btn" onclick="sendCommand('backward')">⬇️ Backward</button>
                        <button class="control-btn" onclick="sendCommand('spin_left')">🔄 Spin Left</button>
                        <button class="control-btn" onclick="sendCommand('spin_right')">🔄 Spin Right</button>
                    </div>
                    
                    <h3 style="margin-top: 20px;">🦾 Arm Controls</h3>
                    <div class="control-grid">
                        <button class="control-btn" onclick="sendCommand('pickup')">🤲 Pickup</button>
                        <button class="control-btn" onclick="sendCommand('rest')">🔄 Rest</button>
                        <button class="control-btn" onclick="sendCommand('open_lid')">📂 Open Lid</button>
                        <button class="control-btn" onclick="sendCommand('close_lid')">📁 Close Lid</button>
                    </div>
                    
                    <div class="keyboard-hint">
                        <kbd>↑</kbd> Forward | <kbd>↓</kbd> Backward | <kbd>←</kbd> Left | <kbd>→</kbd> Right | 
                        <kbd>Q</kbd> Spin Left | <kbd>E</kbd> Spin Right | <kbd>Space</kbd> Stop
                    </div>
                </div>
                
                <!-- Servo Toggle -->
                <div class="info-panel">
                    <h3>🔧 Servo Control</h3>
                    <button class="control-btn" id="servoToggleBtn" onclick="toggleServos()" style="width: 100%;">
                        {'🔴 Disable Servos' if SERVOS_ENABLED else '🟢 Enable Servos'}
                    </button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // State variables
        let currentMode = 'AUTO_SEARCH';
        let servosEnabled = {str(SERVOS_ENABLED).lower()};
        let manualSpeed = {MANUAL_SPEED};
        
        // Update video overlay and status every second
        function updateStatus() {{
            fetch('/status')
                .then(response => response.json())
                .then(data => {{
                    // Update overlay
                    document.getElementById('overlayMode').textContent = data.mode;
                    document.getElementById('overlayState').textContent = data.state;
                    
                    // Update status table
                    document.getElementById('statusMode').textContent = data.mode;
                    document.getElementById('statusState').textContent = data.state;
                    document.getElementById('statusDetected').innerHTML = data.detected ? '✅ FOUND' : '❌ Searching';
                    document.getElementById('statusConfidence').textContent = data.confidence.toFixed(1) + '%';
                    document.getElementById('statusDistance').textContent = data.distance + ' cm';
                    document.getElementById('statusServos').innerHTML = data.servos_enabled ? 'ENABLED' : 'DISABLED';
                    document.getElementById('statusServos').className = data.servos_enabled ? 'servo-enabled' : 'servo-disabled';
                    document.getElementById('statusSpeed').textContent = data.manual_speed + '%';
                    
                    // Update stats
                    document.getElementById('statDetection').innerHTML = data.detected ? '✅' : '❌';
                    document.getElementById('statConfidence').textContent = data.confidence.toFixed(1) + '%';
                    document.getElementById('statDistance').textContent = data.distance + ' cm';
                    document.getElementById('statServos').innerHTML = data.servos_enabled ? 'ON' : 'OFF';
                    
                    // Update mode buttons
                    currentMode = data.mode;
                    updateModeButtons();
                    
                    // Update servo toggle button
                    servosEnabled = data.servos_enabled;
                    updateServoButton();
                    
                    // Update speed display
                    manualSpeed = data.manual_speed;
                    document.getElementById('speedDisplay').textContent = manualSpeed + '%';
                    document.getElementById('speedSlider').value = manualSpeed;
                }})
                .catch(error => console.error('Error fetching status:', error));
        }}
        
        // Update mode button styles
        function updateModeButtons() {{
            const autoBtn = document.getElementById('modeAuto');
            const manualBtn = document.getElementById('modeManual');
            
            if (currentMode === 'AUTO_SEARCH') {{
                autoBtn.classList.add('active');
                manualBtn.classList.remove('active');
                document.getElementById('manualControls').style.display = 'none';
            }} else {{
                manualBtn.classList.add('active');
                autoBtn.classList.remove('active');
                document.getElementById('manualControls').style.display = 'block';
            }}
        }}
        
        // Update servo button text
        function updateServoButton() {{
            const btn = document.getElementById('servoToggleBtn');
            if (servosEnabled) {{
                btn.innerHTML = '🔴 Disable Servos';
                btn.style.background = '#f44336';
            }} else {{
                btn.innerHTML = '🟢 Enable Servos';
                btn.style.background = '#4CAF50';
            }}
        }}
        
        // Show notification
        function showNotification(message, isError = false) {{
            const notification = document.getElementById('notification');
            notification.textContent = message;
            notification.className = 'notification' + (isError ? ' error' : '') + ' show';
            setTimeout(() => {{
                notification.classList.remove('show');
            }}, 3000);
        }}
        
        // Send manual command
        function sendCommand(cmd) {{
            if (currentMode !== 'MANUAL_CONTROL') {{
                showNotification('Switch to Manual mode first!', true);
                return;
            }}
            
            fetch('/manual_control?cmd=' + cmd)
                .then(response => response.json())
                .then(data => {{
                    if(data.error) {{
                        showNotification('Error: ' + data.error, true);
                    }} else {{
                        showNotification('Command: ' + cmd);
                    }}
                }})
                .catch(error => {{
                    console.error('Error:', error);
                    showNotification('Failed to send command', true);
                }});
        }}
        
        // Set operation mode
        function setMode(mode) {{
            fetch('/set_mode?mode=' + mode)
                .then(response => response.json())
                .then(data => {{
                    showNotification('Mode switched to ' + data.mode);
                    updateStatus();
                }})
                .catch(error => {{
                    console.error('Error:', error);
                    showNotification('Failed to switch mode', true);
                }});
        }}
        
        // Set manual speed
        function setSpeed(speed) {{
            fetch('/set_speed?speed=' + speed)
                .then(response => response.json())
                .then(data => {{
                    manualSpeed = data.speed;
                    document.getElementById('speedDisplay').textContent = manualSpeed + '%';
                    showNotification('Speed set to ' + manualSpeed + '%');
                }})
                .catch(error => {{
                    console.error('Error:', error);
                    showNotification('Failed to set speed', true);
                }});
        }}
        
        // Update speed display while sliding
        function updateSpeedDisplay(speed) {{
            document.getElementById('speedDisplay').textContent = speed + '%';
        }}
        
        // Toggle servos
        function toggleServos() {{
            fetch('/toggle_servos')
                .then(response => response.json())
                .then(data => {{
                    servosEnabled = data.servos_enabled;
                    updateServoButton();
                    showNotification('Servos ' + (servosEnabled ? 'enabled' : 'disabled'));
                    updateStatus();
                }})
                .catch(error => {{
                    console.error('Error:', error);
                    showNotification('Failed to toggle servos', true);
                }});
        }}
        
        // Keyboard controls
        document.addEventListener('keydown', function(event) {{
            if (currentMode === 'MANUAL_CONTROL') {{
                switch(event.key) {{
                    case 'ArrowUp': sendCommand('forward'); event.preventDefault(); break;
                    case 'ArrowDown': sendCommand('backward'); event.preventDefault(); break;
                    case 'ArrowLeft': sendCommand('left'); event.preventDefault(); break;
                    case 'ArrowRight': sendCommand('right'); event.preventDefault(); break;
                    case ' ': sendCommand('stop'); event.preventDefault(); break;
                    case 'q': case 'Q': sendCommand('spin_left'); event.preventDefault(); break;
                    case 'e': case 'E': sendCommand('spin_right'); event.preventDefault(); break;
                }}
            }}
        }});
        
        // Initial status update and start periodic updates
        updateStatus();
        setInterval(updateStatus, 500);  // Update every 500ms
        
        // Handle video stream errors (reconnect)
        const videoFeed = document.getElementById('videoFeed');
        videoFeed.onerror = function() {{
            console.log('Video stream error, attempting to reconnect...');
            setTimeout(() => {{
                videoFeed.src = '/stream?' + new Date().getTime();  // Add timestamp to prevent caching
            }}, 1000);
        }};
    </script>
</body>
</html>"""
    
    def _serve_image(self):
        """Serve the latest frame as PNG"""
        if self.frame_buffer and self.frame_buffer.get_frame() is not None:
            frame = self.frame_buffer.get_frame()
            ret, png_data = cv2.imencode('.png', frame)
            if ret:
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Content-Length', str(len(png_data)))
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.end_headers()
                self.wfile.write(png_data.tobytes())
            else:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)
    
    def _serve_status(self):
        """Serve JSON status"""
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        
        import json
        # Get latest detection from detection manager
        latest_detection = detection_manager.get_latest_detection() if detection_manager else DetectionData()
        
        status = {
            'state': global_state.get('state', 'UNKNOWN'),
            'mode': operation_mode.value,
            'target': TARGET_WORD,
            'detected': latest_detection.detected and latest_detection.word == TARGET_WORD,
            'confidence': latest_detection.confidence if latest_detection.detected else 0,
            'distance': global_state.get('distance', -1),
            'servos_enabled': SERVOS_ENABLED,
            'manual_speed': MANUAL_SPEED,
            'center_x': latest_detection.center_x if latest_detection.detected else None,
            'center_y': latest_detection.center_y if latest_detection.detected else None,
            'uptime': time.time() - start_time
        }
        self.wfile.write(json.dumps(status).encode())
    
    def _send_json_response(self, data, status=200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _handle_manual_command(self, cmd):
        """Handle manual control commands"""
        
        if cmd == 'forward':
            set_motor_speed('A', MANUAL_SPEED)
            set_motor_speed('B', MANUAL_SPEED)
        elif cmd == 'backward':
            set_motor_speed('A', -MANUAL_SPEED)
            set_motor_speed('B', -MANUAL_SPEED)
        elif cmd == 'left':
            set_motor_speed('A', -MANUAL_SPEED//2)
            set_motor_speed('B', MANUAL_SPEED)
        elif cmd == 'right':
            set_motor_speed('A', MANUAL_SPEED)
            set_motor_speed('B', -MANUAL_SPEED//2)
        elif cmd == 'spin_left':
            set_motor_speed('A', -MANUAL_SPEED)
            set_motor_speed('B', MANUAL_SPEED)
        elif cmd == 'spin_right':
            set_motor_speed('A', MANUAL_SPEED)
            set_motor_speed('B', -MANUAL_SPEED)
        elif cmd == 'stop':
            stop_motors()
        elif cmd == 'pickup':
            if SERVOS_ENABLED:
                arm_pickup_position()
        elif cmd == 'rest':
            if SERVOS_ENABLED:
                arm_rest_position()
        elif cmd == 'open_lid':
            if SERVOS_ENABLED:
                open_lid()
        elif cmd == 'close_lid':
            if SERVOS_ENABLED:
                close_lid()
    
    def log_message(self, format, *args):
        pass
    
    def _get_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "raspberrypi.local"

class FrameBuffer:
    """Thread-safe buffer for the latest frame"""
    
    def __init__(self):
        self.frame = None
        self.lock = Lock()
    
    def update_frame(self, frame):
        if frame is not None:
            with self.lock:
                self.frame = frame.copy()
    
    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

# --- Global Variables ---
frame_buffer = FrameBuffer()
global_state = {
    'state': 'SEARCHING',
    'target': TARGET_WORD,
    'detected': False,
    'confidence': 0,
    'distance': -1,
    'center_x': None,
    'center_y': None,
    'frame_count': 0
}
start_time = time.time()
detection_history = deque(maxlen=5)  # Store recent detections for smoothing
operation_mode = OperationMode.AUTO_SEARCH  # Default mode
detection_manager = None  # Will be initialized after camera

# --- Setup GPIO Modes ---
print("🔧 Initializing GPIO...")

# Motor Pins as Output
motor_pins = [MOTOR_A_EN, MOTOR_A_IN1, MOTOR_A_IN2, MOTOR_B_EN, MOTOR_B_IN3, MOTOR_B_IN4]
for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)

# Ultrasonic Pins
GPIO.setup(ULTRA_TRIG, GPIO.OUT)
GPIO.setup(ULTRA_ECHO, GPIO.IN)

# Main LED
GPIO.setup(MAIN_LED, GPIO.OUT)

# IR Pins as Input
GPIO.setup(IR_LEFT, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(IR_RIGHT, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(IR_BACK, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# --- PWM Setup for Motors ---
pwm_a = GPIO.PWM(MOTOR_A_EN, 100)
pwm_b = GPIO.PWM(MOTOR_B_EN, 100)
pwm_a.start(0)
pwm_b.start(0)

# --- Servo Setup (Conditional) ---
print("🦾 Initializing servos..." if SERVOS_ENABLED else "⚠️ Servos are DISABLED")

if SERVOS_ENABLED:
    if PINS_FACTORY:
        print("   Using software PWM for smoother servo control")
        servo_base = Servo(SERVO_BASE, initial_value=0, pin_factory=PINS_FACTORY())
        servo_shoulder = Servo(SERVO_SHOULDER, initial_value=0, pin_factory=PINS_FACTORY())
        servo_elbow = Servo(SERVO_ELBOW, initial_value=0, pin_factory=PINS_FACTORY())
        servo_lid = Servo(SERVO_LID, initial_value=0, pin_factory=PINS_FACTORY())
    else:
        print("   Using default servo control (may have jitter)")
        servo_base = Servo(SERVO_BASE, initial_value=0)
        servo_shoulder = Servo(SERVO_SHOULDER, initial_value=0)
        servo_elbow = Servo(SERVO_ELBOW, initial_value=0)
        servo_lid = Servo(SERVO_LID, initial_value=0)
    
    time.sleep(1)  # Allow servos to initialize
else:
    # Create dummy servo objects when disabled
    class DummyServo:
        def __init__(self, *args, **kwargs):
            pass
        @property
        def value(self):
            return 0
        @value.setter
        def value(self, val):
            pass
    
    servo_base = DummyServo()
    servo_shoulder = DummyServo()
    servo_elbow = DummyServo()
    servo_lid = DummyServo()
    print("   Using dummy servo objects (no hardware control)")

# --- Camera Setup ---
print("📷 Initializing camera...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()
time.sleep(2)

# --- Initialize Detection Manager ---
detection_manager = DetectionManager(DETECTION_SERVER_URLS)

# --- Motor Control Functions ---
def set_motor_speed(motor, speed):
    """Controls motor direction and speed."""
    if motor == 'A':
        in1 = MOTOR_A_IN1
        in2 = MOTOR_A_IN2
        pwm = pwm_a
    elif motor == 'B':
        in1 = MOTOR_B_IN3
        in2 = MOTOR_B_IN4
        pwm = pwm_b
    else:
        return

    if speed > 0:
        GPIO.output(in1, GPIO.HIGH)
        GPIO.output(in2, GPIO.LOW)
    elif speed < 0:
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.HIGH)
    else:
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.LOW)

    duty_cycle = min(100, abs(speed))
    pwm.ChangeDutyCycle(duty_cycle)

def stop_motors():
    set_motor_speed('A', 0)
    set_motor_speed('B', 0)

# --- Sensor Functions ---
def get_distance():
    """Returns distance in cm from ultrasonic sensor."""
    global last_distance
    
    GPIO.output(ULTRA_TRIG, True)
    time.sleep(0.00001)
    GPIO.output(ULTRA_TRIG, False)

    pulse_start = time.time()
    pulse_end = time.time()
    timeout = time.time() + 0.1

    while GPIO.input(ULTRA_ECHO) == 0:
        pulse_start = time.time()
        if time.time() > timeout:
            return last_distance if last_distance > 0 else -1

    while GPIO.input(ULTRA_ECHO) == 1:
        pulse_end = time.time()
        if time.time() > timeout:
            return last_distance if last_distance > 0 else -1

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    
    if 2 < distance < 400:
        last_distance = round(distance, 2)
        return last_distance
    else:
        return last_distance if last_distance > 0 else -1

def check_ir_sensors():
    """Returns dictionary with obstacle status."""
    return {
        "left": not GPIO.input(IR_LEFT),
        "right": not GPIO.input(IR_RIGHT),
        "back": not GPIO.input(IR_BACK)
    }

# --- Servo Control Functions (Conditional) ---
def arm_pickup_position():
    """Move arm to pickup position (if servos enabled)."""
    if SERVOS_ENABLED:
        print("🦾 Moving arm to pickup position")
        servo_base.value = 0.0
        servo_shoulder.value = 0.5
        servo_elbow.value = -0.3
        time.sleep(1)
    else:
        print("⚠️ Servos disabled - skipping arm movement")

def arm_rest_position():
    """Move arm to rest position (if servos enabled)."""
    if SERVOS_ENABLED:
        print("🦾 Restoring arm to rest position")
        servo_base.value = 0.0
        servo_shoulder.value = -0.5
        servo_elbow.value = 0.0
        time.sleep(1)
    else:
        print("⚠️ Servos disabled - skipping arm movement")

def open_lid():
    """Open dustbin lid (if servos enabled)."""
    if SERVOS_ENABLED:
        print("🗑️ Opening lid")
        servo_lid.value = 0.8
        time.sleep(1)
    else:
        print("⚠️ Servos disabled - skipping lid operation")

def close_lid():
    """Close dustbin lid (if servos enabled)."""
    if SERVOS_ENABLED:
        print("🗑️ Closing lid")
        servo_lid.value = -0.8
        time.sleep(1)
    else:
        print("⚠️ Servos disabled - skipping lid operation")

# --- Frame Annotation Function ---
def annotate_frame(frame, state, detection=None, distance=None, obstacles=None):
    """Add detection overlays to frame"""
    if frame is None:
        return None
        
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    
    # Add operation mode
    mode_text = f"Mode: {operation_mode.value}"
    mode_color = (0, 255, 0) if operation_mode == OperationMode.AUTO_SEARCH else (255, 255, 0)
    cv2.putText(annotated, mode_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
    
    # Add state information
    cv2.putText(annotated, f"State: {state}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add target info
    cv2.putText(annotated, f"Target: {TARGET_WORD}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Add servo status
    servo_status = "SERVOS: ON" if SERVOS_ENABLED else "SERVOS: OFF"
    servo_color = (0, 255, 0) if SERVOS_ENABLED else (0, 0, 255)
    cv2.putText(annotated, servo_status, (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, servo_color, 1)
    
    # Add manual speed if in manual mode
    if operation_mode == OperationMode.MANUAL_CONTROL:
        cv2.putText(annotated, f"Manual Speed: {MANUAL_SPEED}%", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Add detection info
    y_offset = 180
    if detection and detection.detected and detection.word == TARGET_WORD:
        # Draw bounding box
        if detection.bbox:
            x0 = int(detection.bbox.get('x0', 0))
            y0 = int(detection.bbox.get('y0', 0))
            x1 = int(detection.bbox.get('x1', w))
            y1 = int(detection.bbox.get('y1', h))
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 2)
        
        # Draw center point
        if detection.center_x and detection.center_y:
            cx = int(detection.center_x)
            cy = int(detection.center_y)
            cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
            cv2.line(annotated, (cx-15, cy), (cx+15, cy), (0, 0, 255), 1)
            cv2.line(annotated, (cx, cy-15), (cx, cy+15), (0, 0, 255), 1)
        
        # Add confidence
        cv2.putText(annotated, f"Confidence: {detection.confidence:.1f}%", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 30
    
    # Add distance
    if distance and distance > 0:
        if distance <= DANGER_DISTANCE:
            color = (0, 0, 255)
        elif distance <= CAUTION_DISTANCE:
            color = (0, 165, 255)
        else:
            color = (0, 255, 0)
            
        cv2.putText(annotated, f"Front: {distance}cm", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Add timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(annotated, timestamp, (w - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return annotated

# --- Navigation Functions ---
def avoid_obstacles(ir_status, distance):
    """Handle obstacle avoidance"""
    print("🚨 Avoiding obstacles")
    stop_motors()
    time.sleep(0.2)
    
    # Back up a bit
    set_motor_speed('A', -40)
    set_motor_speed('B', -40)
    time.sleep(0.5)
    
    # Turn away from obstacles
    if ir_status['left'] and not ir_status['right']:
        # Obstacle on left, turn right
        set_motor_speed('A', -60)
        set_motor_speed('B', 30)
    elif ir_status['right'] and not ir_status['left']:
        # Obstacle on right, turn left
        set_motor_speed('A', 30)
        set_motor_speed('B', -60)
    else:
        # Obstacles on both sides or front, random turn
        set_motor_speed('A', 60)
        set_motor_speed('B', -60)
    
    time.sleep(0.5)
    stop_motors()
    
    return True

def approach_target(detection, distance):
    """Approach the detected target using center coordinates"""
    if not detection or not detection.center_x:
        return False
    
    frame_center = 320  # Assuming 640x480
    target_x = detection.center_x
    
    # Calculate error from center
    error = target_x - frame_center
    
    # Adjust speed based on distance
    if 0 < distance < CAUTION_DISTANCE:
        base_speed = SLOW_SPEED
    else:
        base_speed = APPROACH_SPEED
    
    # Steering logic
    turn_gain = 0.4
    correction = error * turn_gain
    
    left_speed = base_speed - correction
    right_speed = base_speed + correction
    
    # Clamp speeds
    left_speed = max(-60, min(60, left_speed))
    right_speed = max(-60, min(60, right_speed))
    
    set_motor_speed('A', left_speed)
    set_motor_speed('B', right_speed)
    
    # Log steering
    if abs(error) > 50:
        direction = "LEFT" if error < 0 else "RIGHT"
        print(f"🔄 Steering {direction} (error: {error:.0f}, dist: {distance}cm)")
    
    return True

# --- State Machine Functions ---
def search_for_text():
    """Searching state - rotate and look for TRASH"""
    global current_state, detection_history, global_state
    
    # Get sensor data
    distance = get_distance()
    ir_status = check_ir_sensors()
    
    # Check for immediate danger - THIS RUNS EVERY CYCLE
    if (0 < distance < DANGER_DISTANCE) or ir_status['left'] or ir_status['right']:
        avoid_obstacles(ir_status, distance)
        return
    
    # Capture frame for detection
    frame = picam2.capture_array()
    
    # Request detection (non-blocking)
    if detection_manager:
        detection_manager.request_detection(frame)
    
    # Get latest detection result (non-blocking)
    latest_detection = detection_manager.get_latest_detection() if detection_manager else DetectionData()

    # Check any detection and toggle Approach rover state
    if latest_detection.detected:
        stop_motors()
        GPIO.output(MAIN_LED, GPIO.HIGH)
        time.sleep(3)
        current_state = RoverState.APPROACHING
        print(f"✅ Found target {TARGET_WORD} with confidence { latest_detection.confidence:.1f}%")
        arm_pickup_position()
        while True:
            print("Trash Found")
            time.sleep(1)
    
    # Check if we have consistent detections from history
    # valid_detections = [d for d in detection_history if d.is_valid()]
    # if len(valid_detections) >= 1:  # Need 1 consistent detection
    #     current_state = RoverState.APPROACHING
    #     print(f"✅ Found target {TARGET_WORD} with confidence {valid_detections[-1].confidence:.1f}%")
    #     stop_motors()
    
    # Update global state for web interface
    global_state.update({
        'state': current_state.value,
        'detected': latest_detection.detected and latest_detection.word == TARGET_WORD,
        'confidence': latest_detection.confidence if latest_detection.detected else 0,
        'distance': distance,
        'center_x': latest_detection.center_x if latest_detection.detected else None,
        'center_y': latest_detection.center_y if latest_detection.detected else None
    })
    
    # Annotate frame
    annotated_frame = annotate_frame(frame, current_state.value, latest_detection, distance, ir_status)
    frame_buffer.update_frame(annotated_frame)
    
    # Rotate while searching
    if 0 < distance < CAUTION_DISTANCE:
        rotate_speed = 15
    else:
        rotate_speed = 25
    
    set_motor_speed('A', rotate_speed)
    set_motor_speed('B', rotate_speed)  # Opposite directions for rotation

def approach_text():
    """Approaching state - move towards detected text"""
    global current_state, detection_history, global_state
    
    # Get sensor data
    distance = get_distance()
    ir_status = check_ir_sensors()
    
    # Check for immediate danger - THIS RUNS EVERY CYCLE
    if (0 < distance < DANGER_DISTANCE) or ir_status['left'] or ir_status['right']:
        avoid_obstacles(ir_status, distance)
        return
    
    # Capture frame for detection
    frame = picam2.capture_array()
    
    # Request detection (non-blocking)
    if detection_manager:
        detection_manager.request_detection(frame)
    
    # Get latest detection result
    latest_detection = detection_manager.get_latest_detection() if detection_manager else DetectionData()
    
    # Update history with latest detection if valid
    if latest_detection.detected and latest_detection.word == TARGET_WORD:
        detection_history.append(latest_detection)
    
    # Check if we lost the target
    valid_detections = [d for d in detection_history if d.is_valid()]
    if len(valid_detections) == 0:
        print("❌ Lost target, returning to search")
        current_state = RoverState.SEARCHING
        detection_history.clear()
        stop_motors()
        return
    
    # Use the most recent valid detection
    latest_detection = valid_detections[-1]
    
    # Update global state
    global_state.update({
        'state': current_state.value,
        'detected': True,
        'confidence': latest_detection.confidence,
        'distance': distance,
        'center_x': latest_detection.center_x,
        'center_y': latest_detection.center_y
    })
    
    # Annotate frame
    annotated_frame = annotate_frame(frame, current_state.value, latest_detection, distance, ir_status)
    frame_buffer.update_frame(annotated_frame)
    
    # Check distance for pickup
    if 0 < distance < 20:  # Close enough to pick up
        print(f"📏 Target in range ({distance}cm)")
        current_state = RoverState.PICKUP
        stop_motors()
        while true:
            print("Trash Found")
            time.sleep(1)
    
    # Approach the target
    approach_target(latest_detection, distance)

def execute_pickup():
    """Pickup and dropoff sequence"""
    global current_state, detection_history, global_state
    
    print(f"📦 Executing pickup sequence")
    stop_motors()
    
    global_state.update({
        'state': current_state.value,
        'frame_count': global_state.get('frame_count', 0) + 1
    })
    
    # Capture pre-pickup frame
    frame = picam2.capture_array()
    annotated_frame = annotate_frame(frame, "PICKUP - PREPARING")
    cv2.putText(annotated_frame, "Preparing for pickup...", (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    frame_buffer.update_frame(annotated_frame)
    time.sleep(1)
    
    # Pickup sequence (servos may be disabled)
    arm_pickup_position()
    
    # Final approach
    distance = get_distance()
    if distance > 10:
        set_motor_speed('A', 15)
        set_motor_speed('B', 15)
        time.sleep(0.3)
        stop_motors()
    
    # Close gripper
    frame = picam2.capture_array()
    annotated_frame = annotate_frame(frame, "PICKUP - GRIPPING")
    cv2.putText(annotated_frame, "Closing gripper...", (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    frame_buffer.update_frame(annotated_frame)
    
    servo_elbow.value = 0.5
    time.sleep(1)
    
    # Lift arm
    frame = picam2.capture_array()
    annotated_frame = annotate_frame(frame, "PICKUP - LIFTING")
    cv2.putText(annotated_frame, "Lifting object...", (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    frame_buffer.update_frame(annotated_frame)
    
    servo_shoulder.value = 0.0
    time.sleep(1)
    
    # Move to bin
    frame = picam2.capture_array()
    annotated_frame = annotate_frame(frame, "MOVING TO BIN")
    cv2.putText(annotated_frame, "Moving to dustbin...", (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    frame_buffer.update_frame(annotated_frame)
    
    set_motor_speed('A', -35)
    set_motor_speed('B', 35)
    time.sleep(1.2)
    stop_motors()
    
    # Drop sequence
    frame = picam2.capture_array()
    annotated_frame = annotate_frame(frame, "DROPPING")
    cv2.putText(annotated_frame, "Opening bin...", (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    frame_buffer.update_frame(annotated_frame)
    
    open_lid()
    
    frame = picam2.capture_array()
    annotated_frame = annotate_frame(frame, "DROPPING")
    cv2.putText(annotated_frame, "Releasing object...", (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    frame_buffer.update_frame(annotated_frame)
    
    servo_base.value = -0.3
    servo_shoulder.value = 0.4
    time.sleep(1)
    servo_elbow.value = -0.5
    time.sleep(1)
    
    # Reset
    frame = picam2.capture_array()
    annotated_frame = annotate_frame(frame, "RESETTING")
    cv2.putText(annotated_frame, "Resetting...", (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    frame_buffer.update_frame(annotated_frame)
    
    arm_rest_position()
    close_lid()
    
    print("✅ Pickup complete!")
    current_state = RoverState.SEARCHING
    detection_history.clear()

def manual_control_loop():
    """Handle manual control mode - just monitor sensors and update display"""
    global global_state
    
    # Get sensor data
    distance = get_distance()
    ir_status = check_ir_sensors()
    
    # Capture frame
    frame = picam2.capture_array()
    
    # Request detection (non-blocking) for info display
    if detection_manager:
        detection_manager.request_detection(frame)
    
    # Get latest detection result
    latest_detection = detection_manager.get_latest_detection() if detection_manager else DetectionData()
    
    # Update global state
    global_state.update({
        'state': RoverState.MANUAL.value,
        'detected': latest_detection.detected and latest_detection.word == TARGET_WORD,
        'confidence': latest_detection.confidence if latest_detection.detected else 0,
        'distance': distance,
        'center_x': latest_detection.center_x if latest_detection.detected else None,
        'center_y': latest_detection.center_y if latest_detection.detected else None
    })
    
    # Annotate frame
    annotated_frame = annotate_frame(frame, RoverState.MANUAL.value, latest_detection, distance, ir_status)
    frame_buffer.update_frame(annotated_frame)

# --- HTTP Server Thread ---
def start_http_server():
    """Start the HTTP image server in a background thread"""
    handler = lambda *args, **kwargs: ImageServerHandler(*args, frame_buffer=frame_buffer, **kwargs)
    
    try:
        server = ThreadedHTTPServer(("0.0.0.0", HTTP_PORT), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"🌐 HTTP Image Server started on port {HTTP_PORT}")
        return server
    except Exception as e:
        print(f"❌ Failed to start HTTP server: {e}")
        return None

def get_ip_address():
    """Get the local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "raspberrypi.local"

def print_network_info():
    """Display network connection information"""
    print("\n🌐 HTTP Image Server Information")
    print("=" * 60)
    
    ip = get_ip_address()
    print(f"\n📱 Access the web interface:")
    print(f"   • Live Stream: http://{ip}:{HTTP_PORT}")
    print(f"   • MJPEG Stream: http://{ip}:{HTTP_PORT}/stream")
    print(f"   • JSON Status: http://{ip}:{HTTP_PORT}/status")
    print(f"   • Toggle Servos: http://{ip}:{HTTP_PORT}/toggle_servos")
    print(f"   • Set Speed: http://{ip}:{HTTP_PORT}/set_speed?speed=50")
    print(f"   • Set Mode: http://{ip}:{HTTP_PORT}/set_mode?mode=AUTO")
    
    print(f"\n🔧 Current Settings:")
    print(f"   • Servo Status: {'ENABLED' if SERVOS_ENABLED else 'DISABLED'}")
    print(f"   • Manual Speed: {MANUAL_SPEED}%")
    print(f"   • Operation Mode: {operation_mode.value}")
    print("=" * 60)

# --- Cleanup Function ---
def cleanup(signum=None, frame=None):
    """Cleanup resources on exit"""
    global running
    print("\n🧹 Cleaning up...")
    running = False
    
    # Stop detection manager
    if detection_manager:
        detection_manager.stop()
    
    if http_server:
        http_server.shutdown()
    
    stop_motors()
    pwm_a.stop()
    pwm_b.stop()
    
    picam2.stop()
    
    # Only attempt servo operations if they're enabled
    if SERVOS_ENABLED:
        arm_rest_position()
        close_lid()
    
    GPIO.cleanup()
    
    print("👋 Shutdown complete")
    sys.exit(0)

# --- Main Execution ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 ROVER WITH EXTERNAL TEXT DETECTION AND LIVE STREAMING")
    print("="*60)
    
    # Initialize
    print("\n🔧 System Configuration:")
    print(f"   • Target Word: {TARGET_WORD}")
    print(f"   • Detection Servers: {', '.join(DETECTION_SERVER_URLS)}")
    print(f"   • Servos: {'ENABLED' if SERVOS_ENABLED else 'DISABLED'}")
    print(f"   • Min Confidence: {MIN_CONFIDENCE}%")
    print(f"   • Default Mode: {operation_mode.value}")
    print(f"   • Detection Cooldown: {DETECTION_COOLDOWN}s")
    print(f"   • Max Queue Size: {MAX_DETECTION_QUEUE_SIZE}")
    
    # Initialize servos to rest position (if enabled)
    if SERVOS_ENABLED:
        arm_rest_position()
        close_lid()
    time.sleep(1)
    
    # Start HTTP server
    http_server = start_http_server()
    
    # Print network info
    print_network_info()
    
    # Signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    # State variables
    current_state = RoverState.SEARCHING
    last_distance = 0
    running = True
    
    print("\n🚀 Rover started! Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    try:
        while running:
            if operation_mode == OperationMode.AUTO_SEARCH:
                if current_state == RoverState.SEARCHING:
                    search_for_text()
                elif current_state == RoverState.APPROACHING:
                    approach_text()
                elif current_state == RoverState.PICKUP:
                    execute_pickup()
                else:
                    # Default to searching if state is invalid
                    current_state = RoverState.SEARCHING
            else:  # Manual control mode
                manual_control_loop()
            
            # Small delay to prevent CPU overload
            time.sleep(0.03)  # ~30Hz control loop
            
    except Exception as e:
        print(f"❌ Error in main loop: {e}")
        import traceback
        traceback.print_exc()
        cleanup()