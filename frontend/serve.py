#!/usr/bin/env python3
"""Simple HTTP server for the TTS Studio frontend.

Usage:
    python serve.py [port]
    
Default port is 8080.
"""

import http.server
import socketserver
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

# Change to the frontend directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

Handler = http.server.SimpleHTTPRequestHandler

# Add CORS headers for local development
class CORSRequestHandler(Handler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

HOST = "0.0.0.0"  # Bind to all interfaces for LAN access

with socketserver.TCPServer((HOST, PORT), CORSRequestHandler) as httpd:
    # Get local IP for display
    import socket
    local_ip = socket.gethostbyname(socket.gethostname())
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        pass
    
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║                    TTS Studio                             ║
╠═══════════════════════════════════════════════════════════╣
║  Local:   http://localhost:{PORT}                           ║
║  LAN:     http://{local_ip}:{PORT}                       ║
║                                                           ║
║  Press Ctrl+C to stop                                     ║
╚═══════════════════════════════════════════════════════════╝
""")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
