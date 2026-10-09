import os
import sys
import subprocess

# Ensure workspace root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>GridLedger - Smart NTL Detection</title>
          <style>
            body {
              font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
              background: #0f172a;
              color: #f8fafc;
              display: flex;
              justify-content: center;
              align-items: center;
              min-height: 100vh;
              margin: 0;
              padding: 20px;
            }
            .card {
              background: #1e293b;
              border: 1px solid #334155;
              border-radius: 12px;
              padding: 32px;
              max-width: 600px;
              box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
              text-align: center;
            }
            h1 { color: #38bdf8; margin-top: 0; font-size: 28px; }
            p { color: #94a3b8; font-size: 16px; line-height: 1.6; }
            .badge {
              display: inline-block;
              background: #0369a1;
              color: #e0f2fe;
              padding: 4px 12px;
              border-radius: 9999px;
              font-size: 13px;
              font-weight: 600;
              margin-bottom: 16px;
            }
            .btn {
              display: inline-block;
              background: #0284c7;
              color: white;
              text-decoration: none;
              padding: 12px 24px;
              border-radius: 8px;
              font-weight: bold;
              margin-top: 20px;
              transition: background 0.2s;
            }
            .btn:hover {
              background: #0369a1;
            }
            .repo-link {
              margin-top: 15px;
              display: block;
              color: #38bdf8;
              text-decoration: none;
              font-size: 14px;
            }
          </style>
        </head>
        <body>
          <div class="card">
            <span class="badge">Streamlit + Vercel Deployment</span>
            <h1>⚡ GridLedger</h1>
            <p><strong>Physics-First Non-Technical Loss (NTL) Detection in Electrical Distribution Grids</strong></p>
            <p>Reconciling energy at every grid boundary (Feeder &rarr; Transformer &rarr; Customer Meter).</p>
            <a class="btn" href="https://github.com/ajaysurya1/GridLedger" target="_blank">View GitHub Repository</a>
          </div>
        </body>
        </html>
        """
        self.wfile.write(html.encode('utf-8'))
        return
