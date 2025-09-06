#mst challenge 

from flask import Flask, request, jsonify
import base64
from io import BytesIO
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import math
from collections import defaultdict

app = Flask(__name__)

def decode_image(base64_str):
    """Decode base64 string to OpenCV image"""
    img_data = base64.b64decode(base64_str)
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    return img

def detect_nodes(img):
    """Detect nodes (black circles) using Hough Circle Transform"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=30
    )
    
    if circles is None:
        return []
    
    # Convert to integers and remove duplicates
    circles = np.uint16(np.around(circles))[0]
    unique_nodes = []
    seen = set()
    for (x, y, r) in circles:
        key = (round(x/5), round(y/5))  # Quantize to reduce duplicates
        if key not in seen:
            seen.add(key)
            unique_nodes.append((x, y))
    
    return unique_nodes

def detect_edges(img):
    """Detect edges (colored lines) using color thresholding and Canny edge detection"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for common edge colors (adjust based on actual edge colors)
    lower = np.array([0, 0, 50])    # Example: non-black colors
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    # Find contours of edges
    edges = cv2.Canny(mask, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    edge_segments = []
    for contour in contours:
        if len(contour) < 2:
            continue
        # Get bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(contour)
        edge_segments.append((x, y, x + w, y + h))  # (x1, y1, x2, y2)
    
    return edge_segments

def extract_edge_weights(img, edge_segments):
    """Extract weight values from edge segments using OCR"""
    weights = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    for (x1, y1, x2, y2) in edge_segments:
        # Extract region around the middle of the edge (where weight is located)
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        # Define a region around the midpoint to capture the weight
        roi = gray[max(0, mid_y - 15):min(img.shape[0], mid_y + 15),
                   max(0, mid_x - 15):min(img.shape[1], mid_x + 15)]
        
        # Preprocess for OCR
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Use Tesseract to extract text
        text = pytesseract.image_to_string(thresh, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
        text = text.strip()
        
        if text.isdigit():
            weights.append(int(text))
        else:
            weights.append(1)  # Fallback if OCR fails (not ideal, but for demonstration)
    
    return weights

def associate_edges_with_nodes(edges, nodes, threshold=30):
    """Associate edge segments with their connected nodes"""
    node_edges = []
    for (x1, y1, x2, y2) in edges:
        # Find closest nodes to each end of the edge
        start_node = None
        end_node = None
        min_dist_start = float('inf')
        min_dist_end = float('inf')
        
        for i, (nx, ny) in enumerate(nodes):
            dist_start = math.hypot(nx - x1, ny - y1)
            dist_end = math.hypot(nx - x2, ny - y2)
            
            if dist_start < min_dist_start and dist_start < threshold:
                min_dist_start = dist_start
                start_node = i
            if dist_end < min_dist_end and dist_end < threshold:
                min_dist_end = dist_end
                end_node = i
        
        if start_node is not None and end_node is not None and start_node != end_node:
            node_edges.append((start_node, end_node))
    
    return node_edges

def kruskal_mst(edges, num_nodes):
    """Compute MST weight using Kruskal's algorithm"""
    parent = list(range(num_nodes))
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # Path compression
            u = parent[u]
        return u
    
    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root == v_root:
            return False  # Already in the same set
        parent[v_root] = u_root
        return True
    
    # Sort edges by weight
    edges_sorted = sorted(edges, key=lambda x: x[2])
    mst_weight = 0
    edges_used = 0
    
    for u, v, w in edges_sorted:
        if union(u, v):
            mst_weight += w
            edges_used += 1
            if edges_used == num_nodes - 1:
                break
    
    return mst_weight

def process_graph(image_base64):
    """Process a single graph image and return MST weight"""
    # Decode image
    img = decode_image(image_base64)
    if img is None:
        return 0
    
    # Detect nodes and edges
    nodes = detect_nodes(img)
    if not nodes:
        return 0
    num_nodes = len(nodes)
    
    edges = detect_edges(img)
    if not edges:
        return 0
    
    # Extract edge weights
    weights = extract_edge_weights(img, edges)
    if len(weights) != len(edges):
        return 0
    
    # Associate edges with nodes
    node_edges = associate_edges_with_nodes(edges, nodes)
    if len(node_edges) != len(edges):
        return 0  # Mismatch in edge detection
    
    # Create list of (u, v, weight)
    graph_edges = []
    for i, (u, v) in enumerate(node_edges):
        graph_edges.append((u, v, weights[i]))
    
    # Compute MST
    mst_weight = kruskal_mst(graph_edges, num_nodes)
    return mst_weight

@app.route('/mst-calculation', methods=['POST'])
def mst_calculation():
    try:
        data = request.get_json()
        if not data or len(data) != 2:
            return jsonify({"error": "Invalid input format"}), 400
        
        results = []
        for test_case in data:
            image_base64 = test_case.get('image', '')
            mst_value = process_graph(image_base64)
            results.append({"value": mst_value})
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
