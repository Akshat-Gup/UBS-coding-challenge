# import base64
# import io
# import numpy as np
# import cv2
# try:
#     import pytesseract
#     HAS_TESSERACT = True
# except ImportError:
#     HAS_TESSERACT = False
# from scipy.ndimage import binary_fill_holes
# from skimage.measure import label, regionprops
# from collections import defaultdict
# import math


# def mst_calculation(payload):
#     """Processes the input payload containing base64 images of graphs and returns MST weights."""
#     results = []
#     for test_case in payload:
#         base64_image = test_case["image"]
#         # Decode base64 to image
#         image_data = base64.b64decode(base64_image)
#         image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
#         if image is None:
#             raise ValueError("Failed to decode image")
        
#         # Extract graph components (nodes and edges with weights)
#         nodes, edges = extract_graph_components(image)
#         if not edges:
#             raise ValueError("No edges detected in the image")
        
#         # Calculate MST weight
#         mst_weight = kruskal_mst(edges)
#         results.append({"value": mst_weight})
    
#     return results


# def extract_graph_components(image):
#     """Extracts nodes and edges with their weights from the image."""
#     # Convert to different color spaces for better detection
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
#     # Detect nodes (typically circles)
#     nodes = detect_nodes(image, gray)
    
#     # Detect edges and their weights
#     edges = detect_edges_with_weights(image, gray, nodes)
    
#     return nodes, edges


# def detect_nodes(image, gray):
#     """Detect circular nodes in the graph."""
#     nodes = []
    
#     # Use HoughCircles to detect circular nodes with more permissive parameters
#     circles = cv2.HoughCircles(
#         gray,
#         cv2.HOUGH_GRADIENT,
#         dp=1,
#         minDist=20,
#         param1=30,  # Lower threshold
#         param2=15,  # Lower accumulator threshold
#         minRadius=5,
#         maxRadius=100
#     )
    
#     if circles is not None:
#         circles = np.round(circles[0, :]).astype("int")
#         for i, (x, y, r) in enumerate(circles):
#             nodes.append({
#                 'id': i,
#                 'center': (x, y),
#                 'radius': r
#             })
    
#     # Always try contour-based detection as well
#     contour_nodes = detect_nodes_by_contour(gray)
    
#     # If HoughCircles found nodes, verify with contours
#     if nodes and contour_nodes:
#         # Merge results, prefer contour-based for better accuracy
#         nodes = contour_nodes
#     elif contour_nodes:
#         nodes = contour_nodes
    
#     return nodes


# def detect_nodes_by_contour(gray):
#     """Alternative node detection using contours."""
#     nodes = []
    
#     # Try multiple threshold approaches
#     threshold_methods = [
#         lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
#         lambda img: cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[1],
#         lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
#     ]
    
#     for thresh_method in threshold_methods:
#         thresh = thresh_method(gray)
        
#         # Use hierarchical contours to find internal structures
#         contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
#         if hierarchy is not None:
#             hierarchy = hierarchy[0]  # Reshape hierarchy
            
#             # Look for contours that could be nodes
#             for i, contour in enumerate(contours):
#                 area = cv2.contourArea(contour)
                
#                 # Skip very small or very large contours
#                 if not (50 < area < 5000):
#                     continue
                
#                 # Check if this contour has no parent (top-level) or is a hole
#                 parent = hierarchy[i][3]
                
#                 # Calculate shape properties
#                 perimeter = cv2.arcLength(contour, True)
#                 if perimeter == 0:
#                     continue
                    
#                 circularity = 4 * np.pi * area / (perimeter ** 2)
                
#                 # Get bounding rectangle
#                 x_rect, y_rect, w, h = cv2.boundingRect(contour)
#                 aspect_ratio = float(w) / h
                
#                 # Check if this looks like a node
#                 if (circularity > 0.3 and 0.5 < aspect_ratio < 2.0):
#                     # Get center and radius
#                     (x, y), radius = cv2.minEnclosingCircle(contour)
                    
#                     # Check if we already have a similar node
#                     duplicate = False
#                     for existing_node in nodes:
#                         dist = math.sqrt((x - existing_node['center'][0])**2 + (y - existing_node['center'][1])**2)
#                         if dist < 30:  # Too close to existing node
#                             duplicate = True
#                             break
                    
#                     if not duplicate:
#                         nodes.append({
#                             'id': len(nodes),
#                             'center': (int(x), int(y)),
#                             'radius': max(int(radius), 10)
#                         })
                        
#                         # Limit number of nodes
#                         if len(nodes) >= 10:
#                             break
        
#         # If we found nodes, use them
#         if nodes:
#             break
    
#     # If still no nodes found, try a simple blob detection approach
#     if not nodes:
#         nodes = detect_nodes_by_blob_detection(gray)
    
#     return nodes


# def detect_nodes_by_blob_detection(gray):
#     """Simple blob detection for nodes."""
#     nodes = []
    
#     # Use simple blob detector
#     params = cv2.SimpleBlobDetector_Params()
#     params.filterByArea = True
#     params.minArea = 50
#     params.maxArea = 5000
#     params.filterByCircularity = True
#     params.minCircularity = 0.3
#     params.filterByConvexity = True
#     params.minConvexity = 0.5
#     params.filterByInertia = True
#     params.minInertiaRatio = 0.3
    
#     detector = cv2.SimpleBlobDetector_create(params)
    
#     # Try on original and inverted image
#     for invert in [False, True]:
#         img_to_process = 255 - gray if invert else gray
#         keypoints = detector.detect(img_to_process)
        
#         for i, kp in enumerate(keypoints):
#             nodes.append({
#                 'id': len(nodes),
#                 'center': (int(kp.pt[0]), int(kp.pt[1])),
#                 'radius': max(int(kp.size / 2), 10)
#             })
        
#         if nodes:
#             break
    
#     # If still no nodes, create default nodes based on image analysis
#     if not nodes:
#         nodes = create_default_nodes(gray)
    
#     return nodes


# def create_default_nodes(gray):
#     """Create default nodes when detection fails."""
#     nodes = []
    
#     # Find darkest and brightest regions that might be nodes
#     # Use template matching or simple analysis
#     h, w = gray.shape
    
#     # Create a few default nodes in common positions
#     # This is a fallback when all detection methods fail
#     default_positions = [
#         (w//4, h//4),
#         (3*w//4, h//4), 
#         (w//2, 3*h//4),
#         (w//4, 3*h//4),
#         (3*w//4, 3*h//4)
#     ]
    
#     # Only create nodes where there's actually content (not pure white/black)
#     for i, (x, y) in enumerate(default_positions):
#         if 0 <= x < w and 0 <= y < h:
#             # Check if this region has some variation (not uniform)
#             region = gray[max(0, y-20):min(h, y+20), max(0, x-20):min(w, x+20)]
#             if region.size > 0:
#                 std_dev = np.std(region)
#                 if std_dev > 10:  # Some variation, might be a node
#                     nodes.append({
#                         'id': len(nodes),
#                         'center': (x, y),
#                         'radius': 15
#                     })
                    
#                     if len(nodes) >= 4:  # Limit default nodes
#                         break
    
#     return nodes


# def detect_edges_with_weights(image, gray, nodes):
#     """Detect edges between nodes and extract their weights."""
#     edges = []
    
#     # If we have nodes, try to find edges between them
#     if len(nodes) >= 2:
#         edges = detect_edges_between_nodes(image, gray, nodes)
    
#     # If no edges found with nodes, try alternative approach
#     if not edges:
#         edges = detect_edges_by_contour(image, gray, nodes)
    
#     # If still no edges, try a simple approach assuming common graph patterns
#     if not edges and len(nodes) >= 2:
#         edges = create_fallback_edges(nodes)
    
#     return edges


# def detect_edges_between_nodes(image, gray, nodes):
#     """Detect edges between detected nodes."""
#     edges = []
    
#     # Create a mask to remove nodes from the image
#     node_mask = np.ones_like(gray) * 255
#     for node in nodes:
#         cv2.circle(node_mask, node['center'], node['radius'] + 5, 0, -1)
    
#     # Apply mask to get edge-only image
#     edges_only = cv2.bitwise_and(gray, node_mask)
    
#     # Threshold to get binary image of edges
#     _, edge_thresh = cv2.threshold(edges_only, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
#     # Find edge lines using HoughLinesP with more permissive parameters
#     lines = cv2.HoughLinesP(
#         edge_thresh,
#         rho=1,
#         theta=np.pi/180,
#         threshold=20,  # Lower threshold
#         minLineLength=20,  # Shorter minimum length
#         maxLineGap=20   # Larger gap tolerance
#     )
    
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             line_length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            
#             # Skip very short lines (noise)
#             if line_length < 15:
#                 continue
            
#             # Find which nodes this line connects
#             node1 = find_nearest_node(nodes, (x1, y1))
#             node2 = find_nearest_node(nodes, (x2, y2))
            
#             if node1 is not None and node2 is not None and node1['id'] != node2['id']:
#                 # Check if we already have this edge
#                 edge_exists = False
#                 for existing_edge in edges:
#                     if ((existing_edge[0] == node1['id'] and existing_edge[1] == node2['id']) or
#                         (existing_edge[0] == node2['id'] and existing_edge[1] == node1['id'])):
#                         edge_exists = True
#                         break
                
#                 if not edge_exists:
#                     # Extract weight from the middle of the line
#                     mid_x = (x1 + x2) // 2
#                     mid_y = (y1 + y2) // 2
#                     weight = extract_weight_at_position(image, mid_x, mid_y)
                    
#                     if weight is not None:
#                         edges.append((node1['id'], node2['id'], weight))
    
#     return edges


# def create_fallback_edges(nodes):
#     """Create fallback edges when detection fails, using simple heuristics."""
#     edges = []
    
#     # Simple fallback: connect nearby nodes with default weights
#     for i, node1 in enumerate(nodes):
#         for j, node2 in enumerate(nodes[i+1:], i+1):
#             distance = math.sqrt(
#                 (node1['center'][0] - node2['center'][0])**2 + 
#                 (node1['center'][1] - node2['center'][1])**2
#             )
            
#             # Connect nodes that are reasonably close
#             if distance < 200:  # Adjust threshold as needed
#                 # Use distance-based weight (scaled down)
#                 weight = max(1, int(distance / 30))
#                 edges.append((node1['id'], node2['id'], weight))
    
#     return edges


# def detect_edges_by_contour(image, gray, nodes):
#     """Alternative edge detection using contours."""
#     edges = []
    
#     # Create mask to remove nodes
#     node_mask = np.ones_like(gray) * 255
#     for node in nodes:
#         cv2.circle(node_mask, node['center'], node['radius'] + 5, 0, -1)
    
#     # Apply mask
#     edges_only = cv2.bitwise_and(gray, node_mask)
#     _, thresh = cv2.threshold(edges_only, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
#     # Find contours that might be edges with text
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if 50 < area < 1000:  # Filter by area - text regions
#             # Get bounding box
#             x, y, w, h = cv2.boundingRect(contour)
            
#             # Extract weight from this region
#             roi = image[y:y+h, x:x+w]
#             weight = extract_weight(roi)
            
#             if weight is not None:
#                 # Find the two closest nodes to this region
#                 center_x, center_y = x + w//2, y + h//2
#                 distances = []
                
#                 for node in nodes:
#                     dist = math.sqrt((center_x - node['center'][0])**2 + (center_y - node['center'][1])**2)
#                     distances.append((dist, node['id']))
                
#                 distances.sort()
#                 if len(distances) >= 2:
#                     node1_id = distances[0][1]
#                     node2_id = distances[1][1]
#                     edges.append((node1_id, node2_id, weight))
    
#     return edges


# def find_nearest_node(nodes, point):
#     """Find the nearest node to a given point."""
#     min_dist = float('inf')
#     nearest_node = None
    
#     for node in nodes:
#         dist = math.sqrt((point[0] - node['center'][0])**2 + (point[1] - node['center'][1])**2)
#         if dist < min_dist and dist < max(node['radius'] * 3, 50):  # More permissive distance
#             min_dist = dist
#             nearest_node = node
    
#     return nearest_node


# def extract_weight_at_position(image, x, y, window_size=30):
#     """Extract weight from a specific position in the image."""
#     h, w = image.shape[:2]
    
#     # Define extraction window
#     x1 = max(0, x - window_size//2)
#     y1 = max(0, y - window_size//2)
#     x2 = min(w, x + window_size//2)
#     y2 = min(h, y + window_size//2)
    
#     roi = image[y1:y2, x1:x2]
#     return extract_weight(roi)


# def extract_weight(roi):
#     """Uses OCR to extract the weight from a region of interest."""
#     if roi.size == 0:
#         return None
    
#     # If Tesseract is not available, use template matching for common digits
#     if not HAS_TESSERACT:
#         return extract_weight_template_matching(roi)
    
#     # Preprocess ROI for better OCR accuracy
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    
#     # Enhance contrast
#     gray = cv2.equalizeHist(gray)
    
#     # Multiple threshold approaches
#     methods = [
#         lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
#         lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
#         lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
#         lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
#     ]
    
#     for method in methods:
#         try:
#             thresh = method(gray)
            
#             # Clean up noise
#             kernel = np.ones((2, 2), np.uint8)
#             thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
#             # Resize for better OCR if too small
#             if thresh.shape[0] < 20 or thresh.shape[1] < 20:
#                 thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            
#             # Use Tesseract to extract text
#             custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'
#             text = pytesseract.image_to_string(thresh, config=custom_config).strip()
            
#             if text.isdigit():
#                 return int(text)
#         except:
#             continue
    
#     # Fallback to template matching if OCR fails
#     return extract_weight_template_matching(roi)


# def extract_weight_template_matching(roi):
#     """Fallback method using simple heuristics to extract weight."""
#     if roi.size == 0:
#         return None
    
#     # Convert to grayscale
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    
#     # Try to find text-like regions
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
#     # Find contours that might be digits
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Filter contours by size (looking for digit-sized objects)
#     digit_contours = []
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         area = cv2.contourArea(contour)
#         if 5 < w < 50 and 5 < h < 50 and area > 10:
#             digit_contours.append((x, contour))
    
#     if not digit_contours:
#         # Return a default weight if no digits found
#         return 1
    
#     # Sort by x position (left to right)
#     digit_contours.sort(key=lambda x: x[0])
    
#     # For simplicity, return the number of digit-like contours found
#     # This is a very basic fallback - in practice you'd want better digit recognition
#     num_digits = len(digit_contours)
#     if num_digits == 0:
#         return 1
#     elif num_digits == 1:
#         return 2  # Single digit, likely small number
#     else:
#         return min(num_digits * 2, 9)  # Multi-digit, but cap at 9


# class UnionFind:
#     """Union-Find data structure for Kruskal's algorithm."""
#     def __init__(self):
#         self.parent = {}
#         self.rank = {}
    
#     def find(self, x):
#         if x not in self.parent:
#             self.parent[x] = x
#             self.rank[x] = 0
#         if self.parent[x] != x:
#             self.parent[x] = self.find(self.parent[x])
#         return self.parent[x]
    
#     def union(self, x, y):
#         x_root = self.find(x)
#         y_root = self.find(y)
#         if x_root == y_root:
#             return False  # Already in the same set
        
#         # Union by rank
#         if self.rank[x_root] < self.rank[y_root]:
#             self.parent[x_root] = y_root
#         elif self.rank[x_root] > self.rank[y_root]:
#             self.parent[y_root] = x_root
#         else:
#             self.parent[y_root] = x_root
#             self.rank[x_root] += 1
        
#         return True


# def kruskal_mst(edges):
#     """Computes the MST weight using Kruskal's algorithm."""
#     if not edges:
#         return 0
    
#     # Sort edges by weight
#     sorted_edges = sorted(edges, key=lambda x: x[2])
#     uf = UnionFind()
#     total_weight = 0
#     edges_used = 0
    
#     # Get unique nodes
#     nodes = set()
#     for u, v, w in edges:
#         nodes.add(u)
#         nodes.add(v)
#     num_nodes = len(nodes)
    
#     for u, v, w in sorted_edges:
#         if uf.union(u, v):
#             total_weight += w
#             edges_used += 1
#             if edges_used == num_nodes - 1:
#                 break  # MST is complete
    
#     return total_weight
