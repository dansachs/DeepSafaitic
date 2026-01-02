#!/usr/bin/env python3
"""
Interactive Safaitic Labeler with Ductus Tracking - Enhanced Version

Features:
- Click boxes to label them with ground truth characters
- Automatic spline path tracking through box centers
- Directional arrows showing text flow
- Manual box rotation (right-click)
- Path restart (R key)
- Zoom in/out (mouse wheel, +/- keys)
- Pan (middle mouse button or space+drag)
- Split view: Original + Binary Mask
- Export with angles and path IDs
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import csv
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import sqlite3
import re
from scipy.interpolate import interp1d
import math
import cv2


class Box:
    """Represents a detected glyph box."""
    def __init__(self, x: int, y: int, w: int, h: int, box_id: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.box_id = box_id
        self.label = ""  # Ground truth character
        self.path_id = 0  # Which path this box belongs to
        self.order_in_path = -1  # Order within path (0-indexed)
        self.angle = 0.0  # Direction angle in degrees (0 = right, 90 = down, -90 = up)
        self.center = (x + w // 2, y + h // 2)
        self.original_x = x
        self.original_y = y
        self.original_w = w
        self.original_h = h
    
    def contains_point(self, px: int, py: int) -> bool:
        """Check if point is inside this box."""
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h
    
    def get_rotation_arrow(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get arrow endpoints based on angle."""
        cx, cy = self.center
        # Arrow length is 1/3 of box size
        length = min(self.w, self.h) // 3
        
        # Convert angle to radians (0 = right, positive = clockwise)
        angle_rad = math.radians(self.angle)
        
        # Calculate arrow endpoint
        end_x = int(cx + length * math.cos(angle_rad))
        end_y = int(cy + length * math.sin(angle_rad))
        
        return (cx, cy), (end_x, end_y)
    
    def scale(self, scale_x: float, scale_y: float):
        """Scale box coordinates."""
        self.x = int(self.original_x * scale_x)
        self.y = int(self.original_y * scale_y)
        self.w = int(self.original_w * scale_x)
        self.h = int(self.original_h * scale_y)
        self.center = (self.x + self.w // 2, self.y + self.h // 2)


class InteractiveLabeler:
    def __init__(self, root: tk.Tk, image_path: str, boxes: List[Tuple[int, int, int, int]], 
                 ground_truth: str = "", db_path: str = ""):
        self.root = root
        self.image_path = image_path
        self.original_image = Image.open(image_path).convert('RGB')
        self.boxes = [Box(x, y, w, h, i) for i, (x, y, w, h) in enumerate(boxes)]
        self.ground_truth = ground_truth
        self.db_path = db_path
        
        # Path tracking
        self.current_path_id = 0
        self.paths = {}  # path_id -> list of box indices in order
        self.selected_box_idx = -1
        self.ground_truth_index = 0  # Current position in ground truth string
        
        # Zoom and pan
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.space_pressed = False
        
        # View mode: 'single' or 'split'
        self.view_mode = 'single'
        
        # Generate binary mask
        self.binary_mask = self.create_binary_mask()
        
        # Store original image dimensions
        self.orig_width = self.original_image.width
        self.orig_height = self.original_image.height
        
        # UI setup
        self.setup_ui()
        self.load_image()
        self.update_display()
        
        # Bind events
        self.setup_bindings()
    
    def create_binary_mask(self) -> np.ndarray:
        """Create binary mask from original image."""
        # Convert PIL to numpy
        img_array = np.array(self.original_image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2
        )
        
        return binary
    
    def setup_bindings(self):
        """Set up all event bindings."""
        # Mouse events
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-3>", self.on_right_click)  # Right click
        self.canvas.bind("<ButtonPress-2>", self.start_pan)  # Middle mouse button
        self.canvas.bind("<B2-Motion>", self.on_pan)
        self.canvas.bind("<ButtonRelease-2>", self.end_pan)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)  # macOS/Windows
        self.canvas.bind("<Button-4>", self.on_mousewheel)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mousewheel)  # Linux scroll down
        
        # Keyboard bindings - bind to root window for global access
        self.root.bind_all("<KeyPress-r>", self.restart_path)
        self.root.bind_all("<KeyPress-R>", self.restart_path)
        self.root.bind_all("<KeyPress-s>", self.save_results)
        self.root.bind_all("<KeyPress-S>", self.save_results)
        self.root.bind_all("<KeyPress-Left>", self.rotate_left)
        self.root.bind_all("<KeyPress-Right>", self.rotate_right)
        self.root.bind_all("<KeyPress-plus>", self.zoom_in)
        self.root.bind_all("<KeyPress-equal>", self.zoom_in)  # + without shift
        self.root.bind_all("<KeyPress-minus>", self.zoom_out)
        self.root.bind_all("<KeyPress-0>", self.zoom_reset)
        self.root.bind_all("<KeyPress-v>", self.toggle_view)
        self.root.bind_all("<KeyPress-V>", self.toggle_view)
        self.root.bind_all("<KeyPress-space>", self.on_space_press)
        self.root.bind_all("<KeyRelease-space>", self.on_space_release)
        
        # Ensure canvas can receive focus
        self.canvas.focus_set()
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())
    
    def setup_ui(self):
        """Set up the user interface."""
        self.root.title("Safaitic Interactive Labeler - Ductus Tracker")
        self.root.geometry("1400x900")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Ground truth display
        ttk.Label(control_frame, text="Ground Truth:").grid(row=0, column=0, padx=5)
        self.gt_var = tk.StringVar(value=self.ground_truth)
        gt_entry = ttk.Entry(control_frame, textvariable=self.gt_var, width=50)
        gt_entry.grid(row=0, column=1, padx=5)
        gt_entry.bind("<KeyRelease>", lambda e: self.update_ground_truth())
        
        # Current character display
        ttk.Label(control_frame, text="Next Char:").grid(row=0, column=2, padx=5)
        self.next_char_var = tk.StringVar(value="")
        ttk.Label(control_frame, textvariable=self.next_char_var, font=("Arial", 14, "bold")).grid(row=0, column=3, padx=5)
        
        # Path info
        ttk.Label(control_frame, text="Path:").grid(row=1, column=0, padx=5)
        self.path_var = tk.StringVar(value="Path 0")
        ttk.Label(control_frame, textvariable=self.path_var).grid(row=1, column=1, padx=5)
        
        # Zoom info
        ttk.Label(control_frame, text="Zoom:").grid(row=1, column=2, padx=5)
        self.zoom_var = tk.StringVar(value="100%")
        ttk.Label(control_frame, textvariable=self.zoom_var).grid(row=1, column=3, padx=5)
        
        # Status
        ttk.Label(control_frame, text="Status:").grid(row=2, column=0, padx=5)
        self.status_var = tk.StringVar(value="Ready - Click boxes to label")
        ttk.Label(control_frame, textvariable=self.status_var).grid(row=2, column=1, padx=5, columnspan=3, sticky=tk.W)
        
        # Instructions
        instructions = (
            "R=Restart path | S=Save | ←/→=Rotate | +/-=Zoom | 0=Reset zoom | "
            "V=Toggle view | Space+Drag=Pan | Middle mouse=Pan | Scroll=Zoom"
        )
        ttk.Label(control_frame, text=instructions, font=("Arial", 8)).grid(
            row=3, column=0, columnspan=4, pady=5
        )
        
        # Canvas for image
        self.canvas = tk.Canvas(main_frame, bg="gray", cursor="crosshair")
        self.canvas.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.canvas.yview)
        v_scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        h_scrollbar = ttk.Scrollbar(main_frame, orient="horizontal", command=self.canvas.xview)
        h_scrollbar.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
    
    def load_image(self):
        """Load and prepare images for display."""
        # Resize if too large for initial display
        max_size = 2000
        if self.original_image.width > max_size or self.original_image.height > max_size:
            ratio = min(max_size / self.original_image.width, max_size / self.original_image.height)
            new_size = (int(self.original_image.width * ratio), int(self.original_image.height * ratio))
            self.display_image = self.original_image.resize(new_size, Image.Resampling.LANCZOS)
            # Scale binary mask
            self.display_binary = Image.fromarray(self.binary_mask).resize(new_size, Image.Resampling.NEAREST)
            # Scale boxes proportionally
            scale_x = new_size[0] / self.orig_width
            scale_y = new_size[1] / self.orig_height
            for box in self.boxes:
                box.scale(scale_x, scale_y)
        else:
            self.display_image = self.original_image.copy()
            self.display_binary = Image.fromarray(self.binary_mask)
        
        self.image_width = self.display_image.width
        self.image_height = self.display_image.height
    
    def update_ground_truth(self):
        """Update ground truth from entry field."""
        self.ground_truth = self.gt_var.get()
        self.update_next_char()
    
    def update_next_char(self):
        """Update the next character display."""
        if self.ground_truth_index < len(self.ground_truth):
            char = self.ground_truth[self.ground_truth_index]
            self.next_char_var.set(f"'{char}'")
        else:
            self.next_char_var.set("(end)")
    
    def get_display_image(self) -> Image.Image:
        """Get the current display image based on view mode."""
        if self.view_mode == 'split':
            # Create side-by-side view
            img1 = self.display_image.copy()
            img2 = self.display_binary.convert('RGB')  # Convert binary to RGB for side-by-side
            
            # Resize to same height
            h = max(img1.height, img2.height)
            w1 = int(img1.width * h / img1.height)
            w2 = int(img2.width * h / img2.height)
            
            img1 = img1.resize((w1, h), Image.Resampling.LANCZOS)
            img2 = img2.resize((w2, h), Image.Resampling.NEAREST)
            
            # Create combined image
            combined = Image.new('RGB', (w1 + w2, h), (255, 255, 255))
            combined.paste(img1, (0, 0))
            combined.paste(img2, (w1, 0))
            
            return combined
        else:
            return self.display_image.copy()
    
    def update_display(self):
        """Update the canvas display with boxes, splines, and arrows."""
        # Get base image
        base_img = self.get_display_image()
        
        # Apply zoom
        if self.zoom_level != 1.0:
            new_w = int(base_img.width * self.zoom_level)
            new_h = int(base_img.height * self.zoom_level)
            base_img = base_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            # Scale boxes for zoom
            zoom_scale_x = self.zoom_level
            zoom_scale_y = self.zoom_level
        else:
            zoom_scale_x = 1.0
            zoom_scale_y = 1.0
        
        # Create a copy for drawing
        img = base_img.copy()
        draw = ImageDraw.Draw(img)
        
        # Scale boxes temporarily for drawing
        temp_boxes = []
        for box in self.boxes:
            if self.view_mode == 'split' and box.x > self.image_width:
                # Skip boxes on the binary mask side in split view
                continue
            temp_boxes.append(box)
        
        # Draw splines for each path
        for path_id, box_indices in self.paths.items():
            if len(box_indices) < 2:
                continue
            
            # Get centers in order (only for boxes on original image side)
            centers = []
            for idx in box_indices:
                if idx < len(temp_boxes):
                    box = temp_boxes[idx]
                    if self.view_mode == 'split' and box.x > self.image_width:
                        continue
                    centers.append(box.center)
            
            # Draw spline
            if len(centers) >= 2:
                self.draw_spline(draw, centers, path_id, zoom_scale_x, zoom_scale_y)
        
        # Draw boxes
        for i, box in enumerate(temp_boxes):
            # Box color based on state
            if i == self.selected_box_idx:
                color = "cyan"
                width = 3
            elif box.label:
                color = "green"
                width = 2
            else:
                color = "red"
                width = 1
            
            # Scale box coordinates for zoom
            bx = int(box.x * zoom_scale_x)
            by = int(box.y * zoom_scale_y)
            bw = int(box.w * zoom_scale_x)
            bh = int(box.h * zoom_scale_y)
            
            # Draw box
            draw.rectangle([bx, by, bx + bw, by + bh], outline=color, width=width)
            
            # Draw label
            if box.label:
                # Background for text
                text_bbox = draw.textbbox((0, 0), box.label, font=None)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                draw.rectangle([bx, by, bx + text_w + 4, by + text_h + 4], 
                             fill="white", outline=color)
                draw.text((bx + 2, by + 2), box.label, fill="black")
            
            # Draw order number
            if box.order_in_path >= 0:
                order_text = str(box.order_in_path + 1)
                draw.text((bx + bw - 15, by), order_text, fill="yellow", font=None)
            
            # Draw directional arrow
            if box.angle != 0 or box.order_in_path >= 0:
                cx, cy = box.center
                cx = int(cx * zoom_scale_x)
                cy = int(cy * zoom_scale_y)
                length = min(bw, bh) // 3
                angle_rad = math.radians(box.angle)
                end_x = int(cx + length * math.cos(angle_rad))
                end_y = int(cy + length * math.sin(angle_rad))
                draw.line([(cx, cy), (end_x, end_y)], fill="yellow", width=2)
                self.draw_arrowhead(draw, (cx, cy), (end_x, end_y), "yellow")
        
        # Convert to PhotoImage and display
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.photo)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
        # Update zoom display
        self.zoom_var.set(f"{int(self.zoom_level * 100)}%")
    
    def draw_spline(self, draw: ImageDraw.Draw, centers: List[Tuple[int, int]], path_id: int,
                   scale_x: float = 1.0, scale_y: float = 1.0):
        """Draw a smooth spline through the centers."""
        if len(centers) < 2:
            return
        
        # Scale centers
        scaled_centers = [(int(c[0] * scale_x), int(c[1] * scale_y)) for c in centers]
        
        # Extract x and y coordinates
        x_coords = [c[0] for c in scaled_centers]
        y_coords = [c[1] for c in scaled_centers]
        
        # Create parameter t from 0 to 1
        t = np.linspace(0, 1, len(scaled_centers))
        
        # Interpolate with cubic spline
        try:
            fx = interp1d(t, x_coords, kind='cubic', bounds_error=False, fill_value='extrapolate')
            fy = interp1d(t, y_coords, kind='cubic', bounds_error=False, fill_value='extrapolate')
            
            # Generate smooth curve
            t_smooth = np.linspace(0, 1, max(50, len(scaled_centers) * 10))
            x_smooth = fx(t_smooth)
            y_smooth = fy(t_smooth)
            
            # Draw the curve
            points = list(zip(x_smooth, y_smooth))
            for i in range(len(points) - 1):
                draw.line([points[i], points[i+1]], fill="yellow", width=3)
        except:
            # Fallback to straight lines if spline fails
            for i in range(len(scaled_centers) - 1):
                draw.line([scaled_centers[i], scaled_centers[i+1]], fill="yellow", width=3)
    
    def draw_arrowhead(self, draw: ImageDraw.Draw, start: Tuple[int, int], 
                      end: Tuple[int, int], color: str):
        """Draw an arrowhead at the end point."""
        # Calculate angle
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        angle = math.atan2(dy, dx)
        
        # Arrowhead size
        size = 8
        angle1 = angle + math.pi - math.pi / 6
        angle2 = angle + math.pi + math.pi / 6
        
        # Arrowhead points
        p1 = (int(end[0] + size * math.cos(angle1)), int(end[1] + size * math.sin(angle1)))
        p2 = (int(end[0] + size * math.cos(angle2)), int(end[1] + size * math.sin(angle2)))
        
        draw.polygon([end, p1, p2], fill=color, outline=color)
    
    def on_click(self, event):
        """Handle left click on canvas."""
        if self.space_pressed:
            self.start_pan(event)
            return
        
        # Get canvas coordinates (account for pan and zoom)
        canvas_x = (self.canvas.canvasx(event.x) - self.pan_x) / self.zoom_level
        canvas_y = (self.canvas.canvasy(event.y) - self.pan_y) / self.zoom_level
        
        # In split view, adjust for image width
        if self.view_mode == 'split':
            if canvas_x > self.image_width:
                return  # Clicked on binary mask side
        
        # Find clicked box
        clicked_idx = -1
        for i, box in enumerate(self.boxes):
            if box.contains_point(int(canvas_x), int(canvas_y)):
                clicked_idx = i
                break
        
        if clicked_idx >= 0:
            self.label_box(clicked_idx)
        else:
            self.selected_box_idx = -1
            self.update_display()
    
    def label_box(self, box_idx: int):
        """Label a box with the next character from ground truth."""
        box = self.boxes[box_idx]
        
        # Get next character
        if self.ground_truth_index < len(self.ground_truth):
            char = self.ground_truth[self.ground_truth_index]
            box.label = char
            box.path_id = self.current_path_id
            box.order_in_path = len(self.paths.get(self.current_path_id, []))
            
            # Add to current path
            if self.current_path_id not in self.paths:
                self.paths[self.current_path_id] = []
            self.paths[self.current_path_id].append(box_idx)
            
            # Calculate angle from previous box
            if box.order_in_path > 0:
                prev_idx = self.paths[self.current_path_id][box.order_in_path - 1]
                prev_box = self.boxes[prev_idx]
                dx = box.center[0] - prev_box.center[0]
                dy = box.center[1] - prev_box.center[1]
                box.angle = math.degrees(math.atan2(dy, dx))
            else:
                # First box in path - default to right
                box.angle = 0.0
            
            # Update previous box angle if it exists
            if box.order_in_path > 0:
                prev_idx = self.paths[self.current_path_id][box.order_in_path - 1]
                prev_box = self.boxes[prev_idx]
                dx = box.center[0] - prev_box.center[0]
                dy = box.center[1] - prev_box.center[1]
                prev_box.angle = math.degrees(math.atan2(dy, dx))
            
            self.ground_truth_index += 1
            self.selected_box_idx = box_idx
            self.status_var.set(f"Labeled box {box_idx} as '{char}'")
        else:
            self.status_var.set("No more characters in ground truth!")
            self.selected_box_idx = box_idx
        
        self.update_next_char()
        self.update_display()
    
    def on_right_click(self, event):
        """Handle right click - rotate selected box."""
        canvas_x = (self.canvas.canvasx(event.x) - self.pan_x) / self.zoom_level
        canvas_y = (self.canvas.canvasy(event.y) - self.pan_y) / self.zoom_level
        
        # Find clicked box
        clicked_idx = -1
        for i, box in enumerate(self.boxes):
            if box.contains_point(int(canvas_x), int(canvas_y)):
                clicked_idx = i
                break
        
        if clicked_idx >= 0:
            self.selected_box_idx = clicked_idx
            # Rotate 90 degrees
            self.boxes[clicked_idx].angle = (self.boxes[clicked_idx].angle + 90) % 360
            self.update_display()
            self.status_var.set(f"Rotated box {clicked_idx}")
    
    def rotate_left(self, event):
        """Rotate selected box counter-clockwise."""
        if self.selected_box_idx >= 0:
            self.boxes[self.selected_box_idx].angle = (self.boxes[self.selected_box_idx].angle - 15) % 360
            self.update_display()
    
    def rotate_right(self, event):
        """Rotate selected box clockwise."""
        if self.selected_box_idx >= 0:
            self.boxes[self.selected_box_idx].angle = (self.boxes[self.selected_box_idx].angle + 15) % 360
            self.update_display()
    
    def restart_path(self, event):
        """Start a new path."""
        self.current_path_id += 1
        self.path_var.set(f"Path {self.current_path_id}")
        self.status_var.set(f"Started new path {self.current_path_id}")
        self.update_display()
    
    def zoom_in(self, event):
        """Zoom in."""
        self.zoom_level = min(self.zoom_level * 1.2, 5.0)
        self.update_display()
    
    def zoom_out(self, event):
        """Zoom out."""
        self.zoom_level = max(self.zoom_level / 1.2, 0.1)
        self.update_display()
    
    def zoom_reset(self, event):
        """Reset zoom to 100%."""
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.update_display()
    
    def toggle_view(self, event):
        """Toggle between single and split view."""
        self.view_mode = 'split' if self.view_mode == 'single' else 'single'
        self.update_display()
        mode_name = "Split (Original + Binary)" if self.view_mode == 'split' else "Single (Original)"
        self.status_var.set(f"View mode: {mode_name}")
    
    def on_mousewheel(self, event):
        """Handle mouse wheel for zooming."""
        if event.delta > 0 or event.num == 4:
            self.zoom_in(event)
        elif event.delta < 0 or event.num == 5:
            self.zoom_out(event)
    
    def start_pan(self, event):
        """Start panning."""
        self.is_panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
    
    def on_pan(self, event):
        """Handle panning."""
        if self.is_panning:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            self.pan_x += dx
            self.pan_y += dy
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self.update_display()
    
    def end_pan(self, event):
        """End panning."""
        self.is_panning = False
    
    def on_space_press(self, event):
        """Handle space key press."""
        self.space_pressed = True
        self.canvas.config(cursor="fleur")
    
    def on_space_release(self, event):
        """Handle space key release."""
        self.space_pressed = False
        self.canvas.config(cursor="crosshair")
        self.is_panning = False
    
    def save_results(self, event=None):
        """Save results to CSV and JSON, and export annotated image."""
        # Choose output directory
        output_dir = Path(self.image_path).parent / "labeled_results"
        output_dir.mkdir(exist_ok=True)
        
        image_name = Path(self.image_path).stem
        
        # Save CSV
        csv_path = output_dir / f"{image_name}_labels.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['box_id', 'x', 'y', 'width', 'height', 'label', 
                           'path_id', 'order_in_path', 'angle', 'center_x', 'center_y'])
            for box in self.boxes:
                writer.writerow([
                    box.box_id, box.original_x, box.original_y, box.original_w, box.original_h, box.label,
                    box.path_id, box.order_in_path, box.angle,
                    box.original_x + box.original_w // 2, box.original_y + box.original_h // 2
                ])
        
        # Save JSON
        json_path = output_dir / f"{image_name}_labels.json"
        data = {
            'image_path': str(self.image_path),
            'ground_truth': self.ground_truth,
            'boxes': [
                {
                    'box_id': box.box_id,
                    'x': box.original_x, 'y': box.original_y, 'width': box.original_w, 'height': box.original_h,
                    'label': box.label,
                    'path_id': box.path_id,
                    'order_in_path': box.order_in_path,
                    'angle': box.angle,
                    'center': (box.original_x + box.original_w // 2, box.original_y + box.original_h // 2)
                }
                for box in self.boxes
            ],
            'paths': {str(k): v for k, v in self.paths.items()}
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        # Save annotated image
        img_annotated = self.display_image.copy()
        draw = ImageDraw.Draw(img_annotated)
        
        # Draw splines
        for path_id, box_indices in self.paths.items():
            if len(box_indices) >= 2:
                centers = [self.boxes[idx].center for idx in box_indices]
                self.draw_spline(draw, centers, path_id)
        
        # Draw boxes and labels
        for box in self.boxes:
            color = "green" if box.label else "red"
            draw.rectangle([box.x, box.y, box.x + box.w, box.y + box.h], 
                          outline=color, width=2)
            if box.label:
                draw.text((box.x + 2, box.y + 2), box.label, fill="black")
            if box.order_in_path >= 0:
                draw.text((box.x + box.w - 15, box.y), str(box.order_in_path + 1), 
                         fill="yellow")
            # Arrow
            if box.angle != 0 or box.order_in_path >= 0:
                start, end = box.get_rotation_arrow()
                draw.line([start, end], fill="yellow", width=2)
                self.draw_arrowhead(draw, start, end, "yellow")
        
        img_path = output_dir / f"{image_name}_annotated.png"
        img_annotated.save(img_path)
        
        messagebox.showinfo("Save Complete", 
                          f"Results saved to:\n{csv_path}\n{json_path}\n{img_path}")
        self.status_var.set("Results saved!")


def get_transliteration_from_db(db_path: str, image_id: str) -> Optional[str]:
    """Get transliteration from database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT transliteration FROM inscriptions WHERE id = ?",
            (image_id,)
        )
        result = cursor.fetchone()
        conn.close()
        if result and result[0]:
            return result[0]
    except Exception as e:
        print(f"Warning: Could not query database: {e}")
    return None


def extract_id_from_filename(filename: str) -> Optional[str]:
    """Extract ID from filename like stone_16820.jpg -> 16820"""
    match = re.search(r'stone_(\d+)', filename)
    return match.group(1) if match else None


def load_detection_results(results_path: str) -> List[Tuple[int, int, int, int]]:
    """Load detected boxes from detection_results.txt file."""
    boxes = []
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Skip header and find detection details
            in_detection = False
            for line in lines:
                if "Detection Details:" in line:
                    in_detection = True
                    continue
                if in_detection and line.strip() and not line.startswith("Box") and not line.startswith("-"):
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            x = int(parts[1])
                            y = int(parts[2])
                            w = int(parts[3])
                            h = int(parts[4])
                            boxes.append((x, y, w, h))
                        except ValueError:
                            continue
    except Exception as e:
        print(f"Warning: Could not load detection results: {e}")
    
    return boxes


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive Safaitic Labeler")
    parser.add_argument("image", help="Path to stone image")
    parser.add_argument("--results", help="Path to detection_results.txt file")
    parser.add_argument("--boxes", nargs='+', type=int, metavar='INT',
                       help="Manual box coordinates: x y width height (can repeat for multiple boxes)")
    parser.add_argument("--db", default="/Users/dansachs/Desktop/Safaitic Inscription Reader/data/safaitic.db",
                       help="Path to SQLite database")
    
    args = parser.parse_args()
    
    # Load boxes
    boxes = []
    if args.results:
        boxes = load_detection_results(args.results)
    elif args.boxes:
        # Parse manual boxes
        if len(args.boxes) % 4 != 0:
            print("Error: Box coordinates must be in groups of 4 (x, y, width, height)")
            return
        for i in range(0, len(args.boxes), 4):
            boxes.append((args.boxes[i], args.boxes[i+1], args.boxes[i+2], args.boxes[i+3]))
    else:
        # Try to find detection results automatically
        image_path = Path(args.image)
        image_name = image_path.stem
        
        # Check in current directory first
        detection_dirs = sorted(Path("detection_results").glob(f"{image_name}_*"), reverse=True)
        if not detection_dirs:
            # Check in image directory
            detection_dirs = sorted(image_path.parent.glob(f"detection_results/{image_name}_*"), reverse=True)
        
        if detection_dirs:
            results_file = detection_dirs[0] / "detection_results.txt"
            if results_file.exists():
                boxes = load_detection_results(str(results_file))
                print(f"Loaded boxes from: {results_file}")
    
    if not boxes:
        print("Warning: No boxes found. You can still use the labeler to manually add boxes.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Get ground truth from database
    image_id = extract_id_from_filename(args.image)
    ground_truth = ""
    if image_id:
        ground_truth = get_transliteration_from_db(args.db, image_id) or ""
        print(f"Loaded ground truth from database: {ground_truth}")
    
    # Create GUI
    root = tk.Tk()
    app = InteractiveLabeler(root, args.image, boxes, ground_truth, args.db)
    root.mainloop()


if __name__ == "__main__":
    main()

