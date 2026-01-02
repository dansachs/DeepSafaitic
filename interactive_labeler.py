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
        self.ground_truth_index = 0  # Current position in ground truth string (non-space chars only)
        self.ground_truth_chars = [c for c in ground_truth if c != ' ']  # Filter out spaces
        
        # Zoom and pan (start with 2x zoom)
        self.zoom_level = 2.0
        self.pan_x = 0
        self.pan_y = 0
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.space_pressed = False
        
        # View mode: 'single' or 'split'
        self.view_mode = 'single'
        
        # Add box mode (default to True)
        self.add_box_mode = True
        self.add_box_start = None
        self.add_box_rect = None
        
        # Generate binary mask
        self.binary_mask = self.create_binary_mask()
        
        # Store original image dimensions
        self.orig_width = self.original_image.width
        self.orig_height = self.original_image.height
        
        # UI setup
        self.setup_ui()
        self.load_image()
        
        # Bind events
        self.setup_bindings()
        
        # Auto-number boxes on startup
        self.auto_number()
        
        # Update status to show add box mode is on by default
        if self.add_box_mode:
            self.status_var.set("Add Box Mode ON (default): Click and drag to create boxes")
        
        self.update_display()
    
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
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.on_right_click)  # Right click
        self.canvas.bind("<ButtonPress-2>", self.start_pan)  # Middle mouse button
        self.canvas.bind("<B2-Motion>", self.on_pan)
        self.canvas.bind("<ButtonRelease-2>", self.end_pan)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)  # macOS/Windows
        self.canvas.bind("<Button-4>", self.on_mousewheel)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mousewheel)  # Linux scroll down
        
        # Keyboard bindings - bind to root window for global access
        self.root.bind_all("<KeyPress-r>", self.restart_current_path)
        self.root.bind_all("<KeyPress-R>", self.restart_current_path)
        self.root.bind_all("<KeyPress-n>", self.new_path)
        self.root.bind_all("<KeyPress-N>", self.new_path)
        self.root.bind_all("<KeyPress-s>", self.save_results)
        self.root.bind_all("<KeyPress-S>", self.save_results)
        self.root.bind_all("<KeyPress-Left>", self.rotate_left)
        self.root.bind_all("<KeyPress-Right>", self.rotate_right)
        self.root.bind_all("<KeyPress-Up>", self.move_box_up)
        self.root.bind_all("<KeyPress-Down>", self.move_box_down)
        self.root.bind_all("<KeyPress-u>", self.auto_number)
        self.root.bind_all("<KeyPress-U>", self.auto_number)
        self.root.bind_all("<KeyPress-p>", self.create_path)
        self.root.bind_all("<KeyPress-P>", self.create_path)
        self.root.bind_all("<KeyPress-plus>", self.zoom_in)
        self.root.bind_all("<KeyPress-equal>", self.zoom_in)  # + without shift
        self.root.bind_all("<KeyPress-minus>", self.zoom_out)
        self.root.bind_all("<KeyPress-0>", self.zoom_reset)
        self.root.bind_all("<KeyPress-v>", self.toggle_view)
        self.root.bind_all("<KeyPress-V>", self.toggle_view)
        self.root.bind_all("<KeyPress-space>", self.on_space_press)
        self.root.bind_all("<KeyRelease-space>", self.on_space_release)
        self.root.bind_all("<KeyPress-Delete>", self.delete_selected_box)
        self.root.bind_all("<KeyPress-BackSpace>", self.delete_selected_box)
        self.root.bind_all("<KeyPress-a>", self.toggle_add_mode)
        self.root.bind_all("<KeyPress-A>", self.toggle_add_mode)
        self.root.bind_all("<KeyPress-Escape>", self.cancel_add_mode)
        
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
        self.zoom_var = tk.StringVar(value="200%")
        ttk.Label(control_frame, textvariable=self.zoom_var).grid(row=1, column=3, padx=5)
        
        # Buttons - Row 1
        button_frame1 = ttk.Frame(control_frame)
        button_frame1.grid(row=2, column=0, columnspan=4, pady=5)
        
        ttk.Button(button_frame1, text="New Path (N)", command=self.new_path_btn).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame1, text="Restart Current Path (R)", command=self.restart_current_path_btn).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame1, text="Toggle Add Box (A)", command=self.toggle_add_mode).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame1, text="Delete Box (Del)", command=self.delete_selected_box).pack(side=tk.LEFT, padx=5)
        
        # Buttons - Row 2 (Numbering controls)
        button_frame2 = ttk.Frame(control_frame)
        button_frame2.grid(row=3, column=0, columnspan=4, pady=5)
        
        ttk.Button(button_frame2, text="Auto Number (U)", command=self.auto_number_btn).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame2, text="Create Path (P)", command=self.create_path_btn).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame2, text="Save Results (S)", command=self.save_results).pack(side=tk.LEFT, padx=5)
        
        # Status
        ttk.Label(control_frame, text="Status:").grid(row=3, column=0, padx=5)
        self.status_var = tk.StringVar(value="Ready - Click boxes to label")
        ttk.Label(control_frame, textvariable=self.status_var).grid(row=3, column=1, padx=5, columnspan=3, sticky=tk.W)
        
        # Instructions
        instructions = (
            "R=Restart path | Del=Delete box | A=Add box mode | S=Save | ←/→=Rotate | "
            "+/-=Zoom | 0=Reset zoom | V=Toggle view | Space+Drag=Pan | Middle mouse=Pan | Scroll=Zoom"
        )
        ttk.Label(control_frame, text=instructions, font=("Arial", 8)).grid(
            row=4, column=0, columnspan=4, pady=5
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
        # Filter out spaces for indexing
        self.ground_truth_chars = [c for c in self.ground_truth if c != ' ']
        self.ground_truth_index = min(self.ground_truth_index, len(self.ground_truth_chars))
        self.update_next_char()
    
    def update_next_char(self):
        """Update the next character display."""
        if self.ground_truth_index < len(self.ground_truth_chars):
            char = self.ground_truth_chars[self.ground_truth_index]
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
        
        # Draw temporary add box rectangle
        if self.add_box_mode and self.add_box_rect:
            x, y, w, h = self.add_box_rect
            bx = int(x * zoom_scale_x)
            by = int(y * zoom_scale_y)
            bw = int(w * zoom_scale_x)
            bh = int(h * zoom_scale_y)
            # Draw with dashed effect
            for i in range(0, max(bw, bh), 5):
                if i < bw:
                    draw.line([(bx + i, by), (bx + min(i + 3, bw), by)], fill="blue", width=2)
                    draw.line([(bx + i, by + bh), (bx + min(i + 3, bw), by + bh)], fill="blue", width=2)
                if i < bh:
                    draw.line([(bx, by + i), (bx, by + min(i + 3, bh))], fill="blue", width=2)
                    draw.line([(bx + bw, by + i), (bx + bw, by + min(i + 3, bh))], fill="blue", width=2)
        
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
            
            # Draw order number (show numbering even if not in path yet)
            if box.order_in_path >= 0:
                order_text = str(box.order_in_path + 1)
                # Use different color for numbering mode vs path mode
                num_color = "orange" if self.numbering_mode and idx in self.numbered_boxes else "yellow"
                draw.text((bx + bw - 15, by), order_text, fill=num_color, font=None)
            
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
        """Draw a straighter line through the centers, accounting for vertical positioning."""
        if len(centers) < 2:
            return
        
        # Scale centers
        scaled_centers = [(int(c[0] * scale_x), int(c[1] * scale_y)) for c in centers]
        
        # Use piecewise linear interpolation for straighter lines
        # Group points by similar Y coordinates to handle vertical positioning
        if len(scaled_centers) == 2:
            # Simple straight line for 2 points
            draw.line([scaled_centers[0], scaled_centers[1]], fill="yellow", width=3)
        else:
            # For multiple points, use piecewise linear with Y-grouping
            y_threshold = 20  # Pixels - boxes on same line
            groups = []
            current_group = [scaled_centers[0]]
            
            for i in range(1, len(scaled_centers)):
                prev_y = scaled_centers[i-1][1]
                curr_y = scaled_centers[i][1]
                if abs(curr_y - prev_y) < y_threshold:
                    current_group.append(scaled_centers[i])
                else:
                    groups.append(current_group)
                    current_group = [scaled_centers[i]]
            groups.append(current_group)
            
            # Draw lines within groups (straight horizontal-ish lines)
            for group in groups:
                if len(group) > 1:
                    # Draw straight line through group
                    for i in range(len(group) - 1):
                        draw.line([group[i], group[i+1]], fill="yellow", width=3)
            
            # Connect groups with straight lines (vertical transitions)
            for i in range(len(groups) - 1):
                if groups[i] and groups[i+1]:
                    draw.line([groups[i][-1], groups[i+1][0]], fill="yellow", width=3)
    
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
        
        # Add box mode - start drawing
        if self.add_box_mode:
            self.add_box_start = (int(canvas_x), int(canvas_y))
            self.add_box_rect = None
            self.status_var.set("Drag to create box...")
            return
        
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
    
    def on_drag(self, event):
        """Handle mouse drag (for adding boxes)."""
        if not self.add_box_mode or self.add_box_start is None:
            return
        
        # Get canvas coordinates
        canvas_x = (self.canvas.canvasx(event.x) - self.pan_x) / self.zoom_level
        canvas_y = (self.canvas.canvasy(event.y) - self.pan_y) / self.zoom_level
        
        # Update rectangle
        x1, y1 = self.add_box_start
        x2, y2 = int(canvas_x), int(canvas_y)
        
        # Ensure valid rectangle
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        self.add_box_rect = (x_min, y_min, x_max - x_min, y_max - y_min)
        self.update_display()
    
    def on_release(self, event):
        """Handle mouse release (finish adding box)."""
        if not self.add_box_mode or self.add_box_start is None:
            return
        
        if self.add_box_rect:
            x, y, w, h = self.add_box_rect
            # Only add if box is large enough
            if w > 5 and h > 5:
                new_box = Box(x, y, w, h, len(self.boxes))
                self.boxes.append(new_box)
                self.selected_box_idx = len(self.boxes) - 1
                self.status_var.set(f"Added new box {new_box.box_id}")
            else:
                self.status_var.set("Box too small, not added")
        
        self.add_box_start = None
        self.add_box_rect = None
        self.update_display()
    
    def label_box(self, box_idx: int):
        """Label a box with the next character from ground truth."""
        box = self.boxes[box_idx]
        
        # Get next character (skip spaces)
        if self.ground_truth_index < len(self.ground_truth_chars):
            char = self.ground_truth_chars[self.ground_truth_index]
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
        
        # Re-predict order for remaining unlabeled boxes
        self.predict_order_and_direction()
        
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
    
    def new_path(self, event=None):
        """Start a completely new path (increments path ID)."""
        self.current_path_id += 1
        self.path_var.set(f"Path {self.current_path_id}")
        self.status_var.set(f"Started new path {self.current_path_id}")
        self.update_display()
    
    def new_path_btn(self):
        """Button callback for new path."""
        self.new_path()
    
    def restart_current_path(self, event=None):
        """Restart the current path (clears boxes from current path, keeps same path ID)."""
        if self.current_path_id in self.paths:
            # Clear order and path info for boxes in current path
            for box_idx in self.paths[self.current_path_id]:
                if box_idx < len(self.boxes):
                    box = self.boxes[box_idx]
                    if box.path_id == self.current_path_id:
                        box.order_in_path = -1
                        box.angle = 0.0
                        # Keep the label if it exists, just reset order
        
        # Clear the current path
        self.paths[self.current_path_id] = []
        self.status_var.set(f"Restarted path {self.current_path_id} - boxes cleared from path")
        self.update_display()
    
    def restart_current_path_btn(self):
        """Button callback for restart current path."""
        self.restart_current_path()
    
    def zoom_in(self, event):
        """Zoom in."""
        self.zoom_level = min(self.zoom_level * 1.2, 5.0)
        self.update_display()
    
    def zoom_out(self, event):
        """Zoom out."""
        self.zoom_level = max(self.zoom_level / 1.2, 0.1)
        self.update_display()
    
    def zoom_reset(self, event):
        """Reset zoom to 200% (default)."""
        self.zoom_level = 2.0
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
        if not self.add_box_mode:
            self.canvas.config(cursor="crosshair")
        self.is_panning = False
    
    def delete_selected_box(self, event=None):
        """Delete the currently selected box."""
        if self.selected_box_idx >= 0 and self.selected_box_idx < len(self.boxes):
            deleted_box = self.boxes.pop(self.selected_box_idx)
            
            # Remove from paths
            for path_id, box_indices in list(self.paths.items()):
                if self.selected_box_idx in box_indices:
                    box_indices.remove(self.selected_box_idx)
                    # Reorder remaining boxes in path
                    for i, idx in enumerate(box_indices):
                        if idx > self.selected_box_idx:
                            box_indices[i] = idx - 1
                    # Update order_in_path for remaining boxes
                    for i, idx in enumerate(box_indices):
                        self.boxes[idx].order_in_path = i
                # Update indices that are greater than deleted index
                self.paths[path_id] = [idx - 1 if idx > self.selected_box_idx else idx 
                                      for idx in box_indices]
            
            # Update box IDs
            for i, box in enumerate(self.boxes):
                box.box_id = i
            
            self.selected_box_idx = -1
            self.status_var.set(f"Deleted box {deleted_box.box_id}")
            self.update_display()
        else:
            self.status_var.set("No box selected to delete")
    
    def toggle_add_mode(self, event=None):
        """Toggle add box mode."""
        self.add_box_mode = not self.add_box_mode
        if self.add_box_mode:
            self.canvas.config(cursor="crosshair")
            self.status_var.set("Add Box Mode: Click and drag to create a box (ESC to cancel)")
        else:
            self.canvas.config(cursor="crosshair")
            self.add_box_start = None
            self.add_box_rect = None
            self.status_var.set("Add Box Mode OFF")
        self.update_display()
    
    def cancel_add_mode(self, event=None):
        """Cancel add box mode."""
        if self.add_box_mode:
            self.add_box_mode = False
            self.add_box_start = None
            self.add_box_rect = None
            self.canvas.config(cursor="crosshair")
            self.status_var.set("Add Box Mode cancelled")
            self.update_display()
    
    def predict_order_and_direction(self):
        """Predict reading order and direction for unlabeled boxes."""
        # Get unlabeled boxes
        unlabeled_boxes = [(i, box) for i, box in enumerate(self.boxes) if not box.label]
        
        if len(unlabeled_boxes) == 0:
            return
        
        # Simple heuristic: sort by position (left-to-right, top-to-bottom)
        def sort_key(item):
            idx, box = item
            cx, cy = box.center
            # Primary: X coordinate (left to right)
            # Secondary: Y coordinate (top to bottom)
            return (cx, cy)
        
        sorted_boxes = sorted(unlabeled_boxes, key=sort_key)
        
        # Assign predicted order to unlabeled boxes
        for order, (idx, box) in enumerate(sorted_boxes):
            if not box.label:  # Only update if still unlabeled
                # Calculate direction from previous box
                if order > 0:
                    prev_idx, prev_box = sorted_boxes[order - 1]
                    dx = box.center[0] - prev_box.center[0]
                    dy = box.center[1] - prev_box.center[1]
                    box.angle = math.degrees(math.atan2(dy, dx))
                else:
                    # First box - default to right
                    box.angle = 0.0
        
        # Update paths with predicted order (only for unlabeled boxes)
        if self.current_path_id not in self.paths:
            self.paths[self.current_path_id] = []
        
        # Add unlabeled boxes to current path in predicted order
        for idx, _ in sorted_boxes:
            if idx not in self.paths[self.current_path_id]:
                self.paths[self.current_path_id].append(idx)
                self.boxes[idx].order_in_path = len(self.paths[self.current_path_id]) - 1
                self.boxes[idx].path_id = self.current_path_id
    
    def auto_number(self, event=None):
        """Automatically number all boxes based on position."""
        if len(self.boxes) == 0:
            self.status_var.set("No boxes to number")
            return
        
        # Get all boxes (or unlabeled boxes if some are labeled)
        boxes_to_number = [(i, box) for i, box in enumerate(self.boxes) if not box.label]
        if len(boxes_to_number) == 0:
            # If all boxes are labeled, number all of them
            boxes_to_number = [(i, box) for i, box in enumerate(self.boxes)]
        
        # Sort by position (left-to-right, top-to-bottom)
        def sort_key(item):
            idx, box = item
            cx, cy = box.center
            return (cx, cy)
        
        sorted_boxes = sorted(boxes_to_number, key=sort_key)
        
        # Store numbered order
        self.numbered_boxes = [idx for idx, _ in sorted_boxes]
        
        # Assign temporary order numbers (not path order yet)
        for order, (idx, box) in enumerate(sorted_boxes):
            # Use a temporary order field (we'll use order_in_path temporarily)
            box.order_in_path = order
            # Calculate direction
            if order > 0:
                prev_idx, prev_box = sorted_boxes[order - 1]
                dx = box.center[0] - prev_box.center[0]
                dy = box.center[1] - prev_box.center[1]
                box.angle = math.degrees(math.atan2(dy, dx))
            else:
                box.angle = 0.0
        
        self.numbering_mode = True
        self.status_var.set(f"Auto-numbered {len(self.numbered_boxes)} boxes. Press P to create path or ↑↓ to reorder.")
        self.update_display()
    
    def auto_number_btn(self):
        """Button callback for auto number."""
        self.auto_number()
    
    def move_box_up(self, event=None):
        """Move selected box up in numbering order."""
        if self.selected_box_idx < 0 or not self.numbering_mode:
            return
        
        if self.selected_box_idx not in self.numbered_boxes:
            return
        
        current_pos = self.numbered_boxes.index(self.selected_box_idx)
        if current_pos > 0:
            # Swap with previous box
            self.numbered_boxes[current_pos], self.numbered_boxes[current_pos - 1] = \
                self.numbered_boxes[current_pos - 1], self.numbered_boxes[current_pos]
            
            # Update order numbers
            for order, idx in enumerate(self.numbered_boxes):
                self.boxes[idx].order_in_path = order
                # Update angle
                if order > 0:
                    prev_idx = self.numbered_boxes[order - 1]
                    prev_box = self.boxes[prev_idx]
                    box = self.boxes[idx]
                    dx = box.center[0] - prev_box.center[0]
                    dy = box.center[1] - prev_box.center[1]
                    box.angle = math.degrees(math.atan2(dy, dx))
            
            self.status_var.set(f"Moved box {self.selected_box_idx} up in order")
            self.update_display()
    
    def move_box_down(self, event=None):
        """Move selected box down in numbering order."""
        if self.selected_box_idx < 0 or not self.numbering_mode:
            return
        
        if self.selected_box_idx not in self.numbered_boxes:
            return
        
        current_pos = self.numbered_boxes.index(self.selected_box_idx)
        if current_pos < len(self.numbered_boxes) - 1:
            # Swap with next box
            self.numbered_boxes[current_pos], self.numbered_boxes[current_pos + 1] = \
                self.numbered_boxes[current_pos + 1], self.numbered_boxes[current_pos]
            
            # Update order numbers
            for order, idx in enumerate(self.numbered_boxes):
                self.boxes[idx].order_in_path = order
                # Update angle
                if order > 0:
                    prev_idx = self.numbered_boxes[order - 1]
                    prev_box = self.boxes[prev_idx]
                    box = self.boxes[idx]
                    dx = box.center[0] - prev_box.center[0]
                    dy = box.center[1] - prev_box.center[1]
                    box.angle = math.degrees(math.atan2(dy, dx))
            
            self.status_var.set(f"Moved box {self.selected_box_idx} down in order")
            self.update_display()
    
    def create_path(self, event=None):
        """Create path from numbered boxes."""
        if not self.numbering_mode or len(self.numbered_boxes) == 0:
            self.status_var.set("No numbered boxes. Press U to auto-number first.")
            return
        
        # Clear current path
        if self.current_path_id in self.paths:
            self.paths[self.current_path_id] = []
        
        # Add numbered boxes to current path
        self.paths[self.current_path_id] = self.numbered_boxes.copy()
        
        # Update box properties
        for order, idx in enumerate(self.numbered_boxes):
            box = self.boxes[idx]
            box.path_id = self.current_path_id
            box.order_in_path = order
            # Update angle
            if order > 0:
                prev_idx = self.numbered_boxes[order - 1]
                prev_box = self.boxes[prev_idx]
                dx = box.center[0] - prev_box.center[0]
                dy = box.center[1] - prev_box.center[1]
                box.angle = math.degrees(math.atan2(dy, dx))
        
        self.numbering_mode = False
        self.status_var.set(f"Created path {self.current_path_id} with {len(self.numbered_boxes)} boxes")
        self.update_display()
    
    def create_path_btn(self):
        """Button callback for create path."""
        self.create_path()
    
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

