import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from hopfield import HopfieldNetwork

class HopfieldGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive 8x8 Hopfield Network")
        self.root.geometry("800x600")
        
        # Initialize network
        self.network = HopfieldNetwork(64)  # 8x8 = 64 neurons
        self.grid_size = 8
        self.cell_size = 40
        
        # Current pattern (8x8 grid flattened to 64)
        self.current_pattern = np.zeros(64, dtype=int)
        
        # Storage for learned patterns
        self.stored_patterns = []
        
        # Create GUI components
        self.create_widgets()
        self.update_display()
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="8x8 Hopfield Network", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Left panel - Grid canvas
        grid_frame = ttk.LabelFrame(main_frame, text="Pattern Grid", padding="10")
        grid_frame.grid(row=1, column=0, padx=(0, 20), sticky=(tk.N))
        
        self.canvas = tk.Canvas(grid_frame, 
                               width=self.grid_size * self.cell_size,
                               height=self.grid_size * self.cell_size,
                               bg='white', bd=2, relief='sunken')
        self.canvas.grid(row=0, column=0)
        self.canvas.bind("<Button-1>", self.on_cell_click)
        
        # Middle panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=1, padx=10, sticky=(tk.N))
        
        ttk.Button(control_frame, text="Clear Grid", 
                  command=self.clear_grid).grid(row=0, column=0, pady=5, sticky='ew')
        
        ttk.Button(control_frame, text="Load Pattern", 
                  command=self.load_pattern).grid(row=1, column=0, pady=5, sticky='ew')
        
        ttk.Button(control_frame, text="Train Network", 
                  command=self.train_network).grid(row=2, column=0, pady=5, sticky='ew')
        
        ttk.Button(control_frame, text="Recall Pattern", 
                  command=self.recall_pattern).grid(row=3, column=0, pady=5, sticky='ew')
        
        ttk.Separator(control_frame, orient='horizontal').grid(row=4, column=0, 
                                                              sticky='ew', pady=10)
        
        ttk.Button(control_frame, text="Add Noise", 
                  command=self.add_noise).grid(row=5, column=0, pady=5, sticky='ew')
        
        # Right panel - Information
        info_frame = ttk.LabelFrame(main_frame, text="Network Info", padding="10")
        info_frame.grid(row=1, column=2, sticky=(tk.N))
        
        self.info_text = tk.Text(info_frame, width=30, height=20, 
                                wrap=tk.WORD, state='disabled')
        info_scroll = ttk.Scrollbar(info_frame, orient="vertical", 
                                   command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scroll.set)
        
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        info_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Bottom panel - Pattern storage
        storage_frame = ttk.LabelFrame(main_frame, text="Stored Patterns", padding="10")
        storage_frame.grid(row=2, column=0, columnspan=3, pady=(20, 0), sticky='ew')
        
        self.pattern_listbox = tk.Listbox(storage_frame, height=4)
        self.pattern_listbox.grid(row=0, column=0, sticky='ew')
        
        storage_buttons = ttk.Frame(storage_frame)
        storage_buttons.grid(row=0, column=1, padx=(10, 0))
        
        ttk.Button(storage_buttons, text="Load Selected", 
                  command=self.load_selected_pattern).grid(row=0, column=0, pady=2)
        ttk.Button(storage_buttons, text="Delete Selected", 
                  command=self.delete_selected_pattern).grid(row=1, column=0, pady=2)
        
        # Configure column weights
        main_frame.columnconfigure(0, weight=1)
        storage_frame.columnconfigure(0, weight=1)
        
        # Initialize info
        self.update_info("Hopfield Network initialized.\nClick cells to draw patterns.")
    
    def on_cell_click(self, event):
        """Handle mouse clicks on the grid"""
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        
        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            index = row * self.grid_size + col
            # Toggle cell value
            self.current_pattern[index] = 1 - self.current_pattern[index]
            self.update_display()
    
    def update_display(self):
        """Update the visual grid display"""
        self.canvas.delete("all")
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                index = i * self.grid_size + j
                x1, y1 = j * self.cell_size, i * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                
                # White for 0, Black for 1
                color = 'black' if self.current_pattern[index] == 1 else 'white'
                
                self.canvas.create_rectangle(x1, y1, x2, y2, 
                                           fill=color, outline='gray')
    
    def clear_grid(self):
        """Clear the current pattern"""
        self.current_pattern = np.zeros(64, dtype=int)
        self.update_display()
        self.update_info("Grid cleared.")
    
    def load_pattern(self):
        """Load current pattern into storage"""
        if np.sum(self.current_pattern) == 0:
            messagebox.showwarning("Warning", "Cannot load empty pattern!")
            return
        
        pattern_copy = self.current_pattern.copy()
        self.stored_patterns.append(pattern_copy)
        
        # Update listbox
        pattern_name = f"Pattern {len(self.stored_patterns)}"
        self.pattern_listbox.insert(tk.END, pattern_name)
        
        self.update_info(f"Pattern loaded! Total patterns: {len(self.stored_patterns)}")
    
    def train_network(self):
        """Train the Hopfield network with stored patterns"""
        if not self.stored_patterns:
            messagebox.showwarning("Warning", "No patterns to train on!")
            return
        
        self.network.train(self.stored_patterns)
        
        info_text = f"Network trained on {len(self.stored_patterns)} patterns.\n"
        info_text += "Hebbian learning complete.\n"
        info_text += f"Weight matrix: {self.network.weights.shape}"
        
        self.update_info(info_text)
    
    def recall_pattern(self):
        """Recall pattern from current grid state"""
        if len(self.stored_patterns) == 0:
            messagebox.showwarning("Warning", "Train the network first!")
            return
        
        # Recall pattern
        recalled_pattern, history = self.network.recall(self.current_pattern)
        
        # Update display with recalled pattern
        self.current_pattern = recalled_pattern.astype(int)
        self.update_display()
        
        # Show convergence info
        info_text = f"Pattern recall complete!\n"
        info_text += f"Converged in {len(history)-1} iterations.\n"
        info_text += f"Energy: {self.network.energy(recalled_pattern):.3f}\n"
        info_text += f"Final pattern sum: {np.sum(recalled_pattern)}"
        
        self.update_info(info_text)
    
    def add_noise(self):
        """Add random noise to current pattern"""
        if np.sum(self.current_pattern) == 0:
            messagebox.showwarning("Warning", "Draw a pattern first!")
            return
        
        # Flip 10-20% of the bits randomly
        noise_level = 0.15
        num_flips = int(64 * noise_level)
        
        indices = np.random.choice(64, num_flips, replace=False)
        for idx in indices:
            self.current_pattern[idx] = 1 - self.current_pattern[idx]
        
        self.update_display()
        self.update_info(f"Added noise: flipped {num_flips} bits")
    
    def load_selected_pattern(self):
        """Load selected pattern from storage"""
        selection = self.pattern_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Select a pattern first!")
            return
        
        index = selection[0]
        self.current_pattern = self.stored_patterns[index].copy()
        self.update_display()
        self.update_info(f"Loaded pattern {index + 1}")
    
    def delete_selected_pattern(self):
        """Delete selected pattern from storage"""
        selection = self.pattern_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Select a pattern first!")
            return
        
        index = selection[0]
        del self.stored_patterns[index]
        self.pattern_listbox.delete(index)
        
        # Retrain network if patterns remain
        if self.stored_patterns:
            self.network.train(self.stored_patterns)
        
        self.update_info(f"Deleted pattern {index + 1}")
    
    def update_info(self, text):
        """Update the information display"""
        self.info_text.config(state='normal')
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, text)
        self.info_text.config(state='disabled')
