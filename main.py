import sys
import time
import sqlite3
import os
import psutil
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTextEdit, QLabel, 
                             QListWidget, QSplitter, QTabWidget, QGroupBox,
                             QProgressBar, QMessageBox, QRadioButton, QButtonGroup,
                             QFileDialog, QMenu)
from PyQt6.QtGui import QTextCharFormat, QColor, QFont, QTextCursor, QAction
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
import language_tool_python

# Flag to check if transformers is available
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
    print("✓ Transformers library detected")
except ImportError:
    print("⚠ Transformers not installed (will be tested when installed)")

# Optional: python-docx for .docx support
DOCX_AVAILABLE = False
try:
    from docx import Document
    DOCX_AVAILABLE = True
    print("✓ python-docx detected")
except ImportError:
    print("⚠ python-docx not installed (install for .docx support)")

class ModelLoaderThread(QThread):
    """Background thread for loading heavy models"""
    finished = pyqtSignal(str, float, float)  # message, time_taken, memory_mb
    progress = pyqtSignal(str)
    
    def __init__(self, quantized=False):
        super().__init__()
        self.quantized = quantized
    
    def run(self):
        """Load Flan-T5 model in background"""
        try:
            model_type = "quantized" if self.quantized else "standard"
            self.progress.emit(f"Loading {model_type} Flan-T5 model...")
            
            # Measure memory before
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            start_time = time.time()
            
            # Load instruction-tuned model
            generator = pipeline('text2text-generation', model='google/flan-t5-base')
            
            # Apply quantization if requested
            if self.quantized and TRANSFORMERS_AVAILABLE:
                self.progress.emit("Applying dynamic quantization...")
                generator.model = torch.quantization.quantize_dynamic(
                    generator.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            
            load_time = time.time() - start_time
            
            # Measure memory after
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = mem_after - mem_before
            
            self.progress.emit("Testing text generation...")
            
            # Test generation
            test_output = generator("paraphrase: The weather is nice today")
            
            message = f"✓ Model loaded successfully!\n\nLoad time: {load_time:.2f}s\nMemory used: {memory_used:.1f}MB\n\nTest output: {test_output[0]['generated_text']}"
            self.finished.emit(message, load_time, memory_used)
            
        except Exception as e:
            self.finished.emit(f"✗ Error loading model: {str(e)}", 0.0, 0.0)

class TextGeneratorThread(QThread):
    """Background thread for text generation to keep UI responsive"""
    finished = pyqtSignal(str, float)
    progress = pyqtSignal(str)
    
    def __init__(self, generator, prompt, max_length, task_type="generate"):
        super().__init__()
        self.generator = generator
        self.prompt = prompt
        self.max_length = max_length
        self.task_type = task_type
    
    def run(self):
        """Generate text in background using Flan-T5"""
        try:
            start_time = time.time()
            
            # Flan-T5 task-specific prompts
            if self.task_type == "paraphrase":
                self.progress.emit("Paraphrasing entire document...")
                prompt = f"paraphrase: {self.prompt}"
            elif self.task_type == "rewrite":
                self.progress.emit("Rewriting entire document...")
                prompt = f"grammar: {self.prompt}"
            elif self.task_type == "formal":
                self.progress.emit("Converting entire document to formal tone...")
                prompt = f"Rewrite in formal business language: {self.prompt}"
            elif self.task_type == "casual":
                self.progress.emit("Converting entire document to casual tone...")
                prompt = f"Rewrite in casual friendly language: {self.prompt}"
            elif self.task_type == "technical":
                self.progress.emit("Converting entire document to technical style...")
                prompt = f"Rewrite using technical and scientific terminology: {self.prompt}"
            elif self.task_type == "simple":
                self.progress.emit("Simplifying entire document...")
                prompt = f"Rewrite this in simple words: {self.prompt}"
            else:
                self.progress.emit("Generating text...")
                prompt = f"continue: {self.prompt}"
            
            # Flan-T5 generation
            result = self.generator(
                prompt,
                max_length=512,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
            generated_text = result[0]['generated_text']
            gen_time = time.time() - start_time
            
            self.finished.emit(generated_text, gen_time)
            
        except Exception as e:
            self.finished.emit(f"Error: {str(e)}", 0.0)

class QuillRnD(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QUILL R&D - Technology Validation")
        self.setGeometry(100, 100, 1200, 700)
        
        # Initialize metrics
        self.metrics = {
            'languagetool_load_time': 0.0,
            'model_load_time': 0.0,
            'model_memory_mb': 0.0,
            'quantized_load_time': 0.0,
            'quantized_memory_mb': 0.0,
            'check_times': [],
            'database_operations': [],
            'generation_times': [],
            'documents_processed': 0
        }
        
        # Store grammar matches for right-click menu
        self.grammar_matches = []
        self.current_file_path = None
        
        # Text generation model
        self.text_generator = None
        self.model_loaded = False
        
        # Debounce timer for real-time checking
        self.grammar_check_timer = QTimer()
        self.grammar_check_timer.setSingleShot(True)
        self.grammar_check_timer.timeout.connect(self.auto_check_grammar)
        
        # Initialize database
        self.init_database()
        
        # Initialize LanguageTool
        print("="*60)
        print("INITIALIZING TECHNOLOGIES...")
        print("="*60)
        print("\n1. Loading LanguageTool...")
        start_time = time.time()
        self.grammar_tool = language_tool_python.LanguageTool('en-US')
        load_time = time.time() - start_time
        print(f"   ✓ LanguageTool loaded in {load_time:.2f} seconds")
        self.metrics['languagetool_load_time'] = load_time
        
        self.init_ui()
        
        print("\n" + "="*60)
        print("APPLICATION READY - All core systems initialized")
        print("="*60)
    
    def init_database(self):
        """Initialize SQLite database"""
        print("\n2. Setting up SQLite database...")
        start_time = time.time()
        
        self.db_path = "quill_rnd.db"
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Create tables
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS preferences (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT,
                metric_value REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # NEW: Documents table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                content TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
        
        db_time = time.time() - start_time
        print(f"   ✓ SQLite database created in {db_time:.4f} seconds")
        print(f"   ✓ Database location: {os.path.abspath(self.db_path)}")
        
        self.metrics['database_operations'].append(('init', db_time))
    
    def save_preference(self, key, value):
        """Save a preference to database"""
        start_time = time.time()
        self.cursor.execute('INSERT OR REPLACE INTO preferences (key, value) VALUES (?, ?)', 
                          (key, value))
        self.conn.commit()
        save_time = time.time() - start_time
        self.metrics['database_operations'].append(('save', save_time))
        return save_time
    
    def load_preference(self, key, default=""):
        """Load a preference from database"""
        start_time = time.time()
        self.cursor.execute('SELECT value FROM preferences WHERE key = ?', (key,))
        result = self.cursor.fetchone()
        load_time = time.time() - start_time
        self.metrics['database_operations'].append(('load', load_time))
        return result[0] if result else default
    
    def save_document_to_db(self, filename, content):
        """Save document to database"""
        start_time = time.time()
        self.cursor.execute('''
            INSERT INTO documents (filename, content) VALUES (?, ?)
        ''', (filename, content))
        self.conn.commit()
        save_time = time.time() - start_time
        self.metrics['database_operations'].append(('save_document', save_time))
        self.metrics['documents_processed'] += 1
        print(f"✓ Document saved to database in {save_time*1000:.2f}ms")
        return save_time
    
    def load_text_generator(self, quantized=False):
        """Load text generation model (lazy loading)"""
        if not TRANSFORMERS_AVAILABLE:
            QMessageBox.warning(self, "Model Not Available", 
                              "Transformers library not installed.\n\nRun: pip install transformers torch")
            return False
        
        if self.model_loaded:
            return True
        
        try:
            model_type = "quantized" if quantized else "standard"
            print(f"\nLoading {model_type} Flan-T5 model for first use...")
            
            # Measure memory before
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024
            
            start_time = time.time()
            
            # Load model
            self.text_generator = pipeline('text2text-generation', model='google/flan-t5-base')
            
            # Apply quantization if requested
            if quantized:
                print("Applying dynamic quantization...")
                self.text_generator.model = torch.quantization.quantize_dynamic(
                    self.text_generator.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            
            load_time = time.time() - start_time
            
            # Measure memory after
            mem_after = process.memory_info().rss / 1024 / 1024
            memory_used = mem_after - mem_before
            
            if quantized:
                self.metrics['quantized_load_time'] = load_time
                self.metrics['quantized_memory_mb'] = memory_used
            else:
                self.metrics['model_load_time'] = load_time
                self.metrics['model_memory_mb'] = memory_used
            
            self.model_loaded = True
            
            print(f"✓ {model_type.capitalize()} Flan-T5 loaded in {load_time:.2f}s ({memory_used:.1f}MB)")
            
            # Save to database
            self.cursor.execute(
                'INSERT INTO performance_metrics (metric_name, metric_value) VALUES (?, ?)',
                (f'{model_type}_load_time', load_time)
            )
            self.cursor.execute(
                'INSERT INTO performance_metrics (metric_name, metric_value) VALUES (?, ?)',
                (f'{model_type}_memory_mb', memory_used)
            )
            self.conn.commit()
            
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")
            return False
    
    def init_ui(self):
        """Create the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Title
        title = QLabel("QUILL R&D - Complete Technology Validation")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        
        # Create tabs
        tabs = QTabWidget()
        
        tabs.addTab(self.create_grammar_tab(), "Writing Assistant")
        tabs.addTab(self.create_model_tab(), "AI Model Testing")
        tabs.addTab(self.create_database_tab(), "Database Testing")
        tabs.addTab(self.create_metrics_tab(), "Performance Metrics")
        
        main_layout.addWidget(tabs)
    
    def create_grammar_tab(self):
        """Tab 1: Complete Writing Assistant"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 5, 10, 10)
        layout.setSpacing(5)
        widget.setLayout(layout)
        
        # File operations
        file_btn_layout = QHBoxLayout()
        
        open_btn = QPushButton("📂 Open File")
        open_btn.clicked.connect(self.open_file)
        open_btn.setStyleSheet("background-color: #607D8B; color: white; padding: 8px;")
        file_btn_layout.addWidget(open_btn)
        
        save_btn = QPushButton("💾 Save File")
        save_btn.clicked.connect(self.save_file)
        save_btn.setStyleSheet("background-color: #607D8B; color: white; padding: 8px;")
        file_btn_layout.addWidget(save_btn)
        
        save_db_btn = QPushButton("📥 Save to Database")
        save_db_btn.clicked.connect(self.save_to_database)
        save_db_btn.setStyleSheet("background-color: #607D8B; color: white; padding: 8px;")
        file_btn_layout.addWidget(save_db_btn)
        
        file_btn_layout.addStretch()
        layout.addLayout(file_btn_layout)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Editor
        editor_widget = QWidget()
        editor_layout = QVBoxLayout()
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(8)
        editor_widget.setLayout(editor_layout)
        
        # Text editor with context menu
        self.text_editor = QTextEdit()
        self.text_editor.setPlaceholderText("Type or paste text here...\n\nReal-time grammar checking enabled!\nRight-click on underlined errors to fix them.")
        self.text_editor.setFont(QFont("Arial", 12))
        self.text_editor.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.text_editor.customContextMenuRequested.connect(self.show_context_menu)
        self.text_editor.textChanged.connect(self.on_text_changed)
        editor_layout.addWidget(self.text_editor)
        
        # Real-time toggle
        realtime_layout = QHBoxLayout()
        self.realtime_check = QRadioButton("Real-time Grammar Checking")
        self.realtime_check.setChecked(True)
        self.realtime_check.toggled.connect(self.toggle_realtime_checking)
        realtime_layout.addWidget(self.realtime_check)
        realtime_layout.addStretch()
        editor_layout.addLayout(realtime_layout)
        
        # Grammar Check Button
        self.check_button = QPushButton("Check Grammar")
        self.check_button.setFont(QFont("Arial", 10))
        self.check_button.clicked.connect(self.check_grammar)
        self.check_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        editor_layout.addWidget(self.check_button)
        
        # Text Generation Section
        gen_label = QLabel("Text Generation (processes entire document):")
        gen_label.setStyleSheet("font-weight: bold; color: #2196F3; margin-top: 5px;")
        editor_layout.addWidget(gen_label)
        
        gen_btn_layout = QHBoxLayout()
        gen_btn_layout.setSpacing(5)
        
        self.continue_btn = QPushButton("Continue Writing")
        self.continue_btn.setFont(QFont("Arial", 9))
        self.continue_btn.clicked.connect(self.continue_writing)
        self.continue_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        gen_btn_layout.addWidget(self.continue_btn)
        
        self.paraphrase_btn = QPushButton("Paraphrase All")
        self.paraphrase_btn.setFont(QFont("Arial", 9))
        self.paraphrase_btn.clicked.connect(self.paraphrase_text)
        self.paraphrase_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        gen_btn_layout.addWidget(self.paraphrase_btn)
        
        self.rewrite_btn = QPushButton("Rewrite All")
        self.rewrite_btn.setFont(QFont("Arial", 9))
        self.rewrite_btn.clicked.connect(self.rewrite_text)
        self.rewrite_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        gen_btn_layout.addWidget(self.rewrite_btn)
        
        editor_layout.addLayout(gen_btn_layout)
        
        # Tone Adjustment Section
        tone_label = QLabel("Tone Adjustment (processes entire document):")
        tone_label.setStyleSheet("font-weight: bold; color: #E91E63; margin-top: 5px;")
        editor_layout.addWidget(tone_label)
        
        tone_row1 = QHBoxLayout()
        tone_row1.setSpacing(10)
        self.tone_formal = QRadioButton("Formal")
        self.tone_casual = QRadioButton("Casual")
        self.tone_formal.setChecked(True)
        tone_row1.addWidget(self.tone_formal)
        tone_row1.addWidget(self.tone_casual)
        tone_row1.addStretch()
        editor_layout.addLayout(tone_row1)
        
        tone_row2 = QHBoxLayout()
        tone_row2.setSpacing(10)
        self.tone_technical = QRadioButton("Technical")
        self.tone_simple = QRadioButton("Simple")
        tone_row2.addWidget(self.tone_technical)
        tone_row2.addWidget(self.tone_simple)
        tone_row2.addStretch()
        editor_layout.addLayout(tone_row2)
        
        self.tone_group = QButtonGroup()
        self.tone_group.addButton(self.tone_formal)
        self.tone_group.addButton(self.tone_casual)
        self.tone_group.addButton(self.tone_technical)
        self.tone_group.addButton(self.tone_simple)
        
        self.transform_btn = QPushButton("Transform All Text")
        self.transform_btn.setFont(QFont("Arial", 10))
        self.transform_btn.clicked.connect(self.transform_selection)
        self.transform_btn.setStyleSheet("""
            QPushButton {
                background-color: #E91E63;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #C2185B;
            }
        """)
        editor_layout.addWidget(self.transform_btn)
        
        # Progress bar
        self.gen_progress = QProgressBar()
        self.gen_progress.setRange(0, 0)
        self.gen_progress.hide()
        editor_layout.addWidget(self.gen_progress)
        
        # Status
        self.gen_status = QLabel("")
        self.gen_status.setStyleSheet("color: #2196F3; font-style: italic; font-size: 10px;")
        editor_layout.addWidget(self.gen_status)
        
        splitter.addWidget(editor_widget)
        
        # Right: Errors
        error_widget = QWidget()
        error_layout = QVBoxLayout()
        error_layout.setContentsMargins(0, 0, 0, 0)
        error_layout.setSpacing(5)
        error_widget.setLayout(error_layout)
        
        error_label = QLabel("Grammar Errors & Suggestions:")
        error_layout.addWidget(error_label)
        
        self.error_list = QListWidget()
        error_layout.addWidget(self.error_list)
        
        self.grammar_metrics_label = QLabel(f"Load time: {self.metrics['languagetool_load_time']:.2f}s")
        self.grammar_metrics_label.setStyleSheet("color: gray; font-size: 10px;")
        error_layout.addWidget(self.grammar_metrics_label)
        
        splitter.addWidget(error_widget)
        splitter.setSizes([650, 350])
        
        layout.addWidget(splitter)
        
        return widget
    
    def create_model_tab(self):
        """Tab 2: AI Model loading and testing"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        if not TRANSFORMERS_AVAILABLE:
            warning_label = QLabel("⚠ Transformers library not installed")
            warning_label.setStyleSheet("color: orange; font-weight: bold; font-size: 14px;")
            layout.addWidget(warning_label)
            
            install_info = QLabel("To test this technology, run:\npip install transformers torch")
            install_info.setStyleSheet("color: gray; font-style: italic;")
            layout.addWidget(install_info)
            
            layout.addStretch()
            return widget
        
        info_label = QLabel("✓ Technology Available: Transformers + PyTorch")
        info_label.setStyleSheet("color: green; font-weight: bold;")
        layout.addWidget(info_label)
        
        desc_label = QLabel("This tests loading Flan-T5 with and without quantization")
        layout.addWidget(desc_label)
        
        # Progress bar
        self.model_progress = QProgressBar()
        self.model_progress.setRange(0, 0)
        self.model_progress.hide()
        layout.addWidget(self.model_progress)
        
        # Status label
        self.model_status = QLabel("Load models to compare performance")
        self.model_status.setWordWrap(True)
        layout.addWidget(self.model_status)
        
        # Load buttons
        btn_layout = QHBoxLayout()
        
        load_standard_btn = QPushButton("Load Standard Model")
        load_standard_btn.setFont(QFont("Arial", 12))
        load_standard_btn.clicked.connect(lambda: self.test_model_loading(False))
        load_standard_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        btn_layout.addWidget(load_standard_btn)
        
        load_quantized_btn = QPushButton("Load Quantized Model")
        load_quantized_btn.setFont(QFont("Arial", 12))
        load_quantized_btn.clicked.connect(lambda: self.test_model_loading(True))
        load_quantized_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF5722;
                color: white;
                padding: 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #E64A19;
            }
        """)
        btn_layout.addWidget(load_quantized_btn)
        
        layout.addLayout(btn_layout)
        
        layout.addStretch()
        
        return widget
    
    def create_database_tab(self):
        """Tab 3: Database testing"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        info_label = QLabel("✓ Technology Tested: SQLite Database")
        info_label.setStyleSheet("color: green; font-weight: bold;")
        layout.addWidget(info_label)
        
        db_info = QLabel(f"Database location: {os.path.abspath(self.db_path)}")
        db_info.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(db_info)
        
        # Document stats
        stats_group = QGroupBox("Document Statistics")
        stats_layout = QVBoxLayout()
        stats_group.setLayout(stats_layout)
        
        self.doc_stats_label = QLabel("Loading...")
        stats_layout.addWidget(self.doc_stats_label)
        self.update_doc_stats()
        
        layout.addWidget(stats_group)
        
        # Test save/load
        test_group = QGroupBox("Test Save/Load Preferences")
        test_layout = QVBoxLayout()
        test_group.setLayout(test_layout)
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Test Value:"))
        self.db_test_input = QTextEdit()
        self.db_test_input.setMaximumHeight(60)
        self.db_test_input.setPlaceholderText("Enter text to save...")
        input_layout.addWidget(self.db_test_input)
        test_layout.addLayout(input_layout)
        
        btn_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save to Database")
        save_btn.clicked.connect(self.test_database_save)
        save_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        btn_layout.addWidget(save_btn)
        
        load_btn = QPushButton("Load from Database")
        load_btn.clicked.connect(self.test_database_load)
        load_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 8px;")
        btn_layout.addWidget(load_btn)
        
        test_layout.addLayout(btn_layout)
        
        self.db_status = QLabel("Ready to test database operations")
        self.db_status.setWordWrap(True)
        test_layout.addWidget(self.db_status)
        
        layout.addWidget(test_group)
        layout.addStretch()
        
        return widget
    
    def create_metrics_tab(self):
        """Tab 4: Performance metrics summary"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        title = QLabel("Performance Metrics Summary")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(title)
        
        self.metrics_display = QTextEdit()
        self.metrics_display.setReadOnly(True)
        self.metrics_display.setFont(QFont("Courier", 10))
        layout.addWidget(self.metrics_display)
        
        refresh_btn = QPushButton("Refresh Metrics")
        refresh_btn.clicked.connect(self.update_metrics_display)
        layout.addWidget(refresh_btn)
        
        self.update_metrics_display()
        
        return widget
    
    def toggle_realtime_checking(self, enabled):
        """Toggle real-time grammar checking"""
        if enabled:
            print("✓ Real-time grammar checking enabled")
        else:
            print("✗ Real-time grammar checking disabled")
    
    def on_text_changed(self):
        """Handle text changes - trigger debounced grammar check"""
        if self.realtime_check.isChecked():
            # Restart timer (debounce)
            self.grammar_check_timer.stop()
            self.grammar_check_timer.start(1000)  # 1 second delay
    
    def auto_check_grammar(self):
        """Auto-check grammar (called by timer)"""
        print("🔄 Auto-checking grammar...")
        self.check_grammar()
    
    def check_grammar(self):
        """Check grammar using LanguageTool"""
        text = self.text_editor.toPlainText()
        
        if not text.strip():
            self.error_list.clear()
            self.error_list.addItem("No text to check!")
            self.grammar_matches = []
            return
        
        print("\nChecking grammar...")
        start_time = time.time()
        
        matches = self.grammar_tool.check(text)
        self.grammar_matches = matches  # Store for context menu
        
        check_time = time.time() - start_time
        self.metrics['check_times'].append(check_time)
        print(f"✓ Grammar check completed in {check_time:.2f}s (found {len(matches)} issues)")
        
        avg_check = sum(self.metrics['check_times']) / len(self.metrics['check_times'])
        self.grammar_metrics_label.setText(
            f"Load: {self.metrics['languagetool_load_time']:.2f}s | "
            f"Last check: {check_time:.2f}s | "
            f"Avg: {avg_check:.2f}s | "
            f"Errors: {len(matches)}"
        )
        
        self.clear_highlights()
        self.error_list.clear()
        
        if len(matches) == 0:
            self.error_list.addItem("✓ No grammar errors found!")
            return
        
        cursor = self.text_editor.textCursor()
        
        for match in matches:
            cursor.setPosition(match.offset)
            cursor.setPosition(match.offset + match.error_length, 
                             cursor.MoveMode.KeepAnchor)
            
            error_format = QTextCharFormat()
            error_format.setBackground(QColor(255, 200, 200))
            error_format.setUnderlineColor(QColor(255, 0, 0))
            error_format.setUnderlineStyle(QTextCharFormat.UnderlineStyle.WaveUnderline)
            
            cursor.setCharFormat(error_format)
            
            error_text = f"{match.rule_id}: {match.message}"
            if match.replacements:
                error_text += f" → {match.replacements[0]}"
            self.error_list.addItem(error_text)
    
    def clear_highlights(self):
        """Clear text highlighting"""
        cursor = self.text_editor.textCursor()
        cursor.select(cursor.SelectionType.Document)
        normal_format = QTextCharFormat()
        cursor.setCharFormat(normal_format)
        cursor.clearSelection()
        self.text_editor.setTextCursor(cursor)
    
    def show_context_menu(self, position):
        """Show context menu with grammar suggestions on right-click"""
        cursor = self.text_editor.cursorForPosition(position)
        click_pos = cursor.position()
        
        # Find if we clicked on an error
        matching_errors = []
        for match in self.grammar_matches:
            if match.offset <= click_pos <= match.offset + match.error_length:
                matching_errors.append(match)
        
        if not matching_errors:
            # No error at click position - show default menu
            self.text_editor.createStandardContextMenu().exec(self.text_editor.mapToGlobal(position))
            return
        
        # Create custom context menu with suggestions
        menu = QMenu(self.text_editor)
        
        for match in matching_errors:
            # Add error description as disabled item
            error_action = QAction(f"📝 {match.message}", self.text_editor)
            error_action.setEnabled(False)
            menu.addAction(error_action)
            
            menu.addSeparator()
            
            # Add suggestions
            if match.replacements:
                for i, suggestion in enumerate(match.replacements[:5]):  # Max 5 suggestions
                    action = QAction(f"✓ {suggestion}", self.text_editor)
                    action.triggered.connect(lambda checked, m=match, s=suggestion: self.apply_suggestion(m, s))
                    menu.addAction(action)
            else:
                no_sugg = QAction("No suggestions available", self.text_editor)
                no_sugg.setEnabled(False)
                menu.addAction(no_sugg)
            
            # Add ignore option
            menu.addSeparator()
            ignore_action = QAction("❌ Ignore", self.text_editor)
            ignore_action.triggered.connect(lambda checked, m=match: self.ignore_error(m))
            menu.addAction(ignore_action)
        
        menu.exec(self.text_editor.mapToGlobal(position))
    
    def apply_suggestion(self, match, suggestion):
        """Apply a grammar suggestion"""
        cursor = self.text_editor.textCursor()
        cursor.setPosition(match.offset)
        cursor.setPosition(match.offset + match.error_length, QTextCursor.MoveMode.KeepAnchor)
        cursor.insertText(suggestion)
        
        print(f"✓ Applied suggestion: '{suggestion}'")
        
        # Re-check grammar after change
        QTimer.singleShot(500, self.check_grammar)
    
    def ignore_error(self, match):
        """Ignore an error (just remove highlight)"""
        print(f"✗ Ignored error: {match.message}")
        # Re-check to update display
        QTimer.singleShot(100, self.check_grammar)
    
    def open_file(self):
        """Open a text file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "",
            "Text Files (*.txt *.md);;Word Documents (*.docx);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            if file_path.endswith('.docx') and DOCX_AVAILABLE:
                # Read .docx file
                doc = Document(file_path)
                text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            else:
                # Read plain text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            self.text_editor.setPlainText(text)
            self.current_file_path = file_path
            print(f"✓ Opened file: {file_path}")
            
            # Auto-check grammar
            if self.realtime_check.isChecked():
                QTimer.singleShot(500, self.check_grammar)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open file:\n{str(e)}")
    
    def save_file(self):
        """Save text to a file"""
        if self.current_file_path:
            file_path = self.current_file_path
        else:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save File",
                "",
                "Text Files (*.txt);;Markdown Files (*.md);;All Files (*)"
            )
        
        if not file_path:
            return
        
        try:
            text = self.text_editor.toPlainText()
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            self.current_file_path = file_path
            print(f"✓ Saved file: {file_path}")
            QMessageBox.information(self, "Success", f"File saved: {os.path.basename(file_path)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file:\n{str(e)}")
    
    def save_to_database(self):
        """Save current document to database"""
        text = self.text_editor.toPlainText()
        
        if not text.strip():
            QMessageBox.warning(self, "No Content", "Please enter some text first!")
            return
        
        filename = self.current_file_path if self.current_file_path else "Untitled Document"
        filename = os.path.basename(filename)
        
        save_time = self.save_document_to_db(filename, text)
        self.update_doc_stats()
        
        QMessageBox.information(self, "Success", 
                               f"Document saved to database!\n\nFilename: {filename}\nTime: {save_time*1000:.2f}ms")
    
    def update_doc_stats(self):
        """Update document statistics"""
        self.cursor.execute('SELECT COUNT(*) FROM documents')
        total_docs = self.cursor.fetchone()[0]
        
        self.cursor.execute('SELECT COUNT(*) FROM performance_metrics')
        total_metrics = self.cursor.fetchone()[0]
        
        stats_text = f"Documents saved: {total_docs}\n"
        stats_text += f"Metrics recorded: {total_metrics}\n"
        stats_text += f"Documents processed this session: {self.metrics['documents_processed']}"
        
        self.doc_stats_label.setText(stats_text)
    
    def continue_writing(self):
        """Continue writing from cursor position"""
        if not self.load_text_generator():
            return
        
        cursor = self.text_editor.textCursor()
        position = cursor.position()
        
        cursor.setPosition(0)
        cursor.setPosition(position, QTextCursor.MoveMode.KeepAnchor)
        prompt = cursor.selectedText().replace('\u2029', '\n')
        
        if not prompt.strip():
            QMessageBox.warning(self, "No Text", "Please type some text first!")
            return
        
        print(f"\nGenerating continuation from: '{prompt[-50:]}'...")
        
        self.gen_progress.show()
        self.gen_status.setText("Generating text continuation...")
        self.continue_btn.setEnabled(False)
        
        self.generator_thread = TextGeneratorThread(
            self.text_generator, 
            prompt, 
            max_length=150,
            task_type="generate"
        )
        self.generator_thread.progress.connect(self.update_generation_progress)
        self.generator_thread.finished.connect(self.generation_finished)
        self.generator_thread.start()
    
    def paraphrase_text(self):
        """Paraphrase ALL text"""
        if not self.load_text_generator():
            return
        
        all_text = self.text_editor.toPlainText()
        
        if not all_text.strip():
            QMessageBox.warning(self, "No Text", "Please type some text first!")
            return
        
        print(f"\nParaphrasing entire document ({len(all_text)} chars)...")
        
        self.gen_progress.show()
        self.gen_status.setText("Paraphrasing entire document...")
        self.paraphrase_btn.setEnabled(False)
        
        self.generator_thread = TextGeneratorThread(
            self.text_generator, 
            all_text, 
            max_length=512,
            task_type="paraphrase"
        )
        self.generator_thread.progress.connect(self.update_generation_progress)
        self.generator_thread.finished.connect(lambda text, time: self.replace_all_text_finished(text, time, "paraphrase"))
        self.generator_thread.start()
    
    def rewrite_text(self):
        """Rewrite ALL text"""
        if not self.load_text_generator():
            return
        
        all_text = self.text_editor.toPlainText()
        
        if not all_text.strip():
            QMessageBox.warning(self, "No Text", "Please type some text first!")
            return
        
        print(f"\nRewriting entire document ({len(all_text)} chars)...")
        
        self.gen_progress.show()
        self.gen_status.setText("Rewriting entire document...")
        self.rewrite_btn.setEnabled(False)
        
        self.generator_thread = TextGeneratorThread(
            self.text_generator, 
            all_text, 
            max_length=512,
            task_type="rewrite"
        )
        self.generator_thread.progress.connect(self.update_generation_progress)
        self.generator_thread.finished.connect(lambda text, time: self.replace_all_text_finished(text, time, "rewrite"))
        self.generator_thread.start()
    
    def transform_selection(self):
        """Transform ALL text based on tone"""
        if not self.load_text_generator():
            return
        
        all_text = self.text_editor.toPlainText()
        
        if not all_text.strip():
            QMessageBox.warning(self, "No Text", "Please type some text first!")
            return
        
        task_type = None
        
        if self.tone_formal.isChecked():
            task_type = "formal"
        elif self.tone_casual.isChecked():
            task_type = "casual"
        elif self.tone_technical.isChecked():
            task_type = "technical"
        elif self.tone_simple.isChecked():
            task_type = "simple"
        
        print(f"\nTransforming entire document to {task_type} ({len(all_text)} chars)...")
        
        self.gen_progress.show()
        self.gen_status.setText(f"Transforming entire document to {task_type}...")
        self.transform_btn.setEnabled(False)
        
        self.generator_thread = TextGeneratorThread(
            self.text_generator, 
            all_text, 
            max_length=512,
            task_type=task_type
        )
        self.generator_thread.progress.connect(self.update_generation_progress)
        self.generator_thread.finished.connect(lambda text, time: self.transform_all_finished(text, time, task_type))
        self.generator_thread.start()
    
    def update_generation_progress(self, message):
        """Update generation progress message"""
        self.gen_status.setText(message)
    
    def generation_finished(self, generated_text, gen_time):
        """Handle text generation completion (for continuation)"""
        self.gen_progress.hide()
        self.continue_btn.setEnabled(True)
        
        if generated_text.startswith("Error:"):
            self.gen_status.setText(f"❌ {generated_text}")
            return
        
        cursor = self.text_editor.textCursor()
        cursor.insertText(" " + generated_text)
        
        self.metrics['generation_times'].append(('continue', gen_time))
        
        self.gen_status.setText(f"✓ Generated in {gen_time:.2f}s")
        print(f"✓ Text continuation completed in {gen_time:.2f}s")
        
        self.cursor.execute(
            'INSERT INTO performance_metrics (metric_name, metric_value) VALUES (?, ?)',
            ('text_generation_time', gen_time)
        )
        self.conn.commit()
    
    def replace_all_text_finished(self, generated_text, gen_time, task_type):
        """Handle paraphrase/rewrite completion"""
        self.gen_progress.hide()
        self.paraphrase_btn.setEnabled(True)
        self.rewrite_btn.setEnabled(True)
        
        if generated_text.startswith("Error:"):
            self.gen_status.setText(f"❌ {generated_text}")
            return
        
        self.text_editor.setPlainText(generated_text)
        
        self.metrics['generation_times'].append((task_type, gen_time))
        
        self.gen_status.setText(f"✓ {task_type.capitalize()}d entire document in {gen_time:.2f}s")
        print(f"✓ {task_type.capitalize()} completed in {gen_time:.2f}s")
        print(f"   Result length: {len(generated_text)} characters")
        
        self.cursor.execute(
            'INSERT INTO performance_metrics (metric_name, metric_value) VALUES (?, ?)',
            (f'{task_type}_time', gen_time)
        )
        self.conn.commit()
    
    def transform_all_finished(self, generated_text, gen_time, task_type):
        """Handle transformation completion"""
        self.gen_progress.hide()
        self.transform_btn.setEnabled(True)
        
        if generated_text.startswith("Error:"):
            self.gen_status.setText(f"❌ {generated_text}")
            return
        
        self.text_editor.setPlainText(generated_text)
        
        self.metrics['generation_times'].append((f'transform_{task_type}', gen_time))
        
        self.gen_status.setText(f"✓ Transformed entire document to {task_type} in {gen_time:.2f}s")
        print(f"✓ Transformation to {task_type} completed in {gen_time:.2f}s")
        print(f"   Result length: {len(generated_text)} characters")
        
        self.cursor.execute(
            'INSERT INTO performance_metrics (metric_name, metric_value) VALUES (?, ?)',
            (f'transform_{task_type}_time', gen_time)
        )
        self.conn.commit()
    
    def test_model_loading(self, quantized=False):
        """Test loading transformer model"""
        model_type = "quantized" if quantized else "standard"
        print(f"\nTesting {model_type} model loading...")
        self.model_status.setText(f"Loading {model_type} model... (this may take 30-60 seconds)")
        self.model_progress.show()
        
        self.model_loader_thread = ModelLoaderThread(quantized)
        self.model_loader_thread.progress.connect(self.update_model_progress)
        self.model_loader_thread.finished.connect(lambda msg, time, mem: self.model_loading_finished(msg, time, mem, quantized))
        self.model_loader_thread.start()
    
    def update_model_progress(self, message):
        """Update model loading progress"""
        self.model_status.setText(message)
    
    def model_loading_finished(self, message, load_time, memory_mb, quantized):
        """Handle model loading completion"""
        self.model_progress.hide()
        self.model_status.setText(message)
        
        if load_time > 0:
            if quantized:
                self.metrics['quantized_load_time'] = load_time
                self.metrics['quantized_memory_mb'] = memory_mb
            else:
                self.metrics['model_load_time'] = load_time
                self.metrics['model_memory_mb'] = memory_mb
            
            model_type = "quantized" if quantized else "standard"
            print(f"✓ {model_type.capitalize()} model loaded in {load_time:.2f}s ({memory_mb:.1f}MB)")
            
            self.cursor.execute(
                'INSERT INTO performance_metrics (metric_name, metric_value) VALUES (?, ?)',
                (f'{model_type}_load_time', load_time)
            )
            self.cursor.execute(
                'INSERT INTO performance_metrics (metric_name, metric_value) VALUES (?, ?)',
                (f'{model_type}_memory_mb', memory_mb)
            )
            self.conn.commit()
        
        self.update_metrics_display()
    
    def test_database_save(self):
        """Test saving to database"""
        value = self.db_test_input.toPlainText()
        if not value.strip():
            self.db_status.setText("⚠ Enter some text first!")
            return
        
        save_time = self.save_preference('test_value', value)
        self.db_status.setText(f"✓ Saved to database in {save_time*1000:.2f}ms")
        print(f"✓ Database save completed in {save_time*1000:.2f}ms")
    
    def test_database_load(self):
        """Test loading from database"""
        loaded_value = self.load_preference('test_value', 'No data saved yet')
        self.db_test_input.setText(loaded_value)
        
        load_time = self.metrics['database_operations'][-1][1]
        self.db_status.setText(f"✓ Loaded from database in {load_time*1000:.2f}ms")
        print(f"✓ Database load completed in {load_time*1000:.2f}ms")
    
    def update_metrics_display(self):
        """Update the metrics summary display"""
        report = "="*60 + "\n"
        report += "QUILL R&D - TECHNOLOGY VALIDATION REPORT\n"
        report += "="*60 + "\n\n"
        
        report += "1. PYQT6 (GUI Framework)\n"
        report += "   Status: ✓ TESTED AND WORKING\n"
        report += "   - Text editor with context menu\n"
        report += "   - Multi-tab interface\n"
        report += "   - File open/save dialogs\n"
        report += "   - Real-time updates\n\n"
        
        report += "2. LANGUAGETOOL (Grammar Checking - PILLAR 1)\n"
        report += "   Status: ✓ TESTED AND WORKING\n"
        report += f"   - Load time: {self.metrics['languagetool_load_time']:.2f}s\n"
        if self.metrics['check_times']:
            avg_check = sum(self.metrics['check_times']) / len(self.metrics['check_times'])
            report += f"   - Average check time: {avg_check:.2f}s\n"
            report += f"   - Total checks: {len(self.metrics['check_times'])}\n"
        report += "   - Real-time checking: ✓ Implemented\n"
        report += "   - Right-click context menu: ✓ Implemented\n\n"
        
        report += "3. FLAN-T5 (AI Model - PILLAR 2 & 3)\n"
        if TRANSFORMERS_AVAILABLE:
            if self.metrics['model_load_time'] > 0:
                report += "   Status: ✓ STANDARD MODEL TESTED\n"
                report += f"   - Load time: {self.metrics['model_load_time']:.2f}s\n"
                report += f"   - Memory usage: {self.metrics['model_memory_mb']:.1f}MB\n"
            
            if self.metrics['quantized_load_time'] > 0:
                report += "   Status: ✓ QUANTIZED MODEL TESTED\n"
                report += f"   - Load time: {self.metrics['quantized_load_time']:.2f}s\n"
                report += f"   - Memory usage: {self.metrics['quantized_memory_mb']:.1f}MB\n"
                
                if self.metrics['model_load_time'] > 0:
                    speedup = self.metrics['model_load_time'] / self.metrics['quantized_load_time']
                    mem_reduction = ((self.metrics['model_memory_mb'] - self.metrics['quantized_memory_mb']) 
                                   / self.metrics['model_memory_mb'] * 100)
                    report += f"   - Speedup: {speedup:.2f}x\n"
                    report += f"   - Memory reduction: {mem_reduction:.1f}%\n"
            
            if self.metrics['generation_times']:
                report += f"   - Generations performed: {len(self.metrics['generation_times'])}\n"
                avg_gen = sum(t for _, t in self.metrics['generation_times']) / len(self.metrics['generation_times'])
                report += f"   - Average generation time: {avg_gen:.2f}s\n"
        else:
            report += "   Status: ⚠ NOT INSTALLED\n"
        report += "\n"
        
        report += "4. SQLITE (Database)\n"
        report += "   Status: ✓ TESTED AND WORKING\n"
        report += f"   - Location: {os.path.abspath(self.db_path)}\n"
        report += f"   - Documents saved: {self.metrics['documents_processed']}\n"
        if self.metrics['database_operations']:
            report += f"   - Total operations: {len(self.metrics['database_operations'])}\n"
            avg_db = sum(t for _, t in self.metrics['database_operations']) / len(self.metrics['database_operations'])
            report += f"   - Average op time: {avg_db*1000:.2f}ms\n"
        report += "\n"
        
        report += "="*60 + "\n"
        report += "FEATURE COMPLETION STATUS\n"
        report += "="*60 + "\n"
        report += "✓ Functional QTextEdit widget\n"
        report += "✓ Text selection and cursor tracking\n"
        report += "✓ Programmatic text insertion\n"
        report += "✓ Color-coded highlighting\n"
        report += "✓ Dynamic highlight updates (real-time)\n"
        report += "✓ Suggestion sidebar\n"
        report += "✓ Accept/reject buttons (right-click menu)\n"
        report += "✓ Threading (QThread)\n"
        report += "✓ Loading indicators\n"
        report += "✓ Model loading (Flan-T5)\n"
        report += "✓ Memory measurement\n"
        report += "✓ Text generation\n"
        report += "✓ Dynamic quantization\n"
        report += "✓ Performance comparison\n"
        report += "✓ Grammar checking integration\n"
        report += "✓ Error display\n"
        report += "✓ End-to-end pipeline\n"
        report += "✓ Debounced processing\n"
        report += "✓ SQLite database\n"
        report += "✓ Save/load preferences\n"
        report += "✓ File upload/save\n"
        report += "✓ Document storage\n"
        
        self.metrics_display.setText(report)
    
    def closeEvent(self, event):
        """Clean up when closing"""
        self.conn.close()
        print("\n" + "="*60)
        print("Application closed - Database connection closed")
        print("="*60)
        event.accept()


def main():
    """Run the application"""
    print("\n" + "="*60)
    print("QUILL R&D - COMPLETE TECHNOLOGY VALIDATION")
    print("="*60)
    print("\nFeatures implemented:")
    print("  ✓ PyQt6 (GUI Framework)")
    print("  ✓ LanguageTool (Grammar)")
    print("  ✓ Flan-T5 (AI Model)")
    print("  ✓ SQLite (Database)")
    print("  ✓ Real-time checking")
    print("  ✓ Right-click corrections")
    print("  ✓ File upload/save")
    print("  ✓ Model quantization")
    print("  ✓ Performance metrics")
    print("\n" + "="*60 + "\n")
    
    app = QApplication(sys.argv)
    window = QuillRnD()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()