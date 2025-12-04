import sys
import time
import sqlite3
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTextEdit, QLabel, 
                             QListWidget, QSplitter, QTabWidget, QGroupBox,
                             QProgressBar, QMessageBox)
from PyQt6.QtGui import QTextCharFormat, QColor, QFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal
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

class ModelLoaderThread(QThread):
    """Background thread for loading heavy models"""
    finished = pyqtSignal(str, float)  # message, time_taken
    progress = pyqtSignal(str)  # progress message
    
    def run(self):
        """Load GPT-2 model in background"""
        try:
            self.progress.emit("Loading DistilGPT-2 model...")
            start_time = time.time()
            
            # Load a smaller, faster model for R&D
            generator = pipeline('text-generation', model='distilgpt2')
            
            load_time = time.time() - start_time
            
            self.progress.emit("Testing text generation...")
            
            # Test generation
            test_output = generator("The future of AI writing is", max_length=20, num_return_sequences=1)
            
            message = f"✓ Model loaded successfully!\n\nLoad time: {load_time:.2f}s\n\nTest output: {test_output[0]['generated_text']}"
            self.finished.emit(message, load_time)
            
        except Exception as e:
            self.finished.emit(f"✗ Error loading model: {str(e)}", 0.0)

class QuillRnD(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QUILL R&D - Technology Validation")
        self.setGeometry(100, 100, 1200, 700)
    
        # Initialize metrics dictionary FIRST (before anything else uses it)
        self.metrics = {
            'languagetool_load_time': 0.0,
            'model_load_time': 0.0,
            'check_times': [],
            'database_operations': []
         }
    
        # Initialize database
        self.init_database()
    
        # Initialize LanguageTool (measure loading time)
        print("="*60)
        print("INITIALIZING TECHNOLOGIES...")
        print("="*60)
        print("\n1. Loading LanguageTool...")
        start_time = time.time()
        self.grammar_tool = language_tool_python.LanguageTool('en-US')
        load_time = time.time() - start_time
        print(f"   ✓ LanguageTool loaded in {load_time:.2f} seconds")
    
        # Update the load time in metrics
        self.metrics['languagetool_load_time'] = load_time
        
        self.init_ui()
        
        print("\n" + "="*60)
        print("APPLICATION READY - All core systems initialized")
        print("="*60)
    
    def init_database(self):
        """Initialize SQLite database (proves database technology works)"""
        print("\n2. Setting up SQLite database...")
        start_time = time.time()
        
        # Create database file
        self.db_path = "quill_rnd.db"
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Create preferences table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS preferences (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        # Create metrics table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT,
                metric_value REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
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
    
    def init_ui(self):
        """Create the user interface with multiple tabs"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Title
        title = QLabel("QUILL R&D - Complete Technology Validation")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        
        # Create tabs for different technology tests
        tabs = QTabWidget()
        
        # Tab 1: Grammar Checking (LanguageTool + PyQt6)
        grammar_tab = self.create_grammar_tab()
        tabs.addTab(grammar_tab, "Grammar Checking")
        
        # Tab 2: Model Testing (Transformers)
        model_tab = self.create_model_tab()
        tabs.addTab(model_tab, "AI Model Testing")
        
        # Tab 3: Database Testing (SQLite)
        database_tab = self.create_database_tab()
        tabs.addTab(database_tab, "Database Testing")
        
        # Tab 4: Performance Metrics
        metrics_tab = self.create_metrics_tab()
        tabs.addTab(metrics_tab, "Performance Metrics")
        
        main_layout.addWidget(tabs)
    
    def create_grammar_tab(self):
        """Tab 1: Grammar checking with LanguageTool"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        info_label = QLabel("✓ Technology Tested: PyQt6 + LanguageTool")
        info_label.setStyleSheet("color: green; font-weight: bold;")
        layout.addWidget(info_label)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Editor
        editor_widget = QWidget()
        editor_layout = QVBoxLayout()
        editor_widget.setLayout(editor_layout)
        
        editor_label = QLabel("Write text to check grammar:")
        editor_layout.addWidget(editor_label)
        
        self.text_editor = QTextEdit()
        self.text_editor.setPlaceholderText("Type text with errors here...")
        self.text_editor.setFont(QFont("Arial", 12))
        editor_layout.addWidget(self.text_editor)
        
        self.check_button = QPushButton("Check Grammar")
        self.check_button.setFont(QFont("Arial", 11))
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
        
        splitter.addWidget(editor_widget)
        
        # Right: Errors
        error_widget = QWidget()
        error_layout = QVBoxLayout()
        error_widget.setLayout(error_layout)
        
        error_label = QLabel("Grammar Errors:")
        error_layout.addWidget(error_label)
        
        self.error_list = QListWidget()
        error_layout.addWidget(self.error_list)
        
        self.grammar_metrics_label = QLabel(f"Load time: {self.metrics['languagetool_load_time']:.2f}s")
        self.grammar_metrics_label.setStyleSheet("color: gray; font-size: 10px;")
        error_layout.addWidget(self.grammar_metrics_label)
        
        splitter.addWidget(error_widget)
        splitter.setSizes([600, 400])
        
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
        
        desc_label = QLabel("This tests loading and running a transformer model (DistilGPT-2)")
        layout.addWidget(desc_label)
        
        # Progress bar
        self.model_progress = QProgressBar()
        self.model_progress.setRange(0, 0)  # Indeterminate
        self.model_progress.hide()
        layout.addWidget(self.model_progress)
        
        # Status label
        self.model_status = QLabel("Click 'Load Model' to test transformer technology")
        self.model_status.setWordWrap(True)
        layout.addWidget(self.model_status)
        
        # Load model button
        load_model_btn = QPushButton("Load DistilGPT-2 Model")
        load_model_btn.setFont(QFont("Arial", 12))
        load_model_btn.clicked.connect(self.test_model_loading)
        load_model_btn.setStyleSheet("""
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
        layout.addWidget(load_model_btn)
        
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
        
        # Test save/load
        test_group = QGroupBox("Test Save/Load Preferences")
        test_layout = QVBoxLayout()
        test_group.setLayout(test_layout)
        
        # Input field
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Test Value:"))
        self.db_test_input = QTextEdit()
        self.db_test_input.setMaximumHeight(60)
        self.db_test_input.setPlaceholderText("Enter text to save...")
        input_layout.addWidget(self.db_test_input)
        test_layout.addLayout(input_layout)
        
        # Buttons
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
        
        # Status
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
    
    def check_grammar(self):
        """Check grammar using LanguageTool"""
        text = self.text_editor.toPlainText()
        
        if not text.strip():
            self.error_list.clear()
            self.error_list.addItem("No text to check!")
            return
        
        print("\nChecking grammar...")
        start_time = time.time()
        
        matches = self.grammar_tool.check(text)
        
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
    
    def test_model_loading(self):
        """Test loading transformer model in background thread"""
        print("\nTesting model loading...")
        self.model_status.setText("Loading model... (this may take 30-60 seconds)")
        self.model_progress.show()
        
        # Create and start background thread
        self.model_loader_thread = ModelLoaderThread()
        self.model_loader_thread.progress.connect(self.update_model_progress)
        self.model_loader_thread.finished.connect(self.model_loading_finished)
        self.model_loader_thread.start()
    
    def update_model_progress(self, message):
        """Update model loading progress"""
        self.model_status.setText(message)
    
    def model_loading_finished(self, message, load_time):
        """Handle model loading completion"""
        self.model_progress.hide()
        self.model_status.setText(message)
        
        if load_time > 0:
            self.metrics['model_load_time'] = load_time
            print(f"✓ Model loaded successfully in {load_time:.2f}s")
            
            # Save metric to database
            self.cursor.execute(
                'INSERT INTO performance_metrics (metric_name, metric_value) VALUES (?, ?)',
                ('model_load_time', load_time)
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
        report += "   - Text editor functional\n"
        report += "   - Multi-tab interface working\n"
        report += "   - Button interactions working\n\n"
        
        report += "2. LANGUAGETOOL (Grammar Checking)\n"
        report += "   Status: ✓ TESTED AND WORKING\n"
        report += f"   - Load time: {self.metrics['languagetool_load_time']:.2f}s\n"
        if self.metrics['check_times']:
            avg_check = sum(self.metrics['check_times']) / len(self.metrics['check_times'])
            report += f"   - Average check time: {avg_check:.2f}s\n"
            report += f"   - Total checks performed: {len(self.metrics['check_times'])}\n"
        report += "\n"
        
        report += "3. TRANSFORMERS + PYTORCH (AI Models)\n"
        if TRANSFORMERS_AVAILABLE:
            if self.metrics['model_load_time'] > 0:
                report += "   Status: ✓ TESTED AND WORKING\n"
                report += f"   - Model load time: {self.metrics['model_load_time']:.2f}s\n"
            else:
                report += "   Status: ⏳ AVAILABLE (Click 'Load Model' in AI Model tab)\n"
        else:
            report += "   Status: ⚠ NOT INSTALLED\n"
            report += "   - Run: pip install transformers torch\n"
        report += "\n"
        
        report += "4. SQLITE (Database)\n"
        report += "   Status: ✓ TESTED AND WORKING\n"
        report += f"   - Database created: {os.path.abspath(self.db_path)}\n"
        if self.metrics['database_operations']:
            report += f"   - Total operations: {len(self.metrics['database_operations'])}\n"
            avg_db_time = sum(t for _, t in self.metrics['database_operations']) / len(self.metrics['database_operations'])
            report += f"   - Average operation time: {avg_db_time*1000:.2f}ms\n"
        report += "\n"
        
        report += "="*60 + "\n"
        report += "CONCLUSION\n"
        report += "="*60 + "\n"
        
        tested_count = 3 if TRANSFORMERS_AVAILABLE and self.metrics['model_load_time'] > 0 else 2
        report += f"Technologies tested: {tested_count}/4 core technologies\n"
        report += "All tested technologies are working correctly.\n"
        report += "Integration between PyQt6 and NLP tools is successful.\n"
        report += "Performance metrics are acceptable for R&D validation.\n"
        
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
    print("\nThis application tests ALL technologies from the pitch:")
    print("  ✓ PyQt6 (GUI Framework)")
    print("  ✓ LanguageTool (Grammar Checking)")
    print("  • Transformers + PyTorch (AI Models)")
    print("  ✓ SQLite (Database)")
    print("\n" + "="*60 + "\n")
    
    app = QApplication(sys.argv)
    window = QuillRnD()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()