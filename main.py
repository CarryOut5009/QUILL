import sys
import time
import sqlite3
import os
import psutil
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTextEdit, QLabel, 
                             QListWidget, QSplitter, QTabWidget, QGroupBox,
                             QProgressBar, QMessageBox, QRadioButton, QButtonGroup,
                             QFileDialog, QMenu, QStatusBar, QListWidgetItem, QInputDialog,
                             QScrollArea, QFrame)
from PyQt6.QtGui import QTextCharFormat, QColor, QFont, QTextCursor, QAction, QKeySequence, QShortcut
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

class TextGeneratorThread(QThread):
    """Background thread for text generation to keep UI responsive"""
    finished = pyqtSignal(list, float)  # List of 3 suggestions, time taken
    progress = pyqtSignal(str)
    
    def __init__(self, generator, prompt):
        super().__init__()
        self.generator = generator
        self.prompt = prompt
    
    def run(self):
        """Generate 3 text continuations using FLAN-T5"""
        try:
            start_time = time.time()
            
            self.progress.emit("Generating 3 continuation suggestions...")
            
            # FLAN-T5 instruction format for text continuation
            instruction = f"Write what happens next in the story with 2 to 3 sentences. Continue from here:\n\n{self.prompt}\n\nNext:"
            
            # Generate 3 different suggestions
            suggestions = []
            for i in range(3):
                self.progress.emit(f"Generating suggestion {i+1}/3...")
                
                result = self.generator(
                    instruction,
                    max_new_tokens=80,
                    min_new_tokens=30,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.8 + (i * 0.2),  # Vary temperature for diversity
                    top_p=0.92,
                    top_k=50,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3
                )
                
                generated_text = result[0]['generated_text'].strip()
                suggestions.append(generated_text)
            
            gen_time = time.time() - start_time
            
            self.finished.emit(suggestions, gen_time)
            
        except Exception as e:
            self.finished.emit([f"Error: {str(e)}"], 0.0)

class CollapsiblePanel(QWidget):
    """A collapsible panel for the sidebar"""
    def __init__(self, title, color="#4CAF50"):
        super().__init__()
        self.is_expanded = False
        self.color = color
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)
        
        # Toggle button
        self.toggle_btn = QPushButton(f"▶ {title}")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                padding: 12px;
                text-align: left;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {self.adjust_color(color, -20)};
            }}
            QPushButton:checked {{
                background-color: {self.adjust_color(color, -30)};
            }}
        """)
        self.toggle_btn.clicked.connect(self.toggle)
        layout.addWidget(self.toggle_btn)
        
        # Content widget (initially hidden)
        self.content = QWidget()
        self.content.hide()
        layout.addWidget(self.content)
        
        # Content layout (to be populated by caller)
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        self.content.setLayout(self.content_layout)
    
    def adjust_color(self, hex_color, amount):
        """Darken or lighten a hex color"""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        r = max(0, min(255, r + amount))
        g = max(0, min(255, g + amount))
        b = max(0, min(255, b + amount))
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def toggle(self):
        """Toggle section visibility"""
        self.is_expanded = not self.is_expanded
        if self.is_expanded:
            self.content.show()
            current_text = self.toggle_btn.text()
            self.toggle_btn.setText(current_text.replace("▶", "▼"))
        else:
            self.content.hide()
            current_text = self.toggle_btn.text()
            self.toggle_btn.setText(current_text.replace("▼", "▶"))

class SuggestionCard(QFrame):
    """Clickable card for AI suggestions"""
    clicked = pyqtSignal(str)
    
    def __init__(self, suggestion_text, number):
        super().__init__()
        self.suggestion_text = suggestion_text
        self.setFrameStyle(QFrame.Shape.Box)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Style as card
        self.setStyleSheet("""
            SuggestionCard {
                background-color: white;
                border: 2px solid #2196F3;
                border-radius: 8px;
                padding: 10px;
            }
            SuggestionCard:hover {
                background-color: #E3F2FD;
                border: 2px solid #1976D2;
            }
        """)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Number badge
        number_label = QLabel(f"Option {number}")
        number_label.setStyleSheet("color: #2196F3; font-weight: bold; font-size: 10px;")
        layout.addWidget(number_label)
        
        # Suggestion text
        text_label = QLabel(suggestion_text)
        text_label.setWordWrap(True)
        text_label.setStyleSheet("color: #333; font-size: 11px; margin-top: 5px;")
        layout.addWidget(text_label)
    
    def mousePressEvent(self, event):
        """Handle click"""
        self.clicked.emit(self.suggestion_text)

class QuillRnD(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QUILL Writing Assistant - Iteration 4")
        self.setGeometry(100, 100, 1400, 800)
        
        # Initialize metrics
        self.metrics = {
            'languagetool_load_time': 0.0,
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
        
        # Setup keyboard shortcuts and status bar
        self.setup_keyboard_shortcuts()
        self.setup_status_bar()
        
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
    
    def delete_document_from_db(self, doc_id):
        """Delete document from database"""
        start_time = time.time()
        self.cursor.execute('DELETE FROM documents WHERE id = ?', (doc_id,))
        self.conn.commit()
        delete_time = time.time() - start_time
        self.metrics['database_operations'].append(('delete_document', delete_time))
        print(f"✓ Document deleted from database in {delete_time*1000:.2f}ms")
        return delete_time
    
    def load_text_generator(self):
        """Load text generation model (lazy loading)"""
        if not TRANSFORMERS_AVAILABLE:
            QMessageBox.warning(self, "Model Not Available", 
                              "Transformers library not installed.\n\nRun: pip install transformers torch")
            return False
        
        if self.model_loaded:
            return True
        
        try:
            print(f"\nLoading FLAN-T5-Base model for first use...")
            
            start_time = time.time()
            
            # Load FLAN-T5-Base model
            self.text_generator = pipeline('text2text-generation', model='google/flan-t5-base')
            
            load_time = time.time() - start_time
            
            self.model_loaded = True
            
            print(f"✓ FLAN-T5-Base loaded in {load_time:.2f}s")
            
            # Save to database
            self.cursor.execute(
                'INSERT INTO performance_metrics (metric_name, metric_value) VALUES (?, ?)',
                ('model_load_time', load_time)
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
        title = QLabel("QUILL Writing Assistant")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #2196F3; padding: 10px;")
        main_layout.addWidget(title)
        
        # Create tabs
        tabs = QTabWidget()
        
        tabs.addTab(self.create_editor_tab(), "✍️ Writing Assistant")
        tabs.addTab(self.create_database_tab(), "💾 Database")
        tabs.addTab(self.create_metrics_tab(), "📊 Performance Metrics")
        
        main_layout.addWidget(tabs)
    
    def create_editor_tab(self):
        """Main editor tab with sidebar"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        widget.setLayout(layout)
        
        # File operations toolbar
        file_btn_layout = QHBoxLayout()
        
        open_btn = QPushButton("📂 Open")
        open_btn.clicked.connect(self.open_file)
        open_btn.setStyleSheet("background-color: #607D8B; color: white; padding: 8px; border-radius: 4px;")
        file_btn_layout.addWidget(open_btn)
        
        save_btn = QPushButton("💾 Save")
        save_btn.clicked.connect(self.save_file)
        save_btn.setStyleSheet("background-color: #607D8B; color: white; padding: 8px; border-radius: 4px;")
        file_btn_layout.addWidget(save_btn)
        
        save_as_btn = QPushButton("📄 Save As")
        save_as_btn.clicked.connect(self.save_file_as)
        save_as_btn.setStyleSheet("background-color: #607D8B; color: white; padding: 8px; border-radius: 4px;")
        file_btn_layout.addWidget(save_as_btn)
        
        save_db_btn = QPushButton("💾 Save to DB")
        save_db_btn.clicked.connect(self.save_to_database)
        save_db_btn.setStyleSheet("background-color: #607D8B; color: white; padding: 8px; border-radius: 4px;")
        file_btn_layout.addWidget(save_db_btn)
        
        undo_btn = QPushButton("↶ Undo")
        undo_btn.clicked.connect(self.text_editor_undo)
        undo_btn.setToolTip("Undo (Ctrl+Z)")
        undo_btn.setStyleSheet("background-color: #FF9800; color: white; padding: 8px; border-radius: 4px;")
        file_btn_layout.addWidget(undo_btn)
        
        redo_btn = QPushButton("↷ Redo")
        redo_btn.clicked.connect(self.text_editor_redo)
        redo_btn.setToolTip("Redo (Ctrl+Y)")
        redo_btn.setStyleSheet("background-color: #FF9800; color: white; padding: 8px; border-radius: 4px;")
        file_btn_layout.addWidget(redo_btn)
        
        file_btn_layout.addStretch()
        layout.addLayout(file_btn_layout)
        
        # Main editor area with sidebar
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # LEFT: Text Editor
        editor_container = QWidget()
        editor_layout = QVBoxLayout()
        editor_layout.setContentsMargins(0, 5, 0, 0)
        editor_container.setLayout(editor_layout)
        
        self.text_editor = QTextEdit()
        self.text_editor.setPlaceholderText("Start writing here...\n\n✓ Real-time grammar checking enabled\n✓ Right-click errors to fix them\n✓ Use sidebar for AI suggestions")
        self.text_editor.setFont(QFont("Arial", 12))
        self.text_editor.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.text_editor.customContextMenuRequested.connect(self.show_context_menu)
        self.text_editor.textChanged.connect(self.on_text_changed)
        editor_layout.addWidget(self.text_editor)
        
        main_splitter.addWidget(editor_container)
        
        # RIGHT: Collapsible Sidebar
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setContentsMargins(5, 5, 5, 5)
        sidebar_layout.setSpacing(10)
        sidebar.setLayout(sidebar_layout)
        
        # PANEL 1: Grammar Errors (starts collapsed)
        self.grammar_panel = CollapsiblePanel("📝 Grammar Errors & Suggestions", "#4CAF50")
        
        self.error_list = QListWidget()
        self.error_list.setStyleSheet("font-size: 10px;")
        self.grammar_panel.content_layout.addWidget(self.error_list)
        
        self.grammar_metrics_label = QLabel(f"Load time: {self.metrics['languagetool_load_time']:.2f}s")
        self.grammar_metrics_label.setStyleSheet("color: gray; font-size: 9px;")
        self.grammar_panel.content_layout.addWidget(self.grammar_metrics_label)
        
        sidebar_layout.addWidget(self.grammar_panel)
        
        # PANEL 2: Continue Writing (starts collapsed)
        self.continue_panel = CollapsiblePanel("✨ Continue Writing", "#2196F3")
        
        continue_info = QLabel("Generate AI-powered continuations for your text")
        continue_info.setWordWrap(True)
        continue_info.setStyleSheet("color: #666; font-size: 10px; margin-bottom: 5px;")
        self.continue_panel.content_layout.addWidget(continue_info)
        
        self.generate_btn = QPushButton("🎯 Generate Suggestions")
        self.generate_btn.clicked.connect(self.generate_suggestions)
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 12px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        self.continue_panel.content_layout.addWidget(self.generate_btn)
        
        # Progress bar
        self.gen_progress = QProgressBar()
        self.gen_progress.setRange(0, 0)
        self.gen_progress.setStyleSheet("margin-top: 5px;")
        self.gen_progress.hide()
        self.continue_panel.content_layout.addWidget(self.gen_progress)
        
        # Status label
        self.gen_status = QLabel("")
        self.gen_status.setStyleSheet("color: #2196F3; font-style: italic; font-size: 9px;")
        self.continue_panel.content_layout.addWidget(self.gen_status)
        
        # Suggestions container (cards will be added here)
        self.suggestions_container = QWidget()
        self.suggestions_layout = QVBoxLayout()
        self.suggestions_layout.setSpacing(8)
        self.suggestions_container.setLayout(self.suggestions_layout)
        self.continue_panel.content_layout.addWidget(self.suggestions_container)
        
        sidebar_layout.addWidget(self.continue_panel)
        sidebar_layout.addStretch()
        
        main_splitter.addWidget(sidebar)
        main_splitter.setSizes([1000, 400])  # 70/30 split
        
        layout.addWidget(main_splitter)
        
        return widget
    
    def text_editor_undo(self):
        """Undo last change"""
        self.text_editor.undo()
    
    def text_editor_redo(self):
        """Redo last change"""
        self.text_editor.redo()
    
    def create_database_tab(self):
        """Database management tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        info_label = QLabel("✓ SQLite Database")
        info_label.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
        layout.addWidget(info_label)
        
        db_info = QLabel(f"Location: {os.path.abspath(self.db_path)}")
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
        
        # Document viewer
        viewer_group = QGroupBox("Saved Documents")
        viewer_layout = QVBoxLayout()
        viewer_group.setLayout(viewer_layout)
        
        self.doc_list = QListWidget()
        self.doc_list.itemDoubleClicked.connect(self.load_document_from_db)
        viewer_layout.addWidget(self.doc_list)
        
        btn_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_document_list)
        refresh_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 8px; border-radius: 3px;")
        btn_layout.addWidget(refresh_btn)
        
        load_btn = QPushButton("📂 Load")
        load_btn.clicked.connect(self.load_selected_document)
        load_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px; border-radius: 3px;")
        btn_layout.addWidget(load_btn)
        
        delete_btn = QPushButton("🗑️ Delete")
        delete_btn.clicked.connect(self.delete_selected_document)
        delete_btn.setStyleSheet("background-color: #F44336; color: white; padding: 8px; border-radius: 3px;")
        btn_layout.addWidget(delete_btn)
        
        viewer_layout.addLayout(btn_layout)
        layout.addWidget(viewer_group)
        layout.addStretch()
        
        QTimer.singleShot(100, self.refresh_document_list)
        
        return widget
    
    def create_metrics_tab(self):
        """Performance metrics tab"""
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
    
    def on_text_changed(self):
        """Handle text changes - trigger debounced grammar check"""
        # Restart timer (debounce)
        self.grammar_check_timer.stop()
        self.grammar_check_timer.start(1000)  # 1 second delay
    
    def auto_check_grammar(self):
        """Auto-check grammar"""
        self.check_grammar()
    
    def check_grammar(self):
        """Check grammar using LanguageTool"""
        text = self.text_editor.toPlainText()
        
        if not text.strip():
            self.error_list.clear()
            self.error_list.addItem("No text to check")
            self.grammar_matches = []
            return
        
        start_time = time.time()
        
        matches = self.grammar_tool.check(text)
        self.grammar_matches = matches
        
        check_time = time.time() - start_time
        self.metrics['check_times'].append(check_time)
        
        avg_check = sum(self.metrics['check_times']) / len(self.metrics['check_times'])
        self.grammar_metrics_label.setText(
            f"Errors: {len(matches)} | Last check: {check_time:.2f}s | Avg: {avg_check:.2f}s"
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
            
            error_text = f"{match.message}"
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
        """Show context menu with grammar suggestions"""
        cursor = self.text_editor.cursorForPosition(position)
        click_pos = cursor.position()
        
        matching_errors = []
        for match in self.grammar_matches:
            if match.offset <= click_pos <= match.offset + match.error_length:
                matching_errors.append(match)
        
        if not matching_errors:
            self.text_editor.createStandardContextMenu().exec(self.text_editor.mapToGlobal(position))
            return
        
        menu = QMenu(self.text_editor)
        
        for match in matching_errors:
            error_action = QAction(f"📝 {match.message}", self.text_editor)
            error_action.setEnabled(False)
            menu.addAction(error_action)
            
            menu.addSeparator()
            
            if match.replacements:
                for suggestion in match.replacements[:5]:
                    action = QAction(f"✓ {suggestion}", self.text_editor)
                    action.triggered.connect(lambda checked, m=match, s=suggestion: self.apply_suggestion(m, s))
                    menu.addAction(action)
            else:
                no_sugg = QAction("No suggestions available", self.text_editor)
                no_sugg.setEnabled(False)
                menu.addAction(no_sugg)
            
            menu.addSeparator()
            ignore_action = QAction("✖ Ignore", self.text_editor)
            ignore_action.triggered.connect(lambda checked, m=match: self.ignore_error(m))
            menu.addAction(ignore_action)
        
        menu.exec(self.text_editor.mapToGlobal(position))
    
    def apply_suggestion(self, match, suggestion):
        """Apply grammar suggestion"""
        cursor = self.text_editor.textCursor()
        cursor.setPosition(match.offset)
        cursor.setPosition(match.offset + match.error_length, QTextCursor.MoveMode.KeepAnchor)
        cursor.insertText(suggestion)
        
        QTimer.singleShot(500, self.check_grammar)
    
    def ignore_error(self, match):
        """Ignore error"""
        QTimer.singleShot(100, self.check_grammar)
    
    def generate_suggestions(self):
        """Generate 3 AI continuation suggestions"""
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
        
        # Limit prompt to last 500 characters
        if len(prompt) > 500:
            prompt = prompt[-500:]
        
        print(f"\nGenerating 3 suggestions...")
        
        # Clear previous suggestions
        self.clear_suggestions()
        
        self.gen_progress.show()
        self.gen_status.setText("Generating suggestions...")
        self.generate_btn.setEnabled(False)
        
        self.generator_thread = TextGeneratorThread(self.text_generator, prompt)
        self.generator_thread.progress.connect(self.update_generation_progress)
        self.generator_thread.finished.connect(self.display_suggestions)
        self.generator_thread.start()
    
    def clear_suggestions(self):
        """Clear suggestion cards"""
        while self.suggestions_layout.count():
            item = self.suggestions_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def update_generation_progress(self, message):
        """Update generation progress"""
        self.gen_status.setText(message)
    
    def display_suggestions(self, suggestions, gen_time):
        """Display 3 suggestion cards"""
        self.gen_progress.hide()
        self.generate_btn.setEnabled(True)
        
        if suggestions and suggestions[0].startswith("Error:"):
            self.gen_status.setText(f"✗ {suggestions[0]}")
            return
        
        self.gen_status.setText(f"✓ Generated in {gen_time:.2f}s - Click a suggestion to insert")
        
        # Clear previous cards
        self.clear_suggestions()
        
        # Create 3 clickable cards
        for i, suggestion in enumerate(suggestions, 1):
            card = SuggestionCard(suggestion, i)
            card.clicked.connect(self.insert_suggestion)
            self.suggestions_layout.addWidget(card)
        
        self.metrics['generation_times'].append(('continue', gen_time))
        
        self.cursor.execute(
            'INSERT INTO performance_metrics (metric_name, metric_value) VALUES (?, ?)',
            ('text_generation_time', gen_time)
        )
        self.conn.commit()
    
    def insert_suggestion(self, suggestion_text):
        """Insert selected suggestion into text"""
        cursor = self.text_editor.textCursor()
        cursor.insertText(" " + suggestion_text)
        
        # Clear suggestion cards
        self.clear_suggestions()
        self.gen_status.setText("✓ Suggestion inserted!")
        
        print(f"✓ Inserted suggestion: {suggestion_text[:50]}...")
    
    def open_file(self):
        """Open file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "",
            "Text Files (*.txt *.md);;Word Documents (*.docx);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            if file_path.endswith('.docx') and DOCX_AVAILABLE:
                doc = Document(file_path)
                text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            self.text_editor.setPlainText(text)
            self.current_file_path = file_path
            self.update_status_bar()
            
            QTimer.singleShot(500, self.check_grammar)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open file:\n{str(e)}")
    
    def save_file(self):
        """Save file"""
        if self.current_file_path:
            file_path = self.current_file_path
        else:
            self.save_file_as()
            return
        
        try:
            text = self.text_editor.toPlainText()
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            self.update_status_bar()
            if hasattr(self, 'status_saved'):
                self.status_saved.setText(f"Saved at {time.strftime('%H:%M:%S')}")
                QTimer.singleShot(3000, lambda: self.status_saved.setText(""))
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")
    
    def save_file_as(self):
        """Save file as"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save As", "",
            "Text Files (*.txt);;Markdown (*.md);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            text = self.text_editor.toPlainText()
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            self.current_file_path = file_path
            self.update_status_bar()
            QMessageBox.information(self, "Success", f"Saved: {os.path.basename(file_path)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")
    
    def save_to_database(self):
        """Save to database"""
        text = self.text_editor.toPlainText()
        
        if not text.strip():
            QMessageBox.warning(self, "No Content", "Please enter text first!")
            return
        
        default_name = os.path.basename(self.current_file_path) if self.current_file_path else "Untitled"
        
        filename, ok = QInputDialog.getText(
            self, "Save to Database", "Document name:", text=default_name
        )
        
        if not ok or not filename.strip():
            if not ok:
                return
            filename = default_name
        
        save_time = self.save_document_to_db(filename, text)
        self.update_doc_stats()
        self.refresh_document_list()
        
        QMessageBox.information(self, "Success", 
                               f"Saved to database!\n\nTime: {save_time*1000:.2f}ms")
    
    def update_doc_stats(self):
        """Update document statistics"""
        self.cursor.execute('SELECT COUNT(*) FROM documents')
        total_docs = self.cursor.fetchone()[0]
        
        self.cursor.execute('SELECT COUNT(*) FROM performance_metrics')
        total_metrics = self.cursor.fetchone()[0]
        
        stats_text = f"Documents: {total_docs}\n"
        stats_text += f"Metrics recorded: {total_metrics}\n"
        stats_text += f"This session: {self.metrics['documents_processed']}"
        
        self.doc_stats_label.setText(stats_text)
    
    def refresh_document_list(self):
        """Refresh document list"""
        if not hasattr(self, 'doc_list'):
            return
            
        self.doc_list.clear()
        
        self.cursor.execute('SELECT id, filename, created_at FROM documents ORDER BY created_at DESC')
        documents = self.cursor.fetchall()
        
        for doc_id, filename, created_at in documents:
            item = QListWidgetItem(f"{filename} - {created_at}")
            item.setData(Qt.ItemDataRole.UserRole, doc_id)
            self.doc_list.addItem(item)
    
    def load_selected_document(self):
        """Load selected document"""
        current_item = self.doc_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Select a document first")
            return
        
        doc_id = current_item.data(Qt.ItemDataRole.UserRole)
        self.load_document_from_db_by_id(doc_id)
    
    def delete_selected_document(self):
        """Delete selected document"""
        current_item = self.doc_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Select a document first")
            return
        
        doc_id = current_item.data(Qt.ItemDataRole.UserRole)
        
        reply = QMessageBox.question(
            self, 'Delete Document',
            f'Delete this document?\n\n{current_item.text()}',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.delete_document_from_db(doc_id)
            self.refresh_document_list()
            self.update_doc_stats()
    
    def load_document_from_db(self, item):
        """Load document on double-click"""
        doc_id = item.data(Qt.ItemDataRole.UserRole)
        self.load_document_from_db_by_id(doc_id)
    
    def load_document_from_db_by_id(self, doc_id):
        """Load document by ID"""
        self.cursor.execute('SELECT filename, content FROM documents WHERE id = ?', (doc_id,))
        result = self.cursor.fetchone()
        
        if result:
            filename, content = result
            self.text_editor.setPlainText(content)
            self.current_file_path = None
            self.update_status_bar()
            if hasattr(self, 'status_file'):
                self.status_file.setText(f"DB: {filename}")
    
    def update_metrics_display(self):
        """Update metrics display"""
        report = "="*60 + "\n"
        report += "QUILL WRITING ASSISTANT - PERFORMANCE REPORT\n"
        report += "="*60 + "\n\n"
        
        report += "1. PYQT6 (GUI Framework)\n"
        report += "   Status: ✓ WORKING\n"
        report += "   - Collapsible sidebar panels\n"
        report += "   - Clickable suggestion cards\n"
        report += "   - Real-time updates\n\n"
        
        report += "2. LANGUAGETOOL (Grammar Checking)\n"
        report += "   Status: ✓ WORKING\n"
        report += f"   - Load time: {self.metrics['languagetool_load_time']:.2f}s\n"
        if self.metrics['check_times']:
            avg_check = sum(self.metrics['check_times']) / len(self.metrics['check_times'])
            report += f"   - Average check: {avg_check:.2f}s\n"
            report += f"   - Total checks: {len(self.metrics['check_times'])}\n\n"
        
        report += "3. FLAN-T5-BASE (AI Text Generation)\n"
        if TRANSFORMERS_AVAILABLE:
            if self.metrics['generation_times']:
                report += "   Status: ✓ WORKING\n"
                avg_gen = sum(t for _, t in self.metrics['generation_times']) / len(self.metrics['generation_times'])
                report += f"   - Generations: {len(self.metrics['generation_times'])}\n"
                report += f"   - Average time: {avg_gen:.2f}s\n\n"
            else:
                report += "   Status: ⚠ Not yet used\n\n"
        else:
            report += "   Status: ⚠ Not installed\n\n"
        
        report += "4. SQLITE (Database)\n"
        report += "   Status: ✓ WORKING\n"
        report += f"   - Documents saved: {self.metrics['documents_processed']}\n"
        if self.metrics['database_operations']:
            avg_db = sum(t for _, t in self.metrics['database_operations']) / len(self.metrics['database_operations'])
            report += f"   - Average op: {avg_db*1000:.2f}ms\n\n"
        
        self.metrics_display.setText(report)
    
    def setup_keyboard_shortcuts(self):
        """Setup shortcuts"""
        QShortcut(QKeySequence("Ctrl+O"), self).activated.connect(self.open_file)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.save_file)
        QShortcut(QKeySequence("Ctrl+Shift+S"), self).activated.connect(self.save_file_as)
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.text_editor_undo)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self.text_editor_redo)
    
    def setup_status_bar(self):
        """Setup status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.status_file = QLabel("No file")
        self.status_words = QLabel("Words: 0")
        self.status_chars = QLabel("Chars: 0")
        self.status_saved = QLabel("")
        
        self.status_bar.addWidget(self.status_file)
        self.status_bar.addPermanentWidget(self.status_words)
        self.status_bar.addPermanentWidget(self.status_chars)
        self.status_bar.addPermanentWidget(self.status_saved)
        
        self.text_editor.textChanged.connect(self.update_status_bar)
    
    def update_status_bar(self):
        """Update status bar"""
        if not hasattr(self, 'status_words'):
            return
            
        text = self.text_editor.toPlainText()
        word_count = len(text.split()) if text.strip() else 0
        char_count = len(text)
        
        self.status_words.setText(f"Words: {word_count}")
        self.status_chars.setText(f"Chars: {char_count}")
        
        if self.current_file_path:
            self.status_file.setText(f"File: {os.path.basename(self.current_file_path)}")
        else:
            self.status_file.setText("Untitled")
    
    def closeEvent(self, event):
        """Cleanup"""
        self.conn.close()
        event.accept()


def main():
    print("\n" + "="*60)
    print("QUILL WRITING ASSISTANT - ITERATION 4")
    print("="*60)
    print("\nFeatures:")
    print("  ✓ Real-time grammar checking")
    print("  ✓ Right-click error correction")
    print("  ✓ AI text continuation (3 suggestions)")
    print("  ✓ Collapsible sidebar panels")
    print("  ✓ SQLite database storage")
    print("  ✓ Professional UI/UX")
    print("\n" + "="*60 + "\n")
    
    app = QApplication(sys.argv)
    window = QuillRnD()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()