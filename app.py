from flask import Flask, request, jsonify, Response, redirect, session, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import mediapipe as mp
import numpy as np

# Fix mediapipe solutions compatibility
try:
    mp.solutions.face_mesh
except AttributeError:
    pass
import os
import base64
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:123%40Fi%24%23%26100@db.qsiedryjuusemdwkvcyf.supabase.co:5432/postgres")

# MediaPipe face mesh for embeddings
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

def _get_face_embedding(rgb_image):
    """Extract a simple face embedding using MediaPipe landmarks as a feature vector."""
    try:
        import mediapipe as mp2
        face_mesh = mp2.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        results = face_mesh.process(rgb_image)
        face_mesh.close()
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0].landmark
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32).flatten()
        coords = coords - coords.mean()
        norm = np.linalg.norm(coords)
        if norm == 0:
            return None
        return coords / norm
    except Exception as e:
        print("Embedding error:", e)
        return None

def _compare_embeddings(known_embedding, unknown_embedding, tolerance=0.6):
    """Compare two embeddings using cosine distance."""
    dist = np.linalg.norm(known_embedding - unknown_embedding)
    return dist < tolerance, dist

app = Flask(__name__)
app.secret_key = "school_attendance_v4_secret_key"

DB_FILE = "attendance.db"
IMAGE_DIR = "student_images"
TEACHER_IMAGE_DIR = "teacher_images"

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

known_encodings = []
known_students = []


# =========================================================
# DATABASE
# =========================================================
def get_db():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    return conn


def init_db():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(TEACHER_IMAGE_DIR, exist_ok=True)
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS teachers (
            id SERIAL PRIMARY KEY,
            teacher_name TEXT NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT NOT NULL,
            photo_path TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id SERIAL PRIMARY KEY,
            student_id TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            password TEXT NOT NULL,
            image_file TEXT NOT NULL,
            registered_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS classes (
            id SERIAL PRIMARY KEY,
            class_name TEXT NOT NULL,
            department TEXT,
            course TEXT,
            section_name TEXT,
            subject_name TEXT,
            teacher_id INTEGER NOT NULL,
            teacher_display_name TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (teacher_id) REFERENCES teachers(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS student_classes (
            id SERIAL PRIMARY KEY,
            student_id_fk INTEGER NOT NULL,
            class_id_fk INTEGER NOT NULL,
            UNIQUE(student_id_fk, class_id_fk),
            FOREIGN KEY (student_id_fk) REFERENCES students(id),
            FOREIGN KEY (class_id_fk) REFERENCES classes(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id SERIAL PRIMARY KEY,
            student_id TEXT NOT NULL,
            full_name TEXT NOT NULL,
            class_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            department TEXT,
            course TEXT,
            section_name TEXT,
            subject_name TEXT,
            teacher_name TEXT,
            status TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            UNIQUE(student_id, class_id, date),
            FOREIGN KEY (class_id) REFERENCES classes(id)
        )
    """)
    conn.commit()
    conn.close()


# =========================================================
# HELPERS
# =========================================================
def sanitize_filename(text):
    text = "".join(c for c in text if c.isalnum() or c in (" ", "_", "-")).strip()
    return text.replace(" ", "_")


def is_admin_logged_in():
    return session.get("admin_logged_in") is True


def is_teacher_logged_in():
    return session.get("teacher_logged_in") is True


def is_student_logged_in():
    return session.get("student_logged_in") is True


def get_logged_teacher_id():
    return session.get("teacher_id")


def get_logged_student_db_id():
    return session.get("student_db_id")


def admin_required():
    if not is_admin_logged_in():
        return redirect("/admin-login")
    return None


def teacher_required():
    if not is_teacher_logged_in():
        return redirect("/teacher-login")
    return None


def student_required():
    if not is_student_logged_in():
        return redirect("/student-login")
    return None


def student_exists(student_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM students WHERE lower(student_id)=lower(?)", (student_id.strip(),))
    row = cur.fetchone()
    conn.close()
    return row is not None


def teacher_username_exists(username):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM teachers WHERE lower(username)=lower(?)", (username.strip(),))
    row = cur.fetchone()
    conn.close()
    return row is not None


def get_all_students():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM students ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    return rows


def get_all_teachers():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM teachers ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    return rows


def get_all_classes():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT c.*, t.teacher_name
        FROM classes c
        LEFT JOIN teachers t ON c.teacher_id = t.id
        ORDER BY c.id DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


def get_teacher_classes(teacher_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT c.*, t.teacher_name
        FROM classes c
        LEFT JOIN teachers t ON c.teacher_id = t.id
        WHERE c.teacher_id=?
        ORDER BY c.id DESC
    """, (teacher_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_class_by_id(class_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT c.*, t.teacher_name
        FROM classes c
        LEFT JOIN teachers t ON c.teacher_id = t.id
        WHERE c.id=?
    """, (class_id,))
    row = cur.fetchone()
    conn.close()
    return row


def get_student_row_by_student_id(student_id_text):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM students WHERE student_id=?", (student_id_text,))
    row = cur.fetchone()
    conn.close()
    return row


def get_student_row_by_db_id(student_db_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM students WHERE id=?", (student_db_id,))
    row = cur.fetchone()
    conn.close()
    return row


def get_students_in_class(class_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT s.*
        FROM students s
        INNER JOIN student_classes sc ON sc.student_id_fk = s.id
        WHERE sc.class_id_fk=?
        ORDER BY s.full_name
    """, (class_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_classes_for_student(student_db_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT c.*
        FROM classes c
        INNER JOIN student_classes sc ON sc.class_id_fk = c.id
        WHERE sc.student_id_fk=?
        ORDER BY c.class_name
    """, (student_db_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


def assign_student_to_class(student_db_id, class_id):
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO student_classes (student_id_fk, class_id_fk)
            VALUES (?, ?)
        """, (student_db_id, class_id))
        conn.commit()
    except:
        pass
    conn.close()


def remove_student_from_class(student_db_id, class_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        DELETE FROM student_classes
        WHERE student_id_fk=? AND class_id_fk=?
    """, (student_db_id, class_id))
    conn.commit()
    conn.close()


def get_all_attendance():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM attendance ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    return rows


def get_attendance_for_teacher(teacher_name):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM attendance
        WHERE teacher_name=?
        ORDER BY id DESC
    """, (teacher_name,))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_attendance_for_class(class_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM attendance
        WHERE class_id=?
        ORDER BY date DESC, time DESC
    """, (class_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_attendance_for_student(student_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM attendance
        WHERE student_id=?
        ORDER BY date DESC, time DESC
    """, (student_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


def load_known_faces():
    global known_encodings, known_students
    known_encodings = []
    known_students = []

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, student_id, full_name, image_file FROM students")
    rows = cur.fetchall()
    conn.close()

    for row in rows:
        image_path = os.path.join(IMAGE_DIR, row["image_file"])
        if not os.path.exists(image_path):
            continue
        try:
            bgr = cv2.imread(image_path)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            embedding = _get_face_embedding(rgb)
            if embedding is not None:
                known_encodings.append(embedding)
                known_students.append({
                    "db_id": row["id"],
                    "student_id": row["student_id"],
                    "full_name": row["full_name"],
                    "image_file": row["image_file"]
                })
        except Exception as e:
            print("Face load error:", e)


def student_belongs_to_class(student_db_id, class_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT 1 FROM student_classes
        WHERE student_id_fk=? AND class_id_fk=?
    """, (student_db_id, class_id))
    row = cur.fetchone()
    conn.close()
    return row is not None


def mark_attendance(student_row, class_row, status="Present"):
    # Fetches local system date and time
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT * FROM attendance
        WHERE student_id=? AND class_id=? AND date=?
    """, (student_row["student_id"], class_row["id"], today))
    existing = cur.fetchone()

    teacher_name = class_row["teacher_display_name"] or class_row["teacher_name"] or ""

    if existing:
        cur.execute("""
            UPDATE attendance
            SET status=?, time=?, teacher_name=?
            WHERE id=?
        """, (status, now_time, teacher_name, existing["id"]))
    else:
        cur.execute("""
            INSERT INTO attendance (
                student_id, full_name, class_id, class_name,
                department, course, section_name, subject_name,
                teacher_name, status, date, time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            student_row["student_id"],
            student_row["full_name"],
            class_row["id"],
            class_row["class_name"],
            class_row["department"] or "",
            class_row["course"] or "",
            class_row["section_name"] or "",
            class_row["subject_name"] or "",
            teacher_name,
            status,
            today,
            now_time
        ))

    conn.commit()
    conn.close()


def get_attendance_count_for_student_class(student_id, class_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) as c
        FROM attendance
        WHERE student_id=? AND class_id=? AND status='Present'
    """, (student_id, class_id))
    present_count = cur.fetchone()["c"]

    cur.execute("""
        SELECT COUNT(*) as c
        FROM attendance
        WHERE student_id=? AND class_id=?
    """, (student_id, class_id))
    total_count = cur.fetchone()["c"]
    conn.close()
    return present_count, total_count


def get_percentage(student_id, class_id):
    present_count, total_count = get_attendance_count_for_student_class(student_id, class_id)
    if total_count == 0:
        return 0
    return round((present_count / total_count) * 100, 2)


def get_report_records(period="daily"):
    conn = get_db()
    cur = conn.cursor()

    today = datetime.now().date()

    if period == "daily":
        start_date = today
    elif period == "weekly":
        start_date = today - timedelta(days=7)
    elif period == "monthly":
        start_date = today - timedelta(days=30)
    else:
        start_date = today

    cur.execute("""
        SELECT * FROM attendance
        WHERE date >= ?
        ORDER BY date DESC, time DESC
    """, (start_date.strftime("%Y-%m-%d"),))
    rows = cur.fetchall()
    conn.close()
    return rows


# =========================================================
# SYSTEM PAGE TEMPLATE LAYOUT ENGINE
# =========================================================
def page_wrapper(title, body, is_admin=False, is_student=False, student_context=None, is_teacher=False, teacher_name=""):
    sidebar_html = ""
    
    if is_admin:
        sidebar_html = f"""
        <div class="sidebar-header">🛡️ Admin Dashboard</div>
        <nav class="sidebar-nav">
            <a href="/admin">📊 Dashboard Home</a>
            <a href="/student-register">🧑‍🎓 Register Student</a>
            <a href="/admin/reports">📋 Attendance Reports</a>
            <a href="/settings">⚙️ Account Settings</a>
            <hr style="border:0; border-top: 1px solid #374151; margin:15px 0;">
            <a href="/" style="background:#1f2937;">🏠 Back Main Site</a>
            <a href="/admin-logout" style="background:#991b1b; color:white;">🚪 Secure Logout</a>
        </nav>
        """
    elif is_teacher:
        sidebar_html = f"""
        <div class="sidebar-header">
            <div style="font-size:18px; font-weight:700; color:#38bdf8;">👨‍🏫 Instructor Panel</div>
            <div style="font-size:13px; color:#9ca3af; margin-top:4px;">{teacher_name}</div>
        </div>
        <nav class="sidebar-nav">
            <a href="/teacher">📋 My Classes Home</a>
            <a href="/settings">⚙️ Update Password</a>
            <hr style="border:0; border-top: 1px solid #374151; margin:15px 0;">
            <a href="/" style="background:#1f2937;">🏠 Back Main Site</a>
            <a href="/teacher-logout" style="background:#991b1b; color:white;">🚪 Secure Logout</a>
        </nav>
        """
    elif is_student and student_context:
        sidebar_html = f"""
        <div class="sidebar-header center">
            <img class="student-photo" style="width:70px; height:70px; border-radius:50%; margin-bottom:10px; border:2px solid #2563eb;" src="/student-image/{student_context['image_file']}">
            <div style="font-size: 16px; font-weight:600;">{student_context['full_name']}</div>
            <div style="font-size: 12px; color:#9ca3af; margin-top:2px;">ID: {student_context['student_id']}</div>
        </div>
        <nav class="sidebar-nav">
            <a href="/student" class="active">📚 My Profile Home</a>
            <a href="/student/scan" style="background:#059669; color:white;">📸 Face Check-In</a>
            <a href="/settings">⚙️ Update Password</a>
            <hr style="border:0; border-top: 1px solid #374151; margin:15px 0;">
            <a href="/" style="background:#1f2937;">🏠 Back Main Site</a>
            <a href="/student-logout" style="background:#991b1b; color:white;">🚪 Secure Logout</a>
        </nav>
        """

    if is_admin or is_student or is_teacher:
        content_html = f"""
        <div class="dashboard-wrapper">
            <div class="sidebar-panel" id="sidebarMenu">
                {sidebar_html}
            </div>
            <button class="mobile-menu-trigger" onclick="toggleSidebar(event)">☰ Menu</button>
            <div class="workspace-panel" onclick="closeSidebar()">
                <div class="box">
                    {body}
                </div>
            </div>
        </div>
        """
    else:
        content_html = f"""
        <div class="box" style="margin:40px auto; max-width:1100px;">
            {body}
        </div>
        """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
        <style>
        body{{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg,#eef2ff,#f8fafc);
            margin:0;
            padding:0;
        }}
        .box {{
            background:white;
            padding:24px;
            border-radius:16px;
            box-shadow:0 4px 20px rgba(0,0,0,0.05);
        }}
        .dashboard-wrapper {{
            display: flex;
            min-height: 100vh;
            position: relative;
        }}
        .sidebar-panel {{
            width: 260px;
            background: #111827;
            color: #f3f4f6;
            display: flex;
            flex-direction: column;
            position: fixed;
            top: 0;
            left: 0;
            bottom: 0;
            z-index: 999;
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 4px 0 15px rgba(0,0,0,0.15);
            padding: 20px 0;
            box-sizing: border-box;
        }}
        .sidebar-header {{
            font-size: 18px;
            font-weight: 700;
            padding: 0 20px 20px 20px;
            border-bottom: 1px solid #1f2937;
            color: white;
        }}
        .sidebar-nav {{
            display: flex;
            flex-direction: column;
            gap: 6px;
            padding: 20px 12px;
        }}
        .sidebar-nav a {{
            color: #d1d5db;
            text-decoration: none;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s ease;
        }}
        .sidebar-nav a:hover, .sidebar-nav a.active {{
            background: #2563eb;
            color: white;
        }}
        .workspace-panel {{
            flex: 1;
            margin-left: 260px;
            padding: 30px 24px;
            transition: margin-left 0.3s ease;
            width: calc(100% - 260px);
            box-sizing: border-box;
        }}
        .mobile-menu-trigger {{
            display: none;
            position: fixed;
            top: 15px;
            left: 15px;
            z-index: 1000;
            background: #2563eb;
            color: white;
            border: none;
            padding: 10px 16px;
            border-radius: 8px;
            font-weight: bold;
            cursor: pointer;
        }}
        table{{
            width:100%;
            border-collapse:collapse;
            background:white;
            border-radius:12px;
            overflow:hidden;
            margin-top:12px;
        }}
        th{{
            background:#1e3a8a;
            color:white;
            padding:14px;
            text-align:left;
            font-size:14px;
        }}
        td{{
            padding:12px;
            border-bottom:1px solid #e2e8f0;
            font-size:14px;
        }}
        input,select,textarea{{
            width:100%;
            max-width:400px;
            padding:10px 14px;
            border:1px solid #cbd5e1;
            border-radius:8px;
            margin:6px 0;
            box-sizing:border-box;
        }}
        button,.btn{{
            display:inline-block;
            padding:10px 20px;
            margin:4px 2px;
            border:none;
            border-radius:8px;
            text-decoration:none;
            font-weight:600;
            cursor:pointer;
            background:#2563eb;
            color:white;
        }}
        .btn.green{{background:#10b981;}}
        .btn.orange{{background:#f97316;}}
        .btn.red{{background:#ef4444;}}
        .btn.dark{{background:#1e293b;}}
        @media (max-width: 1024px) {{
            .sidebar-panel {{ transform: translateX(-100%); }}
            .sidebar-panel.visible {{ transform: translateX(0); }}
            .workspace-panel {{ margin-left: 0; width: 100%; padding-top: 75px; }}
            .mobile-menu-trigger {{ display: block; }}
        }}
        </style>
        <script>
            function toggleSidebar(e) {{
                e.stopPropagation();
                document.getElementById('sidebarMenu').classList.toggle('visible');
            }}
            function closeSidebar() {{
                const menu = document.getElementById('sidebarMenu');
                if(menu) menu.classList.remove('visible');
            }}
        </script>
        <title>{title}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body>
        {content_html}
    </body>
    </html>
    """


# =========================================================
# UNIVERSAL CONFIGURATION & SETTINGS
# =========================================================
@app.route("/settings", methods=["GET", "POST"])
def user_settings():
    role = None
    user_id = None
    display_name = ""
    student_ctx = None
    teacher_name_str = ""
    
    if is_admin_logged_in():
        role = "admin"
        display_name = "Administrator"
    elif is_teacher_logged_in():
        role = "teacher"
        user_id = get_logged_teacher_id()
        teacher_name_str = session.get("teacher_name", "Teacher")
        display_name = teacher_name_str
    elif is_student_logged_in():
        role = "student"
        user_id = get_logged_student_db_id()
        student_ctx = get_student_row_by_db_id(user_id)
        display_name = session.get("student_name", "Student")
    else:
        return redirect("/")

    if request.method == "POST":
        old_password = request.form.get("old_password", "").strip()
        new_password = request.form.get("new_password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        if not old_password or not new_password or not confirm_password:
            return page_wrapper("Settings", "<p class='text-red-500 font-bold'>All fields are required.</p>", is_student=(role == "student"), student_context=student_ctx, is_teacher=(role == "teacher"), teacher_name=teacher_name_str)
        
        if new_password != confirm_password:
            return page_wrapper("Settings", "<p class='text-red-500 font-bold'>New entries do not match.</p>", is_student=(role == "student"), student_context=student_ctx, is_teacher=(role == "teacher"), teacher_name=teacher_name_str)

        conn = get_db()
        cur = conn.cursor()

        if role == "admin":
            global ADMIN_PASSWORD
            if old_password != ADMIN_PASSWORD:
                conn.close()
                return page_wrapper("Settings", "<p class='text-red-500 font-bold'>Incorrect existing password.</p>")
            ADMIN_PASSWORD = new_password
            conn.close()
            return "<script>alert('Admin password updated successfully!'); window.location.href='/admin';</script>"

        elif role == "teacher":
            cur.execute("SELECT password FROM teachers WHERE id=?", (user_id,))
            row = cur.fetchone()
            if not row or row["password"] != old_password:
                conn.close()
                return page_wrapper("Settings", "<p class='text-red-500 font-bold'>Incorrect old password.</p>", is_teacher=True, teacher_name=teacher_name_str)
            
            cur.execute("UPDATE teachers SET password=? WHERE id=?", (new_password, user_id))
            conn.commit()
            conn.close()
            return "<script>alert('Teacher password updated successfully!'); window.location.href='/teacher';</script>"

        elif role == "student":
            cur.execute("SELECT password FROM students WHERE id=?", (user_id,))
            row = cur.fetchone()
            if not row or row["password"] != old_password:
                conn.close()
                return page_wrapper("Settings", "<p class='text-red-500 font-bold'>Incorrect old password.</p>", is_student=True, student_context=student_ctx)
            
            cur.execute("UPDATE students SET password=? WHERE id=?", (new_password, user_id))
            conn.commit()
            conn.close()
            return "<script>alert('Student password updated successfully!'); window.location.href='/student';</script>"

    body = f"""
    <div class="max-w-xl">
        <h1 class="text-2xl font-bold text-slate-800 mb-2">Account Portal Settings</h1>
        <p class="text-sm text-slate-500 mb-6">Change local password credentials below for {display_name}</p>
        <form method="POST" action="/settings" class="space-y-4">
            <div>
                <label class="block text-sm font-medium text-slate-700">Current Password</label>
                <input type="password" name="old_password" class="mt-1 block w-full px-3 py-2 border rounded-lg" required>
            </div>
            <div>
                <label class="block text-sm font-medium text-slate-700">New Password</label>
                <input type="password" name="new_password" class="mt-1 block w-full px-3 py-2 border rounded-lg" required>
            </div>
            <div>
                <label class="block text-sm font-medium text-slate-700">Confirm New Password</label>
                <input type="password" name="confirm_password" class="mt-1 block w-full px-3 py-2 border rounded-lg" required>
            </div>
            <div class="pt-2">
                <button type="submit" class="bg-blue-600 text-white font-semibold px-4 py-2 rounded-lg hover:bg-blue-700">Save Configuration</button>
                <a href="/{role}" class="inline-block ml-2 px-4 py-2 bg-slate-100 text-slate-700 font-semibold rounded-lg hover:bg-slate-200">Cancel</a>
            </div>
        </form>
    </div>
    """
    return page_wrapper("Account Settings", body, is_admin=(role == "admin"), is_student=(role == "student"), student_context=student_ctx, is_teacher=(role == "teacher"), teacher_name=teacher_name_str)


# =========================================================
# SYSTEM INDEX LANDING
# =========================================================
@app.route("/")
def home():
    return page_wrapper("School Attendance V4", """
    <div class="text-center py-12 max-w-4xl mx-auto">
        <h1 class="text-4xl font-extrabold text-slate-900 tracking-tight mb-3">🎓 Smart Attendance Portal</h1>
        <p class="text-lg text-slate-600 mb-8">Integrated AI face recognition mapping alongside professional manual proctor sheets.</p>
        <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 justify-center items-center">
            <a class="p-4 bg-orange-500 text-white rounded-xl font-bold hover:bg-orange-600 shadow-md transition" href="/student-register">🧑‍🎓 Register Face</a>
            <a class="p-4 bg-slate-800 text-white rounded-xl font-bold hover:bg-slate-900 shadow-md transition" href="/admin-login">🔐 Admin Login</a>
            <a class="p-4 bg-purple-600 text-white rounded-xl font-bold hover:bg-purple-700 shadow-md transition" href="/teacher-login">👨‍🏫 Teacher Log-In</a>
            <a class="p-4 bg-emerald-600 text-white rounded-xl font-bold hover:bg-emerald-700 shadow-md transition" href="/student-login">📚 Student Log-In</a>
        </div>
        <div class="mt-12 p-6 bg-slate-50 rounded-2xl border border-slate-200 text-left">
            <h3 class="text-lg font-bold text-slate-800 mb-2">Core Features Matrix</h3>
            <p class="text-slate-600 text-sm">Automated live canvas face encodings lookup matching via OpenCV, deep dashboard reporting matrix tables, secure multi-tenant role session isolation guards, and complete batch roster manual ticking sheet submissions.</p>
        </div>
    </div>
    """)


# =========================================================
# PORTAL AUTHENTICATION ROUTING
# =========================================================
def login_page(title, action, user_placeholder, pass_placeholder, error_message="", extra_fields=""):
    return page_wrapper(title, f"""
        <div class="max-w-md mx-auto my-8 p-6 bg-white border border-slate-200 rounded-xl shadow-sm">
            <h2 class="text-2xl font-bold text-slate-800 text-center mb-6">{title}</h2>
            <form method="POST" action="{action}" class="space-y-4">
                <div>
                    <input type="text" name="username" placeholder="{user_placeholder}" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500" required>
                </div>
                <div>
                    <input type="password" name="password" placeholder="{pass_placeholder}" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500" required>
                </div>
                {extra_fields}
                <button class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition" type="submit">Sign In</button>
            </form>
            {f'<div class="mt-3 text-sm text-red-500 text-center font-semibold">{error_message}</div>' if error_message else ''}
            <div class="mt-4 border-t pt-4 text-center">
                <a class="text-sm text-blue-600 hover:underline" href="/">← Back to Landing Home</a>
            </div>
        </div>
    """)


@app.route("/admin-login", methods=["GET", "POST"])
def admin_login():
    if request.method == "GET":
        return login_page("Admin Login", "/admin-login", "Admin Username", "Admin Password")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session.clear()
        session["admin_logged_in"] = True
        return redirect("/admin")

    return login_page("Admin Login", "/admin-login", "Admin Username", "Admin Password", "Invalid admin username or password")


@app.route("/teacher-login", methods=["GET", "POST"])
def teacher_login():
    if request.method == "GET":
        return login_page("Teacher Login", "/teacher-login", "Teacher Username", "Teacher Password")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM teachers WHERE username=? AND password=?", (username, password))
    teacher = cur.fetchone()
    conn.close()

    if teacher:
        session.clear()
        session["teacher_logged_in"] = True
        session["teacher_id"] = teacher["id"]
        session["teacher_name"] = teacher["teacher_name"]
        return redirect("/teacher")

    return login_page("Teacher Login", "/teacher-login", "Teacher Username", "Teacher Password", "Invalid teacher username or password")


@app.route("/student-login", methods=["GET", "POST"])
def student_login():
    if request.method == "GET":
        return login_page("Student Login", "/student-login", "Student ID", "Student Password")

    student_id = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM students WHERE student_id=? AND password=?", (student_id, password))
    student = cur.fetchone()
    conn.close()

    if student:
        session.clear()
        session["student_logged_in"] = True
        session["student_db_id"] = student["id"]
        session["student_id"] = student["student_id"]
        session["student_name"] = student["full_name"]
        return redirect("/student")

    return login_page("Student Login", "/student-login", "Student ID", "Student Password", "Invalid student ID or password")


@app.route("/admin-logout")
def admin_logout():
    session.clear()
    return redirect("/admin-login")


@app.route("/teacher-logout")
def teacher_logout():
    session.clear()
    return redirect("/teacher-login")


@app.route("/student-logout")
def student_logout():
    session.clear()
    return redirect("/student-login")


# =========================================================
# STUDENT REGISTRATION PAGE
# =========================================================
@app.route("/student-register")
def student_register():
    return page_wrapper("Student Registration", """
        <div class="max-w-2xl mx-auto text-center">
            <h1 class="text-3xl font-bold text-slate-800 mb-2">Student Face Registration</h1>
            <p class="text-sm text-slate-500 mb-6">Input student data fields and click register to create face vector profile mappings</p>
            
            <div class="space-y-3 max-w-md mx-auto mb-6 text-left">
                <input type="text" id="studentId" placeholder="Student Alphanumeric ID" class="w-full px-3 py-2 border rounded-lg">
                <input type="text" id="fullName" placeholder="Full Registered Name" class="w-full px-3 py-2 border rounded-lg">
                <input type="password" id="password" placeholder="Roster Login Password" class="w-full px-3 py-2 border rounded-lg">
            </div>

            <video id="video" autoplay playsinline muted class="w-full max-w-md mx-auto bg-black rounded-xl shadow-inner mb-4 border border-slate-300"></video>
            
            <div class="flex justify-center gap-2 flex-wrap mb-4">
                <button class="bg-emerald-600 hover:bg-emerald-700 text-white px-4 py-2 rounded-lg font-semibold" onclick="registerFace()">Capture & Register Face</button>
                <button class="bg-orange-500 hover:bg-orange-600 text-white px-4 py-2 rounded-lg font-semibold" onclick="startCamera()">Restart Stream Feed</button>
                <button class="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg font-semibold" onclick="switchCamera()">Switch Orientation</button>
            </div>

            <div id="status" class="text-sm font-semibold text-slate-700 mt-2">Please allow camera access camera hardware controls</div>
        </div>

<script>
const video = document.getElementById('video');
const statusDiv = document.getElementById('status');
let stream = null;
let currentFacingMode = "user";

async function startCamera() {
    try {
        statusDiv.innerText = "Starting camera...";
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            statusDiv.innerText = "Camera not supported. Please use HTTPS and a modern browser.";
            return;
        }
        if (stream) {
            stream.getTracks().forEach(t => t.stop());
            stream = null;
        }
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: { ideal: currentFacingMode },
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                },
                audio: false
            });
        } catch (constraintErr) {
            stream = await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: false
            });
        }
        video.srcObject = stream;
        video.addEventListener('loadedmetadata', async function onMeta() {
            video.removeEventListener('loadedmetadata', onMeta);
            try {
                await video.play();
                statusDiv.innerText = "Camera ready";
            } catch (playErr) {
                statusDiv.innerText = "Tap the video area frame to play";
                video.onclick = async () => {
                    await video.play();
                    statusDiv.innerText = "Camera ready";
                    video.onclick = null;
                };
            }
        }, { once: true });
    } catch (err) {
        statusDiv.innerText = "Camera system initialization mapping failure: " + err.message;
    }
}

function switchCamera() {
    currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
    startCamera();
}

async function registerFace() {
    try {
        const studentId = document.getElementById('studentId').value.trim();
        const fullName = document.getElementById('fullName').value.trim();
        const password = document.getElementById('password').value.trim();
        
        if (!studentId || !fullName || !password) {
            alert("All values are required prior to scanning");
            return;
        }

        statusDiv.innerText = "Capturing canvas frame matrix...";
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const image = canvas.toDataURL('image/jpeg');

        statusDiv.innerText = "Uploading credentials to database registry...";
        const response = await fetch('/register-face', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ student_id: studentId, full_name: fullName, password: password, image: image })
        });
        const data = await response.json();
        statusDiv.innerText = data.message;
        if (data.success) {
            alert(data.message);
        }
    } catch (err) {
        statusDiv.innerText = "Registration failed: " + err.message;
    }
}
startCamera();
</script>
""", is_admin=is_admin_logged_in())


@app.route("/register-face", methods=["POST"])
def register_face():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No data received"})
        student_id = data.get("student_id", "").strip()
        full_name = data.get("full_name", "").strip()
        password = data.get("password", "").strip()
        image_data = data.get("image", "")

        if not student_id or not full_name or not password or not image_data:
            return jsonify({"success": False, "message": "All database fields are required"})
        if student_exists(student_id):
            return jsonify({"success": False, "message": "This Student ID already holds a target map registry record"})

        safe_id = sanitize_filename(student_id)
        safe_name = sanitize_filename(full_name)
        filename = f"{safe_id}_{safe_name}.jpg"
        file_path = os.path.join(IMAGE_DIR, filename)

        if "," in image_data:
            image_data = image_data.split(",")
        image_data = image_data.replace(" ", "+")
        missing_padding = len(image_data) % 4
        if missing_padding:
            image_data += "=" * (4 - missing_padding)

        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "message": "Invalid image matrix array decoder mapping"})

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if len(face_encodings) == 0:
            return jsonify({"success": False, "message": "No human faces found in frame context. Please try again."})

        cv2.imwrite(file_path, frame)

        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO students (student_id, full_name, password, image_file, registered_at)
            VALUES (?, ?, ?, ?, ?)
        """, (student_id, full_name, password, filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()

        load_known_faces()
        return jsonify({"success": True, "message": f"Successfully registered face map vector profile for {full_name}!"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Internal mapping failure error: {str(e)}"})


# =========================================================
# ADMIN CONTROLLER DASHBOARD
# =========================================================
@app.route("/admin")
def admin_dashboard():
    protect = admin_required()
    if protect:
        return protect

    students = get_all_students()
    teachers = get_all_teachers()
    classes = get_all_classes()
    attendance = get_all_attendance()

    body = f"""
    <div class="space-y-6">
        <div>
            <h1 class="text-3xl font-extrabold text-slate-800">🛡️ System Administration Console</h1>
            <p class="text-sm text-slate-500">Configure global classes, assign instructor roles, and view records metrics logs.</p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="p-6 bg-slate-50 border border-slate-200 rounded-xl">
                <h2 class="text-xl font-bold text-slate-800 mb-3">Register New Instructor Profile</h2>
                <form method="POST" action="/admin/create-teacher" class="space-y-3">
                    <input type="text" name="teacher_name" placeholder="Instructor Display Full Name" class="w-full px-3 py-2 border rounded-lg" required>
                    <input type="text" name="username" placeholder="Login Account Username" class="w-full px-3 py-2 border rounded-lg" required>
                    <input type="text" name="password" placeholder="System Access Password" class="w-full px-3 py-2 border rounded-lg" required>
                    <button class="bg-blue-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-blue-700" type="submit">Create Profile</button>
                </form>
            </div>

            <div class="p-6 bg-slate-50 border border-slate-200 rounded-xl">
                <h2 class="text-xl font-bold text-slate-800 mb-3">Create Dynamic Course Classroom</h2>
                <form method="POST" action="/admin/create-class" class="space-y-2">
                    <input type="text" name="class_name" placeholder="Class Target Label" class="w-full px-3 py-2 border rounded-lg" required>
                    <input type="text" name="department" placeholder="Department Stream Name" class="w-full px-3 py-2 border rounded-lg">
                    <input type="text" name="course" placeholder="Course ID Reference" class="w-full px-3 py-2 border rounded-lg">
                    <input type="text" name="section_name" placeholder="Section Identity Identifier" class="w-full px-3 py-2 border rounded-lg">
                    <input type="text" name="subject_name" placeholder="Subject Topic Code Name" class="w-full px-3 py-2 border rounded-lg">
                    <select name="teacher_id" class="w-full px-3 py-2 border rounded-lg" required>
                        <option value="">Assign Proctor Profile</option>
    """
    for t in teachers:
        body += f'<option value="{t["id"]}">{t["teacher_name"]} ({t["username"]})</option>'
    body += """
                    </select>
                    <button class="bg-blue-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-blue-700 mt-2" type="submit">Create Classroom Matrix</button>
                </form>
            </div>
        </div>

        <div class="p-6 bg-slate-50 border border-slate-200 rounded-xl">
            <h2 class="text-xl font-bold text-slate-800 mb-3">Assign Registered Student to Course Class</h2>
            <form method="POST" action="/admin/assign-student-class" class="grid grid-cols-1 md:grid-cols-3 gap-3 items-center">
                <select name="student_db_id" class="w-full px-3 py-2 border rounded-lg" required>
                    <option value="">Select Student Profile</option>
    """
    for s in students:
        body += f'<option value="{s["id"]}">{s["student_id"]} - {s["full_name"]}</option>'
    body += """
                </select>
                <select name="class_id" class="w-full px-3 py-2 border rounded-lg" required>
                    <option value="">Select Target Class</option>
    """
    for c in classes:
        teacher_name = c["teacher_display_name"] or c["teacher_name"] or ""
        body += f'<option value="{c["id"]}">{c["class_name"]} | {c["subject_name"] or ""} | {teacher_name}</option>'
    body += """
                </select>
                <button class="bg-emerald-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-emerald-700 w-full md:w-auto" type="submit">Enroll Map Student</button>
            </form>
        </div>

        <div class="overflow-x-auto bg-white rounded-xl shadow-sm border">
            <div class="p-4 border-b bg-slate-50 font-bold text-slate-800 text-lg">Active Instructors Registry</div>
            <table class="w-full text-left">
                <thead>
                    <tr class="bg-slate-100 text-slate-700">
                        <th class="p-3">ID</th>
                        <th class="p-3">Teacher Name</th>
                        <th class="p-3">Username</th>
                        <th class="p-3">Password</th>
                        <th class="p-3">Action Controls</th>
                    </tr>
                </thead>
                <tbody>
    """
    if teachers:
        for t in teachers:
            body += f"""
                <tr class="border-b">
                    <td class="p-3">{t["id"]}</td>
                    <td class="p-3 font-medium">{t["teacher_name"]}</td>
                    <td class="p-3">{t["username"]}</td>
                    <td class="p-3 font-mono">{t["password"]}</td>
                    <td class="p-3">
                        <a class="text-blue-600 hover:underline mr-2" href="/admin/edit-teacher/{t['id']}">Edit</a>
                        <a class="text-red-600 hover:underline" href="/admin/delete-teacher/{t['id']}" onclick="return confirm('Purge data profile matrix?')">Delete</a>
                    </td>
                </tr>
            """
    else:
        body += "<tr><td colspan='5' class='p-4 text-center text-slate-400'>No instructor records present inside system state.</td></tr>"
    body += """
                </tbody>
            </table>
        </div>

        <div class="overflow-x-auto bg-white rounded-xl shadow-sm border">
            <div class="p-4 border-b bg-slate-50 font-bold text-slate-800 text-lg">Course Classes Registry</div>
            <table class="w-full text-left">
                <thead>
                    <tr class="bg-slate-100 text-slate-700">
                        <th class="p-3">ID</th>
                        <th class="p-3">Class Target Label</th>
                        <th class="p-3">Subject Code Title</th>
                        <th class="p-3">Assigned Teacher</th>
                        <th class="p-3">Action Control Triggers</th>
                    </tr>
                </thead>
                <tbody>
    """
    if classes:
        for c in classes:
            t_name = c["teacher_display_name"] or c["teacher_name"] or "Unassigned"
            body += f"""
                <tr class="border-b">
                    <td class="p-3">{c["id"]}</td>
                    <td class="p-3 font-semibold"><a class="text-blue-600 hover:underline" href="/admin/class/{c["id"]}">{c["class_name"]}</a></td>
                    <td class="p-3">{c["subject_name"] or "None"}</td>
                    <td class="p-3 text-slate-600">{t_name}</td>
                    <td class="p-3">
                        <a class="text-blue-500 hover:underline mr-2" href="/admin/edit-class/{c['id']}">Edit</a>
                        <a class="text-red-500 hover:underline" href="/admin/delete-class/{c['id']}" onclick="return confirm('Delete classroom mapping?')">Delete</a>
                    </td>
                </tr>
            """
    else:
        body += "<tr><td colspan='5' class='p-4 text-center text-slate-400'>No classroom instances instantiated inside system database state.</td></tr>"
    body += """
                </tbody>
            </table>
        </div>

        <div class="overflow-x-auto bg-white rounded-xl shadow-sm border">
            <div class="p-4 border-b bg-slate-50 font-bold text-slate-800 text-lg">Global Student Enrolled Registry Matrix</div>
            <table class="w-full text-left">
                <thead>
                    <tr class="bg-slate-100 text-slate-700">
                        <th class="p-3">Face Capture Link</th>
                        <th class="p-3">Student Identifier ID</th>
                        <th class="p-3">Full Registered Name</th>
                        <th class="p-3">Portal Pass Key</th>
                        <th class="p-3">Registered At</th>
                        <th class="p-3">Action Overrides</th>
                    </tr>
                </thead>
                <tbody>
    """
    if students:
        for s in students:
            body += f"""
                <tr class="border-b">
                    <td class="p-3"><img class="w-10 h-10 object-cover rounded-full border shadow-sm" src="/student-image/{s["image_file"]}"></td>
                    <td class="p-3 font-mono text-xs">{s["student_id"]}</td>
                    <td class="p-3 font-medium">{s["full_name"]}</td>
                    <td class="p-3 font-mono text-xs">{s["password"]}</td>
                    <td class="p-3 text-xs text-slate-500">{s["registered_at"]}</td>
                    <td class="p-3">
                        <a class="text-blue-500 hover:underline mr-2" href="/admin/edit-student/{s['id']}">Edit</a>
                        <a class="text-red-500 hover:underline" href="/admin/delete-student/{s['student_id']}" onclick="return confirm('Delete student entirely?')">Delete</a>
                    </td>
                </tr>
            """
    else:
        body += "<tr><td colspan='6' class='p-4 text-center text-slate-400'>No profiles detected.</td></tr>"
    body += """
                </tbody>
            </table>
        </div>

        <div class="overflow-x-auto bg-white rounded-xl shadow-sm border">
            <div class="p-4 border-b bg-slate-50 flex justify-between items-center flex-wrap gap-2">
                <span class="font-bold text-slate-800 text-lg">Historical Records Logs</span>
                <a class="bg-emerald-600 text-white font-semibold text-xs px-3 py-1.5 rounded-lg hover:bg-emerald-700 transition" href="/export-attendance">📥 Export Sheet (.CSV)</a>
            </div>
            <table class="w-full text-left">
                <thead>
                    <tr class="bg-slate-100 text-slate-700">
                        <th class="p-3">Date</th>
                        <th class="p-3">Timestamp</th>
                        <th class="p-3">ID</th>
                        <th class="p-3">Student Name</th>
                        <th class="p-3">Classroom Target</th>
                        <th class="p-3">Section</th>
                        <th class="p-3">Subject Topic</th>
                        <th class="p-3">Status Log</th>
                        <th class="p-3">Authorized Proctor</th>
                    </tr>
                </thead>
                <tbody>
    """
    if attendance:
        for a in attendance:
            body += f"""
                <tr class="border-b text-xs">
                    <td class="p-3 whitespace-nowrap">{a["date"]}</td>
                    <td class="p-3 text-slate-500">{a["time"]}</td>
                    <td class="p-3 font-mono">{a["student_id"]}</td>
                    <td class="p-3 font-medium text-slate-800">{a["full_name"]}</td>
                    <td class="p-3 font-semibold">{a["class_name"]}</td>
                    <td class="p-3">{a["section_name"] or ""}</td>
                    <td class="p-3">{a["subject_name"] or ""}</td>
                    <td class="p-3"><span class="px-2 py-0.5 rounded-full text-xs font-bold {'bg-emerald-100 text-emerald-800' if a['status']=='Present' else 'bg-rose-100 text-rose-800'}">{a["status"]}</span></td>
                    <td class="p-3 text-slate-500">{a["teacher_name"] or ""}</td>
                </tr>
            """
    else:
        body += "<tr><td colspan='9' class='p-4 text-center text-slate-400'>No historical logs have been populated.</td></tr>"
    body += """
                </tbody>
            </table>
        </div>
    </div>
    """
    return page_wrapper("Admin Dashboard", body, is_admin=True)


# =========================================================
# ADMIN CONTROLLERS IMPLEMENTATION MUTATION ROUTING
# =========================================================
@app.route("/admin/create-teacher", methods=["POST"])
def admin_create_teacher():
    protect = admin_required()
    if protect:
        return protect
    teacher_name = request.form.get("teacher_name", "").strip()
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    if not teacher_name or not username or not password:
        return "<script>alert('All instructor fields are required');window.location.href='/admin';</script>"
    if teacher_username_exists(username):
        return "<script>alert('Teacher login username already occupies registry namespace');window.location.href='/admin';</script>"

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO teachers (teacher_name, username, password, created_at)
        VALUES (?, ?, ?, ?)
    """, (teacher_name, username, password, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    return "<script>alert('Instructor profile instantiated successfully');window.location.href='/admin';</script>"


@app.route("/admin/edit-teacher/<int:teacher_id>", methods=["GET", "POST"])
def admin_edit_teacher(teacher_id):
    protect = admin_required()
    if protect:
        return protect
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM teachers WHERE id=?", (teacher_id,))
    teacher = cur.fetchone()
    if not teacher:
        conn.close()
        return "Teacher target row index match not located inside dynamic memory state", 404

    if request.method == "POST":
        teacher_name = request.form.get("teacher_name", "").strip()
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        cur.execute("""
            UPDATE teachers SET teacher_name=?, username=?, password=? WHERE id=?
        """, (teacher_name, username, password, teacher_id))
        conn.commit()
        
        cur.execute("UPDATE classes SET teacher_display_name=? WHERE teacher_id=?", (teacher_name, teacher_id))
        conn.commit()
        
        conn.close()
        return "<script>alert('Teacher entity parameters updated successfully');window.location.href='/admin';</script>"

    body = f"""
    <div class="max-w-md">
        <h1 class="text-2xl font-bold mb-4">Modify Instructor Record</h1>
        <form method="POST" class="space-y-4">
            <div>
                <label class="block text-sm font-medium">Instructor Display Full Name</label>
                <input type="text" name="teacher_name" value="{teacher["teacher_name"]}" class="w-full px-3 py-2 border rounded-lg" required>
            </div>
            <div>
                <label class="block text-sm font-medium">Login Username Reference</label>
                <input type="text" name="username" value="{teacher["username"]}" class="w-full px-3 py-2 border rounded-lg" required>
            </div>
            <div>
                <label class="block text-sm font-medium">Access Pass Key Token</label>
                <input type="text" name="password" value="{teacher["password"]}" class="w-full px-3 py-2 border rounded-lg" required>
            </div>
            <button type="submit" class="bg-blue-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-blue-700">Save Modifications</button>
            <a href="/admin" class="ml-2 inline-block bg-slate-100 text-slate-700 font-bold py-2 px-4 rounded-lg">Cancel</a>
        </form>
    </div>
    """
    conn.close()
    return page_wrapper("Edit Teacher Entity Matrix", body, is_admin=True)


@app.route("/admin/delete-teacher/<int:teacher_id>")
def admin_delete_teacher(teacher_id):
    protect = admin_required()
    if protect:
        return protect
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM teachers WHERE id=?", (teacher_id,))
    conn.commit()
    conn.close()
    return "<script>alert('Teacher purged successfully');window.location.href='/admin';</script>"


@app.route("/admin/create-class", methods=["POST"])
def admin_create_class():
    protect = admin_required()
    if protect:
        return protect
    class_name = request.form.get("class_name", "").strip()
    department = request.form.get("department", "").strip()
    course = request.form.get("course", "").strip()
    section_name = request.form.get("section_name", "").strip()
    subject_name = request.form.get("subject_name", "").strip()
    teacher_id = request.form.get("teacher_id", "").strip()

    if not class_name or not teacher_id:
        return "<script>alert('Class name and teacher selection are mandatory');window.location.href='/admin';</script>"

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM teachers WHERE id=?", (teacher_id,))
    teacher = cur.fetchone()
    if not teacher:
        conn.close()
        return "<script>alert('Selected proctor identity mismatch data reference code');window.location.href='/admin';</script>"

    cur.execute("""
        INSERT INTO classes (
            class_name, department, course, section_name, subject_name, teacher_id, teacher_display_name, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (class_name, department, course, section_name, subject_name, teacher["id"], teacher["teacher_name"], datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    return "<script>alert('New classroom created successfully');window.location.href='/admin';</script>"


@app.route("/admin/edit-class/<int:class_id>", methods=["GET", "POST"])
def admin_edit_class(class_id):
    protect = admin_required()
    if protect:
        return protect
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM classes WHERE id=?", (class_id,))
    class_row = cur.fetchone()
    if not class_row:
        conn.close()
        return "Class file matching matrix not detected index inside standard layout references", 404

    if request.method == "POST":
        class_name = request.form.get("class_name", "").strip()
        department = request.form.get("department", "").strip()
        course = request.form.get("course", "").strip()
        section_name = request.form.get("section_name", "").strip()
        subject_name = request.form.get("subject_name", "").strip()
        teacher_id = request.form.get("teacher_id", "").strip()

        cur.execute("SELECT * FROM teachers WHERE id=?", (teacher_id,))
        t = cur.fetchone()
        t_name = t["teacher_name"] if t else ""

        cur.execute("""
            UPDATE classes SET class_name=?, department=?, course=?, section_name=?, subject_name=?, teacher_id=?, teacher_display_name=?
            WHERE id=?
        """, (class_name, department, course, section_name, subject_name, teacher_id, t_name, class_id))
        conn.commit()
        conn.close()
        return "<script>alert('Classroom record adjustments committed!');window.location.href='/admin';</script>"

    teachers = get_all_teachers()
    body = f"""
    <div class="max-w-md">
        <h1 class="text-2xl font-bold mb-4">Edit Classroom Configuration</h1>
        <form method="POST" class="space-y-3">
            <input type="text" name="class_name" value="{class_row["class_name"]}" class="w-full px-3 py-2 border rounded-lg" required>
            <input type="text" name="department" value="{class_row["department"] or ""}" class="w-full px-3 py-2 border rounded-lg">
            <input type="text" name="course" value="{class_row["course"] or ""}" class="w-full px-3 py-2 border rounded-lg">
            <input type="text" name="section_name" value="{class_row["section_name"] or ""}" class="w-full px-3 py-2 border rounded-lg">
            <input type="text" name="subject_name" value="{class_row["subject_name"] or ""}" class="w-full px-3 py-2 border rounded-lg">
            <select name="teacher_id" class="w-full px-3 py-2 border rounded-lg" required>
    """
    for t in teachers:
        sel = "selected" if t["id"] == class_row["teacher_id"] else ""
        body += f'<option value="{t["id"]}" {sel}>{t["teacher_name"]}</option>'
    body += f"""
            </select>
            <button class="bg-blue-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-blue-700" type="submit">Save Adjustments</button>
            <a class="ml-2 inline-block bg-slate-100 text-slate-700 font-bold py-2 px-4 rounded-lg" href="/admin">Cancel</a>
        </form>
    </div>
    """
    conn.close()
    return page_wrapper("Modify Course Class Configuration", body, is_admin=True)


@app.route("/admin/delete-class/<int:class_id>")
def admin_delete_class(class_id):
    protect = admin_required()
    if protect:
        return protect
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM classes WHERE id=?", (class_id,))
    conn.commit()
    conn.close()
    return "<script>alert('Classroom record mapping purged successfully');window.location.href='/admin';</script>"


@app.route("/admin/assign-student-class", methods=["POST"])
def admin_assign_student_class():
    protect = admin_required()
    if protect:
        return protect
    student_db_id = request.form.get("student_db_id", "").strip()
    class_id = request.form.get("class_id", "").strip()
    if not student_db_id or not class_id:
        return "<script>alert('All enrollment parameters are needed');window.location.href='/admin';</script>"
    assign_student_to_class(int(student_db_id), int(class_id))
    return "<script>alert('Student enrollment map updated dynamically!');window.location.href='/admin';</script>"


@app.route("/admin/edit-student/<int:student_db_id>", methods=["GET", "POST"])
def admin_edit_student(student_db_id):
    protect = admin_required()
    if protect:
        return protect
    student = get_student_row_by_db_id(student_db_id)
    if not student:
        return "Student file mismatch", 404
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        password = request.form.get("password", "").strip()
        conn = get_db()
        cur = conn.cursor()
        cur.execute("UPDATE students SET full_name=?, password=? WHERE id=?", (full_name, password, student_db_id))
        conn.commit()
        conn.close()
        load_known_faces()
        return "<script>alert('Student account adjusted');window.location.href='/admin';</script>"

    body = f"""
    <div class="max-w-md">
        <h1 class="text-2xl font-bold mb-4">Modify Student Registration Settings</h1>
        <form method="POST" class="space-y-4">
            <div>
                <label class="block text-sm font-medium">System Identifier (Disabled)</label>
                <input type="text" value="{student["student_id"]}" class="w-full px-3 py-2 border rounded-lg bg-slate-50" disabled>
            </div>
            <div>
                <label class="block text-sm font-medium">Full Registration Name</label>
                <input type="text" name="full_name" value="{student["full_name"]}" class="w-full px-3 py-2 border rounded-lg" required>
            </div>
            <div>
                <label class="block text-sm font-medium">Portal Authentication Password</label>
                <input type="text" name="password" value="{student["password"]}" class="w-full px-3 py-2 border rounded-lg" required>
            </div>
            <button class="bg-blue-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-blue-700" type="submit">Save File Changes</button>
            <a class="inline-block bg-slate-100 text-slate-700 font-bold py-2 px-4 rounded-lg ml-2" href="/admin">Cancel</a>
        </form>
    </div>
    """
    return page_wrapper("Edit Student Account Data", body, is_admin=True)


@app.route("/admin/delete-student/<student_id>")
def admin_delete_student(student_id):
    protect = admin_required()
    if protect:
        return protect
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, image_file FROM students WHERE student_id=?", (student_id,))
    row = cur.fetchone()
    if row:
        db_id = row["id"]
        img = row["image_file"]
        cur.execute("DELETE FROM attendance WHERE student_id=?", (student_id,))
        cur.execute("DELETE FROM student_classes WHERE student_id_fk=?", (db_id,))
        cur.execute("DELETE FROM students WHERE id=?", (db_id,))
        conn.commit()
        try:
            path = os.path.join(IMAGE_DIR, img)
            if os.path.exists(path):
                os.remove(path)
        except:
            pass
    conn.close()
    load_known_faces()
    return "<script>alert('Student account fully deleted');window.location.href='/admin';</script>"


@app.route("/admin/class/<int:class_id>")
def admin_view_class(class_id):
    protect = admin_required()
    if protect:
        return protect
    class_row = get_class_by_id(class_id)
    if not class_row:
        return "Class code record mapping target not found in framework database state", 404
    students = get_students_in_class(class_id)
    attendance = get_attendance_for_class(class_id)

    body = f"""
    <div class="space-y-6">
        <div>
            <h1 class="text-3xl font-bold text-slate-800">Class: {class_row["class_name"]}</h1>
            <p class="text-sm text-slate-500">Subject: {class_row["subject_name"] or "None"} | Section: {class_row["section_name"] or "None"}</p>
        </div>
        <div class="flex gap-2">
            <a class="bg-slate-800 text-white font-bold py-2 px-4 rounded-lg" href="/admin">← Return Dashboard</a>
        </div>
        
        <div class="bg-white border rounded-xl overflow-hidden shadow-sm">
            <div class="p-4 bg-slate-50 font-bold text-slate-800">Enrolled Students Matrix</div>
            <table class="w-full text-left">
                <thead>
                    <tr class="bg-slate-100 text-slate-700">
                        <th class="p-3">Face</th>
                        <th class="p-3">Student ID</th>
                        <th class="p-3">Full Name</th>
                        <th class="p-3">Action Control</th>
                    </tr>
                </thead>
                <tbody>
    """
    if students:
        for s in students:
            body += f"""
                <tr class="border-b">
                    <td class="p-3"><img class="w-8 h-8 object-cover rounded-full border" src="/student-image/{s["image_file"]}"></td>
                    <td class="p-3 font-mono">{s["student_id"]}</td>
                    <td class="p-3 font-medium">{s["full_name"]}</td>
                    <td class="p-3"><a class="text-red-500 hover:underline" href="/admin/remove-student-from-class/{class_id}/{s['id']}" onclick="return confirm('Unmap from current class framework matrix?')">Drop Enrollment</a></td>
                </tr>
            """
    else:
        body += "<tr><td colspan='4' class='p-4 text-center text-slate-400'>No student data currently mapped inside classroom roster.</td></tr>"
    body += """
                </tbody>
            </table>
        </div>

        <div class="bg-white border rounded-xl overflow-hidden shadow-sm">
            <div class="p-4 bg-slate-50 font-bold text-slate-800">Attendance Log</div>
            <table class="w-full text-left">
                <thead>
                    <tr class="bg-slate-100 text-slate-700">
                        <th class="p-3">Date</th>
                        <th class="p-3">Time logged</th>
                        <th class="p-3">Student ID</th>
                        <th class="p-3">Full Name</th>
                        <th class="p-3">Verification Flag</th>
                    </tr>
                </thead>
                <tbody>
    """
    if attendance:
        for a in attendance:
            body += f"""
                <tr class="border-b">
                    <td class="p-3">{a["date"]}</td>
                    <td class="p-3 text-slate-500">{a["time"]}</td>
                    <td class="p-3 font-mono">{a["student_id"]}</td>
                    <td class="p-3 font-medium">{a["full_name"]}</td>
                    <td class="p-3"><span class="px-2 py-0.5 rounded text-xs font-bold {'bg-emerald-100 text-emerald-800' if a['status']=='Present' else 'bg-rose-100 text-rose-800'}">{a["status"]}</span></td>
                </tr>
            """
    else:
        body += "<tr><td colspan='5' class='p-4 text-center text-slate-400'>No scan iterations completed for this stream matrix.</td></tr>"
    body += """
                </tbody>
            </table>
        </div>
    </div>
    """
    return page_wrapper("Class Details Matrix View", body, is_admin=True)


@app.route("/admin/remove-student-from-class/<int:class_id>/<int:student_db_id>")
def admin_remove_student_from_class(class_id, student_db_id):
    protect = admin_required()
    if protect:
        return protect
    remove_student_from_class(student_db_id, class_id)
    return f"<script>alert('Student unmapped from current class matrix');window.location.href='/admin/class/{class_id}';</script>"


@app.route("/admin/reports")
def admin_reports():
    protect = admin_required()
    if protect:
        return protect
    daily = get_report_records("daily")
    weekly = get_report_records("weekly")
    monthly = get_report_records("monthly")

    def render_report_table(title, rows):
        html = f"""
        <div class="bg-white border rounded-xl overflow-hidden shadow-sm mt-4">
            <div class="p-4 bg-slate-50 font-bold text-slate-800">{title}</div>
            <table class="w-full text-left">
                <thead>
                    <tr class="bg-slate-100 text-slate-700">
                        <th class="p-3">Calendar Date</th>
                        <th class="p-3">Student ID</th>
                        <th class="p-3">Student Name</th>
                        <th class="p-3">Course Subject Target</th>
                        <th class="p-3">Topic Description</th>
                        <th class="p-3">Status Flag</th>
                        <th class="p-3">Proctor Name</th>
                    </tr>
                </thead>
                <tbody>
        """
        if rows:
            for r in rows:
                html += f"""
                    <tr class="border-b text-sm">
                        <td class="p-3">{r["date"]}</td>
                        <td class="p-3 font-mono">{r["student_id"]}</td>
                        <td class="p-3 font-medium">{r["full_name"]}</td>
                        <td class="p-3 font-semibold">{r["class_name"]}</td>
                        <td class="p-3">{r["subject_name"] or ""}</td>
                        <td class="p-3"><span class="px-2 py-0.5 rounded text-xs font-bold {'bg-emerald-100 text-emerald-800' if r['status']=='Present' else 'bg-rose-100 text-rose-800'}">{r["status"]}</span></td>
                        <td class="p-3 text-slate-500">{r["teacher_name"] or ""}</td>
                    </tr>
                """
        else:
            html += "<tr><td colspan='7' class='p-4 text-center text-slate-400'>No records compiled inside this time range delta mapping state.</td></tr>"
        html += "</tbody></table></div>"
        return html

    body = """
    <div class="space-y-4">
        <div>
            <h1 class="text-3xl font-bold text-slate-800">System Time Frame Summaries</h1>
            <p class="text-sm text-slate-500">Review standard dynamic breakdowns mapped by date intervals.</p>
        </div>
    """
    body += render_report_table("Daily Metric Report Insights", daily)
    body += render_report_table("Extended Weekly Performance Matrix", weekly)
    body += render_report_table("Rolling Monthly Aggregated Logs Summary", monthly)
    body += "</div>"
    return page_wrapper("Reports Overview", body, is_admin=True)


# =========================================================
# BEAUTIFUL ORGANIZED TEACHER DASHBOARD PORTAL WITH SIDEBAR
# =========================================================
@app.route("/teacher")
def teacher_dashboard():
    protect = teacher_required()
    if protect:
        return protect

    teacher_id = get_logged_teacher_id()
    teacher_name = session.get("teacher_name", "Instructor")
    classes = get_teacher_classes(teacher_id)
    attendance = get_attendance_for_teacher(teacher_name)

    body = f"""
    <div class="space-y-6">
        <div class="bg-gradient-to-r from-blue-700 to-indigo-800 p-6 rounded-2xl text-white shadow-sm flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div>
                <h1 class="text-3xl font-extrabold tracking-tight">Welcome Back, {teacher_name}!</h1>
                <p class="text-blue-100 text-sm mt-1">Manage active classrooms, launch live face-recognition streams, or complete fast manual manual ticking sheets.</p>
            </div>
            <div class="bg-white/10 px-4 py-2 rounded-xl text-xs font-mono backdrop-blur-sm">
                System Session Verified ✓
            </div>
        </div>

        <div>
            <h2 class="text-xl font-bold text-slate-800 mb-4 flex items-center gap-2"><i class="fas fa-chalkboard text-blue-600"></i> My Assigned Academic Classrooms</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
    """
    if classes:
        for c in classes:
            body += f"""
                <div class="bg-white border border-slate-200 rounded-2xl shadow-sm hover:shadow-md transition flex flex-col justify-between overflow-hidden">
                    <div class="p-5">
                        <div class="flex justify-between items-start mb-3">
                            <span class="text-xs font-bold uppercase tracking-wider bg-blue-50 text-blue-700 px-2.5 py-1 rounded-full">{c["department"] or "General"}</span>
                            <span class="text-xs font-mono text-slate-400">#CLS-{c["id"]}</span>
                        </div>
                        <h3 class="text-xl font-bold text-slate-800 mb-1">{c["class_name"]}</h3>
                        <p class="text-sm font-medium text-slate-600 mb-2">{c["subject_name"] or "No Topic Component Attached"}</p>
                        <div class="text-xs text-slate-400 space-y-1">
                            <div><i class="fas fa-layer-group w-4 text-slate-300"></i> <b>Section Block:</b> {c["section_name"] or "N/A"}</div>
                            <div><i class="fas fa-bookmark w-4 text-slate-300"></i> <b>Course Code:</b> {c["course"] or "N/A"}</div>
                        </div>
                    </div>
                    
                    <div class="p-4 bg-slate-50 border-t border-slate-100 grid grid-cols-2 gap-2 text-center">
                        <a href="/teacher/class/{c["id"]}" class="bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold py-2 px-3 rounded-lg flex items-center justify-center gap-1.5 transition">
                            <i class="fas fa-list-check"></i> Tracker Sheet
                        </a>
                        <a href="/teacher/class/{c["id"]}/scan" class="bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold py-2 px-3 rounded-lg flex items-center justify-center gap-1.5 transition">
                            <i class="fas fa-camera"></i> AI Scan
                        </a>
                    </div>
                </div>
            """
    else:
        body += """
            <div class="col-span-full bg-white border border-dashed rounded-2xl p-8 text-center text-slate-400">
                <i class="fas fa-folder-open text-4xl mb-2 text-slate-300"></i>
                <p class="font-medium">No classroom matrix mapping objects assigned to your proctor account identifier.</p>
            </div>
        """
    body += f"""
            </div>
        </div>

        <div class="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
            <div class="p-4 bg-slate-50 border-b border-slate-100 flex justify-between items-center">
                <h3 class="font-bold text-slate-800 flex items-center gap-2"><i class="fas fa-history text-slate-500"></i> Recent Attendance Mappings Log</h3>
                <span class="text-xs text-slate-500 bg-white border px-2 py-1 rounded-lg">Proctor: {teacher_name}</span>
            </div>
            <div class="overflow-x-auto">
                <table class="w-full text-left m-0 border-none shadow-none rounded-none">
                    <thead>
                        <tr class="bg-slate-100/70 border-b text-slate-700 text-xs uppercase font-bold tracking-wider">
                            <th class="p-3 border-none bg-transparent text-slate-700 font-bold">Date Logged</th>
                            <th class="p-3 border-none bg-transparent text-slate-700 font-bold">Timestamp</th>
                            <th class="p-3 border-none bg-transparent text-slate-700 font-bold">Student ID</th>
                            <th class="p-3 border-none bg-transparent text-slate-700 font-bold">Full Name</th>
                            <th class="p-3 border-none bg-transparent text-slate-700 font-bold">Class Framework</th>
                            <th class="p-3 border-none bg-transparent text-slate-700 font-bold">Section</th>
                            <th class="p-3 border-none bg-transparent text-slate-700 font-bold">Verification Status</th>
                        </tr>
                    </thead>
                    <tbody class="text-xs divide-y divide-slate-100">
    """
    if attendance:
        for a in attendance:
            body += f"""
                <tr class="hover:bg-slate-50/80 transition-colors">
                    <td class="p-3 font-medium whitespace-nowrap">{a["date"]}</td>
                    <td class="p-3 text-slate-400">{a["time"]}</td>
                    <td class="p-3 font-mono font-bold text-slate-600">{a["student_id"]}</td>
                    <td class="p-3 font-semibold text-slate-800">{a["full_name"]}</td>
                    <td class="p-3 text-slate-600">{a["class_name"]}</td>
                    <td class="p-3 text-slate-500">{a["section_name"] or "—"}</td>
                    <td class="p-3"><span class="px-2.5 py-1 rounded-full text-[11px] font-extrabold {'bg-emerald-50 text-emerald-700 border border-emerald-200' if a['status']=='Present' else 'bg-rose-50 text-rose-700 border border-rose-200'}">{a["status"]}</span></td>
                </tr>
            """
    else:
        body += "<tr><td colspan='7' class='p-6 text-center text-slate-400 font-medium'>No records processed via your instructor portal session.</td></tr>"
    body += """
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    """
    return page_wrapper("Teacher Dashboard", body, is_teacher=True, teacher_name=teacher_name)


# =========================================================
# MANUAL ATTENDANCE ROSTER SHEET CONTROL WITH BATCH CLICK SUBMIT
# =========================================================
@app.route("/teacher/class/<int:class_id>")
def teacher_view_class(class_id):
    protect = teacher_required()
    if protect:
        return protect
    class_row = get_class_by_id(class_id)
    if not class_row:
        return "Class mismatch mapping object reference layer", 404
    if class_row["teacher_id"] != get_logged_teacher_id():
        return "Unauthorized proctor routing attempt override locked", 403

    students = get_students_in_class(class_id)
    today_str = datetime.now().strftime("%B %d, %Y")

    body = f"""
    <div class="space-y-6">
        <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 bg-white p-4 border border-slate-200 rounded-xl shadow-sm">
            <div>
                <a href="/teacher" class="text-xs font-bold text-blue-600 hover:underline flex items-center gap-1 mb-1"><i class="fas fa-arrow-left"></i> Back to Dashboard Panel</a>
                <h1 class="text-2xl font-extrabold text-slate-800 tracking-tight">Roster Tracker: {class_row["class_name"]}</h1>
                <p class="text-xs text-slate-400 font-medium mt-0.5">Subject code: {class_row["subject_name"] or "N/A"} | Active Proctor Date: {today_str}</p>
            </div>
            <div class="flex gap-2">
                <a class="bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold py-2.5 px-4 rounded-xl flex items-center gap-1.5 transition shadow-sm shadow-emerald-100" href="/teacher/class/{class_id}/scan">
                    <i class="fas fa-camera"></i> Launch AI Scanner
                </a>
            </div>
        </div>

        <div class="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
            <div class="px-5 py-4 bg-slate-50 border-b border-slate-100 flex justify-between items-center">
                <h3 class="font-bold text-slate-700 text-sm flex items-center gap-2"><i class="fas fa-user-check text-slate-400"></i> Interactive Manual Ticking Sheet</h3>
                <div class="flex items-center gap-3 text-xs">
                    <button onclick="checkAll(true)" type="button" class="text-blue-600 hover:text-blue-800 font-semibold bg-white px-2 py-1 rounded border shadow-sm">Mark All Present</button>
                    <span class="text-slate-300">|</span>
                    <button onclick="checkAll(false)" type="button" class="text-slate-500 hover:text-slate-700 font-semibold bg-white px-2 py-1 rounded border shadow-sm">Clear All</button>
                </div>
            </div>

            <form id="attendanceForm" method="POST" action="/teacher/class/{class_id}/manual-submit">
                <div class="divide-y divide-slate-100">
    """
    if students:
        for idx, s in enumerate(students, 1):
            pct = get_percentage(s["student_id"], class_id)
            body += f"""
                    <div class="p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4 hover:bg-slate-50/50 transition-colors">
                        <div class="flex items-center gap-4">
                            <span class="text-xs font-mono font-bold text-slate-300 w-5 text-center">{idx:02d}</span>
                            <img class="w-10 h-10 object-cover rounded-xl border bg-slate-50" src="/student-image/{s["image_file"]}">
                            <div>
                                <h4 class="font-bold text-slate-800 text-sm">{s["full_name"]}</h4>
                                <p class="text-[11px] font-mono text-slate-400">ID: #{s["student_id"]} | Agg. Ratio Score: <span class="text-blue-600 font-bold">{pct}%</span></p>
                            </div>
                        </div>

                        <div class="flex items-center gap-6">
                            <label class="relative inline-flex items-center cursor-pointer select-none">
                                <input type="checkbox" name="present_students" value="{s["student_id"]}" class="sr-only peer" checked>
                                <div class="w-14 h-7 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[4px] after:left-[4px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-emerald-500"></div>
                                <span class="ml-3 text-xs font-bold text-slate-500 peer-checked:text-emerald-600 uppercase tracking-wider w-16">Present</span>
                            </label>
                        </div>
                    </div>
            """
    else:
        body += """
                    <div class="p-8 text-center text-slate-400 font-medium">
                        <i class="fas fa-users-slash text-3xl mb-1 text-slate-300"></i>
                        <p>No student accounts currently assigned to this classroom registry array block.</p>
                    </div>
        """
    body += """
                </div>

                <div class="p-5 bg-slate-50/60 border-t border-slate-100 flex justify-end">
                    <button type="submit" class="bg-indigo-600 hover:bg-indigo-700 text-white font-bold px-6 py-2.5 rounded-xl shadow-md shadow-indigo-100 flex items-center gap-2 transition-all">
                        <i class="fas fa-save text-xs"></i>
                        <span>Submit Attendance Sheet</span>
                    </button>
                </div>
            </form>
        </div>
    </div>

    <script>
    function checkAll(status) {
        const checkboxes = document.querySelectorAll('input[name="present_students"]');
        checkboxes.forEach(cb => cb.checked = status);
    }
    </script>
    """
    return page_wrapper("Classroom Ticking Roster Sheet", body, is_teacher=True, teacher_name=session.get("teacher_name"))


# =========================================================
# MANUAL BATCH TICKING SUBMISSION HANDLER ROUTE ACTION
# =========================================================
@app.route("/teacher/class/<int:class_id>/manual-submit", methods=["POST"])
def teacher_manual_submit_batch(class_id):
    protect = teacher_required()
    if protect:
        return protect
    class_row = get_class_by_id(class_id)
    if not class_row:
        return "Class file record mapping object mismatch context layer", 404
    if class_row["teacher_id"] != get_logged_teacher_id():
        return "Action forbidden access blocked context matrix", 403

    students = get_students_in_class(class_id)
    ticked_present_ids = request.form.getlist("present_students")

    for s in students:
        status = "Present" if s["student_id"] in ticked_present_ids else "Absent"
        mark_attendance(s, class_row, status=status)

    return "<script>alert('Attendance roster processing batch committed successfully!'); window.location.href='/teacher';</script>"


# =========================================================
# BACKWARDS COMPATIBLE FORCE PILL OVERRIDES 
# =========================================================
@app.route("/teacher/manual-mark/<int:class_id>/<int:student_db_id>/<status>")
def teacher_manual_mark_override(class_id, student_db_id, status):
    protect = teacher_required()
    if protect:
        return protect
    class_row = get_class_by_id(class_id)
    if not class_row:
        return "Class mismatch row array object definition state error", 404
    if class_row["teacher_id"] != get_logged_teacher_id():
        return "Unauthorized action trigger attempt configuration locked", 403

    student_row = get_student_row_by_db_id(student_db_id)
    if student_row:
        mark_attendance(student_row, class_row, status)
    return f"<script>alert('Manually forced student record to {status}!');window.location.href='/teacher/class/{class_id}';</script>"


# =========================================================
# FACE SCAN LIVE CAMERA STREAM MODULE VISUAL ENGINE
# =========================================================
@app.route("/teacher/class/<int:class_id>/scan")
def teacher_scan_class(class_id):
    protect = teacher_required()
    if protect:
        return protect
    class_row = get_class_by_id(class_id)
    if not class_row:
        return "Class asset record code definition state error instance missing", 404
    if class_row["teacher_id"] != get_logged_teacher_id():
        return "Proctor verification sequence match locked mismatch exception", 403

    teacher_name = session.get("teacher_name", "Teacher Proctored Session")

    body = f"""
    <div class="max-w-3xl mx-auto text-center space-y-4">
        <div>
            <h1 class="text-3xl font-extrabold text-slate-800">📸 Automatic AI Face Recognition</h1>
            <p class="text-sm text-slate-500 mt-1">Live frame parser sequence linked to <b>{class_row["class_name"]}</b></p>
        </div>

        <video id="video" autoplay playsinline muted class="w-full max-w-lg mx-auto bg-black border border-slate-300 rounded-2xl shadow-lg"></video>
        
        <div id="result" class="text-2xl font-bold text-emerald-600 mt-4 tracking-tight animate-pulse">Scanning feed state framework...</div>
        <div id="status" class="text-xs font-semibold text-slate-400">Please provide camera hardware layout authorizations</div>
        
        <div class="flex justify-center gap-2 flex-wrap pt-2">
            <button class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-xl" onclick="switchCamera()">Switch Camera Orientation</button>
            <button class="bg-orange-500 hover:bg-orange-600 text-white font-bold py-2 px-4 rounded-xl" onclick="startCamera()">Reset Stream Connection</button>
            <a class="bg-slate-800 hover:bg-slate-900 text-white font-bold py-2 px-4 rounded-xl" href="/teacher/class/{class_id}">Return to Tracker Sheet</a>
        </div>
    </div>

<script>
const video = document.getElementById('video');
const resultDiv = document.getElementById('result');
const statusDiv = document.getElementById('status');
let currentFacingMode = "user";
let stream = null;
let intervalId = null;

async function startCamera() {{
    try {{
        statusDiv.innerText = "Initializing target camera device array sequence...";
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {{
            statusDiv.innerText = "Browser core lacks MediaDevices API standard integration layer definitions";
            return;
        }}
        if (stream) {{
            stream.getTracks().forEach(track => track.stop());
        }}
        try {{
            stream = await navigator.mediaDevices.getUserMedia({{
                video: {{ facingMode: {{ ideal: currentFacingMode }}, width: {{ ideal: 640 }}, height: {{ ideal: 480 }} }},
                audio: false
            }});
        }} catch (err) {{
            stream = await navigator.mediaDevices.getUserMedia({{ video: true, audio: false }});
        }}
        video.srcObject = stream;
        video.onloadedmetadata = async () => {{
            try {{
                await video.play();
                statusDiv.innerText = "Active camera connection stream streaming pipeline established.";
                startScanningLoops();
            }} catch (e) {{
                statusDiv.innerText = "Camera decoded buffer but play permission got standard block exceptions";
            }}
        }};
    }} catch (err) {{
        statusDiv.innerText = "Camera framework pipeline hook connection failure context: " + err.message;
    }}
}}

function switchCamera() {{
    currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
    startCamera();
}}

function startScanningLoops() {{
    if(intervalId) clearInterval(intervalId);
    intervalId = setInterval(async () => {{
        if(video.paused || video.ended) return;
        try {{
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const image = canvas.toDataURL('image/jpeg');

            const res = await fetch('/teacher/class/{class_id}/scan-frame', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{ image: image }})
            }});
            const data = await res.json();
            if (data.name && data.name !== "Unknown") {{
                resultDiv.innerText = "🎉 " + data.name;
                statusDiv.innerText = data.message || "Match successfully logged.";
            }} else {{
                resultDiv.innerText = "Scanning framework loops state...";
            }}
        }} catch(err) {{
            console.log("Parsing framework error exception captured context loops", err);
        }}
    }}, 2000);
}}

window.addEventListener('beforeunload', () => {{
    if(intervalId) clearInterval(intervalId);
    if(stream) stream.getTracks().forEach(t => t.stop());
}});

startCamera();
</script>
    """
    return page_wrapper("Face Scanner Live Canvas Pipeline Engine", body, is_teacher=True, teacher_name=teacher_name)


@app.route("/teacher/class/<int:class_id>/scan-frame", methods=["POST"])
def teacher_scan_frame_matrix_lookup(class_id):
    try:
        protect = teacher_required()
        if protect:
            return jsonify({"name": "Unknown", "message": "Authentication token missing"})

        class_row = get_class_by_id(class_id)
        if not class_row or class_row["teacher_id"] != get_logged_teacher_id():
            return jsonify({"name": "Unknown", "message": "Context scope mapping target mismatch access denied"})

        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"name": "Unknown"})

        image_data = data["image"]
        if "," in image_data:
            image_data = image_data.split(",")
        image_data = image_data.replace(" ", "+")
        missing_padding = len(image_data) % 4
        if missing_padding:
            image_data += "=" * (4 - missing_padding)

        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"name": "Unknown"})

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if len(face_encodings) == 0:
            return jsonify({"name": "Unknown"})

        for face_encoding in face_encodings:
            if len(known_encodings) == 0:
                break
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    student = known_students[best_match_index]
                    if not student_belongs_to_class(student["db_id"], class_id):
                        return jsonify({
                            "name": f"{student['student_id']} - {student['full_name']}",
                            "message": "Recognized, but student is NOT enrolled in this specific roster mapping template"
                        })

                    student_row = get_student_row_by_db_id(student["db_id"])
                    mark_attendance(student_row, class_row, "Present")
                    return jsonify({
                        "name": f"{student['student_id']} - {student['full_name']}",
                        "message": f"Attendance verified & logged successfully into database for {class_row['class_name']}"
                    })

        return jsonify({"name": "Unknown"})
    except Exception as e:
        print("SCAN BATCH ENCODING VECTOR LOOKUP EXCEPTION METRIC ERROR:", e)
        return jsonify({"name": "Unknown", "message": "Lookup processing matrix iteration break exception standard error"})


# =========================================================
# ADDITIONS: STUDENT SELF ATTENDANCE (MANUAL & FACE CHECK-IN)
# =========================================================
@app.route("/student/checkin/manual/<int:class_id>")
def student_manual_checkin(class_id):
    protect = student_required()
    if protect:
        return protect
    
    student_db_id = get_logged_student_db_id()
    if not student_belongs_to_class(student_db_id, class_id):
        return "<script>alert('Error: You are not enrolled in this class framework matrix.'); window.location.href='/student';</script>"

    student_row = get_student_row_by_db_id(student_db_id)
    class_row = get_class_by_id(class_id)
    
    if student_row and class_row:
        mark_attendance(student_row, class_row, "Present")
        return "<script>alert('Success: Your manual check-in has been successfully logged!'); window.location.href='/student';</script>"
    
    return "<script>alert('Error updating configuration parameters.'); window.location.href='/student';</script>"


@app.route("/student/scan")
def student_scan_portal():
    protect = student_required()
    if protect:
        return protect

    student_db_id = get_logged_student_db_id()
    student_ctx = get_student_row_by_db_id(student_db_id)
    
    body = f"""
    <div class="max-w-3xl mx-auto text-center space-y-4">
        <div>
            <h1 class="text-3xl font-extrabold text-slate-800">📸 Student Face Check-In</h1>
            <p class="text-sm text-slate-500 mt-1">Look into your camera device stream to verify your identity profile</p>
        </div>

        <video id="video" autoplay playsinline muted class="w-full max-w-lg mx-auto bg-black border border-slate-300 rounded-2xl shadow-lg"></video>
        
        <div id="result" class="text-2xl font-bold text-emerald-600 mt-4 tracking-tight animate-pulse">Initializing face capture feed layer...</div>
        <div id="status" class="text-xs font-semibold text-slate-400">Please provide camera hardware authorization access</div>
        
        <div class="flex justify-center gap-2 flex-wrap pt-2">
            <button class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-xl" onclick="switchCamera()">Switch Orientation</button>
            <button class="bg-orange-500 hover:bg-orange-600 text-white font-bold py-2 px-4 rounded-xl" onclick="startCamera()">Reset Feed Pipeline</button>
            <a class="bg-slate-800 hover:bg-slate-900 text-white font-bold py-2 px-4 rounded-xl" href="/student">Back to Profile</a>
        </div>
    </div>

<script>
const video = document.getElementById('video');
const resultDiv = document.getElementById('result');
const statusDiv = document.getElementById('status');
let currentFacingMode = "user";
let stream = null;
let intervalId = null;

async function startCamera() {{
    try {{
        statusDiv.innerText = "Initializing targeting system hardware loop...";
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {{
            statusDiv.innerText = "Browser lacks MediaDevices API permissions layout.";
            return;
        }}
        if (stream) {{
            stream.getTracks().forEach(track => track.stop());
        }}
        try {{
            stream = await navigator.mediaDevices.getUserMedia({{
                video: {{ facingMode: {{ ideal: currentFacingMode }}, width: {{ ideal: 640 }}, height: {{ ideal: 480 }} }},
                audio: false
            }});
        }} catch (err) {{
            stream = await navigator.mediaDevices.getUserMedia({{ video: true, audio: false }});
        }}
        video.srcObject = stream;
        video.onloadedmetadata = async () => {{
            try {{
                await video.play();
                statusDiv.innerText = "Streaming canvas sequence pipeline online.";
                startScanningLoops();
            }} catch (e) {{
                statusDiv.innerText = "Play permissions locked by hardware profile error.";
            }}
        }};
    }} catch (err) {{
        statusDiv.innerText = "Stream initialization failure: " + err.message;
    }}
}}

function switchCamera() {{
    currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
    startCamera();
}}

function startScanningLoops() {{
    if(intervalId) clearInterval(intervalId);
    intervalId = setInterval(async () => {{
        if(video.paused || video.ended) return;
        try {{
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const image = canvas.toDataURL('image/jpeg');

            const res = await fetch('/student/scan-frame', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{ image: image }})
            }});
            const data = await res.json();
            if (data.success) {{
                resultDiv.innerText = "✓ Verification Locked!";
                statusDiv.innerText = data.message;
                clearInterval(intervalId);
                setTimeout(() => {{ window.location.href = '/student'; }}, 2500);
            }} else {{
                resultDiv.innerText = "Analyzing face match matrix frame loops...";
                if(data.message) statusDiv.innerText = data.message;
            }}
        }} catch(err) {{
            console.log("Scan routine cycle exception error code:", err);
        }}
    }}, 2000);
}}

window.addEventListener('beforeunload', () => {{
    if(intervalId) clearInterval(intervalId);
    if(stream) stream.getTracks().forEach(t => t.stop());
}});

startCamera();
</script>
    """
    return page_wrapper("Student Self Face-Recognition Portal", body, is_student=True, student_context=student_ctx)


@app.route("/student/scan-frame", methods=["POST"])
def student_scan_frame_matrix_lookup():
    try:
        protect = student_required()
        if protect:
            return jsonify({"success": False, "message": "Session verification check failure"})

        student_db_id = get_logged_student_db_id()
        student_row = get_student_row_by_db_id(student_db_id)
        classes = get_classes_for_student(student_db_id)

        if not student_row or not classes:
            return jsonify({"success": False, "message": "No course classrooms assigned roster registry blocks"})

        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"success": False})

        image_data = data["image"]
        if "," in image_data:
            image_data = image_data.split(",")
        image_data = image_data.replace(" ", "+")
        missing_padding = len(image_data) % 4
        if missing_padding:
            image_data += "=" * (4 - missing_padding)

        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False})

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face using MediaPipe
        face_embedding = _get_face_embedding(rgb_frame)

        if face_embedding is None:
            return jsonify({"success": False, "message": "Position your face clearly within the camera frame layout matrix"})

        if len(known_encodings) == 0:
            return jsonify({"success": False, "message": "No registered faces found"})

        best_match_index = None
        best_dist = float("inf")
        for i, known_emb in enumerate(known_encodings):
            matched, dist = _compare_embeddings(known_emb, face_embedding, tolerance=0.6)
            if matched and dist < best_dist:
                best_dist = dist
                best_match_index = i

        if best_match_index is not None:
            matched_student = known_students[best_match_index]

            if matched_student["db_id"] != student_db_id:
                return jsonify({"success": False, "message": "Face profile mapping match mismatch against logged portal session token identifier"})

            for c in classes:
                mark_attendance(student_row, c, "Present")

            return jsonify({
                "success": True,
                "message": f"Identity verified successfully for {student_row['full_name']}. All active course rosters checked!"
            })

        return jsonify({"success": False, "message": "Face trace vector lookup match mismatch code error"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Internal runtime matrix loop failure error: {str(e)}"})


# =========================================================
# STUDENT PROFILE PORTAL VIEW 
# =========================================================
@app.route("/student")
def student_dashboard_portal():
    protect = student_required()
    if protect:
        return protect

    student_db_id = get_logged_student_db_id()
    student_id = session.get("student_id")
    student_ctx = get_student_row_by_db_id(student_db_id)

    classes = get_classes_for_student(student_db_id)
    history = get_attendance_for_student(student_id)

    body = f"""
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div class="lg:col-span-1 space-y-6">
            <div class="bg-white border rounded-xl p-6 text-center shadow-sm">
                <img class="w-24 h-24 object-cover rounded-full mx-auto border-2 border-blue-500 mb-3 shadow-inner" src="/student-image/{student_ctx["image_file"]}">
                <h2 class="text-xl font-bold text-slate-800">{student_ctx["full_name"]}</h2>
                <p class="text-xs font-mono text-slate-400 mt-1">ID: {student_ctx["student_id"]}</p>
                <div class="mt-3 inline-block bg-emerald-50 text-emerald-700 text-xs font-bold px-3 py-1 rounded-full border border-emerald-200">✓ Active Enrolled Student</div>
                
                <div class="mt-6 border-t pt-4">
                    <a href="/student/scan" class="w-full inline-block text-center bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-2.5 px-4 rounded-xl transition shadow-md shadow-emerald-100 text-xs">
                        <i class="fas fa-camera mr-1"></i> Check-In with Face Scanner
                    </a>
                </div>
            </div>
        </div>

        <div class="lg:col-span-2 space-y-6">
            <div class="bg-white border rounded-xl overflow-hidden shadow-sm">
                <div class="p-4 bg-slate-50 border-b font-bold text-slate-700">📚 Registered Course Classes Summary Matrix</div>
                <div class="overflow-x-auto">
                    <table class="w-full text-left m-0 border-none shadow-none rounded-none">
                        <thead>
                            <tr class="bg-slate-100 border-b text-xs font-bold text-slate-600">
                                <th class="p-3">Class Target</th>
                                <th class="p-3">Department</th>
                                <th class="p-3">Course Catalog ID</th>
                                <th class="p-3">Section Identity</th>
                                <th class="p-3">Attendance Ratio</th>
                                <th class="p-3 text-right">Self Check-In</th>
                            </tr>
                        </thead>
                        <tbody class="text-xs">
    """
    if classes:
        for c in classes:
            pct = get_percentage(student_id, c["id"])
            body += f"""
                <tr class="border-b">
                    <td class="p-3 font-semibold text-slate-800">{c["class_name"]}</td>
                    <td class="p-3 text-slate-500">{c["department"] or ""}</td>
                    <td class="p-3 font-mono">{c["course"] or ""}</td>
                    <td class="p-3">{c["section_name"] or ""}</td>
                    <td class="p-3"><strong class="text-blue-600 font-extrabold">{pct}%</strong></td>
                    <td class="p-3 text-right">
                        <a href="/student/checkin/manual/{c["id"]}" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-1 px-2.5 rounded text-[11px] transition inline-block">
                            <i class="fas fa-check mr-1"></i> Mark Present
                        </a>
                    </td>
                </tr>
            """
    else:
        body += "<tr><td colspan='6' class='p-4 text-center text-slate-400 font-medium'>No dynamic tracking assignments detected mapped to profile matrix.</td></tr>"
    body += """
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="bg-white border rounded-xl overflow-hidden shadow-sm">
                <div class="p-4 bg-slate-50 border-b font-bold text-slate-700">📋 My Historical Verification Check-In Logs</div>
                <div class="overflow-x-auto">
                    <table class="w-full text-left m-0 border-none shadow-none rounded-none">
                        <thead>
                            <tr class="bg-slate-100 border-b text-xs font-bold text-slate-600">
                                <th class="p-3">Calendar Date</th>
                                <th class="p-3">Time logged</th>
                                <th class="p-3">Course Target</th>
                                <th class="p-3">Subject Topic Description</th>
                                <th class="p-3">Status Pillar Flag</th>
                                <th class="p-3">Authorized Proctor</th>
                            </tr>
                        </thead>
                        <tbody class="text-xs">
    """
    if history:
        for h in history:
            body += f"""
                <tr class="border-b">
                    <td class="p-3 font-medium">{h["date"]}</td>
                    <td class="p-3 text-slate-400 font-mono">{h["time"]}</td>
                    <td class="p-3 font-bold text-slate-700">{h["class_name"]}</td>
                    <td class="p-3">{h["subject_name"] or ""}</td>
                    <td class="p-3"><span class="px-2 py-0.5 rounded text-[11px] font-bold {'bg-emerald-100 text-emerald-800' if h['status']=='Present' else 'bg-rose-100 text-rose-800'}">{h['status']}</span></td>
                    <td class="p-3 text-slate-500">{h["teacher_name"] or ""}</td>
                </tr>
            """
    else:
        body += "<tr><td colspan='6' class='p-4 text-center text-slate-400 font-medium'>No scan verification histories logged into backend databases rows.</td></tr>"
    body += """
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    """
    return page_wrapper("Student Personal Hub Portal", body, is_student=True, student_context=student_ctx)


# =========================================================
# IMAGES ASSET LOADING HANDLER DISPATCH CONTROLLERS
# =========================================================
@app.route("/student-image/<filename>")
def student_image(filename):
    file_path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(file_path):
        return "Image asset not detected inside file system storage context block", 404
    ext = filename.lower().split(".")[-1]
    mime = "image/png" if ext == "png" else "image/jpeg"
    with open(file_path, "rb") as f:
        return Response(f.read(), mimetype=mime)


@app.route("/export-attendance")
def export_attendance():
    protect = admin_required()
    if protect:
        return protect

    attendance = get_all_attendance()
    csv_data = "StudentID,FullName,ClassName,Department,Course,Section,Subject,Teacher,Status,Date,Time\n"
    for r in attendance:
        csv_data += (
            f'{r["student_id"]},{r["full_name"]},{r["class_name"]},'
            f'{r["department"] or ""},{r["course"] or ""},{r["section_name"] or ""},'
            f'{r["subject_name"] or ""},{r["teacher_name"] or ""},{r["status"]},{r["date"]},{r["time"]}\n'
        )
    return Response(csv_data, mimetype="text/csv", headers={"Content-disposition": "attachment; filename=attendance_sheet.csv"})


if __name__ == '__main__':
    init_db()
    load_known_faces()
    app.run(host='0.0.0.0', port=5000, debug=True)
