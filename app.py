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

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres.qsiedryjuusemdwkvcyf:Attendance%40School2026!@aws-0-eu-west-1.pooler.supabase.com:6543/postgres")

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
    cur.execute("SELECT 1 FROM students WHERE lower(student_id)=lower(%s)", (student_id.strip(),))
    row = cur.fetchone()
    conn.close()
    return row is not None


def teacher_username_exists(username):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM teachers WHERE lower(username)=lower(%s)", (username.strip(),))
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
        WHERE c.teacher_id=%s
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
        WHERE c.id=%s
    """, (class_id,))
    row = cur.fetchone()
    conn.close()
    return row


def get_student_row_by_student_id(student_id_text):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM students WHERE student_id=%s", (student_id_text,))
    row = cur.fetchone()
    conn.close()
    return row


def get_student_row_by_db_id(student_db_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM students WHERE id=%s", (student_db_id,))
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
        WHERE sc.class_id_fk=%s
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
        WHERE sc.student_id_fk=%s
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
            VALUES (%s, %s)
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
        WHERE student_id_fk=%s AND class_id_fk=%s
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
        WHERE teacher_name=%s
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
        WHERE class_id=%s
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
        WHERE student_id=%s
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
        WHERE student_id_fk=%s AND class_id_fk=%s
    """, (student_db_id, class_id))
    row = cur.fetchone()
    conn.close()
    return row is not None


def mark_attendance(student_row, class_row, status="Present"):
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT * FROM attendance
        WHERE student_id=%s AND class_id=%s AND date=%s
    """, (student_row["student_id"], class_row["id"], today))
    existing = cur.fetchone()

    teacher_name = class_row["teacher_display_name"] or class_row["teacher_name"] or ""

    if existing:
        cur.execute("""
            UPDATE attendance
            SET status=%s, time=%s, teacher_name=%s
            WHERE id=%s
        """, (status, now_time, teacher_name, existing["id"]))
    else:
        cur.execute("""
            INSERT INTO attendance (
                student_id, full_name, class_id, class_name,
                department, course, section_name, subject_name,
                teacher_name, status, date, time
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
        WHERE student_id=%s AND class_id=%s AND status='Present'
    """, (student_id, class_id))
    present_count = cur.fetchone()["c"]

    cur.execute("""
        SELECT COUNT(*) as c
        FROM attendance
        WHERE student_id=%s AND class_id=%s
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
        WHERE date >= %s
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
            <img class="student-photo" style="width:70px; height:70px; border-radius:50%; margin-bottom:10px; border:2px solid #2563eb; object-fit: cover;" src="/student-image/{student_context['image_file']}">
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
            cur.execute("SELECT password FROM teachers WHERE id=%s", (user_id,))
            row = cur.fetchone()
            if not row or row["password"] != old_password:
                conn.close()
                return page_wrapper("Settings", "<p class='text-red-500 font-bold'>Incorrect old password.</p>", is_teacher=True, teacher_name=teacher_name_str)
            
            cur.execute("UPDATE teachers SET password=%s WHERE id=%s", (new_password, user_id))
            conn.commit()
            conn.close()
            return "<script>alert('Teacher password updated successfully!'); window.location.href='/teacher';</script>"

        elif role == "student":
            cur.execute("SELECT password FROM students WHERE id=%s", (user_id,))
            row = cur.fetchone()
            if not row or row["password"] != old_password:
                conn.close()
                return page_wrapper("Settings", "<p class='text-red-500 font-bold'>Incorrect old password.</p>", is_student=True, student_context=student_ctx)
            
            cur.execute("UPDATE students SET password=%s WHERE id=%s", (new_password, user_id))
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
    cur.execute("SELECT * FROM teachers WHERE username=%s AND password=%s", (username, password))
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
    cur.execute("SELECT * FROM students WHERE student_id=%s AND password=%s", (student_id, password))
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
            image_data = image_data.split(",")[1]
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
        embedding = _get_face_embedding(rgb_frame)
        if embedding is None:
            return jsonify({"success": False, "message": "No human faces found in frame context. Please try again."})

        cv2.imwrite(file_path, frame)

        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO students (student_id, full_name, password, image_file, registered_at)
            VALUES (%s, %s, %s, %s, %s)
        """, (student_id, full_name, password, filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()

        load_known_faces()
        return jsonify({"success": True, "message": f"Successfully registered face map vector profile for {full_name}!"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Internal mapping failure error: {str(e)}"})


# =========================================================
# CENTRAL ADMINISTRATION PANEL DASHBOARD VIEW
# =========================================================
@app.route("/admin")
def admin_dashboard():
    protect = admin_required()
    if protect:
        return protect

    students = get_all_students()
    teachers = get_all_teachers()
    classes = get_all_classes()

    student_table_rows = ""
    for s in students:
        student_table_rows += f"""
        <tr>
            <td>{s['student_id']}</td>
            <td>{s['full_name']}</td>
            <td>{s['registered_at']}</td>
            <td>
                <form method="POST" action="/admin/delete-student/{s['id']}" onsubmit="return confirm('Are you sure you want to permanently delete this student profile and revoke class links?');" style="display:inline;">
                    <button type="submit" class="btn red" style="padding: 4px 10px; font-size:12px; margin:0;">Delete</button>
                </form>
            </td>
        </tr>
        """

    teacher_table_rows = ""
    for t in teachers:
        teacher_table_rows += f"""
        <tr>
            <td>{t['id']}</td>
            <td>{t['teacher_name']}</td>
            <td>{t['username']}</td>
            <td>
                <form method="POST" action="/admin/delete-teacher/{t['id']}" onsubmit="return confirm('Are you sure you want to delete this teacher profile?');" style="display:inline;">
                    <button type="submit" class="btn red" style="padding: 4px 10px; font-size:12px; margin:0;">Delete</button>
                </form>
            </td>
        </tr>
        """

    class_options_html = ""
    for c in classes:
        class_options_html += f'<option value="{c["id"]}">{c["class_name"]} - {c["subject_name"] or ""} ({c["teacher_name"] or "No Teacher"})</option>'

    teacher_options_html = ""
    for t in teachers:
        teacher_options_html += f'<option value="{t["id"]}">{t["teacher_name"]} ({t["username"]})</option>'

    student_options_html = ""
    for s in students:
        student_options_html += f'<option value="{s["id"]}">{s["full_name"]} (ID: {s["student_id"]})</option>'

    body = f"""
    <div class="space-y-6">
        <div>
            <h1 class="text-3xl font-extrabold text-slate-800">🛡️ System Administration Console</h1>
            <p class="text-sm text-slate-500">Configure core metrics matrices, link proctors, manage users, and view database rosters.</p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="p-6 bg-slate-50 border rounded-xl">
                <h2 class="text-xl font-bold text-slate-800 mb-3">Register New Instructor Profile</h2>
                <form method="POST" action="/admin/create-teacher" class="space-y-3">
                    <input type="text" name="teacher_name" placeholder="Instructor Display Full Name" class="w-full" required>
                    <input type="text" name="username" placeholder="Login Account Username" class="w-full" required>
                    <input type="text" name="password" placeholder="System Access Password" class="w-full" required>
                    <button type="submit">Create Profile</button>
                </form>
            </div>

            <div class="p-6 bg-slate-50 border rounded-xl">
                <h2 class="text-xl font-bold text-slate-800 mb-3">Create Dynamic Course Classroom</h2>
                <form method="POST" action="/admin/create-class" class="space-y-2">
                    <input type="text" name="class_name" placeholder="Class Target Label" class="w-full" required>
                    <input type="text" name="department" placeholder="Department Stream Name" class="w-full">
                    <input type="text" name="course" placeholder="Course ID Reference" class="w-full">
                    <input type="text" name="section_name" placeholder="Section Identity" class="w-full">
                    <input type="text" name="subject_name" placeholder="Subject Topic Code" class="w-full">
                    <select name="teacher_id" class="w-full" required>
                        <option value="">Assign Proctor Profile</option>
                        {teacher_options_html}
                    </select>
                    <button type="submit" class="mt-2">Create Classroom Matrix</button>
                </form>
            </div>
        </div>

        <div class="p-6 bg-slate-50 border rounded-xl">
            <h2 class="text-xl font-bold text-slate-800 mb-3">Link Student to Classroom Matrix</h2>
            <form method="POST" action="/admin/assign-student-class" class="grid grid-cols-1 sm:grid-cols-3 gap-3 items-end">
                <select name="student_db_id" required>
                    <option value="">Select Student Profile</option>
                    {student_options_html}
                </select>
                <select name="class_id" required>
                    <option value="">Select Classroom Target Slot</option>
                    {class_options_html}
                </select>
                <button type="submit">Complete Allocation</button>
            </form>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div class="box p-4 border border-slate-200 rounded-xl">
                <h3 class="text-lg font-bold text-slate-800 mb-2">🧑‍🎓 Registered Students List</h3>
                <div class="overflow-x-auto">
                    <table>
                        <thead>
                            <tr>
                                <th>Student ID</th>
                                <th>Full Name</th>
                                <th>Registered At</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {student_table_rows if student_table_rows else '<tr><td colspan="4" class="text-center text-slate-400">No students found.</td></tr>'}
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="box p-4 border border-slate-200 rounded-xl">
                <h3 class="text-lg font-bold text-slate-800 mb-2">👨‍🏫 System Instructors List</h3>
                <div class="overflow-x-auto">
                    <table>
                        <thead>
                            <tr>
                                <th>DB ID</th>
                                <th>Teacher Name</th>
                                <th>Username</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {teacher_table_rows if teacher_table_rows else '<tr><td colspan="4" class="text-center text-slate-400">No instructors found.</td></tr>'}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    """
    return page_wrapper("Admin Dashboard", body, is_admin=True)


# =========================================================
# ADMIN CONTROLLER PROCESSORS & DELETIONS
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
        return "<script>alert('All parameters are mandatory!'); window.location.href='/admin';</script>"

    if teacher_username_exists(username):
        return "<script>alert('Error: Instructor username already registered!'); window.location.href='/admin';</script>"

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO teachers (teacher_name, username, password, created_at)
        VALUES (%s, %s, %s, %s)
    """, (teacher_name, username, password, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

    return "<script>alert('Instructor profile created successfully!'); window.location.href='/admin';</script>"


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
        return "<script>alert('Class Target Label and Proctor Profile allocation are mandatory!'); window.location.href='/admin';</script>"

    conn = get_db()
    cur = conn.cursor()
    
    cur.execute("SELECT teacher_name FROM teachers WHERE id=%s", (teacher_id,))
    t_row = cur.fetchone()
    teacher_display_name = t_row["teacher_name"] if t_row else ""

    cur.execute("""
        INSERT INTO classes (class_name, department, course, section_name, subject_name, teacher_id, teacher_display_name, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (class_name, department or None, course or None, section_name or None, subject_name or None, int(teacher_id), teacher_display_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

    return "<script>alert('Dynamic course classroom map registered successfully!'); window.location.href='/admin';</script>"


@app.route("/admin/assign-student-class", methods=["POST"])
def admin_assign_student_class():
    protect = admin_required()
    if protect:
        return protect

    student_db_id = request.form.get("student_db_id", "").strip()
    class_id = request.form.get("class_id", "").strip()

    if not student_db_id or not class_id:
        return "<script>alert('Please select both a student and a class roster targeting slot!'); window.location.href='/admin';</script>"

    assign_student_to_class(int(student_db_id), int(class_id))
    return "<script>alert('Student assigned to class successfully!'); window.location.href='/admin';</script>"


@app.route("/admin/delete-student/<int:id>", methods=["POST"])
def admin_delete_student(id):
    protect = admin_required()
    if protect:
        return protect

    conn = get_db()
    cur = conn.cursor()
    try:
        # 1. Clear out intermediate cross references inside enrollment lists
        cur.execute("DELETE FROM student_classes WHERE student_id_fk = %s", (id,))
        # 2. Drop the master student entry row
        cur.execute("DELETE FROM students WHERE id = %s", (id,))
        conn.commit()
    except Exception as e:
        conn.rollback()
        conn.close()
        return f"<script>alert('Failed to drop student execution error: {str(e)}'); window.location.href='/admin';</script>"

    conn.close()
    # Synchronize facial database representation matrices 
    load_known_faces()
    return "<script>alert('Student profile completely discarded from database registry.'); window.location.href='/admin';</script>"


@app.route("/admin/delete-teacher/<int:id>", methods=["POST"])
def admin_delete_teacher(id):
    protect = admin_required()
    if protect:
        return protect

    conn = get_db()
    cur = conn.cursor()
    try:
        # Check if teacher holds active classrooms to prevent violating constraint mechanics
        cur.execute("SELECT 1 FROM classes WHERE teacher_id = %s LIMIT 1", (id,))
        if cur.fetchone():
            conn.close()
            return "<script>alert('Cannot delete instructor because they are currently assigned to an active class registry module.'); window.location.href='/admin';</script>"

        cur.execute("DELETE FROM teachers WHERE id = %s", (id,))
        conn.commit()
    except Exception as e:
        conn.rollback()
        conn.close()
        return f"<script>alert('Failed to delete instructor structural instance: {str(e)}'); window.location.href='/admin';</script>"

    conn.close()
    return "<script>alert('Teacher instance unlinked successfully.'); window.location.href='/admin';</script>"


@app.route("/admin/reports")
def admin_reports():
    protect = admin_required()
    if protect:
        return protect

    period = request.args.get("period", "daily")
    records = get_report_records(period)

    body = f"""
    <div class="space-y-4">
        <div class="flex justify-between items-center flex-wrap gap-2">
            <div>
                <h1 class="text-2xl font-bold text-slate-800">📋 Roster Master Ledger Reports</h1>
                <p class="text-sm text-slate-500">Historical database log entries sorted by execution datetime slots</p>
            </div>
            <div class="flex gap-2">
                <a class="btn dark px-3 py-1 text-xs" href="/admin/reports?period=daily">Daily Log</a>
                <a class="btn dark px-3 py-1 text-xs" href="/admin/reports?period=weekly">Weekly Matrix</a>
                <a class="btn dark px-3 py-1 text-xs" href="/admin/reports?period=monthly">Monthly Block</a>
                <a class="btn green px-3 py-1 text-xs" href="/export-attendance">📥 Export CSV Sheet</a>
            </div>
        </div>
        
        <div class="overflow-x-auto">
            <table>
                <thead>
                    <tr>
                        <th>Student ID</th>
                        <th>Full Name</th>
                        <th>Class Blueprint</th>
                        <th>Instructor</th>
                        <th>Attendance Status</th>
                        <th>System Timestamp Reference</th>
                    </tr>
                </thead>
                <tbody>
    """
    for r in records:
        body += f"""
        <tr>
            <td>{r['student_id']}</td>
            <td>{r['full_name']}</td>
            <td>{r['class_name']} ({r['subject_name'] or ''})</td>
            <td>{r['teacher_name'] or ''}</td>
            <td><span class="px-2 py-0.5 rounded text-xs font-bold {'bg-green-100 text-green-800' if r['status']=='Present' else 'bg-red-100 text-red-800'}">{r['status']}</span></td>
            <td>{r['date']} @ {r['time']}</td>
        </tr>
        """
    if not records:
        body += '<tr><td colspan="6" class="text-center text-slate-400 py-6">No historical records saved for the chosen period scope.</td></tr>'

    body += """
                </tbody>
            </table>
        </div>
    </div>
    """
    return page_wrapper("Attendance Ledger Reports", body, is_admin=True)


# =========================================================
# TEACHER / INSTRUCTOR DISPATCH VIEWS
# =========================================================
@app.route("/teacher")
def teacher_dashboard():
    protect = teacher_required()
    if protect:
        return protect

    teacher_id = get_logged_teacher_id()
    teacher_name = session.get("teacher_name", "Teacher")
    classes = get_teacher_classes(teacher_id)

    body = f"""
    <div class="space-y-4">
        <div>
            <h1 class="text-2xl font-bold text-slate-800">👨‍🏫 Instructor Dashboard Panel</h1>
            <p class="text-sm text-slate-500">Access macro management metrics for assigned course schedules.</p>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
    """
    for c in classes:
        body += f"""
        <div class="p-4 border border-slate-200 rounded-xl bg-slate-50 flex flex-col justify-between">
            <div>
                <span class="text-xs font-bold uppercase tracking-wider text-blue-600">{c['subject_name'] or 'General Subject'}</span>
                <h3 class="text-lg font-bold text-slate-800 mt-1">{c['class_name']}</h3>
                <p class="text-xs text-slate-500 mt-1">Dept: {c['department'] or 'None'} | Course Code: {c['course'] or 'None'} | Sec: {c['section_name'] or 'None'}</p>
            </div>
            <div class="mt-4 pt-2 border-t flex justify-end">
                <a href="/teacher/class/{c['id']}" class="btn bg-blue-600 text-white text-xs px-3 py-1.5 rounded-lg">Open Manual Sheet →</a>
            </div>
        </div>
        """
    if not classes:
        body += '<div class="col-span-2 p-6 text-center text-slate-400">No workspace allocation found for your account reference.</div>'

    body += "</div></div>"
    return page_wrapper("Instructor Management Console", body, is_teacher=True, teacher_name=teacher_name)


@app.route("/teacher/class/<int:class_id>")
def teacher_class_view(class_id):
    protect = teacher_required()
    if protect:
        return protect

    teacher_name = session.get("teacher_name", "Teacher")
    class_row = get_class_by_id(class_id)
    if not class_row:
        return "Target class index data block structure unallocated", 404

    students = get_students_in_class(class_id)

    body = f"""
    <div class="space-y-4">
        <div>
            <h1 class="text-2xl font-bold text-slate-800">📋 Roster Attendance Ticking Grid</h1>
            <p class="text-sm text-slate-500">Class: <b>{class_row['class_name']}</b> | Subject: {class_row['subject_name'] or 'None'}</p>
        </div>
        
        <form method="POST" action="/teacher/mark-attendance">
            <input type="hidden" name="class_id" value="{class_id}">
            <div class="overflow-x-auto">
                <table>
                    <thead>
                        <tr>
                            <th>Student Code ID</th>
                            <th>Full Name</th>
                            <th>Verified Performance Index</th>
                            <th>Presence Status Toggle Selection</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    for s in students:
        pct = get_percentage(s["student_id"], class_id)
        body += f"""
        <tr>
            <td><b>{s['student_id']}</b></td>
            <td>{s['full_name']}</td>
            <td><span class="text-xs font-semibold px-2 py-0.5 rounded {'bg-green-100 text-green-700' if pct>=75 else 'bg-amber-100 text-amber-700'}">{pct}% Present Rate</span></td>
            <td>
                <select name="status_{s['id']}" style="margin:0; width:140px; padding:4px 8px;" class="border rounded-lg text-sm">
                    <option value="Present">Present</option>
                    <option value="Absent">Absent</option>
                </select>
            </td>
        </tr>
        """
    if not students:
        body += '<tr><td colspan="4" class="text-center text-slate-400 py-6">No students assigned to this classroom list yet.</td></tr>'

    body += f"""
                    </tbody>
                </table>
            </div>
            <div class="pt-4 flex gap-2">
                <button type="submit" class="bg-blue-600 hover:bg-blue-700 text-white font-bold px-4 py-2 rounded-lg">Submit Batch Ledger Records</button>
                <a href="/teacher" class="btn bg-slate-100 text-slate-700 hover:bg-slate-200 px-4 py-2 rounded-lg">Go Back</a>
            </div>
        </form>
    </div>
    """
    return page_wrapper("Roster Verification Matrix", body, is_teacher=True, teacher_name=teacher_name)


@app.route("/teacher/mark-attendance", methods=["POST"])
def teacher_mark_attendance():
    protect = teacher_required()
    if protect:
        return protect

    class_id_str = request.form.get("class_id")
    if not class_id_str:
        return "Invalid dynamic parameters form mapping configuration instance missing", 400

    class_id = int(class_id_str)
    class_row = get_class_by_id(class_id)
    students = get_students_in_class(class_id)

    for s in students:
        status_val = request.form.get(f"status_{s['id']}")
        if status_val:
            mark_attendance(s, class_row, status_val)

    return "<script>alert('Batch records logged into database schema matrix successfully!'); window.location.href='/teacher';</script>"


# =========================================================
# STUDENT SUITE APP DISPATCH CONTROLLERS
# =========================================================
@app.route("/student")
def student_dashboard():
    protect = student_required()
    if protect:
        return protect

    student_db_id = get_logged_student_db_id()
    student_ctx = get_student_row_by_db_id(student_db_id)
    classes = get_classes_for_student(student_db_id)

    body = f"""
    <div class="space-y-4">
        <div>
            <h1 class="text-2xl font-bold text-slate-800">📚 My Registered Classroom Metrics Matrix</h1>
            <p class="text-sm text-slate-500">Realtime percentage indicators tracking calculated attendance compliance rates.</p>
        </div>
        
        <div class="overflow-x-auto">
            <table>
                <thead>
                    <tr>
                        <th>Class Identifier Room</th>
                        <th>Subject Title Code</th>
                        <th>Aggregated Verified Presence Rate</th>
                        <th>Roster Evaluation Index</th>
                    </tr>
                </thead>
                <tbody>
    """
    for c in classes:
        pct = get_percentage(student_ctx["student_id"], c["id"])
        body += f"""
        <tr>
            <td><b>{c['class_name']}</b></td>
            <td>{c['subject_name'] or 'General Subject'}</td>
            <td><b class="text-lg text-blue-600">{pct}%</b> verified presence count</td>
            <td><span class="px-2 py-0.5 text-xs font-bold rounded {'bg-green-100 text-green-800' if pct>=75 else 'bg-red-100 text-red-800'}">{'Compliant Status' if pct>=75 else 'Deficit Action Notice'}</span></td>
        </tr>
        """
    if not classes:
        body += '<tr><td colspan="4" class="text-center text-slate-400 py-6">Your profile is not assigned to any classroom layout frameworks yet.</td></tr>'

    body += """
                </tbody>
            </table>
        </div>
    </div>
    """
    return page_wrapper("Student Hub Matrix", body, is_student=True, student_context=student_ctx)


@app.route("/student/scan")
def student_scan_view():
    protect = student_required()
    if protect:
        return protect

    student_db_id = get_logged_student_db_id()
    student_ctx = get_student_row_by_db_id(student_db_id)

    body = f"""
    <div class="max-w-xl mx-auto text-center">
        <h1 class="text-2xl font-bold text-slate-800 mb-2">📸 AI Biometric Verification Entry Point</h1>
        <p class="text-xs text-slate-500 mb-4">Position your face inside the active viewport template frame layer</p>
        
        <video id="v" autoplay playsinline muted class="w-full max-w-sm mx-auto bg-black rounded-2xl shadow-inner mb-4 border border-slate-300"></video>
        <button onclick="scan()" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-6 rounded-xl transition">Authenticate & Check In</button>
        <div id="status" class="text-xs font-semibold text-slate-500 mt-3">Initial webcamera canvas system configuration matrix mapping online...</div>
    </div>

    <script>
    const video = document.getElementById('v');
    const statusDiv = document.getElementById('status');
    
    navigator.mediaDevices.getUserMedia({{ video: true, audio: false }})
        .then(s => {{ video.srcObject = s; statusDiv.innerText = "Camera stream connected successfully."; }})
        .catch(e => {{ statusDiv.innerText = "Hardware connection failure reference: " + e.message; }});
        
    async function scan() {{
        try {{
            statusDiv.innerText = "Extracting snapshot frame vector mappings...";
            const canvas = document.createElement('canvas');
            canvas.width = 640;
            canvas.height = 480;
            canvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
            const dataUrl = canvas.toDataURL('image/jpeg');
            
            statusDiv.innerText = "Sending biometric authentication data packet...";
            const res = await fetch('/verify-scan', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ image: dataUrl }})
            }});
            const d = await res.json();
            statusDiv.innerText = d.message;
            alert(d.message);
        }} catch(err) {{
            statusDiv.innerText = "Scan runtime framework tracking exception error: " + err.message;
        }}
    }}
    </script>
    """
    return page_wrapper("Facial Scan Matching", body, is_student=True, student_context=student_ctx)


@app.route("/verify-scan", methods=["POST"])
def verify_scan():
    try:
        if not is_student_logged_in():
            return jsonify({"success": False, "message": "Session expired authentication context."})
        data = request.get_json() or {}
        image_data = data.get("image", "")
        if "," in image_data:
            image_data = image_data.split(",")[1]
        
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        embedding = _get_face_embedding(rgb_frame)
        if embedding is None:
            return jsonify({"success": False, "message": "No face found inside camera canvas framework layer."})

        s_ctx = get_student_row_by_db_id(get_logged_student_db_id())
        classes = get_classes_for_student(s_ctx["id"])
        if not classes:
            return jsonify({"success": False, "message": "You are not currently registered to any active class rosters."})

        # Match against cached registered baseline face image
        for match in known_students:
            if match["db_id"] == s_ctx["id"]:
                idx = known_students.index(match)
                known_emb = known_encodings[idx]
                matched, distance = _compare_embeddings(known_emb, embedding)
                if matched:
                    for cls in classes:
                        mark_attendance(s_ctx, cls, "Present")
                    return jsonify({"success": True, "message": "Identity successfully verified via Face Vector! Status updated to Present across today's slots."})

        return jsonify({"success": False, "message": "Biometric configuration verification mismatch."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


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
    return Response(csv_data, mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=attendance_export.csv"})


if __name__ == "__main__":
    init_db()
    load_known_faces()
    app.run(host="0.0.0.0", port=5000, debug=True)
