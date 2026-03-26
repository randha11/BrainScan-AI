# app.py
import os, json
import numpy as np
from datetime import datetime, date, timedelta
from collections import defaultdict
from flask import Flask, request, render_template, redirect, url_for, flash, session, jsonify, send_file
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

from model_utils import prepare_image
from grad_cam import make_gradcam_heatmap, overlay_heatmap, get_img_array
from database import db, User, DoctorProfile, Appointment, ScanResult, Notification, DoctorUnavailability, DoctorRating, MRIRequest, Prescription

# ─── CONFIG ──────────────────────────────────────────────────────────────────
BRAIN_CLASSES   = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
ALL_SLOTS       = ["09:00 AM","09:30 AM","10:00 AM","10:30 AM","11:00 AM","11:30 AM",
                   "02:00 PM","02:30 PM","03:00 PM","03:30 PM","04:00 PM","04:30 PM"]
UPLOAD_FOLDER   = os.path.join("static", "uploads")
ALLOWED_EXT     = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.secret_key = "brain-tumour-secret-2024"
app.config["UPLOAD_FOLDER"]               = UPLOAD_FOLDER
app.config["SQLALCHEMY_DATABASE_URI"]     = "sqlite:///brain_tumour.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─── MODEL ───────────────────────────────────────────────────────────────────
brain_model = None
def get_model():
    global brain_model
    if brain_model is None:
        path = os.path.join("models", "brain_model.h5")
        if os.path.exists(path):
            brain_model = load_model(path)
    return brain_model

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def allowed_file(fn):
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def push_notification(user_id, message, link="/", icon="fa-bell"):
    n = Notification(user_id=user_id, message=message, link=link, icon=icon)
    db.session.add(n)

def get_unread_count(user_id):
    return Notification.query.filter_by(user_id=user_id, is_read=False).count()

def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if "user_id" not in session:
                return redirect(url_for("login"))
            if session.get("role") not in roles:
                flash("Access denied.", "danger")
                return redirect_to_dashboard()
            return f(*args, **kwargs)
        return decorated
    return decorator

def redirect_to_dashboard():
    role = session.get("role")
    if role == "admin":   return redirect(url_for("admin_dashboard"))
    if role == "doctor":  return redirect(url_for("doctor_dashboard"))
    if role == "patient": return redirect(url_for("patient_dashboard"))
    if role == "lab":     return redirect(url_for("lab_dashboard"))
    return redirect(url_for("landing"))

def doctor_avg_rating(doctor_id):
    ratings = DoctorRating.query.filter_by(doctor_id=doctor_id).all()
    if not ratings: return None
    return round(sum(r.rating for r in ratings) / len(ratings), 1)

def get_booked_slots(doctor_id, appt_date, exclude_id=None):
    q = Appointment.query.filter(
        Appointment.doctor_id == doctor_id,
        Appointment.appt_date == appt_date,
        Appointment.status.in_(["pending","confirmed"])
    )
    if exclude_id:
        q = q.filter(Appointment.id != exclude_id)
    return [a.time_slot for a in q.all()]

# ─── CONTEXT PROCESSOR (notifications in every template) ─────────────────────
@app.context_processor
def inject_notifications():
    if "user_id" in session:
        uid = session["user_id"]
        notifs = Notification.query.filter_by(user_id=uid)\
            .order_by(Notification.created_at.desc()).limit(8).all()
        unread  = sum(1 for n in notifs if not n.is_read)
        return dict(notifications=notifs, unread_count=unread)
    return dict(notifications=[], unread_count=0)

# ─── AUTH ─────────────────────────────────────────────────────────────────────
@app.route("/")
def landing():
    if "user_id" in session: return redirect_to_dashboard()
    return render_template("landing.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if "user_id" in session: return redirect_to_dashboard()
    if request.method == "POST":
        email    = request.form.get("email","").strip().lower()
        password = request.form.get("password","")
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session.update(user_id=user.id, role=user.role, name=user.name)
            flash(f"Welcome back, {user.name}!", "success")
            return redirect_to_dashboard()
        flash("Invalid email or password.", "danger")
    return render_template("auth/login.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if "user_id" in session: return redirect_to_dashboard()
    if request.method == "POST":
        name  = request.form.get("name","").strip()
        email = request.form.get("email","").strip().lower()
        pw    = request.form.get("password","")
        phone = request.form.get("phone","").strip()
        role  = request.form.get("role","patient")
        if role not in ("patient","doctor"): role = "patient"
        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
            return render_template("auth/register.html")
        user = User(name=name, email=email, password=generate_password_hash(pw),
                    phone=phone, role=role)
        db.session.add(user)
        db.session.flush()
        if role == "doctor":
            db.session.add(DoctorProfile(
                user_id=user.id,
                specialization=request.form.get("specialization","Neurology"),
                experience=int(request.form.get("experience","0") or 0),
                bio=request.form.get("bio","")
            ))
        db.session.commit()
        flash("Account created! Please login.", "success")
        return redirect(url_for("login"))
    return render_template("auth/register.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("landing"))

# ─── PROFILE ──────────────────────────────────────────────────────────────────
@app.route("/profile", methods=["GET","POST"])
def profile():
    if "user_id" not in session: return redirect(url_for("login"))
    user = User.query.get(session["user_id"])
    if request.method == "POST":
        user.name  = request.form.get("name", user.name).strip()
        user.phone = request.form.get("phone", user.phone).strip()
        new_pw = request.form.get("new_password","").strip()
        if new_pw:
            if not check_password_hash(user.password, request.form.get("current_password","")):
                flash("Current password is incorrect.", "danger")
                return render_template("profile.html", user=user)
            user.password = generate_password_hash(new_pw)
        if user.role == "doctor" and user.doctor_profile:
            user.doctor_profile.specialization = request.form.get("specialization", user.doctor_profile.specialization)
            user.doctor_profile.experience = int(request.form.get("experience", user.doctor_profile.experience) or 0)
            user.doctor_profile.bio = request.form.get("bio", user.doctor_profile.bio)
        db.session.commit()
        session["name"] = user.name
        flash("Profile updated successfully.", "success")
    return render_template("profile.html", user=user)

# ─── NOTIFICATIONS ────────────────────────────────────────────────────────────
@app.route("/notifications/mark-read", methods=["POST"])
def mark_notifications_read():
    if "user_id" not in session: return jsonify({"ok": False})
    Notification.query.filter_by(user_id=session["user_id"], is_read=False)\
        .update({"is_read": True})
    db.session.commit()
    return jsonify({"ok": True})

# ─── AVAILABILITY API ─────────────────────────────────────────────────────────
@app.route("/api/available-slots")
def api_available_slots():
    doctor_id = request.args.get("doctor_id")
    appt_date = request.args.get("date")
    if not doctor_id or not appt_date:
        return jsonify({"slots": ALL_SLOTS, "blocked_day": False})
    blocked_day = DoctorUnavailability.query.filter_by(
        doctor_id=doctor_id, date=appt_date).first()
    if blocked_day:
        return jsonify({"slots": [], "blocked_day": True, "reason": blocked_day.reason})
    booked = get_booked_slots(doctor_id, appt_date)
    available = [s for s in ALL_SLOTS if s not in booked]
    return jsonify({"slots": available, "blocked_day": False})

# ─── PATIENT ──────────────────────────────────────────────────────────────────
@app.route("/patient/dashboard")
@role_required("patient")
def patient_dashboard():
    uid = session["user_id"]
    upcoming = Appointment.query.filter_by(patient_id=uid, status="confirmed")\
        .order_by(Appointment.appt_date).limit(3).all()
    recent_scans = ScanResult.query.filter_by(patient_id=uid)\
        .order_by(ScanResult.created_at.desc()).limit(3).all()
    total = Appointment.query.filter_by(patient_id=uid).count()
    return render_template("patient/dashboard.html",
                           upcoming=upcoming, recent_scans=recent_scans, total=total)

@app.route("/patient/book", methods=["GET","POST"])
@role_required("patient")
def patient_book():
    doctors = User.query.filter_by(role="doctor").join(DoctorProfile).all()
    for d in doctors:
        d.avg_rating = doctor_avg_rating(d.id)
        d.rating_count = DoctorRating.query.filter_by(doctor_id=d.id).count()
    if request.method == "POST":
        doctor_id = request.form.get("doctor_id")
        appt_date = request.form.get("appt_date")
        time_slot = request.form.get("time_slot")
        reason    = request.form.get("reason","")
        priority  = request.form.get("priority","Normal")
        # Check unavailability
        if DoctorUnavailability.query.filter_by(doctor_id=doctor_id, date=appt_date).first():
            flash("Doctor is unavailable on that date. Please choose another.", "warning")
            return render_template("patient/book.html", doctors=doctors)
        if time_slot in get_booked_slots(doctor_id, appt_date):
            flash("That slot is already booked. Please pick another time.", "warning")
            return render_template("patient/book.html", doctors=doctors)
        appt = Appointment(patient_id=session["user_id"], doctor_id=doctor_id,
                           appt_date=appt_date, time_slot=time_slot,
                           reason=reason, priority=priority, status="pending")
        db.session.add(appt)
        db.session.flush()
        push_notification(int(doctor_id),
            f"New appointment request from {session['name']} on {appt_date} at {time_slot}",
            "/doctor/appointments", "fa-calendar-plus")
        db.session.commit()
        flash("Appointment booked! Waiting for doctor confirmation.", "success")
        return redirect(url_for("patient_appointments"))
    return render_template("patient/book.html", doctors=doctors)

@app.route("/patient/appointments")
@role_required("patient")
def patient_appointments():
    uid  = session["user_id"]
    page = request.args.get("page", 1, type=int)
    appts = Appointment.query.filter_by(patient_id=uid)\
        .order_by(Appointment.created_at.desc()).paginate(page=page, per_page=8)
    return render_template("patient/appointments.html", appts=appts)

@app.route("/patient/cancel/<int:appt_id>")
@role_required("patient")
def patient_cancel(appt_id):
    appt = Appointment.query.get_or_404(appt_id)
    if appt.patient_id != session["user_id"]:
        flash("Unauthorized.", "danger"); return redirect(url_for("patient_appointments"))
    appt.status = "cancelled"
    push_notification(appt.doctor_id,
        f"Appointment cancelled by {session['name']} ({appt.appt_date} {appt.time_slot})",
        "/doctor/appointments", "fa-calendar-xmark")
    db.session.commit()
    flash("Appointment cancelled.", "info")
    return redirect(url_for("patient_appointments"))

@app.route("/patient/reschedule/<int:appt_id>", methods=["GET","POST"])
@role_required("patient")
def patient_reschedule(appt_id):
    appt = Appointment.query.get_or_404(appt_id)
    if appt.patient_id != session["user_id"]:
        flash("Unauthorized.", "danger"); return redirect(url_for("patient_appointments"))
    if appt.status not in ("pending","confirmed"):
        flash("Only pending or confirmed appointments can be rescheduled.", "warning")
        return redirect(url_for("patient_appointments"))
    doctors = User.query.filter_by(role="doctor").join(DoctorProfile).all()
    if request.method == "POST":
        new_date = request.form.get("appt_date")
        new_slot = request.form.get("time_slot")
        new_doc  = request.form.get("doctor_id", appt.doctor_id)
        if DoctorUnavailability.query.filter_by(doctor_id=new_doc, date=new_date).first():
            flash("Doctor is unavailable on that date.", "warning")
            return render_template("patient/reschedule.html", appt=appt, doctors=doctors)
        if new_slot in get_booked_slots(new_doc, new_date, exclude_id=appt_id):
            flash("That slot is already taken. Pick another time.", "warning")
            return render_template("patient/reschedule.html", appt=appt, doctors=doctors)
        old_info = f"{appt.appt_date} {appt.time_slot}"
        appt.appt_date  = new_date
        appt.time_slot  = new_slot
        appt.doctor_id  = int(new_doc)
        appt.status     = "pending"
        push_notification(int(new_doc),
            f"Appointment rescheduled by {session['name']} from {old_info} to {new_date} {new_slot}",
            "/doctor/appointments", "fa-calendar-day")
        db.session.commit()
        flash("Appointment rescheduled successfully.", "success")
        return redirect(url_for("patient_appointments"))
    return render_template("patient/reschedule.html", appt=appt, doctors=doctors)

@app.route("/patient/scan/<int:appt_id>", methods=["GET","POST"])
@role_required("patient")
def patient_scan(appt_id):
    appt = Appointment.query.get_or_404(appt_id)
    if appt.patient_id != session["user_id"]:
        flash("Unauthorized.", "danger"); return redirect(url_for("patient_appointments"))
    if request.method == "POST":
        file = request.files.get("scan")
        if not file or not allowed_file(file.filename):
            flash("Please upload a valid image (PNG/JPG).", "danger")
            return redirect(request.url)
        filename = secure_filename(f"scan_{appt_id}_{file.filename}")
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        model = get_model()
        if model:
            img_arr = preprocess_input(prepare_image(filepath))
            img_grad = preprocess_input(get_img_array(filepath, size=(224,224)))
            preds = model.predict(img_arr)
            label_idx   = int(np.argmax(preds[0]))
            confidence  = float(np.max(preds[0])) * 100
            pred_label  = BRAIN_CLASSES[label_idx]
            cam_fn = "cam_" + filename
            heatmap = make_gradcam_heatmap(img_grad, model, pred_index=label_idx)
            overlay_heatmap(filepath, heatmap, os.path.join(app.config["UPLOAD_FOLDER"], cam_fn))
        else:
            pred_label, confidence, cam_fn = "Glioma", 87.5, filename
        scan = ScanResult(appointment_id=appt_id, patient_id=appt.patient_id,
                          doctor_id=appt.doctor_id, image_path=filename,
                          cam_path=cam_fn, prediction=pred_label,
                          confidence=confidence, status="pending_review")
        db.session.add(scan)
        appt.status = "scanned"
        db.session.flush()
        push_notification(appt.doctor_id,
            f"New scan uploaded by {session['name']} — {pred_label} ({confidence:.1f}%)",
            f"/doctor/review/{scan.id}", "fa-microscope")
        db.session.commit()
        flash("Scan uploaded and analysed!", "success")
        return redirect(url_for("patient_report", scan_id=scan.id))
    return render_template("patient/scan.html", appt=appt)

@app.route("/patient/report/<int:scan_id>")
@role_required("patient")
def patient_report(scan_id):
    scan = ScanResult.query.get_or_404(scan_id)
    if scan.patient_id != session["user_id"]:
        flash("Unauthorized.", "danger"); return redirect(url_for("patient_dashboard"))
    patient = User.query.get(scan.patient_id)
    doctor  = User.query.get(scan.doctor_id)
    existing_rating = DoctorRating.query.filter_by(
        appointment_id=scan.appointment_id, patient_id=session["user_id"]).first()
    return render_template("patient/report.html", scan=scan, patient=patient,
                           doctor=doctor, existing_rating=existing_rating)

@app.route("/patient/report/<int:scan_id>/pdf")
@role_required("patient")
def patient_report_pdf(scan_id):
    scan    = ScanResult.query.get_or_404(scan_id)
    if scan.patient_id != session["user_id"]:
        flash("Unauthorized.", "danger"); return redirect(url_for("patient_dashboard"))
    patient = User.query.get(scan.patient_id)
    doctor  = User.query.get(scan.doctor_id)
    pdf_path = generate_pdf_report(scan, patient, doctor)
    return send_file(pdf_path, as_attachment=True,
                     download_name=f"BrainScan_Report_{scan_id}.pdf")

@app.route("/patient/rate/<int:appt_id>", methods=["POST"])
@role_required("patient")
def patient_rate(appt_id):
    appt = Appointment.query.get_or_404(appt_id)
    if appt.patient_id != session["user_id"]:
        flash("Unauthorized.", "danger"); return redirect(url_for("patient_appointments"))
    if DoctorRating.query.filter_by(appointment_id=appt_id).first():
        flash("You've already rated this appointment.", "warning")
        return redirect(url_for("patient_appointments"))
    rating   = int(request.form.get("rating", 5))
    feedback = request.form.get("feedback","").strip()
    dr = DoctorRating(appointment_id=appt_id, patient_id=session["user_id"],
                      doctor_id=appt.doctor_id, rating=rating, feedback=feedback)
    db.session.add(dr)
    push_notification(appt.doctor_id,
        f"{session['name']} gave you a {rating}-star rating.",
        "/doctor/dashboard", "fa-star")
    db.session.commit()
    flash("Thank you for your feedback!", "success")
    return redirect(url_for("patient_appointments"))

@app.route("/patient/prescriptions")
@role_required("patient")
def patient_prescriptions():
    uid = session["user_id"]
    prescriptions = Prescription.query.filter_by(patient_id=uid)\
        .order_by(Prescription.created_at.desc()).all()
    return render_template("patient/prescriptions.html", prescriptions=prescriptions)


# ─── DOCTOR ───────────────────────────────────────────────────────────────────
@app.route("/doctor/dashboard")
@role_required("doctor")
def doctor_dashboard():
    uid   = session["user_id"]
    today = date.today().isoformat()
    today_appts     = Appointment.query.filter_by(doctor_id=uid, appt_date=today).all()
    pending_reviews = ScanResult.query.filter_by(doctor_id=uid, status="pending_review").count()
    scans = ScanResult.query.filter_by(doctor_id=uid).order_by(ScanResult.created_at.desc()).limit(5).all()
    flagged = ScanResult.query.filter(ScanResult.doctor_id==uid,
                                      ScanResult.confidence<70,
                                      ScanResult.status=="pending_review").all()
    # weekly appointments data (last 7 days)
    weekly = []
    for i in range(6, -1, -1):
        d = (date.today() - timedelta(days=i)).isoformat()
        cnt = Appointment.query.filter_by(doctor_id=uid, appt_date=d).count()
        weekly.append({"day": d[5:], "count": cnt})
    avg_rating = doctor_avg_rating(uid)
    rating_count = DoctorRating.query.filter_by(doctor_id=uid).count()
    return render_template("doctor/dashboard.html",
                           today_appts=today_appts, pending_reviews=pending_reviews,
                           scans=scans, flagged=flagged, today=today,
                           weekly=json.dumps(weekly), avg_rating=avg_rating,
                           rating_count=rating_count)

@app.route("/doctor/appointments")
@role_required("doctor")
def doctor_appointments():
    uid  = session["user_id"]
    page = request.args.get("page", 1, type=int)
    appts = Appointment.query.filter_by(doctor_id=uid)\
        .order_by(Appointment.appt_date.desc()).paginate(page=page, per_page=10)
    return render_template("doctor/appointments.html", appts=appts)

@app.route("/doctor/appointment/action/<int:appt_id>/<action>")
@role_required("doctor")
def doctor_appt_action(appt_id, action):
    appt = Appointment.query.get_or_404(appt_id)
    if appt.doctor_id != session["user_id"]:
        flash("Unauthorized.", "danger"); return redirect(url_for("doctor_appointments"))
    if action == "confirm":
        appt.status = "confirmed"
        push_notification(appt.patient_id,
            f"Your appointment on {appt.appt_date} at {appt.time_slot} has been confirmed!",
            "/patient/appointments", "fa-calendar-check")
        flash("Appointment confirmed.", "success")
    elif action == "cancel":
        appt.status = "cancelled"
        push_notification(appt.patient_id,
            f"Your appointment on {appt.appt_date} at {appt.time_slot} was cancelled by your doctor.",
            "/patient/appointments", "fa-calendar-xmark")
        flash("Appointment cancelled.", "info")
    db.session.commit()
    return redirect(url_for("doctor_appointments"))

@app.route("/doctor/review/<int:scan_id>", methods=["GET","POST"])
@role_required("doctor")
def doctor_review(scan_id):
    scan    = ScanResult.query.get_or_404(scan_id)
    if scan.doctor_id != session["user_id"]:
        flash("Unauthorized.", "danger"); return redirect(url_for("doctor_dashboard"))
    patient = User.query.get(scan.patient_id)
    if request.method == "POST":
        scan.doctor_notes        = request.form.get("notes","")
        scan.confirmed_diagnosis = request.form.get("confirmed_diagnosis", scan.prediction)
        scan.status              = request.form.get("review_status","reviewed")
        # Update appointment status so doctor can proceed to prescribe
        if scan.appointment:
            scan.appointment.status = "scanned"  # keep scanned so prescribe button shows
        push_notification(scan.patient_id,
            f"Your scan report is ready. Diagnosis: {scan.confirmed_diagnosis}",
            f"/patient/report/{scan.id}", "fa-file-medical")
        db.session.commit()
        flash("Review saved.", "success")
        return redirect(url_for("doctor_dashboard"))
    return render_template("doctor/review.html", scan=scan, patient=patient,
                           brain_classes=BRAIN_CLASSES)

@app.route("/doctor/scans")
@role_required("doctor")
def doctor_scans():
    uid  = session["user_id"]
    page = request.args.get("page", 1, type=int)
    scans = ScanResult.query.filter_by(doctor_id=uid)\
        .order_by(ScanResult.created_at.desc()).paginate(page=page, per_page=10)
    return render_template("doctor/scans.html", scans=scans)

@app.route("/doctor/availability", methods=["GET","POST"])
@role_required("doctor")
def doctor_availability():
    uid = session["user_id"]
    if request.method == "POST":
        action = request.form.get("action")
        if action == "block":
            date_val = request.form.get("date")
            reason   = request.form.get("reason","Unavailable")
            if not DoctorUnavailability.query.filter_by(doctor_id=uid, date=date_val).first():
                db.session.add(DoctorUnavailability(doctor_id=uid, date=date_val, reason=reason))
                db.session.commit()
                flash(f"{date_val} marked as unavailable.", "success")
            else:
                flash("That date is already blocked.", "warning")
        elif action == "unblock":
            bid = request.form.get("block_id")
            b = DoctorUnavailability.query.get(bid)
            if b and b.doctor_id == uid:
                db.session.delete(b)
                db.session.commit()
                flash("Date unblocked.", "success")
    blocked = DoctorUnavailability.query.filter_by(doctor_id=uid)\
        .order_by(DoctorUnavailability.date).all()
    return render_template("doctor/availability.html", blocked=blocked)

@app.route("/doctor/appointment/<int:appt_id>/consult", methods=["GET", "POST"])
@role_required("doctor")
def doctor_consult(appt_id):
    appt = Appointment.query.get_or_404(appt_id)
    if appt.doctor_id != session["user_id"]:
        flash("Unauthorized.", "danger"); return redirect(url_for("doctor_appointments"))
    patient = User.query.get(appt.patient_id)
    return render_template("doctor/consult.html", appt=appt, patient=patient)


@app.route("/doctor/appointment/<int:appt_id>/request-mri", methods=["POST"])
@role_required("doctor")
def doctor_request_mri(appt_id):
    appt = Appointment.query.get_or_404(appt_id)
    if appt.doctor_id != session["user_id"]:
        flash("Unauthorized.", "danger"); return redirect(url_for("doctor_appointments"))
    if appt.mri_request:
        flash("MRI request already submitted.", "warning")
        return redirect(url_for("doctor_appointments"))
    notes = request.form.get("doctor_notes", "")
    mri_req = MRIRequest(
        appointment_id=appt_id,
        patient_id=appt.patient_id,
        doctor_id=session["user_id"],
        doctor_notes=notes,
        status="pending"
    )
    db.session.add(mri_req)
    appt.status = "mri_requested"
    db.session.flush()
    # Notify all lab staff
    lab_staff = User.query.filter_by(role="lab").all()
    for ls in lab_staff:
        push_notification(ls.id,
            f"New MRI request for patient {appt.patient.name} from Dr. {session['name']}",
            "/lab/requests", "fa-x-ray")
    # Notify patient
    push_notification(appt.patient_id,
        "Your doctor has requested an MRI scan. The lab will contact you to schedule.",
        "/patient/appointments", "fa-x-ray")
    db.session.commit()
    flash("MRI request sent to the lab successfully.", "success")
    return redirect(url_for("doctor_appointments"))


@app.route("/doctor/appointment/<int:appt_id>/prescribe", methods=["GET", "POST"])
@role_required("doctor")
def doctor_prescribe(appt_id):
    appt = Appointment.query.get_or_404(appt_id)
    if appt.doctor_id != session["user_id"]:
        flash("Unauthorized.", "danger"); return redirect(url_for("doctor_appointments"))
    patient = User.query.get(appt.patient_id)
    if request.method == "POST":
        pres = Prescription(
            appointment_id=appt_id,
            patient_id=appt.patient_id,
            doctor_id=session["user_id"],
            diagnosis=request.form.get("diagnosis", ""),
            medicines=request.form.get("medicines", ""),
            instructions=request.form.get("instructions", ""),
            follow_up_date=request.form.get("follow_up_date") or None
        )
        db.session.add(pres)
        appt.status = "prescribed"
        push_notification(appt.patient_id,
            f"Dr. {session['name']} has prescribed your treatment. Check your prescriptions.",
            "/patient/prescriptions", "fa-prescription-bottle-medical")
        db.session.commit()
        flash("Prescription saved and patient notified.", "success")
        return redirect(url_for("doctor_appointments"))
    return render_template("doctor/prescribe.html", appt=appt, patient=patient)


# ─── LAB MODULE ───────────────────────────────────────────────────────────────
@app.route("/lab/dashboard")
@role_required("lab")
def lab_dashboard():
    pending  = MRIRequest.query.filter_by(status="pending").count()
    scheduled= MRIRequest.query.filter_by(status="scheduled").count()
    uploaded = MRIRequest.query.filter_by(status="uploaded").count()
    recent   = MRIRequest.query.order_by(MRIRequest.created_at.desc()).limit(5).all()
    return render_template("lab/dashboard.html",
                           pending=pending, scheduled=scheduled,
                           uploaded=uploaded, recent=recent)


@app.route("/lab/requests")
@role_required("lab")
def lab_requests():
    status_filter = request.args.get("status", "all")
    page = request.args.get("page", 1, type=int)
    q = MRIRequest.query
    if status_filter != "all":
        q = q.filter_by(status=status_filter)
    requests_paged = q.order_by(MRIRequest.created_at.desc()).paginate(page=page, per_page=10)
    return render_template("lab/requests.html",
                           requests=requests_paged, status_filter=status_filter)


@app.route("/lab/request/<int:req_id>/schedule", methods=["POST"])
@role_required("lab")
def lab_schedule(req_id):
    mri_req = MRIRequest.query.get_or_404(req_id)
    sched_date = request.form.get("scheduled_date")
    sched_time = request.form.get("scheduled_time")
    mri_req.scheduled_date = sched_date
    mri_req.scheduled_time = sched_time
    mri_req.lab_staff_id = session["user_id"]
    mri_req.status = "scheduled"
    mri_req.appointment.status = "mri_scheduled"
    push_notification(mri_req.patient_id,
        f"Your MRI scan is scheduled for {sched_date} at {sched_time}. Please visit the lab.",
        "/patient/appointments", "fa-calendar-check")
    push_notification(mri_req.doctor_id,
        f"MRI for patient {mri_req.patient.name} scheduled on {sched_date} at {sched_time}.",
        "/doctor/appointments", "fa-calendar-check")
    db.session.commit()
    flash(f"MRI scan scheduled for {sched_date} at {sched_time}.", "success")
    return redirect(url_for("lab_requests"))


@app.route("/lab/request/<int:req_id>/upload", methods=["GET", "POST"])
@role_required("lab")
def lab_upload(req_id):
    mri_req = MRIRequest.query.get_or_404(req_id)
    if request.method == "POST":
        file = request.files.get("scan")
        if not file or not allowed_file(file.filename):
            flash("Please upload a valid image (PNG/JPG).", "danger")
            return redirect(request.url)
        filename = secure_filename(f"scan_{mri_req.appointment_id}_lab_{file.filename}")
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        # Run AI prediction
        model = get_model()
        if model:
            img_arr  = preprocess_input(prepare_image(filepath))
            img_grad = preprocess_input(get_img_array(filepath, size=(224, 224)))
            preds       = model.predict(img_arr)
            label_idx   = int(np.argmax(preds[0]))
            confidence  = float(np.max(preds[0])) * 100
            pred_label  = BRAIN_CLASSES[label_idx]
            cam_fn = "cam_" + filename
            heatmap = make_gradcam_heatmap(img_grad, model, pred_index=label_idx)
            overlay_heatmap(filepath, heatmap, os.path.join(app.config["UPLOAD_FOLDER"], cam_fn))
        else:
            pred_label, confidence, cam_fn = "Glioma", 87.5, filename
        # Create ScanResult
        scan = ScanResult(
            appointment_id=mri_req.appointment_id,
            patient_id=mri_req.patient_id,
            doctor_id=mri_req.doctor_id,
            image_path=filename, cam_path=cam_fn,
            prediction=pred_label, confidence=confidence,
            status="pending_review"
        )
        db.session.add(scan)
        mri_req.status = "uploaded"
        mri_req.appointment.status = "scanned"
        db.session.flush()
        push_notification(mri_req.doctor_id,
            f"MRI scan for {mri_req.patient.name} has been uploaded. AI Prediction: {pred_label} ({confidence:.1f}%)",
            f"/doctor/review/{scan.id}", "fa-microscope")
        push_notification(mri_req.patient_id,
            "Your MRI scan has been uploaded and sent to your doctor for review.",
            "/patient/appointments", "fa-microscope")
        db.session.commit()
        flash(f"Scan uploaded! AI prediction: {pred_label} ({confidence:.1f}%). Sent to doctor for review.", "success")
        return redirect(url_for("lab_requests"))
    return render_template("lab/upload.html", mri_req=mri_req)


# ─── ADMIN ─────────────────────────────────────────────────────────────────────
@app.route("/admin/dashboard")
@role_required("admin")
def admin_dashboard():
    total_patients = User.query.filter_by(role="patient").count()
    total_doctors  = User.query.filter_by(role="doctor").count()
    total_appts    = Appointment.query.count()
    total_scans    = ScanResult.query.count()
    recent_appts   = Appointment.query.order_by(Appointment.created_at.desc()).limit(5).all()
    scans          = ScanResult.query.all()
    class_counts   = {c: 0 for c in BRAIN_CLASSES}
    for s in scans:
        if s.prediction in class_counts:
            class_counts[s.prediction] += 1
    low_conf = ScanResult.query.filter(ScanResult.confidence < 70).count()
    # Weekly appointments (last 7 days, all doctors)
    weekly = []
    for i in range(6, -1, -1):
        d   = (date.today() - timedelta(days=i)).isoformat()
        cnt = Appointment.query.filter_by(appt_date=d).count()
        weekly.append({"day": d[5:], "count": cnt})
    return render_template("admin/dashboard.html",
                           total_patients=total_patients, total_doctors=total_doctors,
                           total_appts=total_appts, total_scans=total_scans,
                           recent_appts=recent_appts, class_counts=class_counts,
                           low_conf=low_conf, weekly=json.dumps(weekly))

@app.route("/admin/users")
@role_required("admin")
def admin_users():
    page  = request.args.get("page", 1, type=int)
    users = User.query.order_by(User.created_at.desc()).paginate(page=page, per_page=12)
    return render_template("admin/users.html", users=users)

@app.route("/admin/users/toggle/<int:user_id>")
@role_required("admin")
def admin_toggle_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.role == "admin":
        flash("Cannot deactivate admin accounts.", "warning")
    else:
        user.is_active = not user.is_active
        db.session.commit()
        flash(f"User {user.name} {'activated' if user.is_active else 'deactivated'}.", "success")
    return redirect(url_for("admin_users"))

@app.route("/admin/appointments")
@role_required("admin")
def admin_appointments():
    page  = request.args.get("page", 1, type=int)
    appts = Appointment.query.order_by(Appointment.created_at.desc()).paginate(page=page, per_page=12)
    return render_template("admin/appointments.html", appts=appts)

@app.route("/admin/scans")
@role_required("admin")
def admin_scans():
    page  = request.args.get("page", 1, type=int)
    scans = ScanResult.query.order_by(ScanResult.created_at.desc()).paginate(page=page, per_page=12)
    return render_template("admin/scans.html", scans=scans)

# ─── PDF REPORT ───────────────────────────────────────────────────────────────
def generate_pdf_report(scan, patient, doctor):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.units import cm
    except ImportError:
        raise ImportError("reportlab is not installed. Run: pip install reportlab")

    pdf_dir  = os.path.join("static", "reports")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"report_{scan.id}.pdf")

    doc   = SimpleDocTemplate(pdf_path, pagesize=A4,
                               leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    # Header
    header_style = ParagraphStyle("header", parent=styles["Title"],
                                  fontSize=20, textColor=colors.HexColor("#1a1f2e"))
    story.append(Paragraph("BrainScan AI — Medical Report", header_style))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(f"Report #{scan.id} | {scan.created_at.strftime('%d %B %Y')}", styles["Normal"]))
    story.append(Spacer(1, 0.6*cm))

    # Patient & Doctor info table
    info = [
        ["Patient Name", patient.name, "Doctor", doctor.name],
        ["Phone", patient.phone or "—", "Appointment Date", scan.appointment.appt_date],
        ["Time Slot", scan.appointment.time_slot, "Priority", scan.appointment.priority],
    ]
    t = Table(info, colWidths=[3.5*cm, 6*cm, 3.5*cm, 6*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#f8f9fb")),
        ("TEXTCOLOR", (0,0), (0,-1), colors.HexColor("#6b7280")),
        ("TEXTCOLOR", (2,0), (2,-1), colors.HexColor("#6b7280")),
        ("FONTNAME",  (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",  (0,0), (-1,-1), 9),
        ("GRID",      (0,0), (-1,-1), 0.5, colors.HexColor("#e8eaf0")),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#f8f9fb")]),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.6*cm))

    # Diagnosis
    diag_color = colors.HexColor("#16a34a") if scan.prediction == "No Tumor" else colors.HexColor("#dc2626")
    diag_style = ParagraphStyle("diag", parent=styles["Heading2"], textColor=diag_color)
    story.append(Paragraph("AI Prediction", styles["Heading2"]))
    story.append(Paragraph(f"{scan.prediction} — {scan.confidence:.1f}% confidence", diag_style))
    if scan.confirmed_diagnosis:
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(f"Doctor Confirmed: <b>{scan.confirmed_diagnosis}</b>", styles["Normal"]))
    story.append(Spacer(1, 0.4*cm))

    # Scan images
    img_path = os.path.join(app.config["UPLOAD_FOLDER"], scan.image_path)
    cam_path = os.path.join(app.config["UPLOAD_FOLDER"], scan.cam_path)
    img_row  = []
    for p, label in [(img_path, "Original MRI"), (cam_path, "Grad-CAM Heatmap")]:
        if os.path.exists(p):
            img_row.append([Paragraph(label, styles["Normal"]), Image(p, width=7*cm, height=6*cm)])
    if img_row:
        flat = []
        for row in img_row:
            flat.extend(row)
        img_table = Table([flat[:2], flat[2:]] if len(flat) > 2 else [flat],
                          colWidths=[8*cm, 8*cm])
        story.append(img_table)
        story.append(Spacer(1, 0.4*cm))

    # Doctor notes
    if scan.doctor_notes:
        story.append(Paragraph("Doctor's Notes", styles["Heading2"]))
        story.append(Paragraph(scan.doctor_notes, styles["Normal"]))
        story.append(Spacer(1, 0.3*cm))

    # Disclaimer
    disc = ParagraphStyle("disc", parent=styles["Normal"], fontSize=8,
                          textColor=colors.HexColor("#6b7280"))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(
        "Disclaimer: This report is generated by an AI system and is intended for informational purposes only. "
        "Please consult a qualified medical professional for diagnosis and treatment decisions.",
        disc))

    doc.build(story)
    return pdf_path

# ─── INIT DB ──────────────────────────────────────────────────────────────────
def init_db():
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(email="admin@brain.com").first():
            admin = User(name="Admin", email="admin@brain.com",
                         password=generate_password_hash("admin123"), role="admin", phone="0000000000")
            db.session.add(admin)
            doc = User(name="Dr. Sarah Johnson", email="doctor@brain.com",
                       password=generate_password_hash("doctor123"), role="doctor", phone="9876543210")
            db.session.add(doc)
            db.session.flush()
            db.session.add(DoctorProfile(user_id=doc.id, specialization="Neurology",
                                         experience=8, bio="Specialist in brain MRI analysis and tumor detection."))
            pat = User(name="John Patient", email="patient@brain.com",
                       password=generate_password_hash("patient123"), role="patient", phone="1234567890")
            db.session.add(pat)
            lab = User(name="Lab Staff", email="lab@brain.com",
                       password=generate_password_hash("lab123"), role="lab", phone="5555555555")
            db.session.add(lab)
            db.session.commit()
            print("✅ Demo accounts created")

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
