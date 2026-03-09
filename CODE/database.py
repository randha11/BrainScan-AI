# database.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)
    phone = db.Column(db.String(20))
    role = db.Column(db.String(20), default="patient")
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    doctor_profile = db.relationship("DoctorProfile", backref="user", uselist=False)


class DoctorProfile(db.Model):
    __tablename__ = "doctor_profiles"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), unique=True)
    specialization = db.Column(db.String(100), default="Neurology")
    experience = db.Column(db.Integer, default=0)
    bio = db.Column(db.Text, default="")


class Appointment(db.Model):
    __tablename__ = "appointments"
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    doctor_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    appt_date = db.Column(db.String(20), nullable=False)
    time_slot = db.Column(db.String(20), nullable=False)
    reason = db.Column(db.Text)
    priority = db.Column(db.String(20), default="Normal")
    status = db.Column(db.String(30), default="pending")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    patient = db.relationship("User", foreign_keys=[patient_id], backref="patient_appts")
    doctor = db.relationship("User", foreign_keys=[doctor_id], backref="doctor_appts")
    scan = db.relationship("ScanResult", backref="appointment", uselist=False)
    rating = db.relationship("DoctorRating", backref="appointment", uselist=False)


class ScanResult(db.Model):
    __tablename__ = "scan_results"
    id = db.Column(db.Integer, primary_key=True)
    appointment_id = db.Column(db.Integer, db.ForeignKey("appointments.id"))
    patient_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    doctor_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    image_path = db.Column(db.String(256))
    cam_path = db.Column(db.String(256))
    prediction = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    doctor_notes = db.Column(db.Text)
    confirmed_diagnosis = db.Column(db.String(50))
    status = db.Column(db.String(30), default="pending_review")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    patient = db.relationship("User", foreign_keys=[patient_id])
    doctor = db.relationship("User", foreign_keys=[doctor_id])


class Notification(db.Model):
    __tablename__ = "notifications"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    message = db.Column(db.String(300), nullable=False)
    link = db.Column(db.String(200), default="/")
    icon = db.Column(db.String(50), default="fa-bell")
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship("User", foreign_keys=[user_id])


class DoctorUnavailability(db.Model):
    __tablename__ = "doctor_unavailability"
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    date = db.Column(db.String(20), nullable=False)
    reason = db.Column(db.String(200), default="Unavailable")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    doctor = db.relationship("User", foreign_keys=[doctor_id])


class DoctorRating(db.Model):
    __tablename__ = "doctor_ratings"
    id = db.Column(db.Integer, primary_key=True)
    appointment_id = db.Column(db.Integer, db.ForeignKey("appointments.id"), unique=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    doctor_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    rating = db.Column(db.Integer, nullable=False)   # 1-5
    feedback = db.Column(db.Text, default="")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    patient = db.relationship("User", foreign_keys=[patient_id])
    doctor = db.relationship("User", foreign_keys=[doctor_id])


class MRIRequest(db.Model):
    __tablename__ = "mri_requests"
    id = db.Column(db.Integer, primary_key=True)
    appointment_id = db.Column(db.Integer, db.ForeignKey("appointments.id"))
    patient_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    doctor_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    lab_staff_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    # status: pending → scheduled → uploaded → completed
    status = db.Column(db.String(30), default="pending")
    doctor_notes = db.Column(db.Text, default="")   # Doctor's clinical notes for lab
    scheduled_date = db.Column(db.String(20), nullable=True)
    scheduled_time = db.Column(db.String(20), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    patient = db.relationship("User", foreign_keys=[patient_id])
    doctor = db.relationship("User", foreign_keys=[doctor_id])
    lab_staff = db.relationship("User", foreign_keys=[lab_staff_id])
    appointment = db.relationship("Appointment", backref="mri_request", uselist=False)


class Prescription(db.Model):
    __tablename__ = "prescriptions"
    id = db.Column(db.Integer, primary_key=True)
    appointment_id = db.Column(db.Integer, db.ForeignKey("appointments.id"), unique=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    doctor_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    diagnosis = db.Column(db.String(200), default="")
    medicines = db.Column(db.Text, default="")
    instructions = db.Column(db.Text, default="")
    follow_up_date = db.Column(db.String(20), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    patient = db.relationship("User", foreign_keys=[patient_id])
    doctor = db.relationship("User", foreign_keys=[doctor_id])
    appointment = db.relationship("Appointment", backref="prescription", uselist=False)
