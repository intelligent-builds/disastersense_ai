# src/alerts.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import traceback


# Gmail configuration (hardcoded for your project)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "aayushgoyal2503@gmail.com"
SMTP_PASS = "zyoy gape jfbo nqnr"   # <-- your 16-char App Password
FROM_EMAIL = SMTP_USER


def send_email(to_email: str, subject: str, body: str):
    """
    Send an email via Gmail SMTP.
    Returns a dict {"status": "ok"|"error", "detail": "..."}.
    """
    try:
        # Build email message
        msg = MIMEMultipart()
        msg["From"] = FROM_EMAIL
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # Connect to Gmail SMTP
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=15)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(SMTP_USER, SMTP_PASS)

        # Send the email
        server.sendmail(FROM_EMAIL, [to_email], msg.as_string())
        server.quit()

        return {"status": "ok", "detail": f"sent to {to_email} via Gmail"}
    except Exception as e:
        tb = traceback.format_exc()
        return {"status": "error", "detail": str(e), "trace": tb}
