import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

class Email:
    sender_mail: str = "100510664@alumnos.uc3m.es"
    sender_pass: str = "swzm cbbb vksy xvjx"
    receiver_mail: str = "cristobalsm1.618@gmail.com"

    def __init__(self):
        self.message = MIMEMultipart()
        self.message["From"] = Email.sender_mail
        self.message["To"] = Email.receiver_mail

    def send_mail(self, subject: str, text_mail: str):
        self.message["Subject"] = subject
        self.message.attach(MIMEText(text_mail, "plain"))
        text = self.message.as_string()
        session = smtplib.SMTP("smtp.gmail.com", 587)
        session.starttls()
        session.login(Email.sender_mail, Email.sender_pass)
        session.sendmail(Email.sender_mail, Email.receiver_mail, text)
        session.quit()

