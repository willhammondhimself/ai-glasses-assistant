"""Gmail API service for email operations."""
import os
import base64
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

# Optional: Google API client
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    logger.warning("Google API client not installed. Run: pip install google-api-python-client google-auth-oauthlib")

SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.modify'
]


@dataclass
class EmailMessage:
    """Represents an email message."""
    id: str
    thread_id: str
    subject: str
    sender: str
    sender_email: str
    snippet: str
    body: str
    date: datetime
    is_unread: bool
    labels: List[str]

    def to_voice_summary(self) -> str:
        """Return voice-friendly summary."""
        sender_name = self.sender.split('<')[0].strip() if '<' in self.sender else self.sender
        return f"From {sender_name}: {self.subject}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "subject": self.subject,
            "sender": self.sender,
            "sender_email": self.sender_email,
            "snippet": self.snippet,
            "body": self.body[:500] if self.body else "",
            "date": self.date.isoformat(),
            "is_unread": self.is_unread,
            "labels": self.labels
        }


class GmailService:
    """Gmail API wrapper for voice-controlled email."""

    def __init__(self, credentials_path: Optional[str] = None, token_path: Optional[str] = None):
        """Initialize Gmail service.

        Args:
            credentials_path: Path to credentials.json from Google Cloud Console
            token_path: Path to store/load OAuth token
        """
        self.credentials_path = credentials_path or os.environ.get(
            "GMAIL_CREDENTIALS_PATH",
            os.path.expanduser("~/.wham/gmail_credentials.json")
        )
        self.token_path = token_path or os.environ.get(
            "GMAIL_TOKEN_PATH",
            os.path.expanduser("~/.wham/gmail_token.json")
        )
        self._service = None
        self._creds = None

    def is_configured(self) -> bool:
        """Check if Gmail API is properly configured."""
        if not GOOGLE_API_AVAILABLE:
            return False
        return os.path.exists(self.credentials_path)

    def is_authenticated(self) -> bool:
        """Check if already authenticated."""
        return os.path.exists(self.token_path)

    async def authenticate(self) -> bool:
        """Authenticate with Gmail API.

        Returns:
            True if authentication successful
        """
        if not GOOGLE_API_AVAILABLE:
            logger.error("Google API client not installed")
            return False

        if not os.path.exists(self.credentials_path):
            logger.error(f"Credentials file not found: {self.credentials_path}")
            return False

        creds = None

        # Load existing token
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)

        # Refresh or get new token
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Save token
            os.makedirs(os.path.dirname(self.token_path), exist_ok=True)
            with open(self.token_path, 'w') as token:
                token.write(creds.to_json())

        self._creds = creds
        self._service = build('gmail', 'v1', credentials=creds)
        logger.info("Gmail API authenticated successfully")
        return True

    def _get_service(self):
        """Get or create Gmail API service."""
        if self._service is None:
            if not GOOGLE_API_AVAILABLE:
                raise RuntimeError("Google API client not installed")

            if not os.path.exists(self.token_path):
                raise RuntimeError("Not authenticated. Call authenticate() first.")

            creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)
            if not creds.valid and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                with open(self.token_path, 'w') as token:
                    token.write(creds.to_json())

            self._service = build('gmail', 'v1', credentials=creds)
            self._creds = creds

        return self._service

    async def get_unread_count(self) -> int:
        """Get count of unread emails in inbox.

        Returns:
            Number of unread emails
        """
        service = self._get_service()
        results = service.users().messages().list(
            userId='me',
            q='is:unread in:inbox',
            maxResults=1
        ).execute()
        return results.get('resultSizeEstimate', 0)

    async def get_unread_emails(self, max_results: int = 5) -> List[EmailMessage]:
        """Get unread emails from inbox.

        Args:
            max_results: Maximum number of emails to fetch

        Returns:
            List of EmailMessage objects
        """
        service = self._get_service()

        # Get message IDs
        results = service.users().messages().list(
            userId='me',
            q='is:unread in:inbox',
            maxResults=max_results
        ).execute()

        messages = results.get('messages', [])
        emails = []

        for msg in messages:
            email = await self._get_message(msg['id'])
            if email:
                emails.append(email)

        return emails

    async def get_emails_from(self, sender: str, max_results: int = 3) -> List[EmailMessage]:
        """Get emails from a specific sender.

        Args:
            sender: Sender name or email to search for
            max_results: Maximum results

        Returns:
            List of EmailMessage objects
        """
        service = self._get_service()

        # Search by sender
        results = service.users().messages().list(
            userId='me',
            q=f'from:{sender}',
            maxResults=max_results
        ).execute()

        messages = results.get('messages', [])
        emails = []

        for msg in messages:
            email = await self._get_message(msg['id'])
            if email:
                emails.append(email)

        return emails

    async def search_emails(self, query: str, max_results: int = 5) -> List[EmailMessage]:
        """Search emails with Gmail query syntax.

        Args:
            query: Gmail search query
            max_results: Maximum results

        Returns:
            List of EmailMessage objects
        """
        service = self._get_service()

        results = service.users().messages().list(
            userId='me',
            q=query,
            maxResults=max_results
        ).execute()

        messages = results.get('messages', [])
        emails = []

        for msg in messages:
            email = await self._get_message(msg['id'])
            if email:
                emails.append(email)

        return emails

    async def _get_message(self, msg_id: str) -> Optional[EmailMessage]:
        """Get full message details.

        Args:
            msg_id: Gmail message ID

        Returns:
            EmailMessage or None
        """
        service = self._get_service()

        try:
            msg = service.users().messages().get(
                userId='me',
                id=msg_id,
                format='full'
            ).execute()

            headers = {h['name']: h['value'] for h in msg['payload'].get('headers', [])}

            # Extract body
            body = self._extract_body(msg['payload'])

            # Parse date
            date_str = headers.get('Date', '')
            try:
                from email.utils import parsedate_to_datetime
                date = parsedate_to_datetime(date_str)
            except Exception:
                date = datetime.now()

            # Extract sender email
            sender = headers.get('From', '')
            sender_email = sender
            if '<' in sender and '>' in sender:
                sender_email = sender.split('<')[1].split('>')[0]

            return EmailMessage(
                id=msg['id'],
                thread_id=msg['threadId'],
                subject=headers.get('Subject', '(No Subject)'),
                sender=sender,
                sender_email=sender_email,
                snippet=msg.get('snippet', ''),
                body=body,
                date=date,
                is_unread='UNREAD' in msg.get('labelIds', []),
                labels=msg.get('labelIds', [])
            )
        except Exception as e:
            logger.error(f"Failed to get message {msg_id}: {e}")
            return None

    def _extract_body(self, payload: dict) -> str:
        """Extract email body from payload."""
        body = ""

        if 'body' in payload and payload['body'].get('data'):
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')
        elif 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    if part['body'].get('data'):
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                        break
                elif 'parts' in part:
                    # Nested multipart
                    body = self._extract_body(part)
                    if body:
                        break

        return body.strip()

    async def send_email(self, to: str, subject: str, body: str) -> bool:
        """Send an email.

        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body (plain text)

        Returns:
            True if sent successfully
        """
        service = self._get_service()

        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')

        try:
            service.users().messages().send(
                userId='me',
                body={'raw': raw}
            ).execute()
            logger.info(f"Email sent to {to}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    async def reply_to_thread(self, thread_id: str, body: str) -> bool:
        """Reply to an email thread.

        Args:
            thread_id: Gmail thread ID
            body: Reply body

        Returns:
            True if sent successfully
        """
        service = self._get_service()

        # Get original message to get Reply-To info
        thread = service.users().threads().get(
            userId='me',
            id=thread_id
        ).execute()

        messages = thread.get('messages', [])
        if not messages:
            return False

        last_msg = messages[-1]
        headers = {h['name']: h['value'] for h in last_msg['payload'].get('headers', [])}

        to = headers.get('Reply-To', headers.get('From', ''))
        subject = headers.get('Subject', '')
        if not subject.lower().startswith('re:'):
            subject = f"Re: {subject}"

        # Create reply
        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject
        message['In-Reply-To'] = headers.get('Message-ID', '')
        message['References'] = headers.get('Message-ID', '')

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')

        try:
            service.users().messages().send(
                userId='me',
                body={'raw': raw, 'threadId': thread_id}
            ).execute()
            logger.info(f"Reply sent to thread {thread_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to send reply: {e}")
            return False

    async def mark_as_read(self, msg_id: str) -> bool:
        """Mark a message as read.

        Args:
            msg_id: Gmail message ID

        Returns:
            True if successful
        """
        service = self._get_service()

        try:
            service.users().messages().modify(
                userId='me',
                id=msg_id,
                body={'removeLabelIds': ['UNREAD']}
            ).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to mark message as read: {e}")
            return False


# Global instance
_gmail_service: Optional[GmailService] = None


def get_gmail_service() -> GmailService:
    """Get or create global Gmail service instance."""
    global _gmail_service
    if _gmail_service is None:
        _gmail_service = GmailService()
    return _gmail_service
