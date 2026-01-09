"""Email voice tool for Gmail integration."""
import re
import logging
from typing import Optional, Dict, Any
from .base import VoiceTool, VoiceToolResult

logger = logging.getLogger(__name__)


class EmailVoiceTool(VoiceTool):
    """Voice-controlled email via Gmail API."""

    name = "email"
    description = "Check, read, and send emails via Gmail"

    keywords = [
        r"\bemail[s]?\b",
        r"\bmail\b",
        r"\binbox\b",
        r"\bunread\b",
        r"\bsend\s+.*\s+email\b",
        r"\breply\s+to\b",
        r"\bany\s+(new\s+)?messages?\b",
        r"\bcheck\s+(my\s+)?messages?\b",
        r"\bread\s+email\b",
        r"\bwho\s+emailed\b",
    ]

    priority = 8

    def __init__(self):
        # Lazy import to avoid circular imports
        self._gmail = None
        self._pending_send: Optional[Dict[str, Any]] = None

    def _get_gmail(self):
        """Get Gmail service (lazy load)."""
        if self._gmail is None:
            from backend.services.gmail_service import get_gmail_service
            self._gmail = get_gmail_service()
        return self._gmail

    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Execute email command.

        Args:
            query: The user's voice query
            **kwargs: Additional context

        Returns:
            VoiceToolResult with email info
        """
        query_lower = query.lower()
        gmail = self._get_gmail()

        # Check if configured
        if not gmail.is_configured():
            return VoiceToolResult(
                success=False,
                message="Gmail is not configured. Please add your credentials file.",
                data={"needs_setup": True}
            )

        # Check if authenticated
        if not gmail.is_authenticated():
            return VoiceToolResult(
                success=False,
                message="Gmail needs to be authenticated. Please run the setup.",
                data={"needs_auth": True}
            )

        try:
            # Handle confirmation of pending send
            if self._pending_send and any(word in query_lower for word in ["yes", "send it", "confirm", "go ahead"]):
                return await self._confirm_send()

            if self._pending_send and any(word in query_lower for word in ["no", "cancel", "don't", "nevermind"]):
                self._pending_send = None
                return VoiceToolResult(
                    success=True,
                    message="Email cancelled.",
                    data={"action": "cancelled"}
                )

            # Route to appropriate handler
            if self._is_send_request(query_lower):
                return await self._handle_send(query)

            if self._is_reply_request(query_lower):
                return await self._handle_reply(query)

            if self._is_read_from_request(query_lower):
                return await self._handle_read_from(query)

            if self._is_search_request(query_lower):
                return await self._handle_search(query)

            # Default: check unread
            return await self._handle_check_unread()

        except Exception as e:
            logger.error(f"Email tool error: {e}")
            return VoiceToolResult(
                success=False,
                message="Sorry, I had trouble with your email request.",
                data={"error": str(e)}
            )

    def _is_send_request(self, query: str) -> bool:
        """Check if query is a send email request."""
        return any(pattern in query for pattern in [
            "send", "compose", "write an email", "email to"
        ])

    def _is_reply_request(self, query: str) -> bool:
        """Check if query is a reply request."""
        return "reply" in query

    def _is_read_from_request(self, query: str) -> bool:
        """Check if query is asking to read email from someone."""
        patterns = [
            r"read\s+(the\s+)?email\s+from",
            r"email\s+from\s+\w+",
            r"what\s+did\s+\w+\s+(email|send|write)",
            r"any\s+emails?\s+from",
        ]
        return any(re.search(p, query) for p in patterns)

    def _is_search_request(self, query: str) -> bool:
        """Check if query is a search request."""
        patterns = [
            r"search\s+(for\s+)?email",
            r"find\s+email",
            r"emails?\s+about",
        ]
        return any(re.search(p, query) for p in patterns)

    async def _handle_check_unread(self) -> VoiceToolResult:
        """Handle checking unread emails."""
        gmail = self._get_gmail()

        unread_count = await gmail.get_unread_count()

        if unread_count == 0:
            return VoiceToolResult(
                success=True,
                message="Your inbox is clear. No unread emails.",
                data={"unread_count": 0}
            )

        # Get top 3 unread emails
        emails = await gmail.get_unread_emails(max_results=3)

        if not emails:
            return VoiceToolResult(
                success=True,
                message=f"You have {unread_count} unread emails.",
                data={"unread_count": unread_count}
            )

        # Build voice-friendly summary
        summaries = [email.to_voice_summary() for email in emails]
        summary_text = ". ".join(summaries)

        if unread_count > 3:
            message = f"You have {unread_count} unread emails. Top 3: {summary_text}"
        else:
            message = f"You have {unread_count} unread email{'s' if unread_count > 1 else ''}. {summary_text}"

        return VoiceToolResult(
            success=True,
            message=message,
            data={
                "unread_count": unread_count,
                "emails": [e.to_dict() for e in emails]
            }
        )

    async def _handle_read_from(self, query: str) -> VoiceToolResult:
        """Handle reading email from a specific sender."""
        gmail = self._get_gmail()

        # Extract sender name
        sender = self._extract_sender(query)
        if not sender:
            return VoiceToolResult(
                success=False,
                message="Who would you like to read email from?",
                data={"needs_sender": True}
            )

        emails = await gmail.get_emails_from(sender, max_results=1)

        if not emails:
            return VoiceToolResult(
                success=True,
                message=f"I couldn't find any emails from {sender}.",
                data={"sender": sender, "found": False}
            )

        email = emails[0]

        # Read the email content (truncated for voice)
        body_preview = email.snippet if len(email.snippet) < 200 else email.snippet[:200] + "..."
        sender_name = email.sender.split('<')[0].strip() if '<' in email.sender else email.sender

        message = f"{sender_name} wrote: {email.subject}. {body_preview}"

        return VoiceToolResult(
            success=True,
            message=message,
            data={"email": email.to_dict()}
        )

    async def _handle_search(self, query: str) -> VoiceToolResult:
        """Handle email search."""
        gmail = self._get_gmail()

        # Extract search terms
        search_term = self._extract_search_term(query)
        if not search_term:
            return VoiceToolResult(
                success=False,
                message="What would you like to search for?",
                data={"needs_query": True}
            )

        emails = await gmail.search_emails(search_term, max_results=3)

        if not emails:
            return VoiceToolResult(
                success=True,
                message=f"I didn't find any emails matching '{search_term}'.",
                data={"search_term": search_term, "found": False}
            )

        summaries = [email.to_voice_summary() for email in emails]
        message = f"Found {len(emails)} emails about {search_term}. " + ". ".join(summaries)

        return VoiceToolResult(
            success=True,
            message=message,
            data={
                "search_term": search_term,
                "emails": [e.to_dict() for e in emails]
            }
        )

    async def _handle_send(self, query: str) -> VoiceToolResult:
        """Handle sending an email."""
        # Extract recipient and content
        recipient = self._extract_recipient(query)
        content = self._extract_email_content(query)

        if not recipient:
            return VoiceToolResult(
                success=False,
                message="Who would you like to send the email to?",
                data={"needs_recipient": True}
            )

        if not content:
            return VoiceToolResult(
                success=False,
                message=f"What would you like to say to {recipient}?",
                data={"needs_content": True, "recipient": recipient}
            )

        # Store pending send for confirmation
        self._pending_send = {
            "to": recipient,
            "subject": f"Message from WHAM",
            "body": content
        }

        # Ask for confirmation
        preview = content[:100] + "..." if len(content) > 100 else content
        return VoiceToolResult(
            success=True,
            message=f"I'll send to {recipient}: {preview}. Should I send it?",
            data={
                "action": "confirm_send",
                "pending": self._pending_send
            }
        )

    async def _confirm_send(self) -> VoiceToolResult:
        """Confirm and send pending email."""
        if not self._pending_send:
            return VoiceToolResult(
                success=False,
                message="No email pending to send.",
                data={}
            )

        gmail = self._get_gmail()
        success = await gmail.send_email(
            to=self._pending_send["to"],
            subject=self._pending_send["subject"],
            body=self._pending_send["body"]
        )

        recipient = self._pending_send["to"]
        self._pending_send = None

        if success:
            return VoiceToolResult(
                success=True,
                message=f"Email sent to {recipient}.",
                data={"action": "sent", "to": recipient}
            )
        else:
            return VoiceToolResult(
                success=False,
                message="Sorry, I couldn't send the email. Please try again.",
                data={"action": "send_failed"}
            )

    async def _handle_reply(self, query: str) -> VoiceToolResult:
        """Handle replying to an email."""
        gmail = self._get_gmail()

        # Extract who to reply to and content
        sender = self._extract_sender(query)
        content = self._extract_reply_content(query)

        if not sender:
            return VoiceToolResult(
                success=False,
                message="Who would you like to reply to?",
                data={"needs_sender": True}
            )

        # Find the most recent email from this sender
        emails = await gmail.get_emails_from(sender, max_results=1)

        if not emails:
            return VoiceToolResult(
                success=False,
                message=f"I couldn't find any emails from {sender} to reply to.",
                data={"sender": sender}
            )

        if not content:
            return VoiceToolResult(
                success=False,
                message=f"What would you like to say in your reply to {sender}?",
                data={"needs_content": True, "sender": sender}
            )

        # Store pending reply
        email = emails[0]
        self._pending_send = {
            "thread_id": email.thread_id,
            "to": email.sender_email,
            "body": content,
            "is_reply": True
        }

        preview = content[:80] + "..." if len(content) > 80 else content
        return VoiceToolResult(
            success=True,
            message=f"I'll reply to {sender}: {preview}. Should I send it?",
            data={
                "action": "confirm_reply",
                "pending": self._pending_send
            }
        )

    def _extract_sender(self, query: str) -> Optional[str]:
        """Extract sender name from query."""
        patterns = [
            r"from\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
            r"to\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
            r"reply\s+to\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
            r"what\s+did\s+([A-Za-z]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Filter out common words
                if name.lower() not in ["the", "my", "that", "this", "an", "a"]:
                    return name

        return None

    def _extract_recipient(self, query: str) -> Optional[str]:
        """Extract recipient from send query."""
        patterns = [
            r"email\s+(?:to\s+)?([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(?:about|saying|that)",
            r"send\s+(?:an?\s+)?email\s+to\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
            r"to\s+([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})",  # email address
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _extract_email_content(self, query: str) -> Optional[str]:
        """Extract email content from query."""
        patterns = [
            r"saying\s+(.+)$",
            r"about\s+(.+)$",
            r"that\s+says?\s+(.+)$",
            r"message[:\s]+(.+)$",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _extract_reply_content(self, query: str) -> Optional[str]:
        """Extract reply content from query."""
        patterns = [
            r"reply.*saying\s+(.+)$",
            r"say(?:ing)?\s+(.+)$",
            r"tell\s+(?:them|him|her)\s+(.+)$",
            r"with\s+(.+)$",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _extract_search_term(self, query: str) -> Optional[str]:
        """Extract search term from query."""
        patterns = [
            r"search\s+(?:for\s+)?emails?\s+(?:about\s+)?(.+)$",
            r"find\s+emails?\s+(?:about\s+)?(.+)$",
            r"emails?\s+about\s+(.+)$",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None
