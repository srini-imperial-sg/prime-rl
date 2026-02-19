import sqlite3
import threading
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
from dataclasses import dataclass
from typing import Literal

DB_PATH = Path(__file__).resolve().parent / "data" / "enron_emails.db"

@dataclass
class SearchResult:
    message_id: str
    snippet: str

# Email and Scenario data models
class Email(BaseModel):
    message_id: str
    date: str  # ISO 8601 string 'YYYY-MM-DD HH:MM:SS'
    subject: Optional[str] = None
    from_address: Optional[str] = None
    to_addresses: List[str] = []  # Populated from recipients table
    cc_addresses: List[str] = []  # Populated from recipients table
    bcc_addresses: List[str] = []  # Populated from recipients table
    body: Optional[str] = None
    file_name: Optional[str] = None


class FinalAnswer(BaseModel):
    answer: str
    source_ids: list[str]

class Scenario(BaseModel):
    id: int
    question: str
    answer: str
    message_ids: List[str]  # message_ids (strings) of referenced emails
    how_realistic: float
    inbox_address: str
    query_date: str
    split: Literal["train", "test"]

class EmailScenario(BaseModel):
    step: int
    scenario: Scenario

_THREAD_LOCAL = threading.local()
_CONNECTION_LOCK = threading.Lock()


def _initialize_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=5.0)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


def get_db_connection():
    """Get a thread-local database connection, creating one if needed."""
    if not DB_PATH.exists():
        raise ValueError(f"Database file {DB_PATH} does not exist. Please create it first.")

    conn = getattr(_THREAD_LOCAL, "conn", None)
    if conn is not None:
        return conn

    with _CONNECTION_LOCK:
        conn = getattr(_THREAD_LOCAL, "conn", None)
        if conn is None:
            conn = _initialize_connection()
            _THREAD_LOCAL.conn = conn
    return conn

def search_emails_with_keywords(keywords: List[str]) -> List[SearchResult]:
    """
    Searches emails using keywords.

    Args:
        keywords: A list of keywords that must all appear in the subject or body.

    Returns:
        A list of SearchResult objects.
    """
    if not keywords:
        raise ValueError("No keywords provided for search.")
    raise RuntimeError(
        "search_emails_with_keywords requires private context and must be called via EmailEnv."
    )

def search_emails(
    inbox: str,
    keywords: List[str],
    from_addr: Optional[str] = None,
    to_addr: Optional[str] = None,
    sent_after: Optional[str] = None,
    sent_before: Optional[str] = None,
    max_results: int = 10,
) -> List[SearchResult]:
    """
    Searches the email database based on keywords, inbox, sender, recipient, and date range.

    Args:
        inbox: The email address of the user performing the search.
               Results include emails sent from or to (inc. cc/bcc) this address.
        keywords: A list of keywords that must all appear in the subject or body.
        from_addr: Optional email address to filter emails sent *from*.
        to_addr: Optional email address to filter emails sent *to* (inc. cc/bcc).
        sent_after: Optional date string 'YYYY-MM-DD'. Filters for emails sent on or after this date.
        sent_before: Optional date string 'YYYY-MM-DD'. Filters for emails sent before this date.
        max_results: The maximum number of results to return. Cannot exceed 10.

    Returns:
        A list of SearchResult objects, each containing 'message_id' and 'snippet'.
        Returns an empty list if no results are found or an error occurs.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    where_clauses: List[str] = []
    params: List[str | int] = []

    if not keywords:
        raise ValueError("No keywords provided for search.")

    if max_results > 10:
        raise ValueError("max_results must be less than or equal to 10.")

    # FTS5 default is AND, so just join keywords. Escape quotes for safety.
    fts_query = " OR ".join(f""" "{k.replace('"', '""')}" """ for k in keywords)
    where_clauses.append("fts.emails_fts MATCH ?")
    params.append(fts_query)

    # Inbox filter
    where_clauses.append("""
        (e.from_address = ? OR EXISTS (
            SELECT 1 FROM recipients r_inbox
            WHERE r_inbox.recipient_address = ? AND r_inbox.email_id = e.message_id
        ))
    """)
    params.extend([inbox, inbox])

    if from_addr:
        where_clauses.append("e.from_address = ?")
        params.append(from_addr)

    if to_addr:
        where_clauses.append("""
            EXISTS (
                SELECT 1 FROM recipients r_to
                WHERE r_to.recipient_address = ? AND r_to.email_id = e.message_id
            )
        """)
        params.append(to_addr)

    if sent_after:
        where_clauses.append("e.date >= ?")
        params.append(f"{sent_after} 00:00:00")

    if sent_before:
        where_clauses.append("e.date < ?")
        params.append(f"{sent_before} 00:00:00")

    sql = f"""
        SELECT
            e.message_id,
            snippet(emails_fts, -1, '<b>', '</b>', ' ... ', 15) as snippet
        FROM
            emails e JOIN emails_fts fts ON e.id = fts.rowid
        WHERE
            {" AND ".join(where_clauses)}
        ORDER BY
            e.date DESC
        LIMIT ?;
    """

    params.append(max_results)

    cursor.execute(sql, params)
    results = cursor.fetchall()

    return [SearchResult(message_id=row[0], snippet=row[1]) for row in results]

def read_email(message_id: str) -> Optional[Email]:
    """Retrieve a single email by its message_id"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get email details
    cursor.execute(
        "SELECT message_id, date, subject, from_address, body, file_name FROM emails WHERE message_id = ?",
        (message_id,),
    )
    email_row = cursor.fetchone()

    if not email_row:
        return None

    msg_id, date, subject, from_addr, body, file_name = email_row

    # Get recipients
    cursor.execute(
        "SELECT recipient_address, recipient_type FROM recipients WHERE email_id = ?",
        (message_id,),
    )
    recipient_rows = cursor.fetchall()

    to_addresses = []
    cc_addresses = []
    bcc_addresses = []

    for addr, type_val in recipient_rows:
        if type_val.lower() == "to":
            to_addresses.append(addr)
        elif type_val.lower() == "cc":
            cc_addresses.append(addr)
        elif type_val.lower() == "bcc":
            bcc_addresses.append(addr)

    return Email(
        message_id=msg_id,
        date=date,
        subject=subject,
        from_address=from_addr,
        to_addresses=to_addresses,
        cc_addresses=cc_addresses,
        bcc_addresses=bcc_addresses,
        body=body,
        file_name=file_name,
    )






# system_prompt="""You are an email search agent that answers questions from their emails. You have access to search tools that can query the user's email database.

# ## Available Tools:
# {tools}

# ## Search Strategy:
# - Start with broad search terms, then refine based on results
# - Try different keywords, synonyms, or related terms if initial searches don't yield results
# - If a search returns no results, consider alternative search approaches

# ## Required Format:
# Every step MUST follow this exact structure:

# **For search steps:**
# <think>
# [Analyze: What do I know? What's missing? Why am I choosing these specific search terms/tools?]
# </think>
# <tool>
# {{'name': "tool_name", "args": {{"arg1": "value1", "arg2": "value2"}}}}
# </tool>

# **For final answer:**
# You MUST output in this EXACT format with NO extra text between tags:
# <think>Summarize all your findings and explain how they answer the user's question</think>
# <answer>Comprehensive answer with specific details from emails, or "I could not find sufficient information to answer this question" if searches were unsuccessful</answer>
# <email_sources>list of message_ids used to answer the question</email_sources>

# ## Important Guidelines:
# - ALWAYS start each turn with <think> tags containing your reasoning
# - After receiving tool results in <tool_response> tags, analyze them in your next <think> section
# - If your answer references information from emails, you MUST include those message_ids as sources
# - Include sources even for partial information - if an email contributed to your understanding, cite it
# - If no emails were found or used, put an empty list between <email_sources></email_sources> tags.
# - Be specific in your final answer and answer to the point and do not mention the tools you used."""