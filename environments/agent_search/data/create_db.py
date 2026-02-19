import sqlite3
from datasets import Features, Sequence, Value, load_dataset
from datetime import datetime
from tqdm import tqdm

DB_PATH = "./enron_emails.db"
EMAIL_DATASET_REPO_ID = "corbt/enron-emails"

def create_email_database():
    """Create the email database from Hugging Face dataset"""
    print("Creating email database from Hugging Face dataset...")
    print(
        "This will download and process the full Enron email dataset - this may take several minutes..."
    )

    # Database schema
    SQL_CREATE_TABLES = """
    DROP TABLE IF EXISTS recipients;
    DROP TABLE IF EXISTS emails_fts;
    DROP TABLE IF EXISTS emails;

    CREATE TABLE emails (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id TEXT UNIQUE,
        subject TEXT,
        from_address TEXT,
        date TEXT,
        body TEXT,
        file_name TEXT
    );

    CREATE TABLE recipients (
        email_id TEXT,
        recipient_address TEXT,
        recipient_type TEXT
    );
    """

    SQL_CREATE_INDEXES_TRIGGERS = """
    CREATE INDEX idx_emails_from ON emails(from_address);
    CREATE INDEX idx_emails_date ON emails(date);
    CREATE INDEX idx_emails_message_id ON emails(message_id);
    CREATE INDEX idx_recipients_address ON recipients(recipient_address);
    CREATE INDEX idx_recipients_type ON recipients(recipient_type);
    CREATE INDEX idx_recipients_email_id ON recipients(email_id);
    CREATE INDEX idx_recipients_address_email ON recipients(recipient_address, email_id);

    CREATE VIRTUAL TABLE emails_fts USING fts5(
        subject,
        body,
        content='emails',
        content_rowid='id'
    );

    CREATE TRIGGER emails_ai AFTER INSERT ON emails BEGIN
        INSERT INTO emails_fts (rowid, subject, body)
        VALUES (new.id, new.subject, new.body);
    END;

    CREATE TRIGGER emails_ad AFTER DELETE ON emails BEGIN
        DELETE FROM emails_fts WHERE rowid=old.id;
    END;

    CREATE TRIGGER emails_au AFTER UPDATE ON emails BEGIN
        UPDATE emails_fts SET subject=new.subject, body=new.body WHERE rowid=old.id;
    END;
    """

    # Create database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript(SQL_CREATE_TABLES)
    conn.commit()

    # Load dataset
    print("Loading full email dataset...")
    expected_features = Features(
        {
            "message_id": Value("string"),
            "subject": Value("string"),
            "from": Value("string"),
            "to": Sequence(Value("string")),
            "cc": Sequence(Value("string")),
            "bcc": Sequence(Value("string")),
            "date": Value("timestamp[us]"),
            "body": Value("string"),
            "file_name": Value("string"),
        }
    )

    dataset = load_dataset(
        EMAIL_DATASET_REPO_ID, features=expected_features, split="train"
    )
    print(f"Dataset contains {len(dataset)} total emails")

    # Populate database with ALL emails (not limited to 1000)
    print("Populating database with all emails...")
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = MEMORY;")
    conn.execute("BEGIN TRANSACTION;")

    record_count = 0
    skipped_count = 0
    duplicate_count = 0
    processed_emails = set()  # Track (subject, body, from) tuples for deduplication

    for email_data in tqdm(dataset, desc="Inserting emails"):
        message_id = email_data["message_id"]
        subject = email_data["subject"]
        from_address = email_data["from"]
        date_obj: datetime = email_data["date"]
        body = email_data["body"]
        file_name = email_data["file_name"]
        to_list = [str(addr) for addr in email_data["to"] if addr]
        cc_list = [str(addr) for addr in email_data["cc"] if addr]
        bcc_list = [str(addr) for addr in email_data["bcc"] if addr]

        # Apply the same filters as the original project
        total_recipients = len(to_list) + len(cc_list) + len(bcc_list)

        # Filter out very long emails and those with too many recipients
        if len(body) > 5000:
            skipped_count += 1
            continue

        if total_recipients > 30:
            skipped_count += 1
            continue

        # Deduplication check (same as original project)
        email_key = (subject, body, from_address)
        if email_key in processed_emails:
            duplicate_count += 1
            continue
        else:
            processed_emails.add(email_key)

        date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(
            """
            INSERT INTO emails (message_id, subject, from_address, date, body, file_name)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (message_id, subject, from_address, date_str, body, file_name),
        )

        # Insert recipients
        recipient_data = []
        for addr in to_list:
            recipient_data.append((message_id, addr, "to"))
        for addr in cc_list:
            recipient_data.append((message_id, addr, "cc"))
        for addr in bcc_list:
            recipient_data.append((message_id, addr, "bcc"))

        if recipient_data:
            cursor.executemany(
                """
                INSERT INTO recipients (email_id, recipient_address, recipient_type)
                VALUES (?, ?, ?)
            """,
                recipient_data,
            )

        record_count += 1

    conn.commit()

    # Create indexes and triggers
    print("Creating indexes and FTS...")
    cursor.executescript(SQL_CREATE_INDEXES_TRIGGERS)
    cursor.execute('INSERT INTO emails_fts(emails_fts) VALUES("rebuild")')
    conn.commit()

    print(f"Successfully created database with {record_count} emails.")
    print(f"Skipped {skipped_count} emails due to length/recipient limits.")
    print(f"Skipped {duplicate_count} duplicate emails.")
    return conn

if __name__ == "__main__":
    create_email_database()
    